//! Model context-window / output-token limits catalog.
//!
//! Resolves `ModelLimits { context, output }` for a model id. Resolution
//! order mirrors opencode's `models.dev` integration (best-effort,
//! never blocks the agent loop):
//!
//! 1. **In-memory table** — populated from one trusted `models.dev`
//!    provider (default: `opencode`) by a background fetch of
//!    `https://models.dev/api.json`. On the first `resolve()` call we
//!    `tokio::spawn` a fire-and-forget fetch; while it is in flight
//!    (typically during early LLM warm-up, before compaction ever needs
//!    the number) callers fall through to step 2/3.
//! 2. **Fallback string-match table** — the legacy hand-encoded table,
//!    kept as the offline / pre-fetch fallback so behavior never
//!    regresses vs. the old `resolve_context_window_tokens`.
//! 3. **Conservative default** — `{ context: 128_000, output: 8_192 }`.
//!
//! Disk cache: a successful fetch writes
//! `<cache_dir>/agent-harness-rs/models.json` (tempfile + atomic
//! `rename`, no cross-process lock — worst case is two processes each
//! fetch once). TTL 5 minutes. On startup we eagerly load the disk
//! cache into the in-memory table so a transient network blip during a
//! session still serves real values.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// `models.dev` endpoint. Overridable via the `AGENT_HARNESS_MODELS_URL`
/// env var (mirrors opencode's `OPENCODE_MODELS_URL`).
const DEFAULT_MODELS_URL: &str = "https://models.dev/api.json";
const ENV_MODELS_URL: &str = "AGENT_HARNESS_MODELS_URL";
/// models.dev provider used as the single trusted source for model ids.
/// OpenCode's catalog uses the short model ids most OpenAI-compatible
/// gateways expose (`glm-5.2`, `gpt-5.5`, `deepseek-v4-pro`, ...), and covers
/// the long-context coding models this harness targets. Override only if the
/// host intentionally mirrors another provider's naming scheme.
const DEFAULT_MODELS_PROVIDER: &str = "opencode";
const ENV_MODELS_PROVIDER: &str = "AGENT_HARNESS_MODELS_PROVIDER";

/// Disk cache TTL — matches opencode. A fresh fetch within this window
/// is skipped; beyond it we re-fetch.
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

/// Per-request timeout for the HTTP fetch.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Max retry attempts (on top of the initial try) with exponential
/// backoff: 200ms, 400ms, 800ms (+ jitter).
const MAX_RETRIES: u32 = 3;
const BACKOFF_BASE_MS: u64 = 200;

/// Conservative fallback when nothing is known about the model.
pub const DEFAULT_CONTEXT_TOKENS: u64 = 128_000;
pub const DEFAULT_OUTPUT_TOKENS: u64 = 8_192;

/// Resolved limits for a single model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLimits {
    /// Maximum prompt + completion context window (input tokens).
    pub context: u64,
    /// Maximum output tokens for a single completion.
    pub output: u64,
}

impl ModelLimits {
    pub const fn default_fallback() -> Self {
        Self {
            context: DEFAULT_CONTEXT_TOKENS,
            output: DEFAULT_OUTPUT_TOKENS,
        }
    }
}

// ---------------------------------------------------------------------------
// models.dev wire shapes — we deserialize only the fields we consume.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CatalogRoot(HashMap<String, ProviderEntry>);

#[derive(Debug, Deserialize)]
struct ProviderEntry {
    #[serde(default)]
    models: HashMap<String, ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    limit: Option<LimitEntry>,
}

#[derive(Debug, Deserialize)]
struct LimitEntry {
    #[serde(default)]
    context: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

// ---------------------------------------------------------------------------
// In-memory catalog (process-global, lazily initialized).
// ---------------------------------------------------------------------------

struct CatalogState {
    /// model_id (lowercased) -> limits, sourced from the trusted models.dev
    /// provider. We intentionally do not flatten all providers: provider-local
    /// model ids collide (`glm-5.2` is 1M on Z.AI/OpenCode but 256K on
    /// Scaleway), so flattening lets unrelated gateways corrupt each other.
    table: HashMap<String, ModelLimits>,
    /// True once a fetch has been attempted (success or final failure),
    /// so we don't keep re-spawning fire-and-forget tasks on every
    /// `resolve()` call when there is no runtime / the endpoint is down.
    fetch_attempted: bool,
}

static CATALOG: OnceLock<RwLock<CatalogState>> = OnceLock::new();

fn catalog() -> &'static RwLock<CatalogState> {
    CATALOG.get_or_init(|| {
        RwLock::new(CatalogState {
            #[cfg(not(test))]
            table: load_disk_cache_into_memory(),
            #[cfg(test)]
            table: HashMap::new(),
            fetch_attempted: false,
        })
    })
}

/// Eagerly load any valid disk cache into the in-memory table so a
/// network failure during this process still serves real values. Errors
/// are silently ignored (best-effort — the fallback table covers us).
#[cfg_attr(test, allow(dead_code))]
fn load_disk_cache_into_memory() -> HashMap<String, ModelLimits> {
    match (|| -> Option<HashMap<String, ModelLimits>> {
        let path = cache_path()?;
        let bytes = std::fs::read(&path).ok()?;
        let parsed: CatalogRoot = serde_json::from_slice(&bytes).ok()?;
        Some(extract_table(&parsed))
    })() {
        Some(t) => {
            debug!(
                "loaded {} models from disk cache at {:?}",
                t.len(),
                cache_path()
            );
            t
        }
        None => HashMap::new(),
    }
}

fn extract_table(root: &CatalogRoot) -> HashMap<String, ModelLimits> {
    let mut out = HashMap::new();
    let provider_key = trusted_models_provider();
    if let Some(provider) = root.0.get(&provider_key) {
        for (model_id, model) in &provider.models {
            if let Some(limit) = &model.limit {
                let context = limit.context.unwrap_or(DEFAULT_CONTEXT_TOKENS);
                let output = limit.output.unwrap_or(DEFAULT_OUTPUT_TOKENS);
                // Skip degenerate entries — a 0 context window would
                // disable compaction forever; treat as "no data".
                if context == 0 {
                    continue;
                }
                out.insert(
                    model_id.to_ascii_lowercase(),
                    ModelLimits { context, output },
                );
            }
        }
    }
    out
}

fn trusted_models_provider() -> String {
    std::env::var(ENV_MODELS_PROVIDER)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_MODELS_PROVIDER.to_string())
}

/// Resolve limits for a model id. Fast, non-async, never blocks.
///
/// Triggers a one-shot background fetch on the first call (if a tokio
/// runtime is available). While the fetch is in flight the caller gets
/// the fallback-table / default value; subsequent calls pick up the
/// fetched table once it lands.
pub fn resolve_limits(model: &str) -> ModelLimits {
    {
        let guard = match catalog().try_read() {
            Ok(g) => g,
            Err(_) => {
                // Lock contention (fetch is writing) — fall through to
                // fallback rather than blocking the loop. Next call wins.
                return fallback_table_lookup(model);
            }
        };
        if let Some(limits) = guard.table.get(&model.to_ascii_lowercase()) {
            let fallback = fallback_table_lookup(model);
            if fallback == ModelLimits::default_fallback() {
                return *limits;
            }
            return ModelLimits {
                context: limits.context.max(fallback.context),
                output: limits.output.max(fallback.output),
            };
        }
        let attempted = guard.fetch_attempted;
        drop(guard);
        if !attempted {
            trigger_background_fetch_if_needed();
        }
    }
    fallback_table_lookup(model)
}

/// Backwards-compatible single-value accessor — equivalent to
/// `resolve_limits(model).context`. Kept so existing call sites and
/// tests keep working during the rollout.
pub fn resolve_context_window_tokens(model: &str) -> u64 {
    resolve_limits(model).context
}

/// Kick off the background fetch (fire-and-forget). Public so hosts that
/// want to warm the cache before the first turn can call it explicitly.
/// No-op if a fetch has already been attempted.
pub fn prefetch() {
    trigger_background_fetch_if_needed();
}

fn trigger_background_fetch_if_needed() {
    let lock = catalog();
    let attempted = match lock.try_read() {
        Ok(g) => g.fetch_attempted,
        Err(_) => return,
    };
    if attempted {
        return;
    }
    // Only spawn if there is a running tokio runtime; otherwise skip
    // silently (library must not panic just because no runtime exists).
    let handle = match tokio::runtime::Handle::try_current() {
        Ok(h) => h,
        Err(_) => return,
    };
    let _join_handle: tokio::task::JoinHandle<()> = handle.spawn(async move {
        fetch_and_populate().await;
    });
    // Intentionally dropped: fire-and-forget. The fetch runs in the
    // background and populates the catalog; we never await its result.
}

/// Main fetch routine: GET models.dev with backoff-retry, parse, write
/// disk cache (tempfile + rename), and populate the in-memory table.
/// Marks `fetch_attempted = true` regardless of outcome so we don't
/// hammer the endpoint on every `resolve()`.
async fn fetch_and_populate() {
    // Skip if the disk cache is still fresh — avoids redundant fetches
    // when the process restarts within the TTL window.
    if cache_is_fresh() {
        let mut guard = catalog().write().await;
        // Older cache files were written under a synthetic `cached` provider
        // after flattening all gateways. With provider-scoped extraction those
        // parse to an empty table; do not let a fresh-but-empty legacy cache
        // suppress a real fetch.
        if !guard.table.is_empty() {
            guard.fetch_attempted = true;
            return;
        }
    }

    let url = std::env::var(ENV_MODELS_URL).unwrap_or_else(|_| DEFAULT_MODELS_URL.to_string());

    match fetch_with_retry(&url).await {
        Ok(table) => {
            if let Some(path) = cache_path() {
                if let Err(e) = write_disk_cache_atomic(&path, &table).await {
                    debug!("disk cache write failed (non-fatal): {e}");
                }
            }
            let mut guard = catalog().write().await;
            for (k, v) in table {
                guard.table.insert(k, v);
            }
            guard.fetch_attempted = true;
            debug!(
                "models.dev fetch ok; {} models in catalog",
                guard.table.len()
            );
        }
        Err(e) => {
            warn!("models.dev fetch failed after retries, using fallback table: {e}");
            let mut guard = catalog().write().await;
            guard.fetch_attempted = true;
        }
    }
}

async fn fetch_with_retry(url: &str) -> Result<HashMap<String, ModelLimits>, String> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;

    let mut last_err = String::from("no attempt made");
    for attempt in 0..=MAX_RETRIES {
        match client.get(url).send().await {
            Ok(resp) => match resp.json::<CatalogRoot>().await {
                Ok(root) => return Ok(extract_table(&root)),
                Err(e) => last_err = format!("decode failed: {e}"),
            },
            Err(e) => last_err = format!("request failed: {e}"),
        }
        if attempt < MAX_RETRIES {
            let backoff_ms = BACKOFF_BASE_MS * 2u64.pow(attempt);
            let jitter = rand_u64_n(backoff_ms / 2 + 1);
            tokio::time::sleep(Duration::from_millis(backoff_ms + jitter)).await;
        }
    }
    Err(last_err)
}

/// Cheap, dependency-free pseudo-jitter: mix the thread-local instant
/// nanos + thread id into a small range. Good enough to decorrelate
/// concurrent retries; we are not doing crypto here.
fn rand_u64_n(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let tid_hash = format!("{:?}", std::thread::current().id())
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    (nanos ^ tid_hash) % n
}

/// Disk cache path: `<cache_dir>/agent-harness-rs/models.json`.
/// Override the whole path with the `AGENT_HARNESS_CACHE_PATH` env var
/// (primarily for tests, but also lets hosts relocate the cache).
/// Returns `None` if the platform cache dir can't be resolved.
fn cache_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AGENT_HARNESS_CACHE_PATH") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let base = dirs::cache_dir()?;
    Some(base.join("agent-harness-rs").join("models.json"))
}

/// Is the on-disk cache within TTL? Best-effort: any FS error → false.
fn cache_is_fresh() -> bool {
    let path = match cache_path() {
        Some(p) => p,
        None => return false,
    };
    let metadata = match std::fs::metadata(&path) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let mtime = match metadata.modified() {
        Ok(t) => t,
        Err(_) => return false,
    };
    match SystemTime::now().duration_since(mtime) {
        Ok(age) => age < CACHE_TTL,
        Err(_) => false,
    }
}

/// Atomic write: write to `<path>.<pid>.<ts>.tmp` then `rename` over
/// the target. `rename` is atomic on POSIX; readers either see the old
/// file or the new one, never a half-written file.
async fn write_disk_cache_atomic(
    path: &PathBuf,
    table: &HashMap<String, ModelLimits>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let body = serialize_table(table);
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("json.{}.{}.tmp", std::process::id(), ts));
    tokio::fs::write(&tmp, body).await?;
    tokio::fs::rename(&tmp, path).await
}

/// Serialize back to the models.dev `{provider:{models:{limit:{...}}}}`
/// envelope so `load_disk_cache_into_memory` can read our own cache.
/// The provider key is the trusted provider, not a synthetic flattened bucket,
/// so future loads keep the same provider-scoped semantics as network fetches.
fn serialize_table(table: &HashMap<String, ModelLimits>) -> String {
    use serde_json::json;
    let mut models = serde_json::Map::new();
    for (id, limits) in table {
        models.insert(
            id.clone(),
            json!({
                "limit": {
                    "context": limits.context,
                    "output": limits.output,
                }
            }),
        );
    }
    let provider = json!({ "models": serde_json::Value::Object(models) });
    let mut providers = serde_json::Map::new();
    providers.insert(trusted_models_provider(), provider);
    serde_json::to_string(&serde_json::Value::Object(providers)).unwrap_or_else(|_| "{}".into())
}

// ---------------------------------------------------------------------------
// Fallback string-match table (legacy offline resolver).
// ---------------------------------------------------------------------------

/// Hand-encoded table kept as the offline / pre-fetch fallback. Same
/// context values as the legacy `resolve_context_window_tokens`. Output
/// values are conservative provider-typical caps.
fn fallback_table_lookup(model: &str) -> ModelLimits {
    let m = model.to_ascii_lowercase();
    // Claude 4.6 / 4.7 1M context window (Sonnet / Opus extended).
    if m.contains("opus-4-7") || m.contains("opus-4-6") || m.contains("sonnet-4-6") {
        return ModelLimits {
            context: 1_000_000,
            output: 32_000,
        };
    }
    // Anthropic Claude 3.x / 4.x: 200K
    if m.contains("claude") {
        return ModelLimits {
            context: 200_000,
            output: 8_192,
        };
    }
    // OpenAI GPT-4 family: 128K
    if m.contains("gpt-4") || m.contains("gpt-4o") || m.contains("gpt-4.1") {
        return ModelLimits {
            context: 128_000,
            output: 16_384,
        };
    }
    // OpenAI o1 / o3 reasoning: 200K
    if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") {
        return ModelLimits {
            context: 200_000,
            output: 100_000,
        };
    }
    // MiniMax / DeepSeek commonly advertise 1M.
    if m.contains("minimax") || m.contains("deepseek") {
        return ModelLimits {
            context: 1_000_000,
            output: 8_192,
        };
    }
    ModelLimits::default_fallback()
}

// ===========================================================================
// Test helpers: `inject_test_table` lets tests bypass the real models.dev
// fetch and drive the in-memory table directly. Production code never
// touches these.
// ===========================================================================

#[cfg(test)]
async fn inject_test_table(table: HashMap<String, ModelLimits>) {
    let mut guard = catalog().write().await;
    guard.table = table;
    guard.fetch_attempted = true;
}

#[cfg(test)]
fn extract_table_for_test(json: &str) -> HashMap<String, ModelLimits> {
    let root: CatalogRoot = serde_json::from_str(json).unwrap();
    extract_table(&root)
}

/// Serializes tests that mutate the global catalog. Rust runs tests on
/// multiple threads by default; without this lock the shared
/// `OnceLock<RwLock<CatalogState>>` would race across tests.
#[cfg(test)]
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_table_known_models() {
        assert_eq!(fallback_table_lookup("claude-opus-4-7").context, 1_000_000);
        assert_eq!(
            fallback_table_lookup("claude-sonnet-4-6").context,
            1_000_000
        );
        assert_eq!(fallback_table_lookup("claude-haiku-4-5").context, 200_000);
        assert_eq!(fallback_table_lookup("claude-3-5-sonnet").context, 200_000);
        assert_eq!(fallback_table_lookup("gpt-4o").context, 128_000);
        assert_eq!(fallback_table_lookup("gpt-4.1-mini").context, 128_000);
        assert_eq!(fallback_table_lookup("o3-mini").context, 200_000);
        assert_eq!(fallback_table_lookup("MiniMax-M2").context, 1_000_000);
    }

    #[test]
    fn fallback_table_unknown_model_uses_default() {
        let limits = fallback_table_lookup("some-unknown-model-xyz");
        assert_eq!(limits.context, DEFAULT_CONTEXT_TOKENS);
        assert_eq!(limits.output, DEFAULT_OUTPUT_TOKENS);
    }

    #[test]
    fn fallback_table_case_insensitive() {
        assert_eq!(
            fallback_table_lookup("CLAUDE-OPUS-4-7").context,
            fallback_table_lookup("claude-opus-4-7").context
        );
    }

    #[tokio::test]
    async fn extract_table_picks_only_trusted_provider_limit_fields() {
        let _guard = TEST_LOCK.lock().await;
        std::env::remove_var(ENV_MODELS_PROVIDER);
        // Mimics the models.dev envelope; extra fields are ignored.
        let json = r#"{
            "scaleway": {
                "models": {
                    "glm-5.2": {
                        "limit": { "context": 256000, "output": 16384 }
                    }
                }
            },
            "opencode": {
                "models": {
                    "claude-opus-4-7": {
                        "name": "Claude Opus 4.7",
                        "limit": { "context": 1000000, "output": 32000 },
                        "cost": { "input": 15.0, "output": 75.0 }
                    },
                    "claude-haiku-4-5": {
                        "limit": { "context": 200000, "output": 8192 }
                    },
                    "claude-broken": {
                        "limit": { "context": 0, "output": 100 }
                    },
                    "glm-5.2": {
                        "limit": { "context": 1000000, "output": 131072 }
                    }
                }
            },
            "openai": {
                "models": {
                    "gpt-4o": { "limit": { "context": 128000, "output": 16384 } }
                }
            }
        }"#;
        let table = extract_table_for_test(json);
        assert_eq!(table.get("claude-opus-4-7").unwrap().context, 1_000_000);
        assert_eq!(table.get("claude-opus-4-7").unwrap().output, 32_000);
        assert_eq!(table.get("claude-haiku-4-5").unwrap().context, 200_000);
        assert_eq!(table.get("glm-5.2").unwrap().context, 1_000_000);
        assert!(!table.contains_key("gpt-4o"));
        // 0-context entry is dropped.
        assert!(!table.contains_key("claude-broken"));
        assert_eq!(table.len(), 3);
    }

    #[tokio::test]
    async fn extract_table_honors_provider_override() {
        let _guard = TEST_LOCK.lock().await;
        std::env::set_var(ENV_MODELS_PROVIDER, "scaleway");
        let json = r#"{
            "opencode": {
                "models": {
                    "glm-5.2": { "limit": { "context": 1000000, "output": 131072 } }
                }
            },
            "scaleway": {
                "models": {
                    "glm-5.2": { "limit": { "context": 256000, "output": 16384 } }
                }
            }
        }"#;
        let table = extract_table_for_test(json);
        assert_eq!(table.get("glm-5.2").unwrap().context, 256_000);
        assert_eq!(table.get("glm-5.2").unwrap().output, 16_384);
        std::env::remove_var(ENV_MODELS_PROVIDER);
    }

    #[tokio::test]
    async fn extract_table_missing_limit_is_skipped() {
        let _guard = TEST_LOCK.lock().await;
        std::env::remove_var(ENV_MODELS_PROVIDER);
        let json = r#"{
            "opencode": {
                "models": {
                    "no-limits-here": { "name": "Mystery Model" }
                }
            }
        }"#;
        let table = extract_table_for_test(json);
        assert!(table.is_empty());
    }

    #[tokio::test]
    async fn resolve_returns_injected_table_value() {
        let _guard = TEST_LOCK.lock().await;
        let mut table = HashMap::new();
        table.insert(
            "injected-model".to_string(),
            ModelLimits {
                context: 42_000,
                output: 4_000,
            },
        );
        inject_test_table(table).await;
        // resolve_limits consults the in-memory table first.
        let limits = resolve_limits("injected-model");
        assert_eq!(limits.context, 42_000);
        assert_eq!(limits.output, 4_000);
    }

    #[tokio::test]
    async fn resolve_context_window_tokens_shim_matches_resolve_limits() {
        let _guard = TEST_LOCK.lock().await;
        // The backwards-compat shim must equal resolve_limits().context.
        let mut table = HashMap::new();
        table.insert(
            "shim-model".to_string(),
            ModelLimits {
                context: 99_999,
                output: 1_000,
            },
        );
        inject_test_table(table).await;
        assert_eq!(resolve_context_window_tokens("shim-model"), 99_999);
    }

    #[tokio::test]
    async fn serialize_then_load_roundtrips() {
        let _guard = TEST_LOCK.lock().await;
        std::env::remove_var(ENV_MODELS_PROVIDER);
        let mut table = HashMap::new();
        table.insert(
            "roundtrip-model".to_string(),
            ModelLimits {
                context: 123_456,
                output: 6_543,
            },
        );
        let body = serialize_table(&table);
        // Re-parse through the production loader logic.
        let parsed = extract_table_for_test(&body);
        assert_eq!(parsed.get("roundtrip-model").unwrap().context, 123_456);
        assert_eq!(parsed.get("roundtrip-model").unwrap().output, 6_543);
        assert!(body.contains("\"opencode\""));
    }

    // -----------------------------------------------------------------------
    // Integration: real HTTP retry + fallback via a local mock server.
    // We drive fetch_with_retry indirectly by pointing AGENT_HARNESS_MODELS_URL
    // at the mock and calling fetch_and_populate().
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn fetch_and_populate_succeeds_on_mock_server() {
        let _guard = TEST_LOCK.lock().await;
        let cache_file =
            std::env::temp_dir().join(format!("ahrs-test-cache-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&cache_file);
        std::env::set_var("AGENT_HARNESS_CACHE_PATH", &cache_file);
        std::env::remove_var(ENV_MODELS_PROVIDER);

        let body = r#"{
            "opencode": {
                "models": {
                    "mock-large": { "limit": { "context": 500000, "output": 8000 } }
                }
            }
        }"#;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/api.json");
        std::env::set_var("AGENT_HARNESS_MODELS_URL", &url);

        let body_clone = body.to_string();
        let server = tokio::spawn(async move {
            // Serve exactly one request then stop. Loop on read until we
            // see the end-of-headers marker so reqwest's request bytes
            // are fully drained before we write the response.
            let (mut stream, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 256];
            let mut got = String::new();
            while !got.contains("\r\n\r\n") {
                let n = stream.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                got.push_str(String::from_utf8_lossy(&buf[..n]).as_ref());
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body_clone.len(),
                body_clone
            );
            stream.write_all(resp.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
        });

        // Clear table so we start empty, reset fetch_attempted.
        {
            let mut guard = catalog().write().await;
            guard.table.clear();
            guard.fetch_attempted = false;
        }
        fetch_and_populate().await;
        let _ = tokio::time::timeout(Duration::from_secs(2), server).await;

        let guard = catalog().read().await;
        assert_eq!(guard.table.get("mock-large").unwrap().context, 500_000);
        assert!(guard.fetch_attempted);
        let _ = std::fs::remove_file(&cache_file);
    }

    #[tokio::test]
    async fn fetch_and_populate_falls_back_when_endpoint_unreachable() {
        let _guard = TEST_LOCK.lock().await;
        let cache_file =
            std::env::temp_dir().join(format!("ahrs-test-cache-fb-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&cache_file);
        std::env::set_var("AGENT_HARNESS_CACHE_PATH", &cache_file);
        std::env::remove_var(ENV_MODELS_PROVIDER);
        // Point at a closed port → all retries fail → fetch_attempted=true,
        // table unchanged (falls back to the static lookup elsewhere).
        std::env::set_var("AGENT_HARNESS_MODELS_URL", "http://127.0.0.1:1/api.json");
        {
            let mut guard = catalog().write().await;
            guard.table.clear();
            guard.fetch_attempted = false;
        }
        fetch_and_populate().await;
        let guard = catalog().read().await;
        assert!(guard.fetch_attempted);
        assert!(guard.table.is_empty());
        let _ = std::fs::remove_file(&cache_file);
    }
}
