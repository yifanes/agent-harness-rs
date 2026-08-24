//! Model capabilities sourced exclusively from `models.dev["opencode"]`.
//!
//! Hosts initialize the catalog explicitly during startup, resolve a short
//! model id once, and keep the resulting [`ResolvedModelConfig`] for the
//! lifetime of the session. There are deliberately no built-in model profiles
//! or cross-provider merges. A model id absent from the catalog (renamed or
//! gateway-local ids) resolves against the conservative built-in defaults and
//! is marked [`LimitsSource::Default`] — callers must surface that marker
//! instead of treating the limits as verified facts, and capability
//! assertions (output cap, reasoning/temperature support, deprecation) are
//! skipped for such models so a conservative guess can never veto a declared
//! request; the provider stays the final authority.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

const DEFAULT_MODELS_URL: &str = "https://models.dev/api.json";
const MODELS_URL_ENV: &str = "AGENT_HARNESS_MODELS_URL";
const CACHE_PATH_ENV: &str = "AGENT_HARNESS_MODELS_CACHE_PATH";
const PROVIDER: &str = "opencode";
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireProtocol {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    #[default]
    Default,
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub mode: ReasoningMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRequestConfig {
    pub model: String,
    pub max_output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLimits {
    pub context: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<u64>,
    pub output: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningOption {
    Toggle,
    Effort {
        values: Vec<String>,
    },
    BudgetTokens {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u64>,
    },
}

impl ReasoningOption {
    /// Lenient decode of one `reasoning_options` entry.
    ///
    /// The live models.dev payload carries data-quality defects that must not
    /// poison the catalog (see the crate-level tolerance contract):
    ///
    /// * `budget_tokens.min`/`max` may be negative, which upstream uses to
    ///   mean "no bound" — decoded as `None`.
    /// * `effort.values` may contain `null` or non-string entries — dropped.
    ///
    /// Anything structurally wrong (missing `type`, non-array `values`, a
    /// bound that is neither null nor an integer) is an error so the
    /// *containing model* is skipped by [`parse_snapshot`] instead of failing
    /// the whole payload.
    fn from_value(value: &Value) -> Result<Self, String> {
        let type_ = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "reasoning option is missing `type`".to_owned())?;
        match type_ {
            "toggle" => Ok(Self::Toggle),
            "effort" => {
                let raw = value
                    .get("values")
                    .ok_or_else(|| "effort option is missing `values`".to_owned())?;
                let entries = raw
                    .as_array()
                    .ok_or_else(|| "effort option `values` is not an array".to_owned())?;
                let values = entries
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect();
                Ok(Self::Effort { values })
            }
            "budget_tokens" => {
                let malformed = |value: &Value, key: &str| -> bool {
                    matches!(
                        value.get(key),
                        Some(v) if !v.is_null() && v.as_i64().is_none() && v.as_u64().is_none()
                    )
                };
                if malformed(value, "min") || malformed(value, "max") {
                    return Err("budget_tokens bounds must be integers".into());
                }
                let bound = |value: &Value, key: &str| -> Option<u64> {
                    match value.get(key) {
                        None | Some(Value::Null) => None,
                        Some(v) => match v.as_i64() {
                            Some(n) if n >= 0 => Some(n as u64),
                            // Negative = "no bound" in upstream data.
                            Some(_) => None,
                            None => v.as_u64(),
                        },
                    }
                };
                Ok(Self::BudgetTokens {
                    min: bound(value, "min"),
                    max: bound(value, "max"),
                })
            }
            other => Err(format!("unknown reasoning option type `{other}`")),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterleavedReasoning {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
}

/// Where a model's capabilities came from.
///
/// [`LimitsSource::Default`] marks the conservative fallback applied to model
/// ids that are absent from the catalog. It is an observability signal, not a
/// quality grade: limits under this marker are a guess and must be surfaced
/// by the caller (log / status event), never treated as verified model facts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitsSource {
    /// Verified against the `models.dev["opencode"]` catalog.
    #[default]
    Catalog,
    /// Conservative built-in defaults; the model id was not in the catalog.
    Default,
}

impl std::fmt::Display for LimitsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitsSource::Catalog => f.write_str("catalog"),
            LimitsSource::Default => f.write_str("default"),
        }
    }
}

/// Conservative context window for models absent from the catalog. Deliberately
/// the low end of modern agent-model windows: an under-estimate compacts early
/// (safe), an over-estimate overflows the real provider window mid-turn (the
/// one direction that must never be guessed).
pub const DEFAULT_CONTEXT_LIMIT: u64 = 128_000;
/// Conservative output limit for models absent from the catalog; matches the
/// bound RD's bootstrap layer applies to legacy sessions (safely below the
/// tightest limit in the deployed legacy model set).
pub const DEFAULT_OUTPUT_LIMIT: u64 = 32_768;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub id: String,
    pub limits: ModelLimits,
    pub reasoning: bool,
    pub reasoning_options: Vec<ReasoningOption>,
    pub temperature: bool,
    pub tool_call: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interleaved: Option<InterleavedReasoning>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default)]
    pub limits_source: LimitsSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedModelConfig {
    pub model: String,
    pub wire_protocol: WireProtocol,
    pub max_output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    pub reasoning: ReasoningConfig,
    pub capabilities: ModelCapabilities,
}

impl ResolvedModelConfig {
    /// Maximum input budget after reserving the requested completion budget.
    pub fn max_input_tokens(&self) -> u64 {
        self.capabilities.limits.input.unwrap_or_else(|| {
            self.capabilities
                .limits
                .context
                .saturating_sub(self.max_output_tokens)
        })
    }
}

#[derive(Debug, Error)]
pub enum ModelCatalogError {
    #[error("models.dev request failed: {0}")]
    Request(String),
    #[error("models.dev response is invalid: {0}")]
    Decode(String),
    #[error("models.dev has no `{PROVIDER}` provider")]
    ProviderMissing,
    /// Retained for API stability. Since the fail-open default resolution,
    /// `resolve` no longer produces this error: unknown ids resolve against
    /// conservative defaults marked [`LimitsSource::Default`] instead.
    #[error("model `{model}` is not present in models.dev[{PROVIDER}]{suggestions}")]
    ModelNotFound { model: String, suggestions: String },
    #[error("model id `{model}` is ambiguous in models.dev[{PROVIDER}]: {matches:?}")]
    AmbiguousModel { model: String, matches: Vec<String> },
    #[error("model `{model}` is deprecated")]
    DeprecatedModel { model: String },
    #[error("invalid model request for `{model}`: {message}")]
    InvalidRequest { model: String, message: String },
}

// The root payload is parsed as a `Value` in [`parse_snapshot`] and only the
// `{PROVIDER}` provider's `models` object is consumed, per-model.

#[derive(Debug, Clone, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: Option<String>,
    limit: ModelLimits,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    reasoning_options: Vec<ReasoningOption>,
    #[serde(default)]
    temperature: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    interleaved: Option<Value>,
    #[serde(default)]
    status: Option<String>,
}

impl ModelEntry {
    fn into_capabilities(self, map_id: String) -> ModelCapabilities {
        let interleaved = match self.interleaved {
            Some(Value::Bool(enabled)) => Some(InterleavedReasoning {
                enabled,
                field: None,
            }),
            Some(Value::Object(object)) => Some(InterleavedReasoning {
                enabled: true,
                field: object
                    .get("field")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            }),
            _ => None,
        };
        ModelCapabilities {
            id: self.id.unwrap_or(map_id),
            limits: self.limit,
            reasoning: self.reasoning,
            reasoning_options: self.reasoning_options,
            temperature: self.temperature,
            tool_call: self.tool_call,
            interleaved,
            status: self.status,
            limits_source: LimitsSource::Catalog,
        }
    }
}

/// Conservative capabilities for a model id that is absent from the catalog.
/// The id is kept exactly as requested (only trimmed) — it is echoed back to
/// the provider as the wire model name, and gateway-local ids may be
/// case-sensitive, unlike the lower-cased catalog keys.
fn default_capabilities(id: String) -> ModelCapabilities {
    ModelCapabilities {
        id,
        limits: ModelLimits {
            context: DEFAULT_CONTEXT_LIMIT,
            input: None,
            output: DEFAULT_OUTPUT_LIMIT,
        },
        reasoning: false,
        reasoning_options: Vec::new(),
        temperature: true,
        tool_call: true,
        interleaved: None,
        status: None,
        limits_source: LimitsSource::Default,
    }
}

/// Decoded, lower-cased model ids of the `{PROVIDER}` provider. Built by
/// [`parse_snapshot`], which parses model entries individually: a single
/// malformed entry is skipped (with a warn) instead of failing the whole
/// payload, so one dirty model in any provider can never take the catalog
/// down.
#[derive(Debug, Clone)]
struct CatalogSnapshot {
    models: HashMap<String, ModelCapabilities>,
}

/// Shared, refreshable view of the fixed `models.dev["opencode"]` catalog.
#[derive(Debug, Clone)]
pub struct ModelCatalog {
    snapshot: Arc<RwLock<CatalogSnapshot>>,
    refresh_lock: Arc<Mutex<()>>,
    refresh_generation: Arc<AtomicU64>,
    cache_path: Option<PathBuf>,
    models_url: String,
}

impl ModelCatalog {
    /// Load the disk cache and refresh it when needed. A fresh cache returns
    /// immediately. A stale cache remains usable while one background refresh
    /// runs; with no usable cache, the network fetch is awaited and required.
    pub async fn initialize() -> Result<Self, ModelCatalogError> {
        let cache_path = cache_path();
        let cached = cache_path
            .as_deref()
            .and_then(read_cache)
            .and_then(|bytes| parse_snapshot(&bytes).ok());
        let cache_fresh = cache_path.as_deref().map(cache_is_fresh).unwrap_or(false);
        let catalog = Self {
            snapshot: Arc::new(RwLock::new(cached.as_ref().cloned().unwrap_or_else(|| {
                CatalogSnapshot {
                    models: HashMap::new(),
                }
            }))),
            refresh_lock: Arc::new(Mutex::new(())),
            refresh_generation: Arc::new(AtomicU64::new(0)),
            cache_path,
            models_url: std::env::var(MODELS_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_MODELS_URL.to_owned()),
        };

        if cached.is_some() && cache_fresh {
            return Ok(catalog);
        }
        if cached.is_some() {
            let refresh_catalog = catalog.clone();
            tokio::spawn(async move {
                if let Err(error) = refresh_catalog.refresh().await {
                    warn!(%error, "models.dev background refresh failed; retaining stale cache");
                }
            });
            return Ok(catalog);
        }

        catalog.refresh().await?;
        Ok(catalog)
    }

    /// Force a refresh. Concurrent callers share a single in-flight fetch.
    pub async fn refresh(&self) -> Result<(), ModelCatalogError> {
        let observed_generation = self.refresh_generation.load(Ordering::Acquire);
        let _guard = self.refresh_lock.lock().await;
        if self.refresh_generation.load(Ordering::Acquire) != observed_generation {
            return Ok(());
        }
        let bytes = fetch_catalog(&self.models_url).await?;
        let snapshot = parse_snapshot(&bytes)?;
        if let Some(path) = &self.cache_path {
            if let Err(error) = write_cache_atomic(path, &bytes).await {
                warn!(%error, ?path, "failed to write models.dev disk cache");
            }
        }
        let count = snapshot.models.len();
        *self.snapshot.write().await = snapshot;
        self.refresh_generation.fetch_add(1, Ordering::Release);
        debug!(count, provider = PROVIDER, "models.dev catalog refreshed");
        Ok(())
    }

    /// Resolve and validate a short model id for one of the three wire
    /// protocols. Exact id match wins; a path-like id may also match by a
    /// unique basename. Contains/fuzzy matches are suggestions only. An id
    /// with no match at all does not error: it resolves against
    /// [`default_capabilities`]-style conservative limits marked
    /// [`LimitsSource::Default`] (see the module docs for the validation
    /// split that follows from that).
    pub async fn resolve(
        &self,
        request: ModelRequestConfig,
        wire_protocol: WireProtocol,
    ) -> Result<ResolvedModelConfig, ModelCatalogError> {
        let capabilities = self.capabilities(&request.model).await?;
        validate_request(&request, &capabilities, wire_protocol)?;
        Ok(ResolvedModelConfig {
            model: capabilities.id.clone(),
            wire_protocol,
            max_output_tokens: request.max_output_tokens,
            temperature: request.temperature,
            reasoning: normalize_reasoning(request.reasoning, &capabilities),
            capabilities,
        })
    }

    /// Look up one model's capabilities without validating a request. Lets
    /// callers clamp `max_output_tokens` or drop unsupported temperature /
    /// reasoning knobs *before* [`Self::resolve`], instead of recovering
    /// from `InvalidRequest` errors after the fact.
    pub async fn capabilities(&self, model: &str) -> Result<ModelCapabilities, ModelCatalogError> {
        let snapshot = self.snapshot.read().await;
        let original = model.trim();
        let requested = original.to_ascii_lowercase();
        match snapshot.models.get(&requested) {
            Some(model) => Ok(model.clone()),
            None => match resolve_unique_basename(&snapshot.models, &requested) {
                Ok(model) => Ok(model),
                Err(ModelCatalogError::ModelNotFound { suggestions, .. }) => {
                    warn!(
                        model = %original,
                        suggestions = %suggestions,
                        context_limit = DEFAULT_CONTEXT_LIMIT,
                        output_limit = DEFAULT_OUTPUT_LIMIT,
                        "model absent from models.dev[{}]; resolving with conservative defaults (limits_source=default)",
                        PROVIDER
                    );
                    Ok(default_capabilities(original.to_owned()))
                }
                Err(error) => Err(error),
            },
        }
    }

    /// Build a catalog from a models.dev-shaped JSON payload — no network
    /// fetch, no disk cache. For embedding a fixed catalog: tests,
    /// air-gapped deployments, and custom gateways whose model ids are
    /// absent from models.dev["opencode"].
    pub fn from_json(bytes: &[u8]) -> Result<Self, ModelCatalogError> {
        Ok(Self {
            snapshot: Arc::new(RwLock::new(parse_snapshot(bytes)?)),
            refresh_lock: Arc::new(Mutex::new(())),
            refresh_generation: Arc::new(AtomicU64::new(0)),
            cache_path: None,
            models_url: String::new(),
        })
    }

    pub async fn model_count(&self) -> usize {
        self.snapshot.read().await.models.len()
    }
}

fn parse_snapshot(bytes: &[u8]) -> Result<CatalogSnapshot, ModelCatalogError> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| ModelCatalogError::Decode(error.to_string()))?;
    let root = root
        .as_object()
        .ok_or_else(|| ModelCatalogError::Decode("payload root is not a JSON object".into()))?;
    let provider = root
        .get(PROVIDER)
        .ok_or(ModelCatalogError::ProviderMissing)?;
    let models = provider
        .get("models")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ModelCatalogError::Decode(format!(
                "{PROVIDER} provider is missing a usable `models` object"
            ))
        })?;
    let mut parsed: HashMap<String, ModelCapabilities> = HashMap::new();
    let mut skipped: Vec<String> = Vec::new();
    for (id, entry) in models {
        match serde_json::from_value::<ModelEntry>(entry.clone()) {
            Ok(entry) => {
                parsed.insert(id.to_ascii_lowercase(), entry.into_capabilities(id.clone()));
            }
            Err(_) => skipped.push(id.clone()),
        }
    }
    if parsed.is_empty() && !models.is_empty() {
        return Err(ModelCatalogError::Decode(format!(
            "no model in {PROVIDER} could be decoded ({} unusable entries)",
            skipped.len()
        )));
    }
    if !skipped.is_empty() {
        let preview: Vec<String> = skipped.iter().take(5).cloned().collect();
        warn!(
            skipped = %skipped.len(),
            first_skipped = %preview.join(", "),
            "models.dev entries with unusable data were skipped"
        );
    }
    Ok(CatalogSnapshot { models: parsed })
}

fn resolve_unique_basename(
    models: &HashMap<String, ModelCapabilities>,
    requested: &str,
) -> Result<ModelCapabilities, ModelCatalogError> {
    let basename = requested.rsplit('/').next().unwrap_or(requested);
    let mut matches = models
        .iter()
        .filter(|(id, _)| id.rsplit('/').next() == Some(basename))
        .map(|(_, model)| model.clone())
        .collect::<Vec<_>>();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        count if count > 1 => Err(ModelCatalogError::AmbiguousModel {
            model: requested.to_owned(),
            matches: matches.into_iter().map(|model| model.id).collect(),
        }),
        _ => {
            let mut suggestions = models
                .keys()
                .filter(|id| id.contains(requested) || requested.contains(id.as_str()))
                .take(5)
                .cloned()
                .collect::<Vec<_>>();
            suggestions.sort();
            let suggestions = if suggestions.is_empty() {
                String::new()
            } else {
                format!("; did you mean {}?", suggestions.join(", "))
            };
            Err(ModelCatalogError::ModelNotFound {
                model: requested.to_owned(),
                suggestions,
            })
        }
    }
}

fn validate_request(
    request: &ModelRequestConfig,
    capabilities: &ModelCapabilities,
    wire_protocol: WireProtocol,
) -> Result<(), ModelCatalogError> {
    let invalid = |message: String| ModelCatalogError::InvalidRequest {
        model: capabilities.id.clone(),
        message,
    };
    // Structural invariants — enforced for every capability source, including
    // [`LimitsSource::Default`]: these describe well-formed requests, not
    // model capabilities.
    if request.max_output_tokens == 0 {
        return Err(invalid(
            "max_output_tokens must be greater than zero".into(),
        ));
    }
    if request.reasoning.effort.is_some() && request.reasoning.budget_tokens.is_some() {
        return Err(invalid(
            "reasoning.effort and reasoning.budget_tokens are mutually exclusive".into(),
        ));
    }
    if matches!(request.reasoning.mode, ReasoningMode::Default)
        && (request.reasoning.effort.is_some() || request.reasoning.budget_tokens.is_some())
    {
        return Err(invalid(
            "reasoning.mode must be enabled when effort or budget_tokens is set".into(),
        ));
    }
    if matches!(request.reasoning.mode, ReasoningMode::Disabled) {
        if request.reasoning.budget_tokens.is_some() {
            return Err(invalid(
                "reasoning.budget_tokens cannot be set when reasoning is disabled".into(),
            ));
        }
        if request
            .reasoning
            .effort
            .as_deref()
            .is_some_and(|effort| !effort.eq_ignore_ascii_case("none"))
        {
            return Err(invalid(
                "reasoning.effort must be `none` when reasoning is disabled".into(),
            ));
        }
    }
    if matches!(request.reasoning.mode, ReasoningMode::Enabled)
        && request
            .reasoning
            .effort
            .as_deref()
            .is_some_and(|effort| effort.eq_ignore_ascii_case("none"))
    {
        return Err(invalid(
            "reasoning.effort `none` conflicts with reasoning.mode `enabled`".into(),
        ));
    }

    // Wire-protocol constraints — enforced for every capability source: they
    // describe the protocol, not the model.
    if matches!(wire_protocol, WireProtocol::OpenAiResponses)
        && request.reasoning.budget_tokens.is_some()
    {
        return Err(invalid(
            "openai_responses cannot express reasoning budget_tokens".into(),
        ));
    }

    // Capability assertions — only when the catalog actually knows this model.
    // Under [`LimitsSource::Default`] the capabilities are a conservative
    // guess; asserting against them would let the guess veto the session
    // manager's declaration (e.g. a real 65K output cap rejected by our 32K
    // guess). The provider is the final authority for such models: an
    // over-declared limit surfaces as an explicit provider error at turn time.
    if capabilities.limits_source != LimitsSource::Catalog {
        return Ok(());
    }

    if capabilities.status.as_deref() == Some("deprecated") {
        return Err(ModelCatalogError::DeprecatedModel {
            model: capabilities.id.clone(),
        });
    }
    if request.max_output_tokens > capabilities.limits.output {
        return Err(invalid(format!(
            "max_output_tokens {} exceeds models.dev output limit {}",
            request.max_output_tokens, capabilities.limits.output
        )));
    }
    if request.temperature.is_some() && !capabilities.temperature {
        return Err(invalid("temperature is not supported".into()));
    }
    if !capabilities.reasoning && !matches!(request.reasoning.mode, ReasoningMode::Default) {
        return Err(invalid("reasoning is not supported".into()));
    }

    let effort_values = capabilities
        .reasoning_options
        .iter()
        .find_map(|option| match option {
            ReasoningOption::Effort { values } => Some(values),
            _ => None,
        });
    let toggle = capabilities
        .reasoning_options
        .iter()
        .any(|option| matches!(option, ReasoningOption::Toggle));
    let budget = capabilities
        .reasoning_options
        .iter()
        .find_map(|option| match option {
            ReasoningOption::BudgetTokens { min, max } => Some((*min, *max)),
            _ => None,
        });

    if let Some(effort) = request.reasoning.effort.as_deref() {
        let values =
            effort_values.ok_or_else(|| invalid("reasoning effort is not supported".into()))?;
        if !values
            .iter()
            .any(|value| value.eq_ignore_ascii_case(effort))
        {
            return Err(invalid(format!(
                "reasoning effort `{effort}` is unsupported; allowed values: {}",
                values.join(", ")
            )));
        }
    }
    if let Some(tokens) = request.reasoning.budget_tokens {
        let (min, max) =
            budget.ok_or_else(|| invalid("reasoning budget_tokens is not supported".into()))?;
        if min.is_some_and(|minimum| tokens < minimum)
            || max.is_some_and(|maximum| tokens > maximum)
        {
            return Err(invalid(format!(
                "reasoning budget_tokens {tokens} is outside models.dev range {}..{}",
                min.map(|value| value.to_string())
                    .unwrap_or_else(|| "0".into()),
                max.map(|value| value.to_string())
                    .unwrap_or_else(|| "unbounded".into())
            )));
        }
    }

    match request.reasoning.mode {
        ReasoningMode::Default => {}
        ReasoningMode::Enabled
            if request.reasoning.effort.is_none()
                && request.reasoning.budget_tokens.is_none()
                && !toggle =>
        {
            return Err(invalid(
                "reasoning cannot be explicitly enabled without an effort or toggle capability"
                    .into(),
            ));
        }
        ReasoningMode::Disabled if !toggle && !supports_none(effort_values) => {
            return Err(invalid("reasoning cannot be explicitly disabled".into()));
        }
        _ => {}
    }

    if matches!(wire_protocol, WireProtocol::OpenAiResponses)
        && !matches!(request.reasoning.mode, ReasoningMode::Default)
        && request.reasoning.effort.is_none()
        && !supports_none(effort_values)
    {
        return Err(invalid(
            "openai_responses requires an effort-based reasoning control".into(),
        ));
    }
    Ok(())
}

fn supports_none(values: Option<&Vec<String>>) -> bool {
    values.is_some_and(|values| {
        values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("none"))
    })
}

fn normalize_reasoning(
    mut reasoning: ReasoningConfig,
    capabilities: &ModelCapabilities,
) -> ReasoningConfig {
    if matches!(reasoning.mode, ReasoningMode::Disabled) && reasoning.effort.is_none() {
        let supports_none = capabilities
            .reasoning_options
            .iter()
            .any(|option| match option {
                ReasoningOption::Effort { values } => values.iter().any(|value| value == "none"),
                _ => false,
            });
        if supports_none {
            reasoning.effort = Some("none".into());
        }
    }
    reasoning
}

async fn fetch_catalog(url: &str) -> Result<Vec<u8>, ModelCatalogError> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|error| ModelCatalogError::Request(error.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| ModelCatalogError::Request(error.to_string()))?
        .error_for_status()
        .map_err(|error| ModelCatalogError::Request(error.to_string()))?;
    response
        .bytes()
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|error| ModelCatalogError::Request(error.to_string()))
}

fn cache_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(CACHE_PATH_ENV) {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    dirs::cache_dir().map(|base| base.join("agent-harness-rs").join("models.dev.json"))
}

fn read_cache(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn cache_is_fresh(path: &Path) -> bool {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age < CACHE_TTL)
}

async fn write_cache_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let suffix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let temporary = path.with_extension(format!("tmp.{}.{}", std::process::id(), suffix));
    tokio::fs::write(&temporary, bytes).await?;
    tokio::fs::rename(temporary, path).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_protocol_uses_stable_external_names() {
        for (protocol, external) in [
            (WireProtocol::OpenAiCompatible, "openai_compatible"),
            (WireProtocol::Anthropic, "anthropic"),
            (WireProtocol::OpenAiResponses, "openai_responses"),
        ] {
            assert_eq!(
                serde_json::to_string(&protocol).unwrap(),
                format!("\"{external}\"")
            );
            assert_eq!(
                serde_json::from_str::<WireProtocol>(&format!("\"{external}\"")).unwrap(),
                protocol
            );
        }
    }

    fn catalog(json: &str) -> ModelCatalog {
        ModelCatalog::from_json(json.as_bytes()).unwrap()
    }

    const FIXTURE: &str = r#"{
      "opencode": {"models": {
        "deepseek-v4-pro": {
          "id":"deepseek-v4-pro", "reasoning":true, "temperature":true,
          "tool_call":true, "interleaved":{"field":"reasoning_content"},
          "reasoning_options":[{"type":"toggle"},{"type":"effort","values":["high","max"]}],
          "limit":{"context":1000000,"output":384000}
        },
        "gpt-5.5": {
          "reasoning":true, "temperature":false, "tool_call":true,
          "reasoning_options":[{"type":"effort","values":["none","low","high"]}],
          "limit":{"context":1050000,"input":922000,"output":128000}
        },
        "old-model": {
          "status":"deprecated", "limit":{"context":1000,"output":100}
        }
      }}
    }"#;

    fn request(model: &str) -> ModelRequestConfig {
        ModelRequestConfig {
            model: model.into(),
            max_output_tokens: 65_536,
            temperature: None,
            reasoning: ReasoningConfig::default(),
        }
    }

    #[tokio::test]
    async fn exact_short_id_resolves_and_preserves_limits() {
        let resolved = catalog(FIXTURE)
            .resolve(request("deepseek-v4-pro"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.model, "deepseek-v4-pro");
        assert_eq!(resolved.capabilities.limits.context, 1_000_000);
        assert_eq!(resolved.max_input_tokens(), 934_464);
        assert_eq!(
            resolved.capabilities.interleaved.unwrap().field.as_deref(),
            Some("reasoning_content")
        );
    }

    #[tokio::test]
    async fn unique_path_basename_resolves_but_contains_does_not() {
        let resolved = catalog(FIXTURE)
            .resolve(
                request("vendor/deepseek-v4-pro"),
                WireProtocol::OpenAiCompatible,
            )
            .await
            .unwrap();
        assert_eq!(resolved.model, "deepseek-v4-pro");
        // A contains-only match is NOT a catalog resolution: it falls through
        // to the conservative defaults, clearly marked, instead of resolving
        // to the wrong model or failing the session.
        let resolved = catalog(FIXTURE)
            .resolve(request("deepseek-v4"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.model, "deepseek-v4");
        assert_eq!(resolved.capabilities.limits_source, LimitsSource::Default);
    }

    #[tokio::test]
    async fn unknown_model_resolves_with_conservative_defaults() {
        let resolved = catalog(FIXTURE)
            .resolve(request("glm-5.1-rdclaw"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        // The id is echoed back trimmed but NOT lower-cased: it is the wire
        // model name sent to the provider, and gateway-local ids may be
        // case-sensitive.
        assert_eq!(resolved.model, "glm-5.1-rdclaw");
        assert_eq!(resolved.capabilities.limits_source, LimitsSource::Default);
        assert_eq!(resolved.capabilities.limits.context, DEFAULT_CONTEXT_LIMIT);
        assert_eq!(resolved.capabilities.limits.output, DEFAULT_OUTPUT_LIMIT);
        assert_eq!(resolved.capabilities.limits.input, None);
        assert!(!resolved.capabilities.reasoning);
        assert!(resolved.capabilities.temperature);
        assert!(resolved.capabilities.tool_call);
        assert_eq!(resolved.capabilities.status, None);
    }

    #[tokio::test]
    async fn unknown_model_keeps_original_case() {
        let resolved = catalog(FIXTURE)
            .resolve(request("My-Glm-5.1"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.model, "My-Glm-5.1");
        assert_eq!(resolved.capabilities.limits_source, LimitsSource::Default);
    }

    #[tokio::test]
    async fn default_source_never_vetoes_a_declared_request() {
        // SM-declared 65K output must not be rejected by our 32K guess.
        let resolved = catalog(FIXTURE)
            .resolve(request("glm-5.1-rdclaw"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.max_output_tokens, 65_536);

        // Explicitly enabled reasoning passes; the provider decides.
        let mut req = request("glm-5.1-rdclaw");
        req.reasoning = ReasoningConfig {
            mode: ReasoningMode::Enabled,
            effort: Some("high".into()),
            budget_tokens: None,
        };
        catalog(FIXTURE)
            .resolve(req, WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn default_source_still_enforces_structural_invariants() {
        let mut req = request("glm-5.1-rdclaw");
        req.max_output_tokens = 0;
        assert!(matches!(
            catalog(FIXTURE)
                .resolve(req, WireProtocol::OpenAiCompatible)
                .await
                .unwrap_err(),
            ModelCatalogError::InvalidRequest { .. }
        ));

        let mut req = request("glm-5.1-rdclaw");
        req.reasoning = ReasoningConfig {
            mode: ReasoningMode::Enabled,
            effort: Some("high".into()),
            budget_tokens: Some(1_000),
        };
        assert!(matches!(
            catalog(FIXTURE)
                .resolve(req, WireProtocol::OpenAiCompatible)
                .await
                .unwrap_err(),
            ModelCatalogError::InvalidRequest { .. }
        ));

        // Wire-protocol constraints still apply to unknown models too.
        let mut req = request("glm-5.1-rdclaw");
        req.reasoning = ReasoningConfig {
            mode: ReasoningMode::Enabled,
            effort: None,
            budget_tokens: Some(1_000),
        };
        assert!(matches!(
            catalog(FIXTURE)
                .resolve(req, WireProtocol::OpenAiResponses)
                .await
                .unwrap_err(),
            ModelCatalogError::InvalidRequest { .. }
        ));
    }

    #[tokio::test]
    async fn catalog_hit_still_marks_catalog_source() {
        let resolved = catalog(FIXTURE)
            .resolve(request("deepseek-v4-pro"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.capabilities.limits_source, LimitsSource::Catalog);
        // And catalog-source requests are still capped by the real limit.
        let mut req = request("gpt-5.5");
        req.max_output_tokens = u64::MAX;
        assert!(matches!(
            catalog(FIXTURE)
                .resolve(req, WireProtocol::OpenAiCompatible)
                .await
                .unwrap_err(),
            ModelCatalogError::InvalidRequest { .. }
        ));
    }

    #[tokio::test]
    async fn capabilities_looks_up_without_request_validation() {
        let source = catalog(FIXTURE);
        let caps = source.capabilities("gpt-5.5").await.unwrap();
        assert_eq!(caps.limits.output, 128_000);
        assert!(!caps.temperature);
        // Deprecated models still resolve at the lookup layer — deprecation
        // is a request-validation concern, not a lookup one.
        let caps = source.capabilities("old-model").await.unwrap();
        assert_eq!(caps.status.as_deref(), Some("deprecated"));
        // Unknown ids no longer error at the lookup layer: they resolve to
        // conservative defaults, marked as such.
        let caps = source.capabilities("missing-model").await.unwrap();
        assert_eq!(caps.limits_source, LimitsSource::Default);
        assert_eq!(caps.id, "missing-model");
    }

    #[test]
    fn from_json_rejects_payloads_without_the_trusted_provider() {
        let error = ModelCatalog::from_json(br#"{"other": {"models": {}}}"#).unwrap_err();
        assert!(matches!(error, ModelCatalogError::ProviderMissing));
        let error = ModelCatalog::from_json(b"garbage").unwrap_err();
        assert!(matches!(error, ModelCatalogError::Decode(_)));
    }

    #[tokio::test]
    async fn validates_limits_temperature_and_deprecation() {
        let source = catalog(FIXTURE);
        let mut too_large = request("gpt-5.5");
        too_large.max_output_tokens = 128_001;
        assert!(matches!(
            source
                .resolve(too_large, WireProtocol::OpenAiResponses)
                .await,
            Err(ModelCatalogError::InvalidRequest { .. })
        ));
        let mut temperature = request("gpt-5.5");
        temperature.temperature = Some(0.2);
        assert!(matches!(
            source
                .resolve(temperature, WireProtocol::OpenAiResponses)
                .await,
            Err(ModelCatalogError::InvalidRequest { .. })
        ));
        let mut old = request("old-model");
        old.max_output_tokens = 10;
        assert!(matches!(
            source.resolve(old, WireProtocol::OpenAiCompatible).await,
            Err(ModelCatalogError::DeprecatedModel { .. })
        ));
    }

    #[tokio::test]
    async fn validates_and_normalizes_reasoning_controls() {
        let source = catalog(FIXTURE);
        let mut deepseek = request("deepseek-v4-pro");
        deepseek.reasoning = ReasoningConfig {
            mode: ReasoningMode::Enabled,
            effort: Some("high".into()),
            budget_tokens: None,
        };
        let resolved = source
            .resolve(deepseek, WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.reasoning.effort.as_deref(), Some("high"));

        let mut disabled = request("gpt-5.5");
        disabled.reasoning.mode = ReasoningMode::Disabled;
        let resolved = source
            .resolve(disabled, WireProtocol::OpenAiResponses)
            .await
            .unwrap();
        assert_eq!(resolved.reasoning.effort.as_deref(), Some("none"));

        let mut invalid = request("deepseek-v4-pro");
        invalid.reasoning = ReasoningConfig {
            mode: ReasoningMode::Enabled,
            effort: Some("medium".into()),
            budget_tokens: None,
        };
        assert!(matches!(
            source
                .resolve(invalid, WireProtocol::OpenAiCompatible)
                .await,
            Err(ModelCatalogError::InvalidRequest { .. })
        ));

        let mut contradictory = request("gpt-5.5");
        contradictory.reasoning = ReasoningConfig {
            mode: ReasoningMode::Disabled,
            effort: Some("high".into()),
            budget_tokens: None,
        };
        assert!(matches!(
            source
                .resolve(contradictory, WireProtocol::OpenAiResponses)
                .await,
            Err(ModelCatalogError::InvalidRequest { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Lenient-decode contract: live models.dev data-quality defects must not
    // poison the catalog. See the crate-level tolerance contract and
    // `parse_snapshot` / `ReasoningOption::from_value`.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn live_payload_negative_budget_min_is_tolerated() {
        // Reproduces the real `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
        // defect: `budget_tokens.min = -1` must decode as "no bound".
        let json = r#"{
          "opencode": {"models": {
            "nemotron": {
              "reasoning": true,
              "reasoning_options": [
                {"type":"budget_tokens","min":-1,"max":32768}
              ],
              "limit": {"context": 200000, "output": 32768}
            }
          }}
        }"#;
        let source = catalog(json);
        let mut req = request("nemotron");
        req.max_output_tokens = 1024;
        let resolved = source
            .resolve(req, WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        let option = resolved.capabilities.reasoning_options.first().unwrap();
        match option {
            ReasoningOption::BudgetTokens { min, max } => {
                assert_eq!(*min, None);
                assert_eq!(*max, Some(32_768));
            }
            other => panic!("expected budget_tokens, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn live_payload_null_effort_values_are_filtered() {
        // Reproduces the real `sarvam-105b` / `sarvam-30b` defect:
        // `effort.values = [null, "low", "medium"]` must drop the null.
        let json = r#"{
          "opencode": {"models": {
            "sarvam-105b": {
              "reasoning": true,
              "reasoning_options": [
                {"type":"effort","values":[null,"low","medium",123]}
              ],
              "limit": {"context": 4096, "output": 1024}
            }
          }}
        }"#;
        let source = catalog(json);
        let mut req = request("sarvam-105b");
        req.max_output_tokens = 1024;
        let resolved = source
            .resolve(req, WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        let option = resolved.capabilities.reasoning_options.first().unwrap();
        assert_eq!(
            option,
            &ReasoningOption::Effort {
                values: vec!["low".into(), "medium".into()]
            }
        );
    }

    #[tokio::test]
    async fn one_unusable_entry_is_skipped_not_fatal() {
        // A single structurally broken model (negative `context`, which is
        // genuinely unusable) must be skipped while its siblings survive.
        let json = r#"{
          "opencode": {"models": {
            "good-a": {
              "limit": {"context": 1000, "output": 100},
              "reasoning_options": []
            },
            "broken": {
              "limit": {"context": -1, "output": 100}
            },
            "good-b": {
              "limit": {"context": 2000, "output": 200}
            }
          }}
        }"#;
        let source = catalog(json);
        // Both good models remain resolvable...
        for name in ["good-a", "good-b"] {
            let mut req = request(name);
            req.max_output_tokens = 100;
            source
                .resolve(req, WireProtocol::OpenAiCompatible)
                .await
                .unwrap();
        }
        // ...and the broken one is a clean per-model miss that resolves to
        // the conservative defaults, not a catalog failure.
        let mut broken = request("broken");
        broken.max_output_tokens = 100;
        let resolved = source
            .resolve(broken, WireProtocol::OpenAiCompatible)
            .await
            .unwrap();
        assert_eq!(resolved.capabilities.limits_source, LimitsSource::Default);
    }

    #[test]
    fn opencode_without_models_object_is_a_data_failure() {
        // `opencode` present but structurally unusable: this IS a data failure
        // and must fail closed (empty catalog would hide the defect).
        let json = r#"{"opencode": {"name": "opencode"}, "other": {"models": {"x":{"limit":{"context":1,"output":1}}}}}"#;
        assert!(matches!(
            ModelCatalog::from_json(json.as_bytes()),
            Err(ModelCatalogError::Decode(_))
        ));
    }

    #[test]
    fn missing_opencode_provider_still_missing() {
        let json = r#"{"other": {"models": {"x":{"limit":{"context":1,"output":1}}}}}"#;
        assert!(matches!(
            ModelCatalog::from_json(json.as_bytes()),
            Err(ModelCatalogError::ProviderMissing)
        ));
    }

    #[test]
    fn all_entries_unusable_is_a_data_failure() {
        let json = r#"{
          "opencode": {"models": {
            "a": {"limit": {"context": -1, "output": 1}},
            "b": {"limit": {"context": -2, "output": 2}}
          }}
        }"#;
        assert!(matches!(
            ModelCatalog::from_json(json.as_bytes()),
            Err(ModelCatalogError::Decode(_))
        ));
    }
}
