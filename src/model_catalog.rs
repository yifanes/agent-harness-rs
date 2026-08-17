//! Model capabilities sourced exclusively from `models.dev["opencode"]`.
//!
//! Hosts initialize the catalog explicitly during startup, resolve a short
//! model id once, and keep the resulting [`ResolvedModelConfig`] for the
//! lifetime of the session. There are deliberately no built-in model profiles,
//! cross-provider merges, or fallback limits.

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
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    OpenAiCompatible,
    Anthropic,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterleavedReasoning {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
}

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
    #[error("model `{model}` is not present in models.dev[{PROVIDER}]{suggestions}")]
    ModelNotFound { model: String, suggestions: String },
    #[error("model id `{model}` is ambiguous in models.dev[{PROVIDER}]: {matches:?}")]
    AmbiguousModel { model: String, matches: Vec<String> },
    #[error("model `{model}` is deprecated")]
    DeprecatedModel { model: String },
    #[error("invalid model request for `{model}`: {message}")]
    InvalidRequest { model: String, message: String },
}

#[derive(Debug, Clone, Deserialize)]
struct CatalogRoot(HashMap<String, ProviderEntry>);

#[derive(Debug, Clone, Deserialize)]
struct ProviderEntry {
    models: HashMap<String, ModelEntry>,
}

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
        }
    }
}

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
    /// unique basename. Contains/fuzzy matches are suggestions only.
    pub async fn resolve(
        &self,
        request: ModelRequestConfig,
        wire_protocol: WireProtocol,
    ) -> Result<ResolvedModelConfig, ModelCatalogError> {
        let snapshot = self.snapshot.read().await;
        let requested = request.model.trim().to_ascii_lowercase();
        let capabilities = match snapshot.models.get(&requested) {
            Some(model) => model.clone(),
            None => resolve_unique_basename(&snapshot.models, &requested)?,
        };
        drop(snapshot);
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

    pub async fn model_count(&self) -> usize {
        self.snapshot.read().await.models.len()
    }
}

fn parse_snapshot(bytes: &[u8]) -> Result<CatalogSnapshot, ModelCatalogError> {
    let root: CatalogRoot = serde_json::from_slice(bytes)
        .map_err(|error| ModelCatalogError::Decode(error.to_string()))?;
    let provider = root
        .0
        .get(PROVIDER)
        .ok_or(ModelCatalogError::ProviderMissing)?;
    let models = provider
        .models
        .clone()
        .into_iter()
        .map(|(id, entry)| {
            let key = id.to_ascii_lowercase();
            (key, entry.into_capabilities(id))
        })
        .collect();
    Ok(CatalogSnapshot { models })
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
    if capabilities.status.as_deref() == Some("deprecated") {
        return Err(ModelCatalogError::DeprecatedModel {
            model: capabilities.id.clone(),
        });
    }
    if request.max_output_tokens == 0 {
        return Err(invalid(
            "max_output_tokens must be greater than zero".into(),
        ));
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
    if !capabilities.reasoning && !matches!(request.reasoning.mode, ReasoningMode::Default) {
        return Err(invalid("reasoning is not supported".into()));
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

    if matches!(wire_protocol, WireProtocol::OpenAiResponses) {
        if request.reasoning.budget_tokens.is_some() {
            return Err(invalid(
                "openai_responses cannot express reasoning budget_tokens".into(),
            ));
        }
        if !matches!(request.reasoning.mode, ReasoningMode::Default)
            && request.reasoning.effort.is_none()
            && !supports_none(effort_values)
        {
            return Err(invalid(
                "openai_responses requires an effort-based reasoning control".into(),
            ));
        }
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

    fn catalog(json: &str) -> ModelCatalog {
        ModelCatalog {
            snapshot: Arc::new(RwLock::new(parse_snapshot(json.as_bytes()).unwrap())),
            refresh_lock: Arc::new(Mutex::new(())),
            refresh_generation: Arc::new(AtomicU64::new(0)),
            cache_path: None,
            models_url: "unused".into(),
        }
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
        let error = catalog(FIXTURE)
            .resolve(request("deepseek-v4"), WireProtocol::OpenAiCompatible)
            .await
            .unwrap_err();
        assert!(matches!(error, ModelCatalogError::ModelNotFound { .. }));
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
}
