use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::tools::{
    invalid_input_failure, ToolFailure, ToolFailureKind, ToolInvocation, ToolOutcome, ToolRuntime,
    ToolRuntimeError, ToolSpec,
};

const DEFAULT_RESULT_COUNT: usize = 5;
const MAX_RESULT_COUNT: usize = 10;
const MAX_TITLE_CHARS: usize = 500;
const MAX_SNIPPET_CHARS: usize = 2_000;
const MAX_PROVIDER_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchRequest {
    pub query: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum WebSearchProviderError {
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("request timed out: {0}")]
    Timeout(String),
    #[error("provider request failed: {0}")]
    Request(String),
    #[error("provider returned an invalid response: {0}")]
    InvalidResponse(String),
    #[error("cancelled")]
    Cancelled,
}

#[async_trait]
pub trait WebSearchProvider: Send + Sync {
    fn id(&self) -> &str;

    async fn search(
        &self,
        request: WebSearchRequest,
        cancel: Option<&CancellationToken>,
    ) -> Result<Vec<WebSearchResult>, WebSearchProviderError>;
}

/// A standalone managed `web_search` tool. Compose it with filesystem or MCP
/// runtimes through `CompositeToolRuntime`; it is intentionally not part of
/// `builtin_tool_specs()` because search credentials are optional.
#[derive(Clone)]
pub struct WebSearchToolRuntime {
    provider: Arc<dyn WebSearchProvider>,
}

impl WebSearchToolRuntime {
    pub fn new(provider: Arc<dyn WebSearchProvider>) -> Self {
        Self { provider }
    }

    pub fn from_provider(provider: impl WebSearchProvider + 'static) -> Self {
        Self::new(Arc::new(provider))
    }
}

#[async_trait]
impl ToolRuntime for WebSearchToolRuntime {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![web_search_spec()]
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        self.invoke_cancellable(invocation, None).await
    }

    async fn invoke_cancellable(
        &self,
        invocation: ToolInvocation,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        if invocation.name != "web_search" {
            return Err(ToolRuntimeError::UnknownTool(invocation.name));
        }

        let request = match parse_request(&invocation) {
            Ok(request) => request,
            Err(failure) => {
                return Ok(ToolOutcome {
                    output: Err(failure),
                    attachments: vec![],
                });
            }
        };
        let query = request.query.clone();
        let requested_count = request.count;
        let provider = self.provider.id().to_string();
        let outcome = self.provider.search(request, cancel).await;
        let output = match outcome {
            Ok(results) => {
                let mut truncated = results.len() > requested_count;
                let results = results
                    .into_iter()
                    .take(requested_count)
                    .filter_map(|result| normalize_result(result, &mut truncated))
                    .collect::<Vec<_>>();
                Ok(json!({
                    "query": query,
                    "provider": provider,
                    "results": results,
                    "count": results.len(),
                    "truncated": truncated,
                    "external_content": {
                        "untrusted": true,
                        "source": "web_search"
                    }
                }))
            }
            Err(error) => Err(provider_failure(error)),
        };
        Ok(ToolOutcome {
            output,
            attachments: vec![],
        })
    }
}

pub fn web_search_spec() -> ToolSpec {
    ToolSpec {
        name: "web_search".into(),
        description: "Search the public web and return current titles, URLs, and snippets. Use web_fetch to read a selected result in full.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query."
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, maximum 10).",
                    "minimum": 1,
                    "maximum": MAX_RESULT_COUNT
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
    }
}

fn parse_request(invocation: &ToolInvocation) -> Result<WebSearchRequest, ToolFailure> {
    let query = invocation
        .input
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .ok_or_else(|| {
            invalid(
                invocation,
                "missing required non-empty string field `query`",
            )
        })?;
    let count = match invocation.input.get("count") {
        Some(value) => value
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| invalid(invocation, "count must be an integer from 1 to 10"))?,
        None => DEFAULT_RESULT_COUNT,
    };
    if !(1..=MAX_RESULT_COUNT).contains(&count) {
        return Err(invalid(invocation, "count must be an integer from 1 to 10"));
    }
    Ok(WebSearchRequest {
        query: query.to_string(),
        count,
    })
}

fn invalid(invocation: &ToolInvocation, message: &str) -> ToolFailure {
    ToolFailure::new(
        ToolFailureKind::InvalidInput,
        invalid_input_failure("web_search", message, &invocation.input, None).message,
    )
}

fn provider_failure(error: WebSearchProviderError) -> ToolFailure {
    let kind = match error {
        WebSearchProviderError::Timeout(_) => ToolFailureKind::Timeout,
        WebSearchProviderError::Cancelled => ToolFailureKind::Runtime,
        WebSearchProviderError::Auth(_)
        | WebSearchProviderError::Request(_)
        | WebSearchProviderError::InvalidResponse(_) => ToolFailureKind::Runtime,
    };
    ToolFailure::new(kind, error.to_string())
}

fn normalize_result(mut result: WebSearchResult, truncated: &mut bool) -> Option<WebSearchResult> {
    let parsed = match reqwest::Url::parse(result.url.trim()) {
        Ok(parsed) => parsed,
        Err(_) => {
            *truncated = true;
            return None;
        }
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        *truncated = true;
        return None;
    }
    result.url = parsed.to_string();
    result.title = wrap_untrusted(&truncate_chars(&result.title, MAX_TITLE_CHARS, truncated));
    result.snippet = wrap_untrusted(&truncate_chars(
        &result.snippet,
        MAX_SNIPPET_CHARS,
        truncated,
    ));
    result.published_at = result
        .published_at
        .as_deref()
        .map(|value| wrap_untrusted(&truncate_chars(value, 100, truncated)));
    if result.title.is_empty() && result.snippet.is_empty() {
        return None;
    }
    Some(result)
}

fn wrap_untrusted(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    // Prevent provider text from forging our own boundary markers.
    let escaped = value.replace("<<<", "< < <").replace(">>>", "> > >");
    format!(
        "<<<EXTERNAL_UNTRUSTED_CONTENT source=\"web_search\">>>\n{escaped}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>"
    )
}

fn truncate_chars(value: &str, max: usize, truncated: &mut bool) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    *truncated = true;
    value.chars().take(max).collect()
}

#[derive(Clone)]
pub struct BraveSearchConfig {
    pub api_key: String,
    pub timeout: Duration,
}

impl BraveSearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

#[derive(Clone)]
pub struct BraveSearchProvider {
    http: reqwest::Client,
    config: BraveSearchConfig,
}

impl BraveSearchProvider {
    pub fn new(config: BraveSearchConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { http, config }
    }
}

#[derive(Deserialize)]
struct BraveResponse {
    #[serde(default)]
    web: Option<BraveWeb>,
}

#[derive(Deserialize)]
struct BraveWeb {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
    age: Option<String>,
}

#[async_trait]
impl WebSearchProvider for BraveSearchProvider {
    fn id(&self) -> &str {
        "brave"
    }

    async fn search(
        &self,
        request: WebSearchRequest,
        cancel: Option<&CancellationToken>,
    ) -> Result<Vec<WebSearchResult>, WebSearchProviderError> {
        if self.config.api_key.trim().is_empty() {
            return Err(WebSearchProviderError::Auth(
                "Brave Search API key is empty".into(),
            ));
        }
        let send = self
            .http
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &self.config.api_key)
            .query(&[("q", request.query), ("count", request.count.to_string())])
            .timeout(self.config.timeout)
            .send();
        let response = if let Some(cancel) = cancel {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(WebSearchProviderError::Cancelled),
                response = send => response,
            }
        } else {
            send.await
        }
        .map_err(|error| {
            if error.is_timeout() {
                WebSearchProviderError::Timeout(error.to_string())
            } else {
                WebSearchProviderError::Request(error.to_string())
            }
        })?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(WebSearchProviderError::Auth(format!(
                "Brave Search returned HTTP {}",
                status.as_u16()
            )));
        }
        if !status.is_success() {
            return Err(WebSearchProviderError::Request(format!(
                "Brave Search returned HTTP {}",
                status.as_u16()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut body = Vec::new();
        loop {
            let next = if let Some(cancel) = cancel {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(WebSearchProviderError::Cancelled),
                    next = stream.next() => next,
                }
            } else {
                stream.next().await
            };
            let Some(chunk) = next else {
                break;
            };
            let chunk =
                chunk.map_err(|error| WebSearchProviderError::Request(error.to_string()))?;
            if body.len() + chunk.len() > MAX_PROVIDER_RESPONSE_BYTES {
                return Err(WebSearchProviderError::InvalidResponse(format!(
                    "response exceeded {MAX_PROVIDER_RESPONSE_BYTES} bytes"
                )));
            }
            body.extend_from_slice(&chunk);
        }
        let payload = serde_json::from_slice::<BraveResponse>(&body)
            .map_err(|error| WebSearchProviderError::InvalidResponse(error.to_string()))?;
        Ok(payload
            .web
            .map(|web| web.results)
            .unwrap_or_default()
            .into_iter()
            .map(|result| WebSearchResult {
                title: result.title,
                url: result.url,
                snippet: result.description,
                published_at: result.age,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct FakeProvider;

    #[async_trait]
    impl WebSearchProvider for FakeProvider {
        fn id(&self) -> &str {
            "fake"
        }

        async fn search(
            &self,
            _request: WebSearchRequest,
            _cancel: Option<&CancellationToken>,
        ) -> Result<Vec<WebSearchResult>, WebSearchProviderError> {
            Ok(vec![
                WebSearchResult {
                    title: "Valid".into(),
                    url: "https://example.com/result".into(),
                    snippet: "Current information".into(),
                    published_at: None,
                },
                WebSearchResult {
                    title: "Unsafe".into(),
                    url: "file:///etc/passwd".into(),
                    snippet: "discard me".into(),
                    published_at: None,
                },
            ])
        }
    }

    fn invocation(input: Value) -> ToolInvocation {
        ToolInvocation {
            id: "search-1".into(),
            name: "web_search".into(),
            input,
            raw_emitted_args: None,
        }
    }

    #[tokio::test]
    async fn runtime_normalizes_results_and_rejects_non_http_urls() {
        let runtime = WebSearchToolRuntime::from_provider(FakeProvider);
        let output = runtime
            .invoke(invocation(json!({"query": "rust", "count": 5})))
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(output["provider"], "fake");
        assert_eq!(output["results"].as_array().unwrap().len(), 1);
        assert_eq!(output["external_content"]["untrusted"], true);
        assert!(output["results"][0]["title"]
            .as_str()
            .unwrap()
            .contains("EXTERNAL_UNTRUSTED_CONTENT"));
    }

    #[tokio::test]
    async fn runtime_rejects_invalid_count() {
        let runtime = WebSearchToolRuntime::from_provider(FakeProvider);
        for count in [json!(11), json!("5")] {
            let failure = runtime
                .invoke(invocation(json!({"query": "rust", "count": count})))
                .await
                .unwrap()
                .output
                .unwrap_err();
            assert_eq!(failure.kind, ToolFailureKind::InvalidInput);
        }
    }
}
