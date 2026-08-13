use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, StreamExt};
use serde_json::{json, Value};

use crate::event::HarnessUsage;
use crate::tools::{ToolInvocation, ToolSpec};

/// One token-level event from the model. The harness loop consumes a
/// `Stream<ModelChunk>` and forwards `TextDelta` straight through as
/// `HarnessInternalEvent::AssistantTextChunk` so callers see live
/// generation; tool-call chunks are accumulated internally and only
/// emitted once `ToolCallEnd` lands so the harness can dispatch the call
/// with a complete `serde_json::Value` input.
///
/// Provider-agnostic: `OpenAiCompatibleModelClient` translates from
/// OpenAI's chat-completions SSE shape, and the future Anthropic client
/// will project Messages-API events into the same enum. `ScriptedModelClient`
/// synthesizes whichever sequence its scripted `ModelResponse` would
/// have implied.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelChunk {
    TextDelta {
        msg_id: String,
        delta: String,
    },
    ThinkingDelta {
        thinking_id: String,
        delta: String,
        /// Provider-emitted signature (Anthropic extended thinking) that
        /// MUST be round-tripped to history verbatim, otherwise some
        /// providers reject the next turn. `None` for OpenAI today.
        signature: Option<String>,
    },
    /// First chunk of a tool call. The harness records `(id, name)` and
    /// starts buffering `ToolCallInputDelta`s under this id.
    ToolCallStart {
        id: String,
        name: String,
    },
    /// Streaming JSON arguments for a previously-announced tool call.
    /// OpenAI emits these as a string that, when concatenated, is valid
    /// JSON. Harness accumulates these into a single string per id, then
    /// parses on `ToolCallEnd`.
    ToolCallInputDelta {
        id: String,
        delta: String,
    },
    /// Tool call finalised. `input` is the parsed JSON value if the
    /// provider sent the full object on this chunk (Anthropic), or a
    /// placeholder if the harness still needs to parse the accumulated
    /// `ToolCallInputDelta` buffer (OpenAI). Either way the harness
    /// treats `input` as authoritative when present.
    ToolCallEnd {
        id: String,
        input: Option<Value>,
    },
    /// Final chunk. `usage` carries the provider's reported token count
    /// for this call (None if the gateway elides it).
    Done {
        stop_reason: String,
        usage: Option<HarnessUsage>,
    },
}

/// Verbatim thinking block emitted by Anthropic extended thinking. The
/// `signature` is a provider-supplied opaque token that MUST be
/// round-tripped on subsequent turns — Anthropic rejects modified
/// thinking blocks. OpenAI does not emit thinking blocks at all, so this
/// field is always `None` on that path.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssistantThinking {
    pub text: String,
    pub signature: Option<String>,
}

/// Image attached to a user message. Both providers accept either inline
/// base64 bytes or a URL the provider fetches; we surface both shapes
/// rather than always inlining (URLs save bandwidth + sandbox upload).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImageSource {
    /// IANA media type. Anthropic accepts `image/jpeg`, `image/png`,
    /// `image/gif`, `image/webp`; OpenAI accepts the same set. Both
    /// validate at request time — junk values surface as 400.
    pub media_type: String,
    pub data: ImageData,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ImageData {
    /// Base64-encoded image bytes. Most universal — supported by every
    /// modern multimodal model. Encoded without the `data:image/...;base64,`
    /// prefix (the projection layer adds it when the provider's wire
    /// format requires it; e.g. OpenAI image_url).
    Base64(String),
    /// URL the provider fetches. Provider-side IP egress / latency
    /// tradeoff — convenient for public assets, brittle for private ones.
    Url(String),
}

/// One attachment on a `ChatMessage::User`. `Image` is the only variant
/// today; documents / file_id round-trips slot in as future variants
/// without breaking pattern matches (callers should use `, ..` rest
/// pattern when destructuring User to stay forward-compat).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UserAttachment {
    Image(ImageSource),
}

/// One conversation entry as seen by the model. Mirrors the
/// `system / user / assistant / tool` set OpenAI chat/completions expects.
/// Anthropic's Messages API uses a different shape (tool_use / tool_result
/// content blocks instead of separate `tool` role) but consumes the same
/// `ChatMessage` history — the projection lives in the per-provider
/// model client, keeping the harness loop provider-agnostic.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ChatMessage {
    User {
        content: String,
        /// Non-text attachments (images today; file_id / documents later).
        /// Renders as additional content blocks alongside the `content`
        /// text on the wire — projection per provider in §8 / §9.
        attachments: Vec<UserAttachment>,
    },
    /// Assistant turn. May carry any combination of: a `thinking` block
    /// (Anthropic extended thinking), final text, and one+ tool calls.
    /// All three render into the assistant message's content array on
    /// the Anthropic wire; OpenAI uses `tool_calls` for tool calls and
    /// ignores thinking entirely.
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolInvocation>,
        /// Thinking block to round-trip verbatim. `None` for OpenAI /
        /// Anthropic-without-extended-thinking turns.
        thinking: Option<AssistantThinking>,
        /// Provider-reported token tally for this assistant turn, when the
        /// provider supplied one. Records the exact context size up to and
        /// including this message, so context-size estimation can anchor on
        /// a measured value and estimate only the messages appended after it.
        /// `None` when the turn carried no reported usage.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<crate::event::HarnessUsage>,
    },
    /// Tool response paired by `tool_call_id`. `content` is the
    /// serialized tool output (model side sees a string regardless of
    /// the underlying JSON shape). `attachments` carries structured
    /// non-text content the tool produced — e.g. a screenshot MCP tool
    /// returning an image. Only providers with a tool-content array
    /// (Anthropic) surface these on the wire; OpenAI tool role is
    /// strictly string-typed, so attachments degrade to a placeholder
    /// in the `content` string there.
    Tool {
        tool_call_id: String,
        content: String,
        is_error: bool,
        attachments: Vec<UserAttachment>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelTurnInput {
    /// Optional system prompt (already composed: spec_snapshot system_prompt
    /// + driver.append_system_prompt). `None` ⇒ no system message sent.
    pub system_prompt: Option<String>,
    /// Full conversation history for this turn. AgentLoopHarness appends
    /// each `Assistant` / `Tool` message as the loop progresses so the
    /// model retains its own prior reasoning across tool round-trips.
    pub messages: Vec<ChatMessage>,
    /// Tool specs available this turn. Sourced from `ToolRuntime::specs()`
    /// so adding / removing a tool changes one place. Empty Vec ⇒ no tools
    /// advertised (final-answer-only mode).
    pub tools: Vec<ToolSpec>,
    /// Provider-executed tools. These are not dispatched through
    /// `ToolRuntime`; the model provider runs them server-side and streams
    /// the final answer back through normal text deltas.
    pub hosted_tools: Vec<HostedTool>,
    /// How the model should pick (or skip) tools. Defaults to `Auto`.
    /// Set via `AgentLoopHarness::with_tool_choice` from
    /// `bootstrap.driver.native_model.tool_choice`.
    pub tool_choice: ToolChoice,
    /// Whether the model may emit multiple tool_use blocks in one
    /// response (OpenAI's `parallel_tool_calls`). `None` ⇒ provider
    /// default (true for OpenAI). Anthropic is always implicitly
    /// multi-tool-capable so this field is OpenAI-only.
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HostedCapability {
    WebSearch,
}

/// Whether a concrete model client can carry a hosted capability over its
/// current wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CapabilitySupport {
    Supported,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HostedTool {
    WebSearch,
}

/// How the model should route tool selection for the current turn.
/// Mapped to each provider's wire field by `chat_request_body`:
///
/// | variant      | OpenAI                                                  | Anthropic                       |
/// |--------------|---------------------------------------------------------|---------------------------------|
/// | `Auto`       | `"auto"` (or omitted)                                   | `{"type":"auto"}` (or omitted)  |
/// | `None`       | `"none"` — model MUST NOT call a tool                   | (degrades: tools field dropped) |
/// | `Required`   | `"required"` — model MUST call at least one tool        | `{"type":"any"}`                |
/// | `Tool(name)` | `{"type":"function","function":{"name":name}}` — forced | `{"type":"tool","name":name}`   |
///
/// The `None` variant has no direct Anthropic equivalent — Anthropic
/// forces tool consideration whenever `tools` is non-empty. We
/// approximate it by dropping the `tools` field entirely for that turn;
/// the model can't call what it doesn't know about.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Tool(String),
}

impl ToolChoice {
    /// Parse from the bootstrap.yaml string form.
    /// Accepts `"auto"`, `"none"`, `"required"`, `"tool:<name>"`.
    /// Empty / unknown strings degrade to `Auto` for forward compat —
    /// callers that want strict validation can match before calling.
    pub fn parse(s: &str) -> Self {
        let trimmed = s.trim();
        if let Some(name) = trimmed.strip_prefix("tool:") {
            return Self::Tool(name.trim().to_string());
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "none" => Self::None,
            "required" | "any" => Self::Required,
            _ => Self::Auto,
        }
    }
}

/// Response from a single model call. `Message` is a final answer (no
/// follow-up tool needed); `ToolCall` hands control back to the harness
/// loop to execute the tool and feed the result back next turn.
///
/// `usage` carries the provider-reported token tally for *this* call.
/// AgentLoopHarness accumulates these across all steps of a turn and
/// attaches the total to the final `HarnessInternalEvent::TurnEnd`.
/// `None` ⇒ provider didn't report usage on this call (e.g. mid-stream
/// chunk, scripted fake client).
#[derive(Debug, Clone, PartialEq)]
pub enum ModelResponse {
    Message {
        text: String,
        stop_reason: String,
        usage: Option<HarnessUsage>,
    },
    ToolCall {
        preface: Option<String>,
        invocation: ToolInvocation,
        usage: Option<HarnessUsage>,
    },
}

impl ModelResponse {
    /// Per-call usage, regardless of variant. Pulled out so AgentLoopHarness
    /// can fold it into the running total without matching on every branch.
    pub fn usage(&self) -> Option<&HarnessUsage> {
        match self {
            ModelResponse::Message { usage, .. } | ModelResponse::ToolCall { usage, .. } => {
                usage.as_ref()
            }
        }
    }
}

/// Categorised model-client failures. Mirrors `NativeHarnessError`'s
/// `Model*` variants 1:1 so the agent loop can map across without
/// pattern-matching gymnastics.
///
/// Retryability:
///   - `RateLimit` / `Network` / `ServerError` → transient, safe to retry with backoff
///   - `Auth` / `ContextOverflow` / `BadRequest` → config error, never retry
#[derive(Debug, thiserror::Error)]
pub enum ModelClientError {
    /// HTTP 429 — back off and retry.
    #[error("rate limit: {0}")]
    RateLimit(String),
    /// HTTP 401 / 403 — wrong key or no permission; do not retry.
    #[error("auth: {0}")]
    Auth(String),
    /// HTTP 400 that looks like context overflow; do not retry.
    #[error("context overflow: {0}")]
    ContextOverflow(String),
    /// HTTP 400 (invalid model, bad params, etc.) — config error; do not retry.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// HTTP 5xx or transport failure — transient; safe to retry.
    #[error("server error: {0}")]
    ServerError(String),
    /// DNS / TCP / TLS failure — transient; safe to retry.
    #[error("network: {0}")]
    Network(String),
    /// Anything else we couldn't bucket; treated as non-retryable.
    #[error("model error: {0}")]
    Other(String),
}

impl ModelClientError {
    /// Returns `true` if the caller should back off and retry this error.
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimit(_) | Self::Network(_) | Self::ServerError(_)
        )
    }
}

#[async_trait]
pub trait ModelClient: Send + Sync {
    /// Report support for provider-executed capabilities on this client's
    /// actual API path. Automatic routing only selects `Supported` values.
    fn hosted_capability(&self, capability: HostedCapability) -> CapabilitySupport;

    /// Stream the model's response as a sequence of `ModelChunk`s. Production
    /// `ScriptedModelClient` / `OpenAiCompatibleModelClient` implement this;
    /// `next()` is provided as a folding convenience for callers that don't
    /// need token-level events.
    async fn stream(
        &self,
        input: ModelTurnInput,
    ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>;

    /// Buffer the stream into a single `ModelResponse`. Default impl
    /// `await`s every chunk and then runs `collect_model_response`. Override
    /// only if a provider has a cheaper non-streaming path (none today —
    /// even OpenAI's non-stream call still goes through the same SSE-or-not
    /// branch on our side).
    async fn next(&self, input: ModelTurnInput) -> Result<ModelResponse, ModelClientError> {
        let stream = self.stream(input).await?;
        collect_model_response(stream).await
    }
}

/// Fold a `ModelChunk` stream into the legacy `ModelResponse` shape. Used
/// by tests, by the default `next()` impl, and by any caller that prefers
/// "one response at a time" over a streaming feed. Provider-agnostic —
/// the same logic works for OpenAI and Anthropic chunk streams because
/// `ModelChunk` is the projection target both clients normalise into.
pub async fn collect_model_response(
    mut stream: BoxStream<'static, Result<ModelChunk, ModelClientError>>,
) -> Result<ModelResponse, ModelClientError> {
    let mut text_buf = String::new();
    let mut text_msg_id: Option<String> = None;
    // Per tool-call-id: name + accumulated argument bytes + early input (Anthropic-style).
    let mut tool_states: Vec<ToolStreamState> = Vec::new();
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<HarnessUsage> = None;

    while let Some(item) = stream.next().await {
        match item? {
            ModelChunk::TextDelta { msg_id, delta } => {
                if text_msg_id.as_deref() != Some(&msg_id) {
                    text_msg_id = Some(msg_id);
                    text_buf.clear();
                }
                text_buf.push_str(&delta);
            }
            ModelChunk::ThinkingDelta { .. } => {
                // collect_model_response is a back-compat surface; thinking
                // blocks don't fit into the simple Message/ToolCall enum,
                // so we silently drop them. Callers that care use the
                // streaming API directly.
            }
            ModelChunk::ToolCallStart { id, name } => {
                tool_states.push(ToolStreamState {
                    id,
                    name,
                    args_buf: String::new(),
                    early_input: None,
                });
            }
            ModelChunk::ToolCallInputDelta { id, delta } => {
                if let Some(state) = tool_states.iter_mut().find(|s| s.id == id) {
                    state.args_buf.push_str(&delta);
                }
            }
            ModelChunk::ToolCallEnd { id, input } => {
                if let Some(state) = tool_states.iter_mut().find(|s| s.id == id) {
                    state.early_input = input;
                }
            }
            ModelChunk::Done {
                stop_reason: sr,
                usage: u,
            } => {
                stop_reason = Some(sr);
                usage = u;
            }
        }
    }

    // Tool calls take precedence — when the model decides to use a tool,
    // the loop must execute it before any final-answer text matters. Pick
    // the first tool call (multiple-tool parallelism is a future
    // extension; today RD-side serialises them).
    if let Some(state) = tool_states.into_iter().next() {
        let parsed_input = match state.early_input {
            Some(v) => v,
            None => serde_json::from_str(state.args_buf.as_str().trim()).map_err(|e| {
                ModelClientError::Other(format!(
                    "decode tool arguments for {id}: {e}",
                    id = state.id
                ))
            })?,
        };
        let raw_emitted_args = raw_args_for_input(&state.args_buf, &parsed_input);
        return Ok(ModelResponse::ToolCall {
            preface: (!text_buf.is_empty()).then(|| text_buf.clone()),
            invocation: ToolInvocation {
                id: state.id,
                name: state.name,
                input: parsed_input,
                raw_emitted_args,
            },
            usage,
        });
    }

    Ok(ModelResponse::Message {
        text: text_buf,
        stop_reason: stop_reason.unwrap_or_else(|| "end_turn".into()),
        usage,
    })
}

/// Per-tool-call buffer used while folding a stream. Lives inline in
/// `collect_model_response`; pulled out as a struct only because Rust
/// closures over a Vec of tuples get noisy fast.
struct ToolStreamState {
    id: String,
    name: String,
    args_buf: String,
    early_input: Option<Value>,
}

fn raw_args_for_input(raw: &str, input: &Value) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(parsed) if parsed == *input => Some(trimmed.to_string()),
        _ => None,
    }
}

fn tool_invocation_args_for_wire(tc: &ToolInvocation) -> String {
    tc.raw_emitted_args
        .as_deref()
        .and_then(|raw| raw_args_for_input(raw, &tc.input))
        .unwrap_or_else(|| tc.input.to_string())
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i32>,
    /// Reasoning effort hint: `"low"`, `"medium"`, or `"high"`. Sent as the
    /// top-level `reasoning_effort` field. `None` omits it (provider default).
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleModelClient {
    http: reqwest::Client,
    config: OpenAiCompatibleConfig,
}

impl OpenAiCompatibleModelClient {
    pub fn new(config: OpenAiCompatibleConfig) -> Self {
        // Fail fast if the TCP handshake takes too long. Do not set a
        // read_timeout: streaming model responses may legitimately pause
        // longer than a fixed per-read timeout between SSE frames.
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { http, config }
    }

    fn endpoint(&self) -> String {
        // `base_url` is the full API prefix supplied by the caller, including
        // any version segment — e.g. `https://api.openai.com/v1` or
        // `https://open.bigmodel.cn/api/paas/v4`. We only append the route;
        // picking the version is the caller's responsibility, since it differs
        // across OpenAI-compatible providers.
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }

    fn request_body(&self, input: &ModelTurnInput) -> Value {
        let mut messages = Vec::with_capacity(input.messages.len() + 1);
        if let Some(sys) = input.system_prompt.as_deref().filter(|s| !s.is_empty()) {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        for msg in &input.messages {
            messages.push(chat_message_to_wire(msg));
        }

        let mut body = json!({
            "model": self.config.model,
            "messages": messages,
        });
        // Tool advertising obeys `tool_choice`:
        //   - Auto / Required / Tool(name) — send tools + matching
        //     `tool_choice` field; model gets the wire-level constraint.
        //   - None — drop the tools entirely. OpenAI does accept
        //     `tool_choice: "none"` to forbid use, but dropping
        //     `tools` is cheaper (fewer prompt tokens) and the model
        //     can't call what it doesn't see.
        let send_tools = !input.tools.is_empty() && !matches!(input.tool_choice, ToolChoice::None);
        if send_tools {
            body["tools"] = json!(input
                .tools
                .iter()
                .map(tool_spec_to_openai_function)
                .collect::<Vec<_>>());
            body["tool_choice"] = openai_tool_choice_value(&input.tool_choice);
            if let Some(parallel) = input.parallel_tool_calls {
                body["parallel_tool_calls"] = json!(parallel);
            }
        }
        if let Some(temperature) = self.config.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(max_tokens) = self.config.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(effort) = self
            .config
            .reasoning_effort
            .as_deref()
            .filter(|s| !s.is_empty())
        {
            body["reasoning_effort"] = json!(effort);
        }
        body
    }
}

fn openai_tool_choice_value(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!("auto"),
        // ToolChoice::None lands in the caller's "drop tools" branch
        // so it never reaches here, but encode it defensively.
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

/// Pull token counts out of an OpenAI chat/completions `usage` block.
/// Returns `None` if the field is missing / empty — some compat gateways
/// (older DeepSeek, certain proxies) elide it. Maps directly:
///   * `prompt_tokens`        → `input_tokens`
///   * `completion_tokens`    → `output_tokens`
///   * `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`
///     (OpenAI semantics: cached_tokens is a subset of prompt_tokens — we
///     keep that view rather than subtracting; matches the proto field's
///     "展示提示" intent).
///   * cache_creation: no OpenAI equivalent → 0.
fn parse_openai_usage(usage: Option<&Value>) -> Option<HarnessUsage> {
    let u = usage?;
    let input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    // Heuristic: if every counter is zero, treat as "no data reported" so
    // downstream telemetry doesn't mis-attribute a no-op response.
    if input == 0 && output == 0 && cache_read == 0 {
        return None;
    }
    Some(HarnessUsage {
        input_tokens: input,
        output_tokens: output,
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: 0,
        // Provider-reported usage on the main step never carries
        // compaction tokens — agent_loop accumulates those separately
        // when a compaction summarize call returns its own usage.
        compaction_input_tokens: 0,
        compaction_output_tokens: 0,
    })
}

/// Render one `ToolSpec` to the OpenAI chat/completions `tools[*]` shape.
/// Pulled out so future Anthropic client can project the same spec to its
/// Messages API `input_schema` form without duplicating tool metadata.
/// Project an `ImageSource` to the OpenAI chat/completions `image_url`
/// content part shape. Inline base64 gets wrapped in a `data:` URI
/// because OpenAI's API expects the full URL form there — the model
/// won't accept raw base64 as a sibling key.
fn image_to_openai_part(src: &ImageSource) -> Value {
    let url = match &src.data {
        ImageData::Base64(b64) => {
            // `data:image/png;base64,xxx`. OpenAI documents this form
            // in the vision quickstart. We don't validate media_type
            // here — invalid values surface as a 400 at request time
            // (caller saw it in classify_openai_http_error).
            format!("data:{};base64,{}", src.media_type, b64)
        }
        ImageData::Url(u) => u.clone(),
    };
    json!({
        "type": "image_url",
        "image_url": { "url": url },
    })
}

fn tool_spec_to_openai_function(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        }
    })
}

/// Replay-compaction thresholds for a single tool result. A result is
/// compacted only when it exceeds the token estimate or the byte budget —
/// small results are never touched. The kept window is split half head /
/// half tail so the model sees the command's start AND its trailing
/// errors / summary.
const MAX_TOOL_RESULT_REPLAY_TOKENS: u64 = 2_000;
const MAX_TOOL_RESULT_REPLAY_BYTES: usize = 12 * 1024;
const COMPACTED_TOOL_RESULT_KEEP_CHARS: usize = 3_000;

/// Compact one oversized tool result for model replay. Applied at wire
/// projection time ONLY — the full result stays verbatim in the in-memory
/// history and in the persisted `messages.jsonl`, so resume and operators
/// keep the raw data while every subsequent model call stops re-paying
/// thousands of tokens for it. Deterministic (same input → same output),
/// which keeps the projected prefix byte-stable for provider prompt caches.
/// Orthogonal to message-level compaction: this trims per-result on every
/// call, without waiting for a context-window trigger.
fn compact_tool_result_for_replay(content: &str) -> std::borrow::Cow<'_, str> {
    let estimated_tokens = crate::compaction::estimate_tokens(content);
    if estimated_tokens <= MAX_TOOL_RESULT_REPLAY_TOKENS
        && content.len() <= MAX_TOOL_RESULT_REPLAY_BYTES
    {
        return std::borrow::Cow::Borrowed(content);
    }
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= COMPACTED_TOOL_RESULT_KEEP_CHARS {
        return std::borrow::Cow::Borrowed(content);
    }
    let head_len = COMPACTED_TOOL_RESULT_KEEP_CHARS / 2;
    let tail_len = COMPACTED_TOOL_RESULT_KEEP_CHARS - head_len;
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[chars.len() - tail_len..].iter().collect();
    let omitted = chars.len() - COMPACTED_TOOL_RESULT_KEEP_CHARS;
    std::borrow::Cow::Owned(format!(
        "[tool result compacted for model replay]\n\
         original_estimated_tokens={estimated_tokens} original_chars={} \
         retained_head_chars={head_len} retained_tail_chars={tail_len}\n\
         The full raw tool result remains in session history; this replay is abbreviated.\n\n\
         --- head ---\n{head}\n\n\
         --- omitted ---\n[... omitted {omitted} chars from tool result replay ...]\n\n\
         --- tail ---\n{tail}",
        chars.len(),
    ))
}

/// Render one `ChatMessage` to the OpenAI chat/completions wire shape.
/// Pulled out so tests and (future) other providers can share / diff the
/// projection.
fn chat_message_to_wire(msg: &ChatMessage) -> Value {
    match msg {
        ChatMessage::User {
            content,
            attachments,
        } => {
            // Fast path: no attachments → content stays as a plain
            // string. Keeps simple text-only requests byte-identical to
            // pre-multimodal output (Anthropic prompt cache friendliness
            // on OpenAI-compatible gateways that mirror that behaviour).
            if attachments.is_empty() {
                json!({ "role": "user", "content": content })
            } else {
                // Mixed multimodal — promote `content` to an array of
                // OpenAI vision parts. Text always lands first (the
                // common chat ordering); image_url parts follow. Empty
                // text is dropped — OpenAI accepts arrays with only
                // image parts.
                let mut parts: Vec<Value> = Vec::with_capacity(attachments.len() + 1);
                if !content.is_empty() {
                    parts.push(json!({ "type": "text", "text": content }));
                }
                for att in attachments {
                    match att {
                        UserAttachment::Image(src) => {
                            parts.push(image_to_openai_part(src));
                        }
                    }
                }
                json!({ "role": "user", "content": parts })
            }
        }
        ChatMessage::Assistant {
            text,
            tool_calls,
            thinking: _,
            usage: _,
        } => {
            // OpenAI chat/completions has no equivalent of Anthropic
            // thinking blocks; we drop the field here. The Anthropic
            // projection (chat_messages_to_anthropic_messages) consumes
            // it.
            let mut obj = json!({ "role": "assistant" });
            if let Some(t) = text.as_deref().filter(|s| !s.is_empty()) {
                obj["content"] = json!(t);
            } else {
                obj["content"] = Value::Null;
            }
            if !tool_calls.is_empty() {
                let calls: Vec<Value> = tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tool_invocation_args_for_wire(tc),
                            },
                        })
                    })
                    .collect();
                obj["tool_calls"] = json!(calls);
            }
            obj
        }
        ChatMessage::Tool {
            tool_call_id,
            content,
            attachments,
            is_error: _,
        } => {
            // OpenAI tool role is strictly string-typed (no content
            // block array). Non-text attachments (e.g. image returned
            // by an MCP screenshot tool) can't ride here — we surface
            // them as a placeholder appended to `content` so the model
            // is at least aware something visual was attached. Lossy
            // by design; if the agent needs to see the image, use the
            // Anthropic provider.
            let mut content_str = compact_tool_result_for_replay(content).into_owned();
            for att in attachments {
                let UserAttachment::Image(src) = att;
                content_str.push_str(&format!(
                    "\n[image attached: {} (not visible via OpenAI tool role)]",
                    src.media_type
                ));
            }
            json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content_str,
            })
        }
    }
}

#[async_trait]
impl ModelClient for OpenAiCompatibleModelClient {
    fn hosted_capability(&self, _capability: HostedCapability) -> CapabilitySupport {
        // Chat Completions has no portable hosted-tool contract. A gateway
        // calling itself OpenAI-compatible is not evidence that Responses
        // hosted tools are accepted.
        CapabilitySupport::Unsupported
    }

    async fn stream(
        &self,
        input: ModelTurnInput,
    ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError> {
        if !input.hosted_tools.is_empty() {
            return Err(ModelClientError::Other(
                "hosted tools are not supported by OpenAiCompatibleModelClient; \
                 OpenAI web_search requires a Responses API client"
                    .into(),
            ));
        }
        // Bolt-on streaming flags. `stream_options.include_usage` is a
        // recent OpenAI addition that makes the final SSE event carry the
        // usage block; without it streaming responses drop usage entirely.
        // Compat gateways (DeepSeek, Groq, etc.) mostly accept the field
        // and either honour it or ignore — passing it is forward-safe.
        let mut body = self.request_body(&input);
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });

        let resp = match self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(classify_reqwest_error(&e, e.to_string())),
        };
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(classify_openai_http_error(status, &body_text));
        }

        // bytes_stream → SSE event_stream → ModelChunk stream. The SSE
        // parser handles UTF-8 boundaries, multi-line `data:` reassembly,
        // and the `[DONE]` sentinel; we layer ModelChunk extraction on
        // top with an explicit state machine because OpenAI ships `id`
        // / `name` only on the first delta of each tool call.
        let event_stream = resp.bytes_stream().eventsource();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ModelChunk, ModelClientError>>(8);

        tokio::spawn(async move {
            let mut state = OpenAiStreamState::default();
            futures::pin_mut!(event_stream);
            while let Some(ev) = event_stream.next().await {
                let chunks = match ev {
                    Ok(event) => match state.feed_data(&event.data) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = tx
                            .send(Err(ModelClientError::Network(format!(
                                "SSE transport error: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                for c in chunks {
                    if tx.send(Ok(c)).await.is_err() {
                        return;
                    }
                }
            }
            // Stream closed. Distinguish a clean end from a premature cut-off:
            //   * clean  — we saw `[DONE]` or a `finish_reason`. Emit the final
            //     Done (covers gateways that close without `[DONE]` but DO send
            //     a finish_reason).
            //   * cut off — neither marker arrived. The connection dropped mid
            //     response (proxy/idle timeout, upstream truncation). Surfacing
            //     this as a (retryable) error instead of fabricating a Done is
            //     critical: otherwise a truncated answer is silently accepted as
            //     complete, and the user sees a half-finished reply with no hint
            //     anything went wrong.
            if state.ended_cleanly() {
                if let Some(final_chunk) = state.finalize() {
                    let _ = tx.send(Ok(final_chunk)).await;
                }
            } else {
                let _ = tx
                    .send(Err(ModelClientError::Network(
                        "model stream closed before completion (no finish_reason or [DONE]) \
                         — connection dropped or upstream truncated the response"
                            .into(),
                    )))
                    .await;
            }
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }
}

/// State machine that turns OpenAI chat.completion.chunk SSE events into
/// `ModelChunk` stream items. Lives outside the trait impl so the parsing
/// logic is unit-testable without standing up an HTTP server.
#[derive(Debug, Default)]
struct OpenAiStreamState {
    /// Carries the assistant message id forward from the first chunk that
    /// supplied it (OpenAI uses `chatcmpl-...`). Falls back to a synthetic
    /// id if the provider doesn't send one — text deltas without an id
    /// would otherwise be dropped on the floor by `native_adapter`'s chunk
    /// accumulator.
    msg_id: Option<String>,
    /// Maps the OpenAI `tool_calls[i].index` to the eventual call id so
    /// subsequent `arguments` deltas can route to the right buffer.
    tool_call_by_index: std::collections::HashMap<u64, String>,
    finish_reason: Option<String>,
    pending_usage: Option<HarnessUsage>,
    /// Set once `[DONE]` or a finish_reason chunk lands; suppresses the
    /// duplicate `Done` we'd otherwise emit from `finalize()`.
    done_emitted: bool,
}

impl OpenAiStreamState {
    fn feed_data(&mut self, data: &str) -> Result<Vec<ModelChunk>, ModelClientError> {
        // `[DONE]` is OpenAI's end-of-stream sentinel; some compat gateways
        // also emit it, others just close. Either path lands in finalize().
        if data.trim() == "[DONE]" {
            if let Some(done) = self.emit_done() {
                return Ok(vec![done]);
            }
            return Ok(vec![]);
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|e| ModelClientError::Other(format!("SSE data not JSON: {e}; raw={data}")))?;

        let mut out: Vec<ModelChunk> = Vec::new();

        // The final chunk on `stream_options.include_usage=true` arrives
        // with empty `choices` and a populated `usage` block.
        if let Some(usage) = parse_openai_usage(value.get("usage")) {
            self.pending_usage = Some(usage);
        }

        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            if self.msg_id.is_none() && !id.is_empty() {
                self.msg_id = Some(id.to_string());
            }
        }

        let Some(choices) = value.get("choices").and_then(|v| v.as_array()) else {
            return Ok(out);
        };
        let Some(choice) = choices.first() else {
            return Ok(out);
        };
        let Some(delta) = choice.get("delta") else {
            // Some gateways send a usage-only chunk with no delta — fine.
            if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                self.finish_reason = Some(reason.to_string());
            }
            return Ok(out);
        };

        // Text token delta. Falls back to "msg_native_default" if the
        // provider hasn't emitted a chunk id; `native_adapter`'s
        // accumulator groups deltas by id so we must keep it stable.
        if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                let msg_id = self
                    .msg_id
                    .clone()
                    .unwrap_or_else(|| "msg_native_default".to_string());
                out.push(ModelChunk::TextDelta {
                    msg_id,
                    delta: text.to_string(),
                });
            }
        }

        // Tool call deltas. Each entry in `tool_calls[]` has an `index`
        // that's stable across chunks; the first chunk for a given index
        // carries `id` + `function.name`, later chunks just stream
        // `function.arguments`. We route every arguments delta through
        // the index→id map.
        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        self.tool_call_by_index.insert(index, id.to_string());
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        out.push(ModelChunk::ToolCallStart {
                            id: id.to_string(),
                            name,
                        });
                    }
                }
                // Streaming argument bytes — concatenation across chunks
                // forms a single JSON object string. Handed back as a
                // delta so collect_model_response (or any other folder)
                // can accumulate.
                if let Some(args) = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    if let Some(id) = self.tool_call_by_index.get(&index).cloned() {
                        if !args.is_empty() {
                            out.push(ModelChunk::ToolCallInputDelta {
                                id,
                                delta: args.to_string(),
                            });
                        }
                    }
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            self.finish_reason = Some(reason.to_string());
            // OpenAI's tool-call streaming ends with finish_reason="tool_calls";
            // emit a `ToolCallEnd` for each open tool call (without a parsed
            // input — `collect_model_response` will parse the accumulated
            // buffer if it needs the value).
            if reason == "tool_calls" {
                for (_idx, id) in self.tool_call_by_index.iter() {
                    out.push(ModelChunk::ToolCallEnd {
                        id: id.clone(),
                        input: None,
                    });
                }
            }
            // Some gateways then close the stream without `[DONE]`. We
            // could emit Done eagerly here, but to handle the include_usage
            // case (usage arrives in a later chunk), defer to finalize().
        }

        Ok(out)
    }

    fn finalize(&mut self) -> Option<ModelChunk> {
        self.emit_done()
    }

    /// True once we've observed a legitimate end-of-response signal — either
    /// the `[DONE]` sentinel (sets `done_emitted`) or a chunk carrying a
    /// `finish_reason` (e.g. `stop` / `length` / `tool_calls`). When a stream
    /// closes WITHOUT either, the response was cut off mid-flight (dropped
    /// connection, proxy/idle timeout, upstream truncation) and must NOT be
    /// treated as a complete answer.
    fn ended_cleanly(&self) -> bool {
        self.done_emitted || self.finish_reason.is_some()
    }

    fn emit_done(&mut self) -> Option<ModelChunk> {
        if self.done_emitted {
            return None;
        }
        self.done_emitted = true;
        let stop_reason = map_openai_finish_reason(self.finish_reason.as_deref());
        Some(ModelChunk::Done {
            stop_reason,
            usage: self.pending_usage.take(),
        })
    }
}

/// Map OpenAI's `finish_reason` to the harness's stop_reason vocabulary
/// used by `dispatch::map_stop_reason`. Unknown / null values fall back
/// to "end_turn" (the model produced something), not "unknown_stop_reason"
/// (which `dispatch::map_stop_reason` would otherwise classify as
/// RUNTIME_ERROR — wrong here, the call succeeded).
fn map_openai_finish_reason(reason: Option<&str>) -> String {
    match reason {
        Some("stop") => "end_turn".into(),
        Some("length") => "max_tokens".into(),
        Some("tool_calls") => "end_turn".into(),
        Some("content_filter") => "refusal".into(),
        Some(other) if !other.is_empty() => other.to_string(),
        _ => "end_turn".into(),
    }
}

/// Bucket an HTTP error into the right `ModelClientError` variant. Looks
/// at status code first; falls back to the response body for the trickier
/// case (BadRequest could be context overflow, malformed prompt, or just
/// a bad parameter).
fn classify_openai_http_error(status: reqwest::StatusCode, body: &str) -> ModelClientError {
    use reqwest::StatusCode;
    let snippet = body.chars().take(512).collect::<String>();

    // Retryable errors
    if status == StatusCode::TOO_MANY_REQUESTS {
        return ModelClientError::RateLimit(format!("HTTP {status}: {snippet}"));
    }
    if status.is_server_error() {
        // 5xx are transient — proxy overloaded, upstream error, etc.
        return ModelClientError::ServerError(format!("HTTP {status}: {snippet}"));
    }

    // Non-retryable config/client errors
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return ModelClientError::Auth(format!("HTTP {status}: {snippet}"));
    }
    if status == StatusCode::BAD_REQUEST && looks_like_context_overflow(body) {
        return ModelClientError::ContextOverflow(format!("HTTP {status}: {snippet}"));
    }
    if status == StatusCode::BAD_REQUEST {
        // "Invalid model name", missing required fields, etc. — config error.
        return ModelClientError::BadRequest(format!("HTTP {status}: {snippet}"));
    }

    ModelClientError::Other(format!("HTTP {status}: {snippet}"))
}

/// Heuristic match against the common phrasings OpenAI-compatible
/// providers use to flag "this prompt is too long for the model". Different
/// gateways word it differently; we widen the net by lowercasing + token
/// search rather than trying to JSON-parse the body (which may be a
/// non-JSON HTML 400 from some proxies).
fn looks_like_context_overflow(body: &str) -> bool {
    let lower = body.to_lowercase();
    lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("context_length_exceeded")
        || lower.contains("too many tokens")
        || lower.contains("exceeds the model")
}

/// Map a `reqwest::Error` to either `Network` (transport-layer / DNS /
/// connect / timeout / body) or `Other` (everything else — e.g. invalid
/// URL built locally, which is a programmer bug rather than transient).
fn classify_reqwest_error(err: &reqwest::Error, msg: String) -> ModelClientError {
    if err.is_connect() || err.is_timeout() || err.is_request() || err.is_body() {
        ModelClientError::Network(msg)
    } else {
        ModelClientError::Other(msg)
    }
}

/// In-process model client used by tests / dev. Inspects the most recent
/// `User` message in `input.messages` and either fires a deterministic
/// tool call (if a prior `Tool` result hasn't landed yet) or emits a
/// summary text reply (once it has).
///
/// Implements `stream` natively so the agent loop exercises the
/// streaming code path even in tests; the chunk sequence it produces
/// matches what `OpenAiCompatibleModelClient` would emit for the same
/// logical response.
#[derive(Debug, Default, Clone)]
pub struct ScriptedModelClient;

#[async_trait]
impl ModelClient for ScriptedModelClient {
    fn hosted_capability(&self, _capability: HostedCapability) -> CapabilitySupport {
        CapabilitySupport::Unsupported
    }

    async fn stream(
        &self,
        input: ModelTurnInput,
    ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError> {
        let chunks = scripted_chunks_for(&input);
        let stream = futures::stream::iter(chunks.into_iter().map(Ok));
        Ok(stream.boxed())
    }
}

/// Decide which `ModelChunk` sequence the scripted client would emit for a
/// given `ModelTurnInput`. Same heuristics as the old `next()` impl, just
/// rendered as a stream so tests that expect tool→summary→done chunking
/// see the right shape.
fn scripted_chunks_for(input: &ModelTurnInput) -> Vec<ModelChunk> {
    // If we already have a tool result in history, produce the final
    // summary message — text-only chunk followed by Done.
    let last_tool = input.messages.iter().rev().find_map(|m| match m {
        ChatMessage::Tool {
            tool_call_id,
            content,
            is_error,
            ..
        } => Some((tool_call_id.clone(), content.clone(), *is_error)),
        _ => None,
    });
    if let Some((id, content, is_error)) = last_tool {
        let summary = if is_error {
            format!("tool {id} failed: {content}")
        } else {
            format!("tool {id} completed: {content}")
        };
        return vec![
            ModelChunk::TextDelta {
                msg_id: "scripted_msg".into(),
                delta: summary,
            },
            ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: None,
            },
        ];
    }

    // Otherwise look at the latest user prompt and pick a tool.
    let user_prompt = input
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            ChatMessage::User { content, .. } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let prompt = user_prompt.trim();
    let (id, name, args) = if let Some(path) = prompt.strip_prefix("read ") {
        ("tc_read_1", "read", json!({"path": path.trim()}))
    } else if let Some(rest) = prompt.strip_prefix("write ") {
        let (path, content) = rest.split_once(' ').unwrap_or((rest, ""));
        (
            "tc_write_1",
            "write",
            json!({"path": path.trim(), "content": content}),
        )
    } else {
        ("tc_bash_1", "bash", json!({"command": prompt}))
    };

    vec![
        ModelChunk::TextDelta {
            msg_id: "scripted_msg".into(),
            delta: format!("native model selected tool: {name}"),
        },
        ModelChunk::ToolCallStart {
            id: id.into(),
            name: name.into(),
        },
        ModelChunk::ToolCallEnd {
            id: id.into(),
            input: Some(args),
        },
        ModelChunk::Done {
            stop_reason: "end_turn".into(),
            usage: None,
        },
    ]
}

// ─── Anthropic Messages API ──────────────────────────────────────────────
//
// Streaming Messages API differs from OpenAI chat/completions in three
// load-bearing places:
//
//   1. Endpoint + auth   POST /v1/messages with `x-api-key` +
//                        `anthropic-version: 2023-06-01`, no Bearer.
//   2. Body              `system` is a top-level field (not a chat role);
//                        `tool_result` lives as a content block inside the
//                        next user message (not a separate `tool` role);
//                        `max_tokens` is required.
//   3. SSE shape         Uses both `event:` and `data:` lines. Event types
//                        partition the chunk types we care about, and tool
//                        arguments arrive as `input_json_delta` strings
//                        that concatenate into a single JSON object.
//
// All of the work in this section is wire-shape translation; the harness
// loop and `ModelChunk` enum stay provider-agnostic.

/// Deployment-side credentials + endpoint for the Anthropic Messages API.
/// `max_tokens` is required by Anthropic on every request, so we hold it
/// here (default below) rather than relying on the agent recipe.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Required by Anthropic. Defaults to 4096 if the spec snapshot
    /// doesn't override; we surface it on the config so HR can tune via
    /// `spec_snapshot.agent_spec_version.model_config.max_tokens`.
    pub max_tokens: i32,
    pub temperature: Option<f64>,
    /// Wire-format version pin. Defaults to "2023-06-01" — the only one
    /// supported by Messages API at time of writing. Override if Anthropic
    /// ever publishes a newer one we want to opt into.
    pub anthropic_version: String,
}

impl AnthropicConfig {
    /// Default `anthropic-version` header value the client sends if HR
    /// doesn't override it.
    pub const DEFAULT_VERSION: &'static str = "2023-06-01";
    /// Default `max_tokens` when the agent recipe doesn't specify.
    /// 4096 is small enough to keep cost-of-mistake bounded; HR is
    /// expected to override on real workloads.
    pub const DEFAULT_MAX_TOKENS: i32 = 4096;
}

#[derive(Debug, Clone)]
pub struct AnthropicModelClient {
    http: reqwest::Client,
    config: AnthropicConfig,
}

impl AnthropicModelClient {
    pub fn new(config: AnthropicConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { http, config }
    }

    fn endpoint(&self) -> String {
        // `base_url` is the full API prefix supplied by the caller, including
        // the version segment — e.g. `https://api.anthropic.com/v1`. We only
        // append the route; picking the version is the caller's job.
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/messages") {
            base.to_string()
        } else {
            format!("{base}/messages")
        }
    }

    /// Build the JSON body for `POST /v1/messages`. Mirrors `OpenAi`'s
    /// `request_body` but projects to the Messages API wire shape — see
    /// the section header for the differences.
    fn request_body(&self, input: &ModelTurnInput) -> Value {
        let messages = chat_messages_to_anthropic_messages(&input.messages);
        // `tool_choice::None` has no direct Anthropic equivalent
        // (Anthropic forces tool consideration once `tools` is non-
        // empty). Best approximation is to drop the tools entirely.
        let tools = if matches!(input.tool_choice, ToolChoice::None) {
            Vec::new()
        } else {
            let mut tools = input
                .tools
                .iter()
                .map(tool_spec_to_anthropic_tool)
                .collect::<Vec<_>>();
            tools.extend(input.hosted_tools.iter().map(hosted_tool_to_anthropic_tool));
            tools
        };
        let system_field = anthropic_system_field(input.system_prompt.as_deref());

        // Cache strategy is applied last so it can decorate the final
        // serialized shape. Builds up to 4 ephemeral breakpoints (system,
        // last tool, last message, optionally mid message for long chats).
        let cached = apply_anthropic_cache_strategy(system_field, tools, messages);

        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": cached.messages,
            "stream": true,
        });
        if let Some(sys) = cached.system {
            body["system"] = sys;
        }
        if !cached.tools.is_empty() {
            body["tools"] = json!(cached.tools);
            // tool_choice only meaningful when we're actually sending
            // tools; Auto is Anthropic's default so we omit it to keep
            // the wire body byte-identical for the common case
            // (matters for prompt cache stability).
            if !matches!(input.tool_choice, ToolChoice::Auto) {
                body["tool_choice"] = anthropic_tool_choice_value(&input.tool_choice);
            }
        }
        // Anthropic doesn't support `parallel_tool_calls` — it's an
        // OpenAI-only knob. We silently ignore input.parallel_tool_calls
        // on this provider. (Anthropic returns multiple tool_use blocks
        // freely when the model decides to.)
        if let Some(t) = self.config.temperature {
            body["temperature"] = json!(t);
        }
        body
    }
}

fn anthropic_tool_choice_value(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::None => json!({"type": "auto"}), // unreachable in practice (caller drops tools)
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
    }
}

#[async_trait]
impl ModelClient for AnthropicModelClient {
    fn hosted_capability(&self, capability: HostedCapability) -> CapabilitySupport {
        match capability {
            HostedCapability::WebSearch => {
                official_endpoint_support(&self.config.base_url, &["api.anthropic.com"])
            }
        }
    }

    async fn stream(
        &self,
        input: ModelTurnInput,
    ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError> {
        let resp = match self
            .http
            .post(self.endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.anthropic_version)
            .header("content-type", "application/json")
            .json(&self.request_body(&input))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(classify_reqwest_error(&e, e.to_string())),
        };
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(classify_anthropic_http_error(status, &body_text));
        }

        let event_stream = resp.bytes_stream().eventsource();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ModelChunk, ModelClientError>>(8);
        tokio::spawn(async move {
            let mut state = AnthropicStreamState::default();
            futures::pin_mut!(event_stream);
            while let Some(ev) = event_stream.next().await {
                let chunks = match ev {
                    Ok(event) => match state.feed_event(&event.event, &event.data) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = tx
                            .send(Err(ModelClientError::Network(format!(
                                "SSE transport error: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                for c in chunks {
                    if tx.send(Ok(c)).await.is_err() {
                        return;
                    }
                }
            }
            if let Some(done) = state.finalize() {
                let _ = tx.send(Ok(done)).await;
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }
}

/// Render `ChatMessage[]` into Anthropic Messages API `messages[]`. The
/// projection has two non-trivial rules:
///
///   1. `ChatMessage::Tool` doesn't have a dedicated role on Anthropic —
///      tool results live as content blocks on the user message that
///      follows them. We buffer pending tool results and flush them
///      into the next user message (or a synthetic user message at the
///      end if the conversation ends on tool results).
///   2. `ChatMessage::Assistant` may carry `thinking`, plain `text`, and
///      tool calls; all three render as content blocks under one
///      assistant message in that exact order (thinking → text →
///      tool_use). Anthropic API rejects modified thinking blocks, so
///      we round-trip the signature verbatim.
fn chat_messages_to_anthropic_messages(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut pending_tool_results: Vec<Value> = Vec::new();

    let flush_tool_results = |bucket: &mut Vec<Value>, out: &mut Vec<Value>| {
        if !bucket.is_empty() {
            let blocks = std::mem::take(bucket);
            out.push(json!({"role": "user", "content": blocks}));
        }
    };

    for msg in messages {
        match msg {
            ChatMessage::User {
                content,
                attachments,
            } => {
                // Build the content block list in this order:
                //   1. flushed tool_result blocks (if any pending)
                //   2. text block (when content non-empty)
                //   3. image blocks per attachment, in original order
                //
                // Anthropic requires the user message body to alternate
                // with assistant, so when tool_results are pending they
                // merge into THIS user message (saving an extra hop).
                let mut blocks: Vec<Value> = std::mem::take(&mut pending_tool_results);
                if !content.is_empty() {
                    blocks.push(json!({"type":"text","text":content}));
                }
                for att in attachments {
                    match att {
                        UserAttachment::Image(src) => {
                            blocks.push(image_to_anthropic_block(src));
                        }
                    }
                }
                // Defensive: if everything was empty (no tool_results,
                // empty content, no attachments), still emit a single
                // empty text block — Anthropic API rejects empty
                // `content` arrays.
                if blocks.is_empty() {
                    blocks.push(json!({"type":"text","text":""}));
                }
                out.push(json!({"role": "user", "content": blocks}));
            }
            ChatMessage::Assistant {
                text,
                tool_calls,
                thinking,
                usage: _,
            } => {
                // tool_results must be flushed before we can append an
                // assistant message (Anthropic rejects two consecutive
                // assistant messages).
                flush_tool_results(&mut pending_tool_results, &mut out);
                let mut blocks: Vec<Value> = Vec::new();
                // Thinking first — both as observed in Anthropic's own
                // serialisations and as a stable position for cache
                // breakpoint placement.
                if let Some(t) = thinking {
                    let mut tb = json!({"type": "thinking", "thinking": t.text});
                    if let Some(sig) = t.signature.as_deref() {
                        if !sig.is_empty() {
                            tb["signature"] = json!(sig);
                        }
                    }
                    blocks.push(tb);
                }
                if let Some(t) = text.as_deref() {
                    if !t.is_empty() {
                        blocks.push(json!({"type": "text", "text": t}));
                    }
                }
                for tc in tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    }));
                }
                if blocks.is_empty() {
                    // Pathological case — assistant turn with nothing
                    // to project. Anthropic rejects empty content arrays,
                    // so skip the message entirely.
                    continue;
                }
                out.push(json!({"role": "assistant", "content": blocks}));
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                is_error,
                attachments,
            } => {
                // Anthropic tool_result CAN carry image content blocks
                // (in addition to text). Render text first, then any
                // image attachments — same `text → image` ordering used
                // for the user message projection (§9 in internals doc).
                let mut blocks: Vec<Value> = Vec::new();
                if !content.is_empty() {
                    let replay = compact_tool_result_for_replay(content);
                    blocks.push(json!({"type": "text", "text": replay}));
                }
                for att in attachments {
                    let UserAttachment::Image(src) = att;
                    blocks.push(image_to_anthropic_block(src));
                }
                // tool_result.content can be either a string or a block
                // array per Anthropic docs. We use the array form
                // uniformly so attachments-or-not codepath is one.
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": ""}));
                }
                pending_tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": blocks,
                    "is_error": is_error,
                }));
            }
        }
    }

    // Conversation ended on tool results without a follow-up user message
    // — synthesize one so Anthropic sees the results.
    flush_tool_results(&mut pending_tool_results, &mut out);
    out
}

/// Anthropic accepts `system` as either a string OR a content-block
/// array (the latter lets us put `cache_control` on it). We always
/// pre-shape it as the array form so the cache strategy layer can apply
/// markers without re-allocating. `None` ⇒ no system field on the wire.
fn anthropic_system_field(prompt: Option<&str>) -> Option<Value> {
    let s = prompt?.trim();
    if s.is_empty() {
        return None;
    }
    Some(json!([{"type": "text", "text": s}]))
}

/// Render a `ToolSpec` into Anthropic's tool definition shape. Same
/// `input_schema` field both providers use, just different envelope.
/// Project an `ImageSource` to an Anthropic `image` content block.
/// Anthropic accepts two source shapes:
///   * `{type:"base64", media_type, data}` for inline bytes
///   * `{type:"url", url}` for fetched-by-server URLs (added 2024)
fn image_to_anthropic_block(src: &ImageSource) -> Value {
    let source = match &src.data {
        ImageData::Base64(b64) => json!({
            "type": "base64",
            "media_type": src.media_type,
            "data": b64,
        }),
        ImageData::Url(url) => json!({
            "type": "url",
            "url": url,
        }),
    };
    json!({"type": "image", "source": source})
}

fn tool_spec_to_anthropic_tool(spec: &ToolSpec) -> Value {
    json!({
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    })
}

fn hosted_tool_to_anthropic_tool(tool: &HostedTool) -> Value {
    match tool {
        HostedTool::WebSearch => json!({
            "type": "web_search_20250305",
            "name": "web_search",
        }),
    }
}

struct AnthropicCached {
    system: Option<Value>,
    tools: Vec<Value>,
    messages: Vec<Value>,
}

/// Apply up-to-four `cache_control: ephemeral` breakpoints to the
/// request, mirroring OMA `default-loop.ts:997-1039`. Anthropic rate-
/// limits cache writes to 4 breakpoints per request; we spend them on:
///
///   1. **System block (last content)** — caches everything before the
///      messages section (system + tools).
///   2. **Last tool definition** — defensive when system also has
///      dynamic content (a system change shouldn't bust the tools cache).
///   3. **Last message** — multi-turn chat tail breakpoint, where most
///      hits come from.
///   4. **Mid message** (only when `messages.len() > 30`) — Anthropic's
///      20-block lookback window means long single turns blow past the
///      tail breakpoint; the intermediate one catches reads inside the
///      turn.
///
/// Empty `system` skips its breakpoint (Anthropic API rejects
/// `cache_control` on empty text blocks).
fn apply_anthropic_cache_strategy(
    system: Option<Value>,
    tools: Vec<Value>,
    messages: Vec<Value>,
) -> AnthropicCached {
    let mut system = system;
    if let Some(sys) = system.as_mut() {
        if let Some(arr) = sys.as_array_mut() {
            if let Some(last) = arr.last_mut() {
                if last
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
                {
                    last["cache_control"] = json!({"type": "ephemeral"});
                }
            }
        }
    }

    let mut tools = tools;
    if let Some(last) = tools.last_mut() {
        last["cache_control"] = json!({"type": "ephemeral"});
    }

    let mut messages = messages;
    // Last message tail breakpoint — applied to the last content block
    // of the last message (mirrors OMA's behaviour and matches what
    // Anthropic docs recommend for chat tails).
    if let Some(last) = messages.last_mut() {
        if let Some(blocks) = last.get_mut("content").and_then(|v| v.as_array_mut()) {
            if let Some(last_block) = blocks.last_mut() {
                last_block["cache_control"] = json!({"type": "ephemeral"});
            }
        }
    }
    // Mid breakpoint for long chats — same shape, applied at the message
    // sitting at the midpoint when there are >30 messages.
    if messages.len() > 30 {
        let mid = messages.len() / 2;
        if let Some(blocks) = messages[mid]
            .get_mut("content")
            .and_then(|v| v.as_array_mut())
        {
            if let Some(last_block) = blocks.last_mut() {
                last_block["cache_control"] = json!({"type": "ephemeral"});
            }
        }
    }

    AnthropicCached {
        system,
        tools,
        messages,
    }
}

/// Anthropic streaming uses `event:` to discriminate frame types. This
/// state machine maps the lifecycle into our `ModelChunk` enum.
///
/// Frames we care about (Anthropic docs § streaming):
///   * `message_start` — carries message id (msg_xxxxx) + initial usage.
///   * `content_block_start` — declares a new block at `index` with
///     `type: text | thinking | tool_use`. tool_use carries id + name.
///   * `content_block_delta` — `delta.type: text_delta | input_json_delta
///     | thinking_delta | signature_delta`.
///   * `content_block_stop` — block at index complete. For tool_use,
///     this is where we emit `ToolCallEnd`.
///   * `message_delta` — final stop_reason + output_tokens.
///   * `message_stop` — frame after which the connection closes.
///   * `ping` — keepalive, ignored.
///   * `error` — provider-side fault.
#[derive(Debug, Default)]
struct AnthropicStreamState {
    msg_id: Option<String>,
    /// Stream-local block index → kind + tool metadata.
    blocks: std::collections::HashMap<u64, AnthropicBlock>,
    stop_reason: Option<String>,
    pending_usage: Option<HarnessUsage>,
    done_emitted: bool,
}

#[derive(Debug)]
enum AnthropicBlock {
    Text,
    Thinking { thinking_id: String },
    ToolUse { id: String },
    Ignored,
}

impl AnthropicStreamState {
    fn feed_event(&mut self, event: &str, data: &str) -> Result<Vec<ModelChunk>, ModelClientError> {
        // Anthropic sends `ping` for keepalive — ignore. Unknown events
        // we also ignore (forward-compat with future event types).
        match event {
            "ping" | "" => return Ok(vec![]),
            "error" => {
                return Err(ModelClientError::Other(format!(
                    "anthropic stream error event: {data}"
                )));
            }
            _ => {}
        }

        let value: Value = serde_json::from_str(data).map_err(|e| {
            ModelClientError::Other(format!(
                "anthropic SSE data not JSON (event={event}): {e}; raw={data}"
            ))
        })?;
        let mut out: Vec<ModelChunk> = Vec::new();

        match event {
            "message_start" => {
                let msg = value.get("message");
                if let Some(id) = msg.and_then(|m| m.get("id")).and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        self.msg_id = Some(id.to_string());
                    }
                }
                if let Some(u) = msg.and_then(|m| m.get("usage")) {
                    self.pending_usage =
                        Some(merge_anthropic_usage(self.pending_usage.clone(), u, true));
                }
            }
            "content_block_start" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let block = value.get("content_block");
                let kind = block.and_then(|b| b.get("type")).and_then(|v| v.as_str());
                match kind {
                    Some("text") => {
                        self.blocks.insert(index, AnthropicBlock::Text);
                    }
                    Some("thinking") => {
                        let thinking_id = self
                            .msg_id
                            .clone()
                            .map(|m| format!("{m}_t{index}"))
                            .unwrap_or_else(|| format!("thinking_{index}"));
                        self.blocks
                            .insert(index, AnthropicBlock::Thinking { thinking_id });
                    }
                    Some("tool_use") => {
                        let id = block
                            .and_then(|b| b.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let name = block
                            .and_then(|b| b.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if !id.is_empty() && !name.is_empty() {
                            out.push(ModelChunk::ToolCallStart {
                                id: id.clone(),
                                name,
                            });
                        }
                        self.blocks.insert(index, AnthropicBlock::ToolUse { id });
                    }
                    Some("server_tool_use") | Some("web_search_tool_result") => {
                        self.blocks.insert(index, AnthropicBlock::Ignored);
                    }
                    _ => {
                        // Unknown block type — record as Text so we
                        // don't panic on later deltas; ignoring them
                        // is the safer forward-compat path.
                        self.blocks.insert(index, AnthropicBlock::Ignored);
                    }
                }
            }
            "content_block_delta" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let delta = match value.get("delta") {
                    Some(d) => d,
                    None => return Ok(out),
                };
                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match (self.blocks.get(&index), delta_type) {
                    (Some(AnthropicBlock::Text), "text_delta") => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                let msg_id = self
                                    .msg_id
                                    .clone()
                                    .unwrap_or_else(|| "msg_anthropic_default".into());
                                out.push(ModelChunk::TextDelta {
                                    msg_id,
                                    delta: text.to_string(),
                                });
                            }
                        }
                    }
                    (Some(AnthropicBlock::Thinking { thinking_id }), "thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                out.push(ModelChunk::ThinkingDelta {
                                    thinking_id: thinking_id.clone(),
                                    delta: text.to_string(),
                                    signature: None,
                                });
                            }
                        }
                    }
                    (Some(AnthropicBlock::Thinking { thinking_id }), "signature_delta") => {
                        if let Some(sig) = delta.get("signature").and_then(|v| v.as_str()) {
                            // Signature deltas land empty-text + signed.
                            // We pass them through as ThinkingDelta with
                            // empty `delta` so agent_loop's signature
                            // latch fires. Empty-delta ThinkingChunk is
                            // suppressed downstream by the empty check.
                            out.push(ModelChunk::ThinkingDelta {
                                thinking_id: thinking_id.clone(),
                                delta: String::new(),
                                signature: Some(sig.to_string()),
                            });
                        }
                    }
                    (Some(AnthropicBlock::ToolUse { id }), "input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            if !partial.is_empty() {
                                out.push(ModelChunk::ToolCallInputDelta {
                                    id: id.clone(),
                                    delta: partial.to_string(),
                                });
                            }
                        }
                    }
                    _ => { /* unknown delta type or block-kind mismatch */ }
                }
            }
            "content_block_stop" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                if let Some(AnthropicBlock::ToolUse { id }) = self.blocks.get(&index) {
                    // Tool argument bytes accumulated as deltas; harness
                    // parses them in `consume_step_stream`. We don't
                    // attach an early input here — Anthropic ships the
                    // arguments only via deltas.
                    out.push(ModelChunk::ToolCallEnd {
                        id: id.clone(),
                        input: None,
                    });
                }
            }
            "message_delta" => {
                if let Some(reason) = value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(u) = value.get("usage") {
                    // message_delta only refreshes the output_tokens —
                    // input_tokens come from message_start. Merge.
                    self.pending_usage =
                        Some(merge_anthropic_usage(self.pending_usage.clone(), u, false));
                }
            }
            "message_stop" => {
                if let Some(done) = self.emit_done() {
                    out.push(done);
                }
            }
            _ => { /* forward-compat: silently ignore unknown event types */ }
        }
        Ok(out)
    }

    fn finalize(&mut self) -> Option<ModelChunk> {
        self.emit_done()
    }

    fn emit_done(&mut self) -> Option<ModelChunk> {
        if self.done_emitted {
            return None;
        }
        self.done_emitted = true;
        let stop_reason = map_anthropic_stop_reason(self.stop_reason.as_deref());
        Some(ModelChunk::Done {
            stop_reason,
            usage: self.pending_usage.take(),
        })
    }
}

/// Merge an Anthropic usage block (from `message_start` or `message_delta`)
/// into the running tally. When `include_input=true`, accept both
/// `input_tokens` and the cache-related fields; otherwise only refresh
/// `output_tokens` (per Anthropic's spec — `message_delta` only carries
/// output deltas).
fn merge_anthropic_usage(
    prior: Option<HarnessUsage>,
    incoming: &Value,
    include_input: bool,
) -> HarnessUsage {
    let mut u = prior.unwrap_or_default();
    if include_input {
        if let Some(v) = incoming.get("input_tokens").and_then(|v| v.as_u64()) {
            u.input_tokens = v;
        }
        if let Some(v) = incoming
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
        {
            u.cache_read_input_tokens = v;
        }
        if let Some(v) = incoming
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
        {
            u.cache_creation_input_tokens = v;
        }
    }
    if let Some(v) = incoming.get("output_tokens").and_then(|v| v.as_u64()) {
        u.output_tokens = v;
    }
    u
}

fn map_anthropic_stop_reason(reason: Option<&str>) -> String {
    // Anthropic Messages API stop_reason values (per docs):
    //   end_turn / max_tokens / stop_sequence / tool_use / refusal
    match reason {
        Some("end_turn") | Some("stop_sequence") | Some("tool_use") => "end_turn".into(),
        Some("max_tokens") => "max_tokens".into(),
        Some("refusal") => "refusal".into(),
        Some(other) if !other.is_empty() => other.to_string(),
        _ => "end_turn".into(),
    }
}

/// HTTP status → `ModelClientError` for Anthropic. Status codes mostly
/// align with OpenAI's so we reuse the classifier logic; the body
/// heuristics differ slightly (Anthropic uses `type:"error"` with a
/// `message` field rather than nested `error.message`).
fn classify_anthropic_http_error(status: reqwest::StatusCode, body: &str) -> ModelClientError {
    use reqwest::StatusCode;
    let snippet = body.chars().take(512).collect::<String>();
    if status == StatusCode::TOO_MANY_REQUESTS {
        return ModelClientError::RateLimit(format!("HTTP {status}: {snippet}"));
    }
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return ModelClientError::Auth(format!("HTTP {status}: {snippet}"));
    }
    if status == StatusCode::BAD_REQUEST && looks_like_context_overflow(body) {
        return ModelClientError::ContextOverflow(format!("HTTP {status}: {snippet}"));
    }
    if status == StatusCode::BAD_REQUEST {
        return ModelClientError::BadRequest(format!("HTTP {status}: {snippet}"));
    }
    if status.is_server_error() {
        return ModelClientError::ServerError(format!("HTTP {status}: {snippet}"));
    }
    ModelClientError::Other(format!("HTTP {status}: {snippet}"))
}

// ─── OpenAI Responses API ────────────────────────────────────────────────
//
// The Responses API (`POST /v1/responses`) is OpenAI's current-generation
// surface for gpt-5 / o-series reasoning models. It differs from
// chat/completions in three load-bearing places:
//
//   1. Request shape   `input[]` (a heterogeneous list of typed items)
//                      replaces `messages[]`; `system` moves to the
//                      top-level `instructions` field; tool calls ride as
//                      `function_call` / `function_call_output` items
//                      (not a `tool` role); tools are FLAT
//                      (`{type:"function", name, ...}`, no nested
//                      `function{}`); `max_output_tokens` replaces
//                      `max_tokens`.
//   2. Reasoning       Reasoning surfaces as `reasoning` items. In the
//                      stateless mode we use (`store:false`) the model's
//                      reasoning must be round-tripped verbatim by echoing
//                      the item `id` + `encrypted_content` back on the next
//                      turn (requested via `include`). We fold both into
//                      the existing `AssistantThinking.signature` field so
//                      `ChatMessage` stays unchanged.
//   3. SSE shape       Events are discriminated by a JSON `type` field
//                      (`response.output_text.delta`, `response.completed`,
//                      …) inside `data:`, not by chat's `choices[].delta`.
//
// As with the Anthropic section, all the work here is wire-shape
// translation into the provider-agnostic `ModelChunk` enum; the harness
// loop stays provider-agnostic.

/// Deployment-side credentials + endpoint + knobs for the OpenAI Responses
/// API.
///
/// The client always operates **statelessly** (`store:false`): this harness
/// owns conversation history itself (persisted to `messages.jsonl`) and
/// replays it in full every turn, so it never references OpenAI's
/// server-side response storage. Consequently reasoning is round-tripped by
/// echoing each item's `encrypted_content` (requested via `include`), not by
/// `previous_response_id` / `item_reference`. A stateful (`store:true`) mode
/// would require that item-reference machinery, which is intentionally not
/// implemented — so no `store` knob is exposed.
#[derive(Debug, Clone)]
pub struct OpenAiResponsesConfig {
    /// Full API prefix including any version segment — e.g.
    /// `https://api.openai.com/v1`. The `/responses` route is appended.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<f64>,
    /// Responses' replacement for `max_tokens`. `None` ⇒ provider default.
    pub max_output_tokens: Option<i32>,
    /// `"low"` / `"medium"` / `"high"` → `reasoning.effort`. `None` omits it.
    pub reasoning_effort: Option<String>,
    /// `"auto"` → `reasoning.summary`, which makes the model stream a
    /// human-readable summary of its reasoning (surfaced as thinking
    /// deltas). `None` omits it — the model still reasons, just silently.
    pub reasoning_summary: Option<String>,
}

impl OpenAiResponsesConfig {
    /// Default API prefix if the caller doesn't override `base_url`.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
}

#[derive(Debug, Clone)]
pub struct OpenAiResponsesModelClient {
    http: reqwest::Client,
    config: OpenAiResponsesConfig,
}

impl OpenAiResponsesModelClient {
    pub fn new(config: OpenAiResponsesConfig) -> Self {
        // Same construction as the other clients: fail fast on a slow TCP
        // handshake, but never cap per-read time — streaming reasoning
        // responses legitimately pause between SSE frames.
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { http, config }
    }

    fn endpoint(&self) -> String {
        // `base_url` carries the version segment; we only append the route.
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/responses") {
            base.to_string()
        } else {
            format!("{base}/responses")
        }
    }

    fn request_body(&self, input: &ModelTurnInput) -> Value {
        let mut body = json!({
            "model": self.config.model,
            "input": chat_messages_to_responses_input(&input.messages),
            "stream": true,
            // Always stateless — see `OpenAiResponsesConfig`. The harness
            // replays full history itself, so it never leans on server-side
            // response storage.
            "store": false,
        });
        // System prompt rides on the top-level `instructions` field — the
        // Responses-recommended home for it (kept out of the `input[]` list).
        if let Some(sys) = input.system_prompt.as_deref().filter(|s| !s.is_empty()) {
            body["instructions"] = json!(sys);
        }

        // `ToolChoice::None` means "no tools this turn" — that must also
        // suppress hosted (provider-run) tools, otherwise the model could
        // still fire e.g. server-side web_search, defeating the caller's
        // intent (and incurring unexpected egress / cost). Mirrors the
        // Anthropic path, which drops all tools for `None`.
        let advertise_tools = !matches!(input.tool_choice, ToolChoice::None);
        let mut tools: Vec<Value> = Vec::new();
        if advertise_tools {
            tools.extend(input.tools.iter().map(tool_spec_to_responses_tool));
            tools.extend(input.hosted_tools.iter().map(hosted_tool_to_responses_tool));
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            // Emit `tool_choice` whenever it's meaningful for the advertised
            // set — NOT only when client function tools exist:
            //   * Auto / Required apply to the whole set (function + hosted),
            //     so send them as long as any tool is advertised. Gating on
            //     function tools would silently downgrade a hosted-only
            //     `Required` to the default auto.
            //   * Tool(name) force-selects a *function* tool, so only send it
            //     when a function tool exists — encoding a hosted tool as
            //     `{type:"function", name}` would be rejected as a 400.
            let send_choice = match input.tool_choice {
                ToolChoice::Auto | ToolChoice::Required => true,
                ToolChoice::Tool(_) => !input.tools.is_empty(),
                ToolChoice::None => false, // unreachable: advertise_tools is false
            };
            if send_choice {
                body["tool_choice"] = responses_tool_choice_value(&input.tool_choice);
            }
            if let Some(parallel) = input.parallel_tool_calls {
                body["parallel_tool_calls"] = json!(parallel);
            }
        }

        if let Some(temperature) = self.config.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(max_output) = self.config.max_output_tokens {
            body["max_output_tokens"] = json!(max_output);
        }

        let effort = self
            .config
            .reasoning_effort
            .as_deref()
            .filter(|s| !s.is_empty());
        let summary = self
            .config
            .reasoning_summary
            .as_deref()
            .filter(|s| !s.is_empty());
        if effort.is_some() || summary.is_some() {
            let mut reasoning = json!({});
            if let Some(e) = effort {
                reasoning["effort"] = json!(e);
            }
            if let Some(s) = summary {
                reasoning["summary"] = json!(s);
            }
            body["reasoning"] = reasoning;
        }

        // Stateless mode always asks OpenAI to include the encrypted
        // reasoning payload so we can echo it back on the next turn (see
        // `chat_messages_to_responses_input`).
        body["include"] = json!(["reasoning.encrypted_content"]);

        body
    }
}

/// Render one `ToolSpec` into the Responses FLAT tool shape. Note the
/// difference from chat/completions (`tool_spec_to_openai_function`), which
/// nests everything under a `function` key.
fn tool_spec_to_responses_tool(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
    })
}

/// Project a `HostedTool` to a Responses hosted-tool definition. Unlike the
/// chat/completions client — which rejects hosted tools outright — Responses
/// runs them server-side and streams the result back inline.
fn hosted_tool_to_responses_tool(tool: &HostedTool) -> Value {
    match tool {
        HostedTool::WebSearch => json!({ "type": "web_search" }),
    }
}

fn responses_tool_choice_value(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!("auto"),
        // Unreachable in practice (caller drops function tools for None),
        // but encode defensively.
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({ "type": "function", "name": name }),
    }
}

/// Render an `ImageSource` to the string the Responses `input_image` part
/// expects. Inline base64 becomes a `data:` URI; a URL passes through.
fn image_to_responses_data_url(src: &ImageSource) -> String {
    match &src.data {
        ImageData::Base64(b64) => format!("data:{};base64,{}", src.media_type, b64),
        ImageData::Url(u) => u.clone(),
    }
}

/// Pack a Responses reasoning item's `id` + `encrypted_content` into the
/// single `AssistantThinking.signature` slot so reasoning round-trips
/// without changing `ChatMessage`. The item id never contains a newline, so
/// a `\n` separator is unambiguous. `decode_reasoning_signature` reverses it;
/// a signature without a `\n` (e.g. an Anthropic thinking signature) decodes
/// to `None` and is simply not re-sent as a reasoning item.
fn encode_reasoning_signature(item_id: &str, encrypted_content: &str) -> String {
    format!("{item_id}\n{encrypted_content}")
}

fn decode_reasoning_signature(sig: &str) -> Option<(String, String)> {
    let (id, enc) = sig.split_once('\n')?;
    if id.is_empty() || enc.is_empty() {
        return None;
    }
    Some((id.to_string(), enc.to_string()))
}

/// Render `ChatMessage[]` into the Responses `input[]` list. Rules:
///
///   1. `User` → `{role:"user", content:[input_text, input_image...]}`.
///   2. `Assistant` expands into up to three item kinds, in order:
///      reasoning (from `thinking`, if its signature round-trips) →
///      `{role:"assistant", content:[output_text]}` → one `function_call`
///      per tool call.
///   3. `Tool` → `function_call_output` keyed by `call_id`. Image
///      attachments ride as an `input_image` content array (Responses
///      supports structured tool output — no lossy placeholder like the
///      chat/completions `tool` role).
fn chat_messages_to_responses_input(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg {
            ChatMessage::User {
                content,
                attachments,
            } => {
                let mut parts: Vec<Value> = Vec::with_capacity(attachments.len() + 1);
                if !content.is_empty() {
                    parts.push(json!({"type": "input_text", "text": content}));
                }
                for att in attachments {
                    let UserAttachment::Image(src) = att;
                    parts.push(json!({
                        "type": "input_image",
                        "image_url": image_to_responses_data_url(src),
                    }));
                }
                // Responses rejects empty content arrays.
                if parts.is_empty() {
                    parts.push(json!({"type": "input_text", "text": ""}));
                }
                out.push(json!({"role": "user", "content": parts}));
            }
            ChatMessage::Assistant {
                text,
                tool_calls,
                thinking,
                usage: _,
            } => {
                // Reasoning first — only re-sent when the signature carries a
                // decodable (item_id, encrypted_content) pair. A plain-text
                // or Anthropic-style signature simply isn't projected.
                if let Some(t) = thinking {
                    if let Some((item_id, enc)) =
                        t.signature.as_deref().and_then(decode_reasoning_signature)
                    {
                        let summary = if t.text.is_empty() {
                            json!([])
                        } else {
                            json!([{"type": "summary_text", "text": t.text}])
                        };
                        out.push(json!({
                            "type": "reasoning",
                            "id": item_id,
                            "summary": summary,
                            "encrypted_content": enc,
                        }));
                    }
                }
                if let Some(t) = text.as_deref().filter(|s| !s.is_empty()) {
                    out.push(json!({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": t}],
                    }));
                }
                for tc in tool_calls {
                    out.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tool_invocation_args_for_wire(tc),
                    }));
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                is_error: _,
                attachments,
            } => {
                if attachments.is_empty() {
                    let replay = compact_tool_result_for_replay(content).into_owned();
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": replay,
                    }));
                } else {
                    let mut parts: Vec<Value> = Vec::new();
                    if !content.is_empty() {
                        let replay = compact_tool_result_for_replay(content).into_owned();
                        parts.push(json!({"type": "input_text", "text": replay}));
                    }
                    for att in attachments {
                        let UserAttachment::Image(src) = att;
                        parts.push(json!({
                            "type": "input_image",
                            "image_url": image_to_responses_data_url(src),
                        }));
                    }
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": parts,
                    }));
                }
            }
        }
    }
    out
}

/// Pull token counts out of a Responses `usage` block. Maps:
///   * `input_tokens`                          → `input_tokens`
///   * `output_tokens`                         → `output_tokens`
///   * `input_tokens_details.cached_tokens`    → `cache_read_input_tokens`
///
/// Returns `None` when every counter is zero / absent (no-op response).
fn parse_responses_usage(usage: Option<&Value>) -> Option<HarnessUsage> {
    let u = usage?;
    let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let cache_read = u
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if input == 0 && output == 0 && cache_read == 0 {
        return None;
    }
    Some(HarnessUsage {
        input_tokens: input,
        output_tokens: output,
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: 0,
        compaction_input_tokens: 0,
        compaction_output_tokens: 0,
    })
}

/// Map a Responses `incomplete_details.reason` to the harness stop_reason
/// vocabulary. A clean finish (no reason) is `end_turn` regardless of
/// whether a tool was called — the harness dispatches tools whenever a
/// tool call landed, independent of stop_reason.
fn map_responses_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("max_output_tokens") => "max_tokens".into(),
        Some("content_filter") => "refusal".into(),
        _ => "end_turn".into(),
    }
}

/// Bucket a Responses stream-level failure (`response.failed` /
/// top-level `error` event) into a `ModelClientError`. Mirrors the HTTP
/// classifier's retryability split.
fn classify_responses_error(code: &str, message: &str) -> ModelClientError {
    let full = if code.is_empty() {
        message.to_string()
    } else {
        format!("{code}: {message}")
    };
    if code == "context_length_exceeded" || looks_like_context_overflow(message) {
        return ModelClientError::ContextOverflow(full);
    }
    if code == "rate_limit_exceeded" {
        return ModelClientError::RateLimit(full);
    }
    ModelClientError::BadRequest(full)
}

#[async_trait]
impl ModelClient for OpenAiResponsesModelClient {
    fn hosted_capability(&self, capability: HostedCapability) -> CapabilitySupport {
        match capability {
            HostedCapability::WebSearch => {
                official_endpoint_support(&self.config.base_url, &["api.openai.com"])
            }
        }
    }

    async fn stream(
        &self,
        input: ModelTurnInput,
    ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError> {
        let body = self.request_body(&input);
        let resp = match self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(classify_reqwest_error(&e, e.to_string())),
        };
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            // Responses shares OpenAI's HTTP error shape, so the same
            // status/body classifier applies.
            return Err(classify_openai_http_error(status, &body_text));
        }

        let event_stream = resp.bytes_stream().eventsource();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ModelChunk, ModelClientError>>(8);
        tokio::spawn(async move {
            let mut state = OpenAiResponsesStreamState::default();
            futures::pin_mut!(event_stream);
            while let Some(ev) = event_stream.next().await {
                let chunks = match ev {
                    Ok(event) => match state.feed_data(&event.data) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = tx
                            .send(Err(ModelClientError::Network(format!(
                                "SSE transport error: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                for c in chunks {
                    if tx.send(Ok(c)).await.is_err() {
                        return;
                    }
                }
            }
            // Same clean-vs-cut-off discipline as the OpenAI chat path: a
            // terminal `response.*` event marks a legitimate end. Without
            // one, the connection dropped mid-response and we must surface a
            // retryable error rather than fabricate a completed answer.
            if state.ended_cleanly() {
                if let Some(done) = state.finalize() {
                    let _ = tx.send(Ok(done)).await;
                }
            } else {
                let _ = tx
                    .send(Err(ModelClientError::Network(
                        "model stream closed before completion (no terminal response event) \
                         — connection dropped or upstream truncated the response"
                            .into(),
                    )))
                    .await;
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }
}

fn official_endpoint_support(base_url: &str, official_hosts: &[&str]) -> CapabilitySupport {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return CapabilitySupport::Unknown;
    };
    let Some(host) = url.host_str() else {
        return CapabilitySupport::Unknown;
    };
    if official_hosts
        .iter()
        .any(|official| host.eq_ignore_ascii_case(official))
    {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unknown
    }
}

/// State machine turning Responses SSE events into `ModelChunk`s. Events are
/// dispatched on the `type` field inside each `data:` payload (more reliable
/// than the SSE `event:` line). Lives outside the trait impl so the parsing
/// is unit-testable without an HTTP server.
#[derive(Debug, Default)]
struct OpenAiResponsesStreamState {
    /// First `response.id` we see; used as a fallback text/msg id.
    response_id: Option<String>,
    /// Maps a `function_call` item id (`fc_...`) to its `call_id`
    /// (`call_...`). Arguments deltas arrive keyed by the item id, but the
    /// harness pairs tool results by `call_id`, so we route through this.
    fc_call_by_item: std::collections::HashMap<String, String>,
    /// `"{item_id}:{summary_index}"` keys for reasoning-summary parts that
    /// have already emitted a text delta. Used to prefix a part boundary to
    /// the first real text of parts after index 0.
    summary_parts_seen: std::collections::HashSet<String>,
    stop_reason: Option<String>,
    pending_usage: Option<HarnessUsage>,
    done_emitted: bool,
    /// Set once a terminal `response.*` event (completed / incomplete /
    /// failed) lands, distinguishing a clean end from a dropped connection.
    saw_terminal: bool,
}

impl OpenAiResponsesStreamState {
    fn feed_data(&mut self, data: &str) -> Result<Vec<ModelChunk>, ModelClientError> {
        let trimmed = data.trim();
        // Responses uses terminal `response.*` events, not `[DONE]`, but
        // tolerate an empty keepalive frame or a stray sentinel.
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Ok(vec![]);
        }
        let value: Value = serde_json::from_str(trimmed).map_err(|e| {
            ModelClientError::Other(format!("Responses SSE data not JSON: {e}; raw={trimmed}"))
        })?;
        let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        // Latch the response id from any event that carries it.
        if let Some(rid) = value
            .get("response")
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
        {
            if self.response_id.is_none() && !rid.is_empty() {
                self.response_id = Some(rid.to_string());
            }
        }

        let mut out: Vec<ModelChunk> = Vec::new();
        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                    if !delta.is_empty() {
                        let msg_id = value
                            .get("item_id")
                            .and_then(|v| v.as_str())
                            .map(String::from)
                            .or_else(|| self.response_id.clone())
                            .unwrap_or_else(|| "msg_responses_default".into());
                        out.push(ModelChunk::TextDelta {
                            msg_id,
                            delta: delta.to_string(),
                        });
                    }
                }
            }
            // Summary reasoning. A summary can arrive as multiple parts, each
            // keyed by an incrementing `summary_index`. The harness thinking
            // model is a single text stream (no per-part structure to fill),
            // so we preserve the boundary by prefixing a blank line to the
            // FIRST real text delta of every part after the first. We key off
            // the text delta — not the structural `..._part.added` event — so
            // the separator only ever rides genuine model output (a bare
            // separator on a structural event would register as "progress" in
            // the agent loop and could persist for an empty part).
            //
            // `response.reasoning_summary.delta` is an alias some
            // Responses-compatible gateways emit instead of the `_text`
            // variant; accept both so their reasoning summaries aren't lost.
            "response.reasoning_summary_text.delta" | "response.reasoning_summary.delta" => {
                if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                    if !delta.is_empty() {
                        let item_id = value
                            .get("item_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("reasoning-0");
                        let idx = value
                            .get("summary_index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        // `insert` returns true the first time we see this
                        // (item, part) — that's the part's first text delta.
                        let first_of_part =
                            self.summary_parts_seen.insert(format!("{item_id}:{idx}"));
                        let text = if first_of_part && idx > 0 {
                            format!("\n\n{delta}")
                        } else {
                            delta.to_string()
                        };
                        out.push(ModelChunk::ThinkingDelta {
                            thinking_id: item_id.to_string(),
                            delta: text,
                            signature: None,
                        });
                    }
                }
            }
            // Non-summary reasoning text is a single stream — no part
            // boundaries to bridge.
            "response.reasoning_text.delta" => {
                if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                    if !delta.is_empty() {
                        let thinking_id = value
                            .get("item_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("reasoning-0")
                            .to_string();
                        out.push(ModelChunk::ThinkingDelta {
                            thinking_id,
                            delta: delta.to_string(),
                            signature: None,
                        });
                    }
                }
            }
            "response.output_item.added" => {
                if let Some(item) = value.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let fc_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .unwrap_or(fc_id);
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !call_id.is_empty() {
                            if !fc_id.is_empty() {
                                self.fc_call_by_item
                                    .insert(fc_id.to_string(), call_id.to_string());
                            }
                            out.push(ModelChunk::ToolCallStart {
                                id: call_id.to_string(),
                                name,
                            });
                        }
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = value.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                    if !delta.is_empty() {
                        let call_id = self
                            .fc_call_by_item
                            .get(item_id)
                            .cloned()
                            .unwrap_or_else(|| item_id.to_string());
                        out.push(ModelChunk::ToolCallInputDelta {
                            id: call_id,
                            delta: delta.to_string(),
                        });
                    }
                }
            }
            "response.output_item.done" => {
                if let Some(item) = value.get("item") {
                    let itype = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if itype == "function_call" {
                        let fc_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let call_id = self
                            .fc_call_by_item
                            .get(fc_id)
                            .cloned()
                            .or_else(|| {
                                item.get("call_id")
                                    .and_then(|v| v.as_str())
                                    .filter(|s| !s.is_empty())
                                    .map(String::from)
                            })
                            .unwrap_or_else(|| fc_id.to_string());
                        // Defensive: if we never saw `output_item.added` for
                        // this call, synthesize the start so the folder has a
                        // tool_state to attach the end to.
                        if !fc_id.is_empty() && !self.fc_call_by_item.contains_key(fc_id) {
                            let name = item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            out.push(ModelChunk::ToolCallStart {
                                id: call_id.clone(),
                                name,
                            });
                            self.fc_call_by_item
                                .insert(fc_id.to_string(), call_id.clone());
                        }
                        // The done item carries the complete argument string;
                        // parse it as authoritative early input when present.
                        let input = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .and_then(|s| serde_json::from_str::<Value>(s).ok());
                        out.push(ModelChunk::ToolCallEnd { id: call_id, input });
                    } else if itype == "reasoning" {
                        // Capture the encrypted reasoning payload and fold it
                        // into a signature-only ThinkingDelta so the final
                        // ChatMessage::Assistant.thinking carries it for
                        // round-trip on the next turn.
                        let rid = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        if let Some(enc) = item
                            .get("encrypted_content")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                        {
                            if !rid.is_empty() {
                                out.push(ModelChunk::ThinkingDelta {
                                    thinking_id: rid.to_string(),
                                    delta: String::new(),
                                    signature: Some(encode_reasoning_signature(rid, enc)),
                                });
                            }
                        }
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                self.saw_terminal = true;
                let resp = value.get("response");
                self.pending_usage = parse_responses_usage(resp.and_then(|r| r.get("usage")));
                let reason = resp
                    .and_then(|r| r.get("incomplete_details"))
                    .and_then(|d| d.get("reason"))
                    .and_then(|v| v.as_str());
                self.stop_reason = Some(map_responses_stop_reason(reason));
                if let Some(done) = self.emit_done() {
                    out.push(done);
                }
            }
            "response.failed" => {
                self.saw_terminal = true;
                let err = value.get("response").and_then(|r| r.get("error"));
                let code = err
                    .and_then(|e| e.get("code"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let message = err
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Responses response failed");
                return Err(classify_responses_error(code, message));
            }
            "error" => {
                self.saw_terminal = true;
                let code = value.get("code").and_then(|v| v.as_str()).unwrap_or("");
                let message = value
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Responses stream error");
                return Err(classify_responses_error(code, message));
            }
            // forward-compat: ignore unknown / display-only event types
            // (response.created, *.output_text.done, *_part.added, ping, …)
            _ => {}
        }
        Ok(out)
    }

    fn finalize(&mut self) -> Option<ModelChunk> {
        self.emit_done()
    }

    fn ended_cleanly(&self) -> bool {
        self.saw_terminal || self.done_emitted
    }

    fn emit_done(&mut self) -> Option<ModelChunk> {
        if self.done_emitted {
            return None;
        }
        self.done_emitted = true;
        Some(ModelChunk::Done {
            stop_reason: self
                .stop_reason
                .clone()
                .unwrap_or_else(|| "end_turn".into()),
            usage: self.pending_usage.take(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(prompt: &str) -> ModelTurnInput {
        ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: prompt.into(),
                attachments: vec![],
            }],
            tools: vec![],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }
    }

    fn bash_spec() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command inside the sandbox.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    #[test]
    fn openai_client_builds_chat_completions_request() {
        // base_url is the full prefix: the route is appended verbatim, the
        // version segment is NOT injected — a trailing slash is tolerated.
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test/v1/".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        assert_eq!(
            client.endpoint(),
            "https://example.test/v1/chat/completions"
        );
        let client_with_v1 = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        assert_eq!(
            client_with_v1.endpoint(),
            "https://example.test/v1/chat/completions"
        );
        // Providers whose version segment is not `/v1` (e.g. GLM's `/v4`) are
        // honored verbatim — regression guard for the `/v1`-injection bug.
        let glm = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://open.bigmodel.cn/api/coding/paas/v4".into(),
            api_key: "sk-test".into(),
            model: "glm-4.6".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        assert_eq!(
            glm.endpoint(),
            "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
        );
        let body = client.request_body(&user("hello"));
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        // tools omitted ⇒ no `tools` / `tool_choice` keys
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());

        // Now pass an actual ToolSpec and confirm it renders as a function.
        let with_tools = ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "hello".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let body = client.request_body(&with_tools);
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"][0],
            "command"
        );
        assert_eq!(body["tool_choice"], "auto");
        // parallel_tool_calls defaults to None ⇒ field omitted (let
        // OpenAI server-side default apply).
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn openai_client_emits_tool_choice_required() {
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(false),
        });
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn openai_client_emits_tool_choice_named_tool() {
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Tool("bash".into()),
            parallel_tool_calls: None,
        });
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "bash");
    }

    #[test]
    fn openai_client_drops_tools_when_choice_is_none() {
        // tool_choice: None — we drop the tools entirely so the model
        // can't call what it doesn't see (cheaper prompt + equivalent
        // semantic).
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        });
        assert!(body.get("tools").is_none(), "tools should be dropped");
        assert!(body.get("tool_choice").is_none());
    }

    #[tokio::test]
    async fn openai_compatible_rejects_hosted_web_search() {
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        let err = match client
            .stream(ModelTurnInput {
                system_prompt: None,
                messages: vec![ChatMessage::User {
                    content: "search".into(),
                    attachments: vec![],
                }],
                tools: vec![],
                hosted_tools: vec![HostedTool::WebSearch],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
            })
            .await
        {
            Ok(_) => panic!("expected hosted tool rejection"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Responses API"));
    }

    #[test]
    fn tool_choice_parse_handles_canonical_strings() {
        assert!(matches!(ToolChoice::parse(""), ToolChoice::Auto));
        assert!(matches!(ToolChoice::parse("auto"), ToolChoice::Auto));
        assert!(matches!(ToolChoice::parse("AUTO"), ToolChoice::Auto));
        assert!(matches!(ToolChoice::parse("none"), ToolChoice::None));
        assert!(matches!(
            ToolChoice::parse("required"),
            ToolChoice::Required
        ));
        assert!(matches!(ToolChoice::parse("any"), ToolChoice::Required));
        match ToolChoice::parse("tool:bash") {
            ToolChoice::Tool(name) => assert_eq!(name, "bash"),
            other => panic!("expected Tool(bash), got {other:?}"),
        }
        // Unknown degrades to Auto (forward-compat).
        assert!(matches!(ToolChoice::parse("garbage"), ToolChoice::Auto));
    }

    #[test]
    fn openai_client_prepends_system_when_set() {
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        let input = ModelTurnInput {
            system_prompt: Some("you are concise".into()),
            messages: vec![ChatMessage::User {
                content: "hi".into(),
                attachments: vec![],
            }],
            tools: vec![],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let body = client.request_body(&input);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "you are concise");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn parse_openai_usage_extracts_token_counts() {
        let u = parse_openai_usage(Some(&json!({
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 4}
        })))
        .expect("usage parsed");
        assert_eq!(u.input_tokens, 12);
        assert_eq!(u.output_tokens, 7);
        assert_eq!(u.cache_read_input_tokens, 4);
        assert_eq!(u.cache_creation_input_tokens, 0);
    }

    #[test]
    fn parse_openai_usage_without_cache_details() {
        let u = parse_openai_usage(Some(&json!({
            "prompt_tokens": 200,
            "completion_tokens": 30
        })))
        .expect("usage parsed");
        assert_eq!(u.input_tokens, 200);
        assert_eq!(u.output_tokens, 30);
        assert_eq!(u.cache_read_input_tokens, 0);
    }

    #[test]
    fn openai_stream_state_emits_text_deltas_then_done() {
        let mut state = OpenAiStreamState::default();
        // First chunk seeds id + role.
        let out = state
            .feed_data(
                r#"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
            )
            .unwrap();
        assert!(out.is_empty(), "empty content shouldn't emit");
        // Text deltas.
        let out = state
            .feed_data(r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::TextDelta { msg_id, delta } => {
                assert_eq!(msg_id, "chatcmpl-1");
                assert_eq!(delta, "Hello");
            }
            other => panic!("expected TextDelta, got {other:?}"),
        }
        let out = state
            .feed_data(r#"{"choices":[{"index":0,"delta":{"content":" world"}}]}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        // finish_reason in penultimate chunk (no Done yet — usage may
        // follow if include_usage was set).
        let out = state
            .feed_data(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#)
            .unwrap();
        assert!(out.is_empty());
        // include_usage final chunk (empty choices, usage populated).
        let out = state
            .feed_data(r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#)
            .unwrap();
        assert!(out.is_empty());
        // [DONE] sentinel produces the final Done chunk.
        let out = state.feed_data("[DONE]").unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::Done { stop_reason, usage } => {
                assert_eq!(stop_reason, "end_turn");
                let u = usage.as_ref().expect("usage propagated");
                assert_eq!(u.input_tokens, 10);
                assert_eq!(u.output_tokens, 3);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn openai_stream_state_emits_tool_call_chunks() {
        let mut state = OpenAiStreamState::default();
        // First tool-call chunk: id + name + initial arguments.
        let out = state
            .feed_data(
                r#"{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"bash","arguments":""}}]}}]}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "call_x");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        // Streaming arguments come split into multiple chunks.
        let out = state
            .feed_data(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}}]}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        let ModelChunk::ToolCallInputDelta { delta, .. } = &out[0] else {
            panic!("expected ToolCallInputDelta");
        };
        assert_eq!(delta, "{\"");

        let out = state
            .feed_data(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cmd\":\"pwd\"}"}}]}}]}"#,
            )
            .unwrap();
        let ModelChunk::ToolCallInputDelta { delta, .. } = &out[0] else {
            panic!("expected ToolCallInputDelta");
        };
        assert_eq!(delta, "cmd\":\"pwd\"}");

        // finish_reason="tool_calls" should emit ToolCallEnd for every
        // open tool call.
        let out = state
            .feed_data(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::ToolCallEnd { id, input } => {
                assert_eq!(id, "call_x");
                assert!(input.is_none(), "OpenAI streaming defers parsing");
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }

        // Stream closes (no [DONE] sentinel from some gateways) → finalize().
        let final_chunk = state.finalize().expect("finalize emits Done");
        match final_chunk {
            ModelChunk::Done { stop_reason, .. } => assert_eq!(stop_reason, "end_turn"),
            other => panic!("expected Done from finalize, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn collect_model_response_preserves_raw_tool_arguments() {
        let chunks = vec![
            Ok(ModelChunk::ToolCallStart {
                id: "call_x".into(),
                name: "bash".into(),
            }),
            Ok(ModelChunk::ToolCallInputDelta {
                id: "call_x".into(),
                delta: r#"{ "b": 2, "#.into(),
            }),
            Ok(ModelChunk::ToolCallInputDelta {
                id: "call_x".into(),
                delta: r#""a": 1 }"#.into(),
            }),
            Ok(ModelChunk::ToolCallEnd {
                id: "call_x".into(),
                input: None,
            }),
            Ok(ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: None,
            }),
        ];

        let response = collect_model_response(futures::stream::iter(chunks).boxed())
            .await
            .unwrap();
        let ModelResponse::ToolCall { invocation, .. } = response else {
            panic!("expected tool call response");
        };

        assert_eq!(invocation.input, json!({"b": 2, "a": 1}));
        assert_eq!(
            invocation.raw_emitted_args.as_deref(),
            Some(r#"{ "b": 2, "a": 1 }"#)
        );
    }

    #[test]
    fn openai_projection_uses_raw_tool_arguments_when_matching_input() {
        let msg = ChatMessage::Assistant {
            text: None,
            tool_calls: vec![ToolInvocation {
                id: "call_1".into(),
                name: "bash".into(),
                input: json!({"b": 2, "a": 1}),
                raw_emitted_args: Some(r#"{ "b": 2, "a": 1 }"#.into()),
            }],
            thinking: None,
            usage: None,
        };

        let wire = chat_message_to_wire(&msg);
        assert_eq!(
            wire["tool_calls"][0]["function"]["arguments"],
            r#"{ "b": 2, "a": 1 }"#
        );
    }

    #[test]
    fn openai_projection_ignores_stale_raw_tool_arguments() {
        let msg = ChatMessage::Assistant {
            text: None,
            tool_calls: vec![ToolInvocation {
                id: "call_1".into(),
                name: "bash".into(),
                input: json!({"command": "pwd"}),
                raw_emitted_args: Some(r#"{"command": "rm -rf /"}"#.into()),
            }],
            thinking: None,
            usage: None,
        };

        let wire = chat_message_to_wire(&msg);
        assert_eq!(
            wire["tool_calls"][0]["function"]["arguments"],
            json!({"command": "pwd"}).to_string()
        );
    }

    #[test]
    fn map_openai_finish_reason_table() {
        assert_eq!(map_openai_finish_reason(Some("stop")), "end_turn");
        assert_eq!(map_openai_finish_reason(Some("length")), "max_tokens");
        assert_eq!(map_openai_finish_reason(Some("tool_calls")), "end_turn");
        assert_eq!(map_openai_finish_reason(Some("content_filter")), "refusal");
        assert_eq!(map_openai_finish_reason(None), "end_turn");
        assert_eq!(map_openai_finish_reason(Some("")), "end_turn");
    }

    // ── Anthropic Messages API ────────────────────────────────────────

    #[test]
    fn chat_message_to_openai_wire_text_only_keeps_string_content() {
        // Fast path: no attachments → content stays as a plain string
        // (byte-identical to pre-multimodal behaviour).
        let msg = ChatMessage::User {
            content: "hello".into(),
            attachments: vec![],
        };
        let v = chat_message_to_wire(&msg);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello");
        // Not an array — that distinction matters for OpenAI-compatible
        // gateways that strict-parse the chat schema.
        assert!(v["content"].is_string());
    }

    #[test]
    fn chat_message_to_openai_wire_with_base64_image() {
        let msg = ChatMessage::User {
            content: "describe this".into(),
            attachments: vec![UserAttachment::Image(ImageSource {
                media_type: "image/png".into(),
                data: ImageData::Base64("iVBORw0KG...".into()),
            })],
        };
        let v = chat_message_to_wire(&msg);
        let parts = v["content"].as_array().expect("content array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe this");
        assert_eq!(parts[1]["type"], "image_url");
        // base64 must be wrapped in a data URI for OpenAI.
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.contains("iVBORw0KG..."));
    }

    #[test]
    fn chat_message_to_openai_wire_with_url_image() {
        let msg = ChatMessage::User {
            content: "".into(), // empty text, image-only
            attachments: vec![UserAttachment::Image(ImageSource {
                media_type: "image/jpeg".into(),
                data: ImageData::Url("https://cdn.example.com/cat.jpg".into()),
            })],
        };
        let v = chat_message_to_wire(&msg);
        let parts = v["content"].as_array().unwrap();
        // Empty text dropped — only image part survives.
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(
            parts[0]["image_url"]["url"],
            "https://cdn.example.com/cat.jpg"
        );
    }

    #[test]
    fn chat_message_to_openai_tool_role_degrades_image_to_placeholder() {
        // OpenAI's `tool` role is strictly string-typed — images coming
        // out of MCP tool calls have to degrade. We append a placeholder
        // line per attachment so the model still notices something
        // visual was returned, even if it can't see it.
        let msg = ChatMessage::Tool {
            tool_call_id: "call_x".into(),
            content: "ok".into(),
            is_error: false,
            attachments: vec![UserAttachment::Image(ImageSource {
                media_type: "image/png".into(),
                data: ImageData::Base64("AAA".into()),
            })],
        };
        let v = chat_message_to_wire(&msg);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_x");
        let content = v["content"].as_str().unwrap();
        assert!(content.starts_with("ok\n"));
        assert!(content.contains("image attached: image/png"));
        // Base64 bytes must NOT leak into the wire — degradation, not
        // smuggling.
        assert!(!content.contains("AAA"));
    }

    // ── E7: tool_result replay compaction ──

    #[test]
    fn replay_compaction_leaves_small_results_untouched() {
        let small = "x".repeat(1_000);
        assert!(matches!(
            compact_tool_result_for_replay(&small),
            std::borrow::Cow::Borrowed(_)
        ));
        // Over the token budget (10_000 ASCII chars / 4 = 2_500 > 2_000)
        // → compacted even though bytes are under 12 KB.
        let medium = "word ".repeat(2_000);
        let out = compact_tool_result_for_replay(&medium);
        assert!(out.contains("compacted for model replay"));
    }

    #[test]
    fn replay_compaction_keeps_head_and_tail_deterministically() {
        let body = format!("HEAD_MARK{}TAIL_MARK", "x".repeat(20_000));
        let first = compact_tool_result_for_replay(&body).into_owned();
        let second = compact_tool_result_for_replay(&body).into_owned();
        // Deterministic: byte-identical across calls (prompt-cache safety).
        assert_eq!(first, second);
        assert!(first.starts_with("[tool result compacted for model replay]"));
        assert!(first.contains("HEAD_MARK"), "head survives");
        assert!(first.contains("TAIL_MARK"), "tail survives");
        assert!(first.contains("omitted"), "omission marker present");
        // Massively smaller than the original.
        assert!(first.len() < body.len() / 2);
    }

    #[test]
    fn openai_projection_compacts_oversized_tool_result() {
        let big = format!("START{}END", "y".repeat(20_000));
        let msg = ChatMessage::Tool {
            tool_call_id: "call_big".into(),
            content: big.clone(),
            is_error: false,
            attachments: vec![],
        };
        let v = chat_message_to_wire(&msg);
        let content = v["content"].as_str().unwrap();
        assert!(content.contains("compacted for model replay"));
        assert!(content.contains("START") && content.contains("END"));
        // Projection-only: the source message still holds the full result.
        match &msg {
            ChatMessage::Tool { content, .. } => assert_eq!(content.len(), big.len()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn anthropic_projection_compacts_oversized_tool_result() {
        let big = "z".repeat(20_000);
        let msgs = vec![
            ChatMessage::Assistant {
                text: None,
                tool_calls: vec![crate::tools::ToolInvocation {
                    id: "tc_big".into(),
                    name: "bash".into(),
                    input: json!({}),
                    raw_emitted_args: None,
                }],
                thinking: None,
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "tc_big".into(),
                content: big,
                is_error: false,
                attachments: vec![],
            },
        ];
        let wire = chat_messages_to_anthropic_messages(&msgs);
        let rendered = serde_json::to_string(&wire).unwrap();
        assert!(rendered.contains("compacted for model replay"));
    }

    #[test]
    fn chat_messages_to_anthropic_tool_result_carries_image_block() {
        // After an assistant tool_use, a Tool message that carries an
        // image attachment (e.g. MCP screenshot) should project to a
        // tool_result whose content array contains both the text and
        // the image content block.
        let msgs = vec![
            ChatMessage::Assistant {
                text: None,
                tool_calls: vec![ToolInvocation {
                    id: "tc_img".into(),
                    name: "screenshot".into(),
                    input: json!({}),
                    raw_emitted_args: None,
                }],
                thinking: None,
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "tc_img".into(),
                content: "see image".into(),
                is_error: false,
                attachments: vec![UserAttachment::Image(ImageSource {
                    media_type: "image/png".into(),
                    data: ImageData::Base64("PNGBYTES".into()),
                })],
            },
        ];
        let out = chat_messages_to_anthropic_messages(&msgs);
        // [assistant tool_use, user(tool_result containing text+image)]
        assert_eq!(out.len(), 2);
        let user = &out[1];
        assert_eq!(user["role"], "user");
        let outer = user["content"].as_array().unwrap();
        assert_eq!(outer.len(), 1);
        assert_eq!(outer[0]["type"], "tool_result");
        assert_eq!(outer[0]["tool_use_id"], "tc_img");
        let inner = outer[0]["content"].as_array().unwrap();
        // tool_result content is now block-array form (not a string)
        // when attachments are present.
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0]["type"], "text");
        assert_eq!(inner[0]["text"], "see image");
        assert_eq!(inner[1]["type"], "image");
        assert_eq!(inner[1]["source"]["type"], "base64");
        assert_eq!(inner[1]["source"]["media_type"], "image/png");
        assert_eq!(inner[1]["source"]["data"], "PNGBYTES");
    }

    #[test]
    fn chat_messages_to_anthropic_renders_user_text_with_image_block() {
        let msgs = vec![ChatMessage::User {
            content: "what is this".into(),
            attachments: vec![UserAttachment::Image(ImageSource {
                media_type: "image/png".into(),
                data: ImageData::Base64("AAAA".into()),
            })],
        }];
        let out = chat_messages_to_anthropic_messages(&msgs);
        assert_eq!(out.len(), 1);
        let blocks = out[0]["content"].as_array().unwrap();
        // Text first, then image — same ordering as OpenAI parts.
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "what is this");
        assert_eq!(blocks[1]["type"], "image");
        // base64 source shape
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn chat_messages_to_anthropic_renders_url_image() {
        let msgs = vec![ChatMessage::User {
            content: "".into(),
            attachments: vec![UserAttachment::Image(ImageSource {
                media_type: "image/jpeg".into(),
                data: ImageData::Url("https://example.com/x.jpg".into()),
            })],
        }];
        let out = chat_messages_to_anthropic_messages(&msgs);
        let blocks = out[0]["content"].as_array().unwrap();
        // Empty-text user with image: only one block (the image).
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["type"], "url");
        assert_eq!(blocks[0]["source"]["url"], "https://example.com/x.jpg");
    }

    #[test]
    fn chat_message_to_anthropic_merges_tool_results_and_image() {
        // Tool result pending + a user message that also carries an image:
        // both must land in the same user message's content array.
        let msgs = vec![
            ChatMessage::Assistant {
                text: None,
                tool_calls: vec![ToolInvocation {
                    id: "tc_1".into(),
                    name: "screenshot".into(),
                    input: json!({}),
                    raw_emitted_args: None,
                }],
                thinking: None,
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "tc_1".into(),
                content: "captured".into(),
                is_error: false,
                attachments: vec![],
            },
            ChatMessage::User {
                content: "what changed?".into(),
                attachments: vec![UserAttachment::Image(ImageSource {
                    media_type: "image/png".into(),
                    data: ImageData::Base64("ZZ".into()),
                })],
            },
        ];
        let out = chat_messages_to_anthropic_messages(&msgs);
        // [assistant tool_use, user(tool_result + text + image)]
        assert_eq!(out.len(), 2);
        let blocks = out[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "tc_1");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "what changed?");
        assert_eq!(blocks[2]["type"], "image");
    }

    #[test]
    fn chat_messages_to_anthropic_renders_simple_user_assistant() {
        let msgs = vec![
            ChatMessage::User {
                content: "hi".into(),
                attachments: vec![],
            },
            ChatMessage::Assistant {
                text: Some("hello".into()),
                tool_calls: vec![],
                thinking: None,
                usage: None,
            },
        ];
        let out = chat_messages_to_anthropic_messages(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"][0]["type"], "text");
        assert_eq!(out[0]["content"][0]["text"], "hi");
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn chat_messages_to_anthropic_folds_tool_results_into_next_user() {
        // After an assistant tool_use, the next user message is the one
        // carrying the tool_result content block — no separate user/tool
        // hop on the wire.
        let msgs = vec![
            ChatMessage::User {
                content: "do it".into(),
                attachments: vec![],
            },
            ChatMessage::Assistant {
                text: None,
                tool_calls: vec![ToolInvocation {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: json!({"command": "pwd"}),
                    raw_emitted_args: None,
                }],
                thinking: None,
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "call_1".into(),
                content: "{\"stdout\":\"/\"}".into(),
                is_error: false,
                attachments: vec![],
            },
            ChatMessage::User {
                content: "explain".into(),
                attachments: vec![],
            },
        ];
        let out = chat_messages_to_anthropic_messages(&msgs);
        // user, assistant(tool_use), user(tool_result + "explain")
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"][0]["type"], "tool_use");
        assert_eq!(out[1]["content"][0]["id"], "call_1");
        assert_eq!(out[2]["role"], "user");
        assert_eq!(out[2]["content"][0]["type"], "tool_result");
        assert_eq!(out[2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(out[2]["content"][1]["type"], "text");
        assert_eq!(out[2]["content"][1]["text"], "explain");
    }

    #[test]
    fn chat_messages_to_anthropic_renders_thinking_then_text_then_tool_use() {
        let msgs = vec![ChatMessage::Assistant {
            text: Some("preface".into()),
            tool_calls: vec![ToolInvocation {
                id: "t".into(),
                name: "n".into(),
                input: json!({"a": 1}),
                raw_emitted_args: None,
            }],
            thinking: Some(AssistantThinking {
                text: "deep thought".into(),
                signature: Some("sig123".into()),
            }),
            usage: None,
        }];
        let out = chat_messages_to_anthropic_messages(&msgs);
        let blocks = out[0]["content"].as_array().unwrap();
        // Order: thinking → text → tool_use.
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "deep thought");
        assert_eq!(blocks[0]["signature"], "sig123");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "preface");
        assert_eq!(blocks[2]["type"], "tool_use");
    }

    #[test]
    fn chat_messages_to_anthropic_trailing_tool_results_flushed() {
        // Conversation ends on tool results without a follow-up user
        // turn — we still need to send the results to the model.
        let msgs = vec![
            ChatMessage::Assistant {
                text: None,
                tool_calls: vec![ToolInvocation {
                    id: "t".into(),
                    name: "n".into(),
                    input: json!({}),
                    raw_emitted_args: None,
                }],
                thinking: None,
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "t".into(),
                content: "ok".into(),
                is_error: false,
                attachments: vec![],
            },
        ];
        let out = chat_messages_to_anthropic_messages(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["role"], "user");
        assert_eq!(out[1]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn apply_anthropic_cache_strategy_marks_system_last_tool_and_last_message() {
        let system = anthropic_system_field(Some("system prompt"));
        let tools = vec![
            json!({"name": "a", "description": "", "input_schema": {"type": "object"}}),
            json!({"name": "b", "description": "", "input_schema": {"type": "object"}}),
        ];
        let messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "hello"}]}),
        ];
        let out = apply_anthropic_cache_strategy(system, tools, messages);
        let sys_block = &out.system.as_ref().unwrap()[0];
        assert_eq!(sys_block["cache_control"]["type"], "ephemeral");
        // Last tool (only the LAST one) cached, not the first.
        assert!(out.tools[0].get("cache_control").is_none());
        assert_eq!(out.tools[1]["cache_control"]["type"], "ephemeral");
        // Last message's last content block cached.
        let last_msg_blocks = out.messages.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(
            last_msg_blocks.last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn apply_anthropic_cache_strategy_skips_empty_system() {
        // Empty system → omitted entirely (Anthropic rejects cache_control
        // on empty text blocks).
        let out = apply_anthropic_cache_strategy(None, vec![], vec![]);
        assert!(out.system.is_none());
    }

    fn anthropic_client_for_tool_choice_tests() -> AnthropicModelClient {
        AnthropicModelClient::new(AnthropicConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "claude-test".into(),
            max_tokens: 1024,
            temperature: None,
            anthropic_version: AnthropicConfig::DEFAULT_VERSION.into(),
        })
    }

    #[test]
    fn anthropic_client_omits_tool_choice_when_auto() {
        // Auto is Anthropic's default — keep the wire byte-identical
        // by omitting the field (prompt cache stability).
        let client = anthropic_client_for_tool_choice_tests();
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        assert!(body["tools"].as_array().unwrap().len() > 0);
        assert!(body.get("tool_choice").is_none());
        // parallel_tool_calls is OpenAI-only — Anthropic body must
        // never carry it.
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn anthropic_client_emits_tool_choice_required_as_any() {
        let client = anthropic_client_for_tool_choice_tests();
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
        });
        assert_eq!(body["tool_choice"]["type"], "any");
        // OpenAI-only knob must NOT leak into Anthropic body even when
        // caller set it (harness passes it uniformly to both providers).
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn anthropic_client_projects_hosted_web_search_tool() {
        let client = anthropic_client_for_tool_choice_tests();
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "research current AI market".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![HostedTool::WebSearch],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let tools = body["tools"].as_array().unwrap();
        let web = tools
            .iter()
            .find(|tool| tool.get("name").and_then(Value::as_str) == Some("web_search"))
            .unwrap();
        assert_eq!(web["type"], "web_search_20250305");
        assert!(web.get("max_uses").is_none());
    }

    #[test]
    fn clients_report_web_search_support_by_protocol_and_endpoint() {
        let chat = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
        });
        assert_eq!(
            chat.hosted_capability(HostedCapability::WebSearch),
            CapabilitySupport::Unsupported
        );

        let responses = OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            reasoning_summary: None,
        });
        assert_eq!(
            responses.hosted_capability(HostedCapability::WebSearch),
            CapabilitySupport::Supported
        );

        let gateway = OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
            base_url: "https://gateway.example/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            reasoning_summary: None,
        });
        assert_eq!(
            gateway.hosted_capability(HostedCapability::WebSearch),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn anthropic_client_emits_tool_choice_named_tool() {
        let client = anthropic_client_for_tool_choice_tests();
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Tool("bash".into()),
            parallel_tool_calls: None,
        });
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "bash");
    }

    #[test]
    fn anthropic_client_drops_tools_when_choice_is_none() {
        // Anthropic has no native "tool_choice: none" — best
        // approximation is dropping the tools array. The body's
        // `tools` key must be absent and `tool_choice` too.
        let client = anthropic_client_for_tool_choice_tests();
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![ChatMessage::User {
                content: "go".into(),
                attachments: vec![],
            }],
            tools: vec![bash_spec()],
            hosted_tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        });
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn anthropic_stream_state_text_only() {
        let mut s = AnthropicStreamState::default();
        // message_start with id + usage.
        let _ = s
            .feed_event(
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_01","usage":{"input_tokens":10,"output_tokens":0}}}"#,
            )
            .unwrap();
        let _ = s
            .feed_event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            )
            .unwrap();
        let out = s
            .feed_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::TextDelta { msg_id, delta } => {
                assert_eq!(msg_id, "msg_01");
                assert_eq!(delta, "Hello");
            }
            other => panic!("expected TextDelta, got {other:?}"),
        }
        let _ = s.feed_event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
        );
        let out = s
            .feed_event("message_stop", r#"{"type":"message_stop"}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::Done { stop_reason, usage } => {
                assert_eq!(stop_reason, "end_turn");
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input_tokens, 10);
                assert_eq!(u.output_tokens, 5);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_stream_state_thinking_block_emits_delta_and_signature() {
        let mut s = AnthropicStreamState::default();
        let _ = s.feed_event(
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_t"}}"#,
        );
        let _ = s.feed_event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        );
        let out = s
            .feed_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning..."}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        let ModelChunk::ThinkingDelta {
            delta, signature, ..
        } = &out[0]
        else {
            panic!("expected ThinkingDelta");
        };
        assert_eq!(delta, "reasoning...");
        assert!(signature.is_none());

        let out = s
            .feed_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}"#,
            )
            .unwrap();
        let ModelChunk::ThinkingDelta {
            delta, signature, ..
        } = &out[0]
        else {
            panic!("expected ThinkingDelta");
        };
        assert_eq!(delta, "");
        assert_eq!(signature.as_deref(), Some("sig_abc"));
    }

    #[test]
    fn anthropic_stream_state_tool_use_streamed_input() {
        let mut s = AnthropicStreamState::default();
        let _ = s.feed_event(
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_x"}}"#,
        );
        let out = s
            .feed_event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            ModelChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        let out = s
            .feed_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":"}}"#,
            )
            .unwrap();
        let ModelChunk::ToolCallInputDelta { id, delta } = &out[0] else {
            panic!("expected ToolCallInputDelta");
        };
        assert_eq!(id, "toolu_1");
        assert_eq!(delta, "{\"cmd\":");

        let out = s
            .feed_event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            )
            .unwrap();
        match &out[0] {
            ModelChunk::ToolCallEnd { id, input } => {
                assert_eq!(id, "toolu_1");
                assert!(input.is_none());
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_stream_state_finalises_on_close_without_message_stop() {
        // Some gateways drop `message_stop`; the harness still needs a
        // Done. finalize() bridges that gap.
        let mut s = AnthropicStreamState::default();
        let _ = s
            .feed_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":100}}"#,
            )
            .unwrap();
        let done = s.finalize().unwrap();
        match done {
            ModelChunk::Done { stop_reason, .. } => assert_eq!(stop_reason, "max_tokens"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_stop_reason_mapping() {
        assert_eq!(map_anthropic_stop_reason(Some("end_turn")), "end_turn");
        assert_eq!(map_anthropic_stop_reason(Some("tool_use")), "end_turn");
        assert_eq!(map_anthropic_stop_reason(Some("max_tokens")), "max_tokens");
        assert_eq!(map_anthropic_stop_reason(Some("stop_sequence")), "end_turn");
        assert_eq!(map_anthropic_stop_reason(Some("refusal")), "refusal");
        assert_eq!(map_anthropic_stop_reason(None), "end_turn");
    }

    #[test]
    fn classify_anthropic_http_error_buckets_by_status() {
        use reqwest::StatusCode;
        assert!(matches!(
            classify_anthropic_http_error(StatusCode::TOO_MANY_REQUESTS, "{}"),
            ModelClientError::RateLimit(_)
        ));
        assert!(matches!(
            classify_anthropic_http_error(StatusCode::UNAUTHORIZED, "{}"),
            ModelClientError::Auth(_)
        ));
        assert!(matches!(
            classify_anthropic_http_error(
                StatusCode::BAD_REQUEST,
                "{\"error\":{\"message\":\"prompt is too long; context_length_exceeded\"}}"
            ),
            ModelClientError::ContextOverflow(_)
        ));
        assert!(matches!(
            classify_anthropic_http_error(StatusCode::BAD_REQUEST, "invalid model"),
            ModelClientError::BadRequest(_)
        ));
        assert!(matches!(
            classify_anthropic_http_error(StatusCode::INTERNAL_SERVER_ERROR, "oops"),
            ModelClientError::ServerError(_)
        ));
    }

    #[tokio::test]
    async fn collect_model_response_folds_streamed_tool_call_arguments() {
        // Simulate an OpenAI-style streaming tool call whose arguments
        // arrive in three chunks. collect_model_response should glue them
        // back into a parsed JSON value and ignore the leading text.
        let chunks = vec![
            Ok(ModelChunk::TextDelta {
                msg_id: "m".into(),
                delta: "ok ".into(),
            }),
            Ok(ModelChunk::ToolCallStart {
                id: "call_1".into(),
                name: "bash".into(),
            }),
            Ok(ModelChunk::ToolCallInputDelta {
                id: "call_1".into(),
                delta: "{\"command\":".into(),
            }),
            Ok(ModelChunk::ToolCallInputDelta {
                id: "call_1".into(),
                delta: "\"pwd\"}".into(),
            }),
            Ok(ModelChunk::ToolCallEnd {
                id: "call_1".into(),
                input: None,
            }),
            Ok(ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: None,
            }),
        ];
        let stream = futures::stream::iter(chunks).boxed();
        let response = collect_model_response(stream).await.unwrap();
        let ModelResponse::ToolCall {
            invocation,
            preface,
            ..
        } = response
        else {
            panic!("expected ToolCall");
        };
        assert_eq!(invocation.name, "bash");
        assert_eq!(invocation.input["command"], "pwd");
        assert_eq!(preface.as_deref(), Some("ok "));
    }

    #[test]
    fn classify_openai_http_error_buckets_by_status_and_body() {
        use reqwest::StatusCode;
        assert!(matches!(
            classify_openai_http_error(StatusCode::TOO_MANY_REQUESTS, "rate limit hit"),
            ModelClientError::RateLimit(_)
        ));
        assert!(matches!(
            classify_openai_http_error(StatusCode::UNAUTHORIZED, "bad key"),
            ModelClientError::Auth(_)
        ));
        assert!(matches!(
            classify_openai_http_error(StatusCode::FORBIDDEN, "no access"),
            ModelClientError::Auth(_)
        ));
        assert!(matches!(
            classify_openai_http_error(
                StatusCode::BAD_REQUEST,
                "{\"error\":{\"message\":\"this model's maximum context length is 8192\"}}"
            ),
            ModelClientError::ContextOverflow(_)
        ));
        // BAD_REQUEST without context_overflow tell-tale → BadRequest.
        assert!(matches!(
            classify_openai_http_error(StatusCode::BAD_REQUEST, "missing argument"),
            ModelClientError::BadRequest(_)
        ));
        // 5xx → ServerError (retryable with backoff).
        assert!(matches!(
            classify_openai_http_error(StatusCode::INTERNAL_SERVER_ERROR, "oops"),
            ModelClientError::ServerError(_)
        ));
    }

    #[test]
    fn looks_like_context_overflow_matches_common_phrasings() {
        assert!(looks_like_context_overflow(
            "context_length_exceeded: this model has a maximum context length of 8192"
        ));
        assert!(looks_like_context_overflow("too many tokens in prompt"));
        assert!(looks_like_context_overflow(
            "Prompt exceeds the model's maximum context"
        ));
        assert!(!looks_like_context_overflow("invalid api key"));
    }

    #[test]
    fn parse_openai_usage_returns_none_for_missing_or_all_zero() {
        // Field absent entirely.
        assert!(parse_openai_usage(None).is_none());
        // Field present but all zeros — treat as "provider didn't report".
        assert!(parse_openai_usage(Some(&json!({
            "prompt_tokens": 0,
            "completion_tokens": 0
        })))
        .is_none());
    }

    #[test]
    fn openai_client_renders_multi_turn_history() {
        let client = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
            base_url: "https://example.test".into(),
            api_key: "sk-test".into(),
            model: "gpt-test".into(),
            temperature: Some(0.2),
            max_tokens: Some(128),
            reasoning_effort: None,
        });
        let body = client.request_body(&ModelTurnInput {
            system_prompt: None,
            messages: vec![
                ChatMessage::User {
                    content: "run pwd".into(),
                    attachments: vec![],
                },
                ChatMessage::Assistant {
                    text: None,
                    tool_calls: vec![ToolInvocation {
                        id: "call_1".into(),
                        name: "bash".into(),
                        input: json!({"command": "pwd"}),
                        raw_emitted_args: None,
                    }],
                    thinking: None,
                    usage: None,
                },
                ChatMessage::Tool {
                    tool_call_id: "call_1".into(),
                    content: "{\"stdout\":\"/home/user\"}".into(),
                    is_error: false,
                    attachments: vec![],
                },
            ],
            tools: vec![],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "bash"
        );
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
    }

    #[tokio::test]
    async fn scripted_client_emits_tool_call_then_summary() {
        let scripted = ScriptedModelClient;
        let first = scripted
            .next(user("read README.md"))
            .await
            .expect("scripted first");
        let ModelResponse::ToolCall { invocation, .. } = first else {
            panic!("expected tool call on first step");
        };
        assert_eq!(invocation.name, "read");

        // Simulate the loop appending Assistant + Tool messages, then ask
        // the scripted client for its next move — should be a summary.
        let history = ModelTurnInput {
            system_prompt: None,
            messages: vec![
                ChatMessage::User {
                    content: "read README.md".into(),
                    attachments: vec![],
                },
                ChatMessage::Assistant {
                    text: None,
                    tool_calls: vec![invocation.clone()],
                    thinking: None,
                    usage: None,
                },
                ChatMessage::Tool {
                    tool_call_id: invocation.id.clone(),
                    content: "{\"content\":\"hi\"}".into(),
                    is_error: false,
                    attachments: vec![],
                },
            ],
            tools: vec![],
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let second = scripted.next(history).await.expect("scripted second");
        let ModelResponse::Message { text, .. } = second else {
            panic!("expected final message after tool result");
        };
        assert!(text.contains("completed"));
    }

    // ─── OpenAI Responses API ────────────────────────────────────────────

    #[test]
    fn responses_client_builds_responses_endpoint_and_body() {
        let mk = |base: &str| {
            OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
                base_url: base.into(),
                api_key: "sk-test".into(),
                model: "gpt-5".into(),
                temperature: None,
                max_output_tokens: Some(2048),
                reasoning_effort: Some("high".into()),
                reasoning_summary: None,
            })
        };
        // Route appended; version segment respected; no double-append.
        assert_eq!(
            mk("https://api.openai.com/v1").endpoint(),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            mk("https://api.openai.com/v1/").endpoint(),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            mk("https://api.openai.com/v1/responses").endpoint(),
            "https://api.openai.com/v1/responses"
        );

        let client = mk("https://api.openai.com/v1");
        let mut input = user("hello");
        input.system_prompt = Some("be terse".into());
        input.tools = vec![bash_spec()];
        let body = client.request_body(&input);
        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["instructions"], "be terse");
        assert_eq!(body["max_output_tokens"], 2048);
        assert_eq!(body["reasoning"]["effort"], "high");
        // Stateless mode requests encrypted reasoning for round-trip.
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        // Flat tool shape (not nested under `function`).
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "bash");
        assert!(body["tools"][0]["parameters"].is_object());
        assert_eq!(body["tool_choice"], "auto");
        // User message projects to an input_text content part.
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(body["input"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn responses_tool_choice_none_drops_client_and_hosted_tools() {
        let client = OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-5".into(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            reasoning_summary: None,
        });
        let mut input = user("hi");
        input.tools = vec![bash_spec()];
        input.hosted_tools = vec![HostedTool::WebSearch];
        input.tool_choice = ToolChoice::None;
        let body = client.request_body(&input);
        // `None` means no tools this turn — that MUST also suppress hosted
        // (provider-run) tools, else server-side web_search could still fire.
        assert!(
            body.get("tools").is_none(),
            "tools should be absent, got {:?}",
            body.get("tools")
        );
        assert!(body.get("tool_choice").is_none());

        // Sanity: with Auto, both the function tool and the hosted web_search
        // are advertised.
        input.tool_choice = ToolChoice::Auto;
        let body = client.request_body(&input);
        let tools = body["tools"].as_array().expect("tools present");
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t["type"] == "function"));
        assert!(tools.iter().any(|t| t["type"] == "web_search"));
    }

    #[test]
    fn responses_tool_choice_required_sent_for_hosted_only() {
        let client = OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-5".into(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            reasoning_summary: None,
        });
        // Hosted tool only — no client function tools.
        let mut input = user("search the web");
        input.hosted_tools = vec![HostedTool::WebSearch];

        // Required must be sent even without function tools, else it silently
        // downgrades to the default auto.
        input.tool_choice = ToolChoice::Required;
        let body = client.request_body(&input);
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert_eq!(body["tool_choice"], "required");

        // Tool(name) force-selects a FUNCTION tool; with only a hosted tool
        // present there's nothing to force, so it must NOT be sent (encoding a
        // hosted tool as {type:function,name} would 400).
        input.tool_choice = ToolChoice::Tool("bash".into());
        let body = client.request_body(&input);
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert!(
            body.get("tool_choice").is_none(),
            "Tool(name) must not be sent for a hosted-only turn, got {:?}",
            body.get("tool_choice")
        );

        // With a function tool present, Tool(name) IS sent as a function choice.
        input.tools = vec![bash_spec()];
        let body = client.request_body(&input);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["name"], "bash");
    }

    #[test]
    fn responses_stream_state_accepts_reasoning_summary_alias() {
        // Some gateways emit `response.reasoning_summary.delta` instead of the
        // `_text` variant — both must surface as thinking.
        let mut st = OpenAiResponsesStreamState::default();
        let chunks = st
            .feed_data(
                r#"{"type":"response.reasoning_summary.delta","item_id":"rs_1","summary_index":0,"delta":"aliased"}"#,
            )
            .expect("feed_data");
        assert!(
            matches!(&chunks[0], ModelChunk::ThinkingDelta { delta, .. } if delta == "aliased")
        );
    }

    #[test]
    fn responses_stream_state_separates_reasoning_summary_parts() {
        let mut st = OpenAiResponsesStreamState::default();
        let mut chunks: Vec<ModelChunk> = Vec::new();
        let mut feed = |st: &mut OpenAiResponsesStreamState, data: &str| {
            chunks.extend(st.feed_data(data).expect("feed_data"));
        };
        // The separator rides the FIRST real text delta of a later part — not
        // the structural `part.added` event — so it only ever attaches to
        // genuine model output. Part 0 gets no prefix; part 1's first delta
        // is prefixed with a blank line; subsequent deltas of a part are not.
        feed(
            &mut st,
            r#"{"type":"response.reasoning_summary_part.added","item_id":"rs_1","summary_index":0}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":"first"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":" more"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.reasoning_summary_part.added","item_id":"rs_1","summary_index":1}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":1,"delta":"second"}"#,
        );
        let deltas: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c {
                ModelChunk::ThinkingDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        // `part.added` produces nothing; the boundary is folded into the
        // first text delta of part 1.
        assert_eq!(deltas, vec!["first", " more", "\n\nsecond"]);
    }

    #[test]
    fn responses_input_projection_round_trips_tool_calls_and_reasoning() {
        let sig = encode_reasoning_signature("rs_123", "ENC==");
        let messages = vec![
            ChatMessage::User {
                content: "hi".into(),
                attachments: vec![],
            },
            ChatMessage::Assistant {
                text: Some("looked".into()),
                tool_calls: vec![ToolInvocation {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                    raw_emitted_args: None,
                }],
                thinking: Some(AssistantThinking {
                    text: "let me look".into(),
                    signature: Some(sig),
                }),
                usage: None,
            },
            ChatMessage::Tool {
                tool_call_id: "call_1".into(),
                content: "file.txt".into(),
                is_error: false,
                attachments: vec![],
            },
        ];
        let input = chat_messages_to_responses_input(&messages);
        assert_eq!(input[0]["role"], "user");
        // Reasoning item precedes text + tool call, carrying encrypted payload.
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["id"], "rs_123");
        assert_eq!(input[1]["encrypted_content"], "ENC==");
        assert_eq!(input[1]["summary"][0]["text"], "let me look");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[2]["content"][0]["type"], "output_text");
        // Tool call keyed by call_id (the harness's pairing key).
        assert_eq!(input[3]["type"], "function_call");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["name"], "bash");
        assert_eq!(input[4]["type"], "function_call_output");
        assert_eq!(input[4]["call_id"], "call_1");
        assert_eq!(input[4]["output"], "file.txt");
    }

    #[test]
    fn responses_stream_state_emits_text_toolcall_and_done() {
        let mut st = OpenAiResponsesStreamState::default();
        let mut chunks: Vec<ModelChunk> = Vec::new();
        let mut feed = |st: &mut OpenAiResponsesStreamState, data: &str| {
            chunks.extend(st.feed_data(data).expect("feed_data"));
        };
        feed(
            &mut st,
            r#"{"type":"response.created","response":{"id":"resp_1"}}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"Hel"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"lo"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"bash"}}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"command\":"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"\"ls\"}"}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}"}}"#,
        );
        feed(
            &mut st,
            r#"{"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":2}}}}"#,
        );

        assert!(matches!(&chunks[0], ModelChunk::TextDelta { delta, .. } if delta == "Hel"));
        assert!(matches!(&chunks[1], ModelChunk::TextDelta { delta, .. } if delta == "lo"));
        assert!(
            matches!(&chunks[2], ModelChunk::ToolCallStart { id, name } if id == "call_1" && name == "bash")
        );
        assert!(matches!(&chunks[3], ModelChunk::ToolCallInputDelta { id, .. } if id == "call_1"));
        assert!(matches!(&chunks[4], ModelChunk::ToolCallInputDelta { id, .. } if id == "call_1"));
        match &chunks[5] {
            ModelChunk::ToolCallEnd { id, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(input.as_ref().expect("early input")["command"], "ls");
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        match chunks.last().expect("done chunk") {
            ModelChunk::Done { stop_reason, usage } => {
                assert_eq!(stop_reason, "end_turn");
                let u = usage.as_ref().expect("usage");
                assert_eq!(u.input_tokens, 10);
                assert_eq!(u.output_tokens, 5);
                assert_eq!(u.cache_read_input_tokens, 2);
            }
            other => panic!("expected Done, got {other:?}"),
        }
        assert!(st.ended_cleanly());
    }

    #[test]
    fn responses_stream_state_round_trips_reasoning_signature() {
        let mut st = OpenAiResponsesStreamState::default();
        let mut chunks: Vec<ModelChunk> = Vec::new();
        chunks.extend(
            st.feed_data(
                r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"pondering"}"#,
            )
            .unwrap(),
        );
        chunks.extend(
            st.feed_data(
                r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","encrypted_content":"ENC=="}}"#,
            )
            .unwrap(),
        );
        assert!(
            matches!(&chunks[0], ModelChunk::ThinkingDelta { delta, signature, .. } if delta == "pondering" && signature.is_none())
        );
        match &chunks[1] {
            ModelChunk::ThinkingDelta {
                delta, signature, ..
            } => {
                assert!(delta.is_empty());
                let (id, enc) =
                    decode_reasoning_signature(signature.as_ref().expect("signature")).unwrap();
                assert_eq!(id, "rs_1");
                assert_eq!(enc, "ENC==");
            }
            other => panic!("expected ThinkingDelta, got {other:?}"),
        }
    }

    #[test]
    fn responses_stream_state_reports_cutoff_without_terminal_event() {
        let mut st = OpenAiResponsesStreamState::default();
        st.feed_data(r#"{"type":"response.output_text.delta","item_id":"m","delta":"hi"}"#)
            .unwrap();
        // No terminal `response.*` event arrived → the stream was cut off.
        assert!(!st.ended_cleanly());
    }

    #[test]
    fn reasoning_signature_encode_decode_roundtrip() {
        let sig = encode_reasoning_signature("rs_abc", "base64==payload");
        assert_eq!(
            decode_reasoning_signature(&sig),
            Some(("rs_abc".into(), "base64==payload".into()))
        );
        // A plain (non-Responses) signature has no newline → not decodable,
        // so it is never mis-projected as a reasoning item.
        assert_eq!(decode_reasoning_signature("anthropic-sig-no-newline"), None);
    }
}
