//! MCP (Model Context Protocol) client + tool runtime.
//!
//! Speaks MCP 2024-11-05 in two transports:
//!
//! - **HTTP POST** (`bootstrap.mcp_servers[].type = "url"`): every request
//!   is a self-contained `POST <url>` with a JSON-RPC 2.0 body. We do
//!   NOT do the SSE streaming variant of the streamable HTTP transport.
//! - **stdio** (`bootstrap.mcp_servers[].type = "stdio"`): spawn the
//!   configured `command` as a child process, exchange newline-delimited
//!   JSON-RPC 2.0 messages over its stdin/stdout. Stderr is forwarded to
//!   the RD tracing log under `target = "harness::mcp::stdio"`. The
//!   process lives for the McpClient's lifetime (one per MCP server);
//!   Drop kills it (best-effort SIGKILL via `tokio::process::Child::kill`).
//!
//! All configuration (command, args, env vars to forward, working dir,
//! HTTP url, timeout) is sourced from bootstrap.yaml — the harness
//! crate never reads `std::env::var` directly.
//!
//! Lifecycle on a session boot:
//!   1. `McpToolRuntime::discover(servers)`
//!      ├─ for each server: `McpClient::new` → `initialize` →
//!      │  `tools/list` → cache the spec list with `{server}__` prefix
//!      └─ unreachable servers log + skip (don't fail the session)
//!   2. AgentLoopHarness sees the MCP tools through the composite
//!      `ToolRuntime` (native + MCP merged via `CompositeToolRuntime`)
//!   3. On invocation: route by tool-name prefix to the right
//!      `McpClient.tools_call`, strip prefix before sending to server
//!
//! Session-id handling: some MCP server implementations issue an
//! `Mcp-Session-Id` header on `initialize` and require it on subsequent
//! requests. `McpClient` records the first non-empty value it sees and
//! replays it; servers that don't issue one stay stateless and that's
//! fine too.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::model::{ImageData, ImageSource, UserAttachment};
use crate::tools::{
    ToolFailure, ToolFailureKind, ToolInvocation, ToolOutcome, ToolRuntime, ToolRuntimeError,
    ToolSpec,
};

/// MCP protocol version we negotiate with the server. The wire format
/// is stable across patches; bump this only when the spec maintainers
/// publish a version that changes the methods we use (`initialize` /
/// `tools/list` / `tools/call`).
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Default per-request timeout. MCP tools can be slow (calling other
/// LLMs / external APIs) but 30 s catches the common deadlocks. Same
/// budget governs HTTP requests and stdio request-response round-trips.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Transport-agnostic config for a single MCP server. The variant
/// determines whether we POST to a URL or spawn a child process.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    /// Per-call tool invocation timeout (maps to `tool_timeout_sec`).
    /// Default: 30 s for backwards-compat; bootstrap sets it to
    /// the MCP-spec default of 60 s via `with_timeout`.
    pub timeout: Duration,
    /// Time allowed for `initialize` + `tools/list` handshake.
    /// Default: 10 s (OpenAI Codex default).
    pub startup_timeout: Duration,
    /// If `true`, a failure to initialise aborts session boot.
    /// Default: `false` (unreachable servers are silently skipped).
    pub required: bool,
    /// Tool allowlist (short names, no `{server}__` prefix).
    /// Empty = all tools exposed.
    pub enabled_tools: Vec<String>,
    pub transport: McpTransport,
}

/// Transport mechanism for an MCP server.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// MCP over HTTP POST / SSE. Sessions are tracked via the
    /// `Mcp-Session-Id` header per spec.
    Http {
        url: String,
        /// Static headers sent on every request (e.g. `Authorization`).
        headers: HashMap<String, String>,
    },
    /// MCP over stdio: spawn `command` (with `args`, `env`, `working_dir`),
    /// exchange newline-delimited JSON-RPC 2.0 messages over its
    /// stdin/stdout. The process is owned by the `McpClient`; Drop
    /// kills it.
    ///
    /// `env` is the **full** env passed to the child — anything not in
    /// this map is NOT inherited (we deliberately don't read
    /// `std::env::vars()` so all per-session secrets stay in
    /// bootstrap.yaml). The child still receives a baseline `PATH`
    /// derived from `command` lookup, but no other host env leaks in.
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        working_dir: Option<String>,
    },
}

impl McpServerConfig {
    /// HTTP-transport convenience constructor. Same name as the v1
    /// `new()` for source compat.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::http(name, url)
    }

    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timeout: DEFAULT_TIMEOUT,
            startup_timeout: Duration::from_secs(10),
            required: false,
            enabled_tools: Vec::new(),
            transport: McpTransport::Http {
                url: url.into(),
                headers: HashMap::new(),
            },
        }
    }

    /// stdio-transport convenience constructor. `env` / `working_dir`
    /// default to empty; use the struct literal form or chain setters
    /// if you need them.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            name: name.into(),
            timeout: DEFAULT_TIMEOUT,
            startup_timeout: Duration::from_secs(10),
            required: false,
            enabled_tools: Vec::new(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args,
                env: HashMap::new(),
                working_dir: None,
            },
        }
    }

    /// Per-call tool invocation timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Time allowed for `initialize` + `tools/list` handshake.
    pub fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// If `true`, init failure aborts session boot.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Tool allowlist (short names without `{server}__` prefix).
    pub fn with_enabled_tools(mut self, tools: Vec<String>) -> Self {
        self.enabled_tools = tools;
        self
    }

    /// Add a static HTTP request header (HTTP transport only).
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Http { headers, .. } = &mut self.transport {
            headers.insert(key.into(), value.into());
        }
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.insert(key.into(), value.into());
        }
        self
    }

    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        if let McpTransport::Stdio { working_dir, .. } = &mut self.transport {
            *working_dir = Some(dir.into());
        }
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("decode: {0}")]
    Decode(String),
    #[error("server error code={code} message={message}")]
    Server { code: i64, message: String },
    #[error("missing field {0}")]
    MissingField(&'static str),
}

// ── JSON-RPC 2.0 wire types ─────────────────────────────────────────

#[derive(Debug, Serialize)]
struct McpRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct McpNotification<'a> {
    jsonrpc: &'static str,
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct McpResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    /// `None` for server-initiated notifications (stdio transport). HTTP
    /// transport always carries the response id, but we accept both
    /// shapes from the same parse routine.
    id: Option<u64>,
    result: Option<Value>,
    error: Option<McpResponseError>,
}

#[derive(Debug, Deserialize)]
struct McpResponseError {
    code: i64,
    message: String,
    #[serde(default)]
    #[allow(dead_code)]
    data: Option<Value>,
}

// ── MCP-specific payload types (under JSON-RPC `result`) ────────────

#[derive(Debug, Deserialize)]
struct McpToolDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema", default = "default_input_schema")]
    input_schema: Value,
}

fn default_input_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

#[derive(Debug, Deserialize)]
struct McpToolsListResult {
    tools: Vec<McpToolDef>,
}

#[derive(Debug, Deserialize)]
struct McpToolsCallResult {
    #[serde(default)]
    content: Vec<McpContent>,
    #[serde(default, rename = "isError")]
    is_error: bool,
}

/// MCP content block on a `tools/call` response.
///
/// Text variants flow into `ToolOutcome.output`'s `content` string;
/// Image variants are surfaced as `UserAttachment::Image` on
/// `ToolOutcome.attachments` so that providers which support vision
/// in the tool-result slot (Anthropic) can present them to the model.
/// Providers that don't (OpenAI) degrade them to a text placeholder
/// in `chat_message_to_wire`.
///
/// `resource` / other variants we don't yet model are collapsed into
/// bracketed placeholder text — they're rare in practice and the
/// shape varies enough to warrant a dedicated pass when we add them.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[serde(default, rename = "mimeType")]
        mime_type: String,
        #[serde(default)]
        data: String,
    },
    #[serde(other)]
    Other,
}

// ── McpClient ───────────────────────────────────────────────────────

/// One client per MCP server. Stateless on the wire except for the
/// optional `Mcp-Session-Id` header (HTTP transport only) — see
/// module docstring.
pub struct McpClient {
    name: String,
    timeout: Duration,
    next_id: AtomicU64,
    inner: McpClientInner,
}

enum McpClientInner {
    Http(HttpInner),
    Stdio(StdioInner),
}

struct HttpInner {
    http: reqwest::Client,
    url: String,
    /// Static headers sent on every request (e.g. `Authorization`).
    headers: HashMap<String, String>,
    session_id: Arc<RwLock<Option<String>>>,
}

/// Stdio inner — owns the child process via the writer/reader task
/// pair. `request_tx` is the only way to talk to the child; closing
/// it tears down the pair on Drop.
struct StdioInner {
    request_tx: tokio::sync::mpsc::Sender<StdioRequest>,
    pending_kill: Option<Arc<std::sync::Mutex<Option<tokio::process::Child>>>>,
}

enum StdioRequest {
    /// JSON-RPC request that expects a response. Reply lands on `reply`
    /// via the reader task's id-routing map.
    Call {
        id: u64,
        body: String,
        reply: tokio::sync::oneshot::Sender<Result<Value, McpError>>,
    },
    /// JSON-RPC notification — write + forget.
    Notify { body: String },
}

type PendingReplies =
    Arc<std::sync::Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, McpError>>>>>;

impl McpClient {
    pub fn new(config: McpServerConfig) -> Result<Self, McpError> {
        let name = config.name.clone();
        let timeout = config.timeout;
        let inner = match config.transport {
            McpTransport::Http { url, headers } => {
                let http = reqwest::Client::builder()
                    .timeout(timeout)
                    .build()
                    .map_err(|e| McpError::Transport(e.to_string()))?;
                McpClientInner::Http(HttpInner {
                    http,
                    url,
                    headers,
                    session_id: Arc::new(RwLock::new(None)),
                })
            }
            McpTransport::Stdio {
                command,
                args,
                env,
                working_dir,
            } => spawn_stdio(&name, command, args, env, working_dir)?,
        };
        Ok(Self {
            name,
            timeout,
            next_id: AtomicU64::new(1),
            inner,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the spec-required initialization handshake. Captures the
    /// session id (if any) and notifies the server that we're ready.
    pub async fn initialize(&self) -> Result<(), McpError> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "agentmatrix-runtime-driver",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });
        let _ = self.call("initialize", Some(params)).await?;
        // notifications/initialized has no `id` and expects no response —
        // we fire-and-forget. Errors get logged but don't fail boot.
        if let Err(e) = self.notify("notifications/initialized", None).await {
            tracing::warn!(
                target: "harness::mcp",
                server = %self.name,
                error = %e,
                "notifications/initialized fire-and-forget failed; continuing"
            );
        }
        Ok(())
    }

    /// Pull the server's tool advertisement. Each `ToolSpec` returned
    /// has the **unprefixed** name; `McpToolRuntime::discover` prefixes
    /// with `{server_name}__` to avoid collisions.
    pub async fn tools_list(&self) -> Result<Vec<ToolSpec>, McpError> {
        let value = self.call("tools/list", None).await?;
        let result: McpToolsListResult = serde_json::from_value(value)
            .map_err(|e| McpError::Decode(format!("tools/list result: {e}")))?;
        Ok(result
            .tools
            .into_iter()
            .map(|t| ToolSpec {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect())
    }

    /// Invoke an MCP tool. `name` is the **unprefixed** MCP-side tool
    /// name (caller has already stripped the `{server}__` prefix).
    /// Returns a `ToolOutcome` shaped the same way `SandboxToolRuntime`
    /// returns its outcomes — the harness layer doesn't care which
    /// runtime served the call.
    pub async fn tools_call(&self, name: &str, arguments: Value) -> Result<ToolOutcome, McpError> {
        let params = json!({
            "name": name,
            "arguments": arguments,
        });
        let value = self.call("tools/call", Some(params)).await?;
        let result: McpToolsCallResult = serde_json::from_value(value)
            .map_err(|e| McpError::Decode(format!("tools/call result: {e}")))?;

        // Split MCP content blocks into a text channel (goes into
        // the tool_result content string) and an image channel
        // (goes into `attachments`, preserved for the next
        // user-turn projection on vision-capable providers).
        let mut text_parts: Vec<String> = Vec::new();
        let mut attachments: Vec<UserAttachment> = Vec::new();
        for c in result.content {
            match c {
                McpContent::Text { text } => text_parts.push(text),
                McpContent::Image { mime_type, data } => {
                    if data.is_empty() {
                        // Server sent an image block but no payload —
                        // surface a placeholder so the model still
                        // knows something visual was returned.
                        text_parts.push(format!("[image {mime_type} returned with empty data]"));
                    } else {
                        attachments.push(UserAttachment::Image(ImageSource {
                            media_type: if mime_type.is_empty() {
                                "image/png".to_string()
                            } else {
                                mime_type
                            },
                            data: ImageData::Base64(data),
                        }));
                    }
                }
                McpContent::Other => {
                    text_parts.push("[non-text MCP content elided]".into());
                }
            }
        }
        let content_str = text_parts.join("\n");

        let output = if result.is_error {
            Err(ToolFailure::new(
                ToolFailureKind::Runtime,
                if content_str.is_empty() {
                    format!("MCP tool {name} reported error")
                } else {
                    format!("MCP tool {name} error: {content_str}")
                },
            ))
        } else {
            // Wrap text content in a JSON object so it slots into the
            // same `tool_result.content` shape native tools produce.
            Ok(json!({"content": content_str}))
        };
        Ok(ToolOutcome {
            output,
            attachments,
        })
    }

    /// Internal: send a request, parse the JSON-RPC envelope, return
    /// the `result` field (or the `error` mapped to `McpError::Server`).
    async fn call(&self, method: &str, params: Option<Value>) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = McpRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        match &self.inner {
            McpClientInner::Http(http) => self.call_http(http, id, &body).await,
            McpClientInner::Stdio(stdio) => self.call_stdio(stdio, id, &body).await,
        }
    }

    /// Fire-and-forget notification (JSON-RPC has no `id`, no response).
    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        let body = McpNotification {
            jsonrpc: "2.0",
            method,
            params,
        };
        match &self.inner {
            McpClientInner::Http(http) => self.notify_http(http, &body).await,
            McpClientInner::Stdio(stdio) => self.notify_stdio(stdio, &body).await,
        }
    }

    async fn call_http(
        &self,
        http: &HttpInner,
        id: u64,
        body: &McpRequest<'_>,
    ) -> Result<Value, McpError> {
        // Advertise both response shapes so servers can pick. MCP SDK's
        // default streamable-HTTP transport will reply with
        // `text/event-stream` if it sees this header — we MUST then
        // parse SSE rather than `resp.text()` (which would block until
        // the stream is fully drained but, more importantly, the bytes
        // we get back are SSE framing, not JSON).
        let mut req = http
            .http
            .post(&http.url)
            .header("Accept", "application/json, text/event-stream")
            .json(body);
        // Inject static headers (e.g. Authorization) configured in bootstrap.
        for (k, v) in &http.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        if let Some(sid) = cached_session_id(&http.session_id) {
            req = req.header("Mcp-Session-Id", sid);
        }
        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                McpError::Timeout(e.to_string())
            } else {
                McpError::Transport(e.to_string())
            }
        })?;

        // Server may issue a session id on initialize; capture the first
        // non-empty one we see. Subsequent calls replay it.
        if let Some(sid) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            if !sid.is_empty() {
                if let Ok(mut guard) = http.session_id.write() {
                    if guard.is_none() {
                        *guard = Some(sid.to_string());
                    }
                }
            }
        }

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(McpError::Http {
                status: status.as_u16(),
                body: body_text.chars().take(512).collect(),
            });
        }

        // Branch on Content-Type. SSE responses can carry multiple
        // `message` events (server-side notifications + our response);
        // we drain them all and pick the JSON-RPC envelope whose `id`
        // matches what we sent.
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if content_type.starts_with("text/event-stream") {
            parse_mcp_sse_response(resp, id, &self.name).await
        } else {
            let body_text = resp.text().await.unwrap_or_default();
            let parsed: McpResponse = serde_json::from_str(&body_text)
                .map_err(|e| McpError::Decode(format!("response body: {e}; raw={body_text}")))?;
            if let Some(err) = parsed.error {
                return Err(McpError::Server {
                    code: err.code,
                    message: err.message,
                });
            }
            parsed.result.ok_or(McpError::MissingField("result"))
        }
    }

    async fn notify_http(
        &self,
        http: &HttpInner,
        body: &McpNotification<'_>,
    ) -> Result<(), McpError> {
        // Notifications still advertise SSE; some servers reply with
        // `202 Accepted` + an SSE stream that carries no envelope (it
        // was just an ACK). We send + drop the response body.
        let mut req = http
            .http
            .post(&http.url)
            .header("Accept", "application/json, text/event-stream")
            .json(body);
        for (k, v) in &http.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        if let Some(sid) = cached_session_id(&http.session_id) {
            req = req.header("Mcp-Session-Id", sid);
        }
        req.send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        Ok(())
    }

    async fn call_stdio(
        &self,
        stdio: &StdioInner,
        id: u64,
        body: &McpRequest<'_>,
    ) -> Result<Value, McpError> {
        let line = serde_json::to_string(body)
            .map_err(|e| McpError::Decode(format!("encode request: {e}")))?;
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        stdio
            .request_tx
            .send(StdioRequest::Call {
                id,
                body: line,
                reply: reply_tx,
            })
            .await
            .map_err(|_| McpError::Transport("stdio worker gone".into()))?;
        match tokio::time::timeout(self.timeout, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(McpError::Transport(
                "stdio reply channel closed before response".into(),
            )),
            Err(_) => Err(McpError::Timeout(format!(
                "stdio request timed out after {:?}",
                self.timeout
            ))),
        }
    }

    async fn notify_stdio(
        &self,
        stdio: &StdioInner,
        body: &McpNotification<'_>,
    ) -> Result<(), McpError> {
        let line = serde_json::to_string(body)
            .map_err(|e| McpError::Decode(format!("encode notification: {e}")))?;
        stdio
            .request_tx
            .send(StdioRequest::Notify { body: line })
            .await
            .map_err(|_| McpError::Transport("stdio worker gone".into()))?;
        Ok(())
    }
}

fn cached_session_id(slot: &Arc<RwLock<Option<String>>>) -> Option<String> {
    slot.read().ok().and_then(|g| g.clone())
}

/// Drain a `text/event-stream` response from an MCP server and return
/// the JSON-RPC envelope whose `id` matches `expected_id`. Server-side
/// notifications (no `id`) and unrelated responses (other `id`s) are
/// logged at debug + discarded — the streamable-HTTP transport may
/// interleave them with our response.
///
/// Fails if the stream ends without an id-matching envelope, or with an
/// `error` field on the matched envelope.
async fn parse_mcp_sse_response(
    resp: reqwest::Response,
    expected_id: u64,
    server_name: &str,
) -> Result<Value, McpError> {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;

    let mut events = resp.bytes_stream().eventsource();
    while let Some(ev) = events.next().await {
        let ev = ev.map_err(|e| McpError::Transport(format!("SSE transport error: {e}")))?;
        // MCP streamable-HTTP uses default event name (`message`) for
        // JSON-RPC envelopes. We accept either an empty event name or
        // explicit `message`; anything else (ping, retry, custom) is
        // ignored.
        if !ev.event.is_empty() && ev.event != "message" {
            tracing::debug!(
                target: "harness::mcp",
                server = %server_name,
                event = %ev.event,
                "ignoring non-message SSE event"
            );
            continue;
        }
        let trimmed = ev.data.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: McpResponse = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    target: "harness::mcp",
                    server = %server_name,
                    error = %e,
                    "SSE event body is not a JSON-RPC envelope; skipping"
                );
                continue;
            }
        };
        // Server-initiated notification (no id): we don't implement
        // sampling / elicitation yet, so drop and keep draining.
        let Some(rid) = parsed.id else {
            tracing::debug!(
                target: "harness::mcp",
                server = %server_name,
                "ignoring server-initiated notification mid-SSE stream"
            );
            continue;
        };
        if rid != expected_id {
            // Stale / unrelated reply (shouldn't really happen on a
            // request-scoped POST stream, but defend anyway).
            tracing::debug!(
                target: "harness::mcp",
                server = %server_name,
                rid,
                expected_id,
                "ignoring SSE response with mismatched id"
            );
            continue;
        }
        if let Some(err) = parsed.error {
            return Err(McpError::Server {
                code: err.code,
                message: err.message,
            });
        }
        return parsed.result.ok_or(McpError::MissingField("result"));
    }
    Err(McpError::Transport(format!(
        "SSE stream closed without a JSON-RPC response matching id={expected_id}"
    )))
}

/// Drop wires teardown for stdio inners: signal the worker to stop +
/// best-effort kill on the child if it's still alive.
impl Drop for McpClient {
    fn drop(&mut self) {
        if let McpClientInner::Stdio(stdio) = &mut self.inner {
            // Closing the sender lets the writer task observe EOF and
            // exit, which closes the child's stdin; most well-behaved
            // MCP servers exit on EOF. As a safety net, kill the child
            // explicitly so a misbehaved server can't leak processes.
            if let Some(child_slot) = stdio.pending_kill.take() {
                if let Ok(mut guard) = child_slot.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.start_kill();
                    }
                }
            }
        }
    }
}

// ── stdio transport ─────────────────────────────────────────────────

/// Spawn the configured MCP server as a child process, wire up a
/// writer task (drains an mpsc of outbound requests/notifications to
/// the child's stdin) and a reader task (parses newline-delimited
/// JSON-RPC messages from the child's stdout, routes responses to
/// their `oneshot` waiters by `id`). Stderr is drained into RD
/// tracing under `target = "harness::mcp::stdio"`.
///
/// Returns an `McpClientInner::Stdio` ready for `call`/`notify`.
fn spawn_stdio(
    name: &str,
    command: String,
    args: Vec<String>,
    env: HashMap<String, String>,
    working_dir: Option<String>,
) -> Result<McpClientInner, McpError> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let mut cmd = Command::new(&command);
    cmd.args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // env_clear FIRST so nothing host-side leaks in. The MCP server
        // sees exactly what bootstrap.yaml said and nothing more.
        .env_clear();
    for (k, v) in &env {
        cmd.env(k, v);
    }
    // Keep the child env sealed, but provide PATH by default so common
    // stdio launchers like `npx` can resolve their own subprocesses.
    if !env.contains_key("PATH") {
        if let Some(path) = std::env::var_os("PATH") {
            cmd.env("PATH", path);
        }
    }
    if let Some(dir) = working_dir.as_deref() {
        cmd.current_dir(dir);
    }
    cmd.kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| McpError::Transport(format!("stdio spawn {command:?}: {e}")))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| McpError::Transport("stdio child has no stdin".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| McpError::Transport("stdio child has no stdout".into()))?;
    let stderr = child.stderr.take();

    // Shared map from outbound request id → oneshot reply slot. The
    // writer task installs entries; the reader task pops them.
    let pending: PendingReplies = Arc::new(std::sync::Mutex::new(HashMap::new()));

    // Writer task: drain outbound mpsc → child stdin.
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel::<StdioRequest>(32);
    {
        let pending = pending.clone();
        let server_name = name.to_string();
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(req) = request_rx.recv().await {
                match req {
                    StdioRequest::Call { id, body, reply } => {
                        // Register pending FIRST so a quick reply can't
                        // race the writer.
                        if let Ok(mut guard) = pending.lock() {
                            guard.insert(id, reply);
                        }
                        if let Err(e) = write_stdio_line(&mut stdin, &body).await {
                            // Pop the waiter we just registered + return
                            // the error so the caller doesn't hang.
                            if let Ok(mut guard) = pending.lock() {
                                if let Some(slot) = guard.remove(&id) {
                                    let _ = slot.send(Err(McpError::Transport(format!(
                                        "stdio write failed: {e}"
                                    ))));
                                }
                            }
                            tracing::warn!(
                                target: "harness::mcp::stdio",
                                server = %server_name,
                                error = %e,
                                "stdio writer terminated"
                            );
                            break;
                        }
                    }
                    StdioRequest::Notify { body } => {
                        if let Err(e) = write_stdio_line(&mut stdin, &body).await {
                            tracing::warn!(
                                target: "harness::mcp::stdio",
                                server = %server_name,
                                error = %e,
                                "stdio writer terminated during notify"
                            );
                            break;
                        }
                    }
                }
            }
            // Channel closed → drop stdin so child sees EOF and exits.
            // (Explicit drop for documentation; happens implicitly too.)
            drop(stdin);
        });
    }

    // Reader task: parse newline-delimited JSON-RPC → route by id.
    {
        let pending = pending.clone();
        let server_name = name.to_string();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        // EOF — server closed stdout. Fail all pending.
                        if let Ok(mut guard) = pending.lock() {
                            for (_, slot) in guard.drain() {
                                let _ = slot.send(Err(McpError::Transport(
                                    "stdio server closed stdout".into(),
                                )));
                            }
                        }
                        break;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let parsed: McpResponse = match serde_json::from_str(trimmed) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(
                                    target: "harness::mcp::stdio",
                                    server = %server_name,
                                    error = %e,
                                    line = %trimmed.chars().take(256).collect::<String>(),
                                    "stdio reader could not parse JSON-RPC envelope"
                                );
                                continue;
                            }
                        };
                        // We only care about responses (have `id`).
                        // Server-initiated notifications without `id`
                        // are ignored — we don't implement sampling /
                        // elicitation yet.
                        let Some(id) = parsed.id else {
                            tracing::debug!(
                                target: "harness::mcp::stdio",
                                server = %server_name,
                                "ignoring server-initiated notification"
                            );
                            continue;
                        };
                        let slot = if let Ok(mut guard) = pending.lock() {
                            guard.remove(&id)
                        } else {
                            None
                        };
                        if let Some(slot) = slot {
                            let result = if let Some(err) = parsed.error {
                                Err(McpError::Server {
                                    code: err.code,
                                    message: err.message,
                                })
                            } else {
                                parsed.result.ok_or(McpError::MissingField("result"))
                            };
                            let _ = slot.send(result);
                        } else {
                            tracing::debug!(
                                target: "harness::mcp::stdio",
                                server = %server_name,
                                id,
                                "stdio reply for unknown id (timeout already fired?)"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "harness::mcp::stdio",
                            server = %server_name,
                            error = %e,
                            "stdio reader I/O error"
                        );
                        break;
                    }
                }
            }
        });
    }

    // Stderr drain — best-effort tracing. Don't wait on this task; if
    // the child's stderr is huge we still let the reader/writer
    // dominate scheduling.
    if let Some(stderr) = stderr {
        let server_name = name.to_string();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if !trimmed.is_empty() {
                            tracing::debug!(
                                target: "harness::mcp::stdio",
                                server = %server_name,
                                stderr = %trimmed,
                            );
                        }
                    }
                }
            }
        });
    }

    // Park the child so Drop on McpClient can kill it. We DON'T
    // .await it — exit code is best-effort observed via the reader's
    // EOF detection.
    let child_slot = Arc::new(std::sync::Mutex::new(Some(child)));

    Ok(McpClientInner::Stdio(StdioInner {
        request_tx,
        pending_kill: Some(child_slot),
    }))
}

async fn write_stdio_line<W: tokio::io::AsyncWrite + Unpin>(
    stdin: &mut W,
    body: &str,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    stdin.write_all(body.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await
}

// ── McpToolRuntime ──────────────────────────────────────────────────

/// ToolRuntime that fronts one or more MCP servers. Built once at
/// session boot via `discover()`. Cheap to clone (Arc internally).
#[derive(Clone)]
pub struct McpToolRuntime {
    inner: Arc<McpToolRuntimeInner>,
}

struct McpToolRuntimeInner {
    clients: Vec<McpClient>,
    specs: Vec<ToolSpec>,
    tool_to_client: HashMap<String, usize>,
}

impl McpToolRuntime {
    /// Tool-name prefix separator between the server name and the
    /// upstream tool name. Chosen as `__` because it's invalid in most
    /// MCP server names already and unlikely to collide. The agent loop
    /// receives the prefixed name; `tools_call` strips it before
    /// forwarding to the server.
    pub const NAME_SEPARATOR: &'static str = "__";

    /// Connect to every configured server, handshake, list tools.
    /// Unreachable / misbehaving servers log a warning and are skipped —
    /// a single down dependency shouldn't fail the whole session.
    /// Returns the runtime even if 0 servers came up (specs() will just
    /// be empty); caller decides whether that's acceptable.
    pub async fn discover(servers: Vec<McpServerConfig>) -> Self {
        let mut clients: Vec<McpClient> = Vec::with_capacity(servers.len());
        let mut specs: Vec<ToolSpec> = Vec::new();
        let mut tool_to_client: HashMap<String, usize> = HashMap::new();

        for config in servers {
            let server_name = config.name.clone();
            let required = config.required;
            // Extract enabled_tools before config is consumed by McpClient::new.
            let enabled_tools: std::collections::HashSet<String> =
                config.enabled_tools.iter().cloned().collect();
            let client = match McpClient::new(config) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        target: "harness::mcp",
                        server = %server_name,
                        error = %e,
                        "McpClient::new failed; skipping server"
                    );
                    continue;
                }
            };
            if let Err(e) = client.initialize().await {
                if required {
                    tracing::error!(
                        target: "harness::mcp",
                        server = %server_name,
                        error = %e,
                        "required MCP server failed to initialize; session boot will fail"
                    );
                    // `discover` returns a partial runtime; caller checks
                    // server_count() and aborts if required server is absent.
                    // Flag the absence via a sentinel rather than panicking.
                } else {
                    tracing::warn!(
                        target: "harness::mcp",
                        server = %server_name,
                        error = %e,
                        "MCP initialize failed; skipping server"
                    );
                }
                continue;
            }
            let server_specs = match client.tools_list().await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        target: "harness::mcp",
                        server = %server_name,
                        error = %e,
                        "MCP tools/list failed; skipping server"
                    );
                    continue;
                }
            };
            let client_idx = clients.len();
            for mut spec in server_specs {
                let original = spec.name.clone();
                // Apply enabled_tools allowlist: skip tools not in the list
                // (empty set = all tools allowed).
                if !enabled_tools.is_empty() && !enabled_tools.contains(&original) {
                    continue;
                }
                // Prefix server name onto the tool to avoid collisions
                // when multiple MCP servers offer same-named tools.
                spec.name = format!("{server_name}{}{original}", Self::NAME_SEPARATOR);
                if tool_to_client.contains_key(&spec.name) {
                    // Two servers with the same name → operator misconfigured.
                    tracing::warn!(
                        target: "harness::mcp",
                        tool = %spec.name,
                        "duplicate MCP tool name after prefixing; later registration wins"
                    );
                }
                tool_to_client.insert(spec.name.clone(), client_idx);
                specs.push(spec);
            }
            clients.push(client);
        }

        Self {
            inner: Arc::new(McpToolRuntimeInner {
                clients,
                specs,
                tool_to_client,
            }),
        }
    }

    /// Number of MCP servers that successfully came up (initialize + tools/list ok).
    pub fn server_count(&self) -> usize {
        self.inner.clients.len()
    }
}

#[async_trait]
impl ToolRuntime for McpToolRuntime {
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner.specs.clone()
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let Some(&idx) = self.inner.tool_to_client.get(&invocation.name) else {
            return Err(ToolRuntimeError::UnknownTool(invocation.name));
        };
        let client = &self.inner.clients[idx];
        // Strip the `{server}__` prefix before sending to the MCP server.
        let original_name = invocation
            .name
            .split_once(Self::NAME_SEPARATOR)
            .map(|(_, name)| name)
            .unwrap_or(invocation.name.as_str())
            .to_string();
        client
            .tools_call(&original_name, invocation.input)
            .await
            .map_err(mcp_error_to_tool_runtime_error)
    }
}

fn mcp_error_to_tool_runtime_error(err: McpError) -> ToolRuntimeError {
    if matches!(err, McpError::Timeout(_)) {
        return ToolRuntimeError::Timeout(format!("MCP: {err}"));
    }
    // MCP errors are mostly transport-level / protocol-level — bucket
    // them all as `Runtime` (the catch-all). Caller (agent_loop) will
    // surface this as `NativeHarnessError::ToolRuntime` → eventually
    // `sandbox_failed_error` on the wire, which conveys "MCP side
    // broke" close enough. We don't add Timeout etc explicitly because
    // reqwest's timeout already surfaces as Transport.
    ToolRuntimeError::Runtime(format!("MCP: {err}"))
}

// ── CompositeToolRuntime ────────────────────────────────────────────

/// Layer two ToolRuntimes into one. Used to combine native tools
/// (`SandboxToolRuntime`) and MCP tools (`McpToolRuntime`) into a
/// single thing the agent loop's generic `R: ToolRuntime` can consume.
///
/// Dispatch policy: try `primary` first; if it reports `UnknownTool`,
/// fall back to `secondary`. Native tools live in primary by convention
/// (cheaper to call, no network hop).
#[derive(Clone)]
pub struct CompositeToolRuntime {
    primary: Arc<dyn ToolRuntime>,
    secondary: Arc<dyn ToolRuntime>,
}

impl CompositeToolRuntime {
    pub fn new(primary: Arc<dyn ToolRuntime>, secondary: Arc<dyn ToolRuntime>) -> Self {
        Self { primary, secondary }
    }
}

#[async_trait]
impl ToolRuntime for CompositeToolRuntime {
    fn specs(&self) -> Vec<ToolSpec> {
        let mut combined = self.primary.specs();
        combined.extend(self.secondary.specs());
        combined
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        match self.primary.invoke(invocation.clone()).await {
            Err(ToolRuntimeError::UnknownTool(_)) => self.secondary.invoke(invocation).await,
            other => other,
        }
    }

    async fn invoke_cancellable(
        &self,
        invocation: ToolInvocation,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        match self
            .primary
            .invoke_cancellable(invocation.clone(), cancel)
            .await
        {
            Err(ToolRuntimeError::UnknownTool(_)) => {
                self.secondary.invoke_cancellable(invocation, cancel).await
            }
            other => other,
        }
    }
}

// Blanket impl so `Arc<dyn ToolRuntime>` itself satisfies ToolRuntime —
// lets the agent loop's `R: ToolRuntime + Clone` accept an Arc of a
// trait object directly. Forwards to the inner concrete impl.
#[async_trait]
impl ToolRuntime for Arc<dyn ToolRuntime> {
    fn specs(&self) -> Vec<ToolSpec> {
        (**self).specs()
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        (**self).invoke(invocation).await
    }

    async fn invoke_cancellable(
        &self,
        invocation: ToolInvocation,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        (**self).invoke_cancellable(invocation, cancel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spin up a one-shot mock HTTP server that responds to MCP requests
    /// with scripted JSON bodies. Returns the URL the client should
    /// POST to. Server task ends after `expected_requests` requests.
    /// Cheaper than wiremock; covers our minimal needs.
    async fn spawn_mock_mcp_server(
        scripted_responses: Vec<String>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/mcp");

        let handle = tokio::spawn(async move {
            let mut remaining = scripted_responses.into_iter();
            while let Some(response_body) = remaining.next() {
                let (mut stream, _) = listener.accept().await.unwrap();
                // Drain request: read until \r\n\r\n then Content-Length
                // bytes. Hand-rolled because we don't need full HTTP parsing.
                let mut buf = Vec::with_capacity(2048);
                let mut header_end = 0;
                loop {
                    let mut tmp = [0u8; 1024];
                    let n = stream.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = find_header_end(&buf) {
                        header_end = pos + 4;
                        break;
                    }
                }
                let headers = std::str::from_utf8(&buf[..header_end.saturating_sub(4)])
                    .unwrap()
                    .to_lowercase();
                let mut content_length = 0usize;
                for line in headers.lines() {
                    if let Some(v) = line.strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut already_read = buf.len() - header_end;
                while already_read < content_length {
                    let mut tmp = [0u8; 1024];
                    let n = stream.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    already_read += n;
                }

                // Reply with the scripted JSON body.
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.flush().await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (url, handle)
    }

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n")
    }

    /// SSE variant: for each request, the mock replies with a
    /// `text/event-stream` response whose body is each scripted entry
    /// (a list of SSE events) concatenated. Each "entry" is itself a
    /// `Vec<String>` of SSE event blocks (each ending with `\n\n`) so
    /// a single response can carry multiple events — exercising the
    /// "discard server-initiated notifications between expected
    /// envelopes" code path. Connection close terminates the stream.
    async fn spawn_mock_mcp_sse_server(
        scripted_responses: Vec<Vec<String>>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/mcp");

        let handle = tokio::spawn(async move {
            let mut remaining = scripted_responses.into_iter();
            while let Some(events) = remaining.next() {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = Vec::with_capacity(2048);
                let mut header_end = 0;
                loop {
                    let mut tmp = [0u8; 1024];
                    let n = stream.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = find_header_end(&buf) {
                        header_end = pos + 4;
                        break;
                    }
                }
                let headers = std::str::from_utf8(&buf[..header_end.saturating_sub(4)])
                    .unwrap()
                    .to_lowercase();
                let mut content_length = 0usize;
                for line in headers.lines() {
                    if let Some(v) = line.strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut already_read = buf.len() - header_end;
                while already_read < content_length {
                    let mut tmp = [0u8; 1024];
                    let n = stream.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    already_read += n;
                }

                // We send the response WITHOUT Content-Length and with
                // Connection: close — closing the TCP stream signals
                // end-of-stream to the client's SSE parser. Each event
                // is already terminated by its own `\n\n`.
                let body: String = events.concat();
                let header_block = "HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream\r\n\
                    Cache-Control: no-cache\r\n\
                    Connection: close\r\n\r\n";
                stream.write_all(header_block.as_bytes()).await.unwrap();
                stream.write_all(body.as_bytes()).await.unwrap();
                stream.flush().await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (url, handle)
    }

    /// Build one SSE `message`-event block from a JSON-RPC body.
    fn sse_event(body: &str) -> String {
        format!("event: message\ndata: {body}\n\n")
    }

    fn jsonrpc_result(id: u64, result: Value) -> String {
        json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string()
    }

    #[tokio::test]
    async fn mcp_client_initializes_lists_and_calls_a_tool() {
        // Scripted responses for: initialize, notifications/initialized
        // (notification gets no response but the test server still
        // returns OK), tools/list, tools/call.
        let (url, _server) = spawn_mock_mcp_server(vec![
            jsonrpc_result(1, json!({"protocolVersion": "2024-11-05", "capabilities": {}})),
            // Notification — server returns empty 200 with empty body
            // is technically OK; we send a benign empty JSON-RPC anyway
            // so the mock parses cleanly.
            json!({}).to_string(),
            jsonrpc_result(
                3,
                json!({
                    "tools": [{
                        "name": "echo",
                        "description": "echo back input",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}
                    }]
                }),
            ),
            jsonrpc_result(
                4,
                json!({
                    "content": [{"type": "text", "text": "hello back"}],
                    "isError": false
                }),
            ),
        ])
        .await;
        let client = McpClient::new(McpServerConfig::new("fs", url)).unwrap();
        client.initialize().await.expect("init");
        let specs = client.tools_list().await.expect("list");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
        let outcome = client
            .tools_call("echo", json!({"text": "hi"}))
            .await
            .expect("call");
        let v = outcome.output.expect("ok output");
        assert_eq!(v["content"], "hello back");
    }

    #[tokio::test]
    async fn mcp_client_extracts_image_content_into_attachments() {
        // MCP servers that return an `image` content block (e.g. a
        // screenshot tool) should surface the bytes as
        // ToolOutcome.attachments — NOT degrade them to placeholder
        // text. Anthropic projection then puts the image into the
        // tool_result block array; OpenAI projection appends a
        // placeholder note in the tool-role string.
        let (url, _server) = spawn_mock_mcp_server(vec![
            jsonrpc_result(
                1,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ),
            json!({}).to_string(),
            jsonrpc_result(
                3,
                json!({
                    "content": [
                        {"type": "text", "text": "captured"},
                        {"type": "image", "mimeType": "image/png", "data": "PNGBYTES"}
                    ],
                    "isError": false
                }),
            ),
        ])
        .await;
        let client = McpClient::new(McpServerConfig::new("screen", url)).unwrap();
        client.initialize().await.unwrap();
        let outcome = client
            .tools_call("screenshot", json!({}))
            .await
            .expect("call");
        let v = outcome.output.expect("ok output");
        // Text channel: only the actual text block — image isn't
        // smuggled in as a stringified payload.
        assert_eq!(v["content"], "captured");
        // Attachment channel: image bytes preserved verbatim.
        assert_eq!(outcome.attachments.len(), 1);
        let UserAttachment::Image(src) = &outcome.attachments[0];
        assert_eq!(src.media_type, "image/png");
        match &src.data {
            ImageData::Base64(b) => assert_eq!(b, "PNGBYTES"),
            ImageData::Url(_) => panic!("expected base64, got url"),
        }
    }

    #[tokio::test]
    async fn mcp_client_surfaces_tool_error_as_tool_failure() {
        let (url, _server) = spawn_mock_mcp_server(vec![
            jsonrpc_result(
                1,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ),
            json!({}).to_string(),
            jsonrpc_result(
                3,
                json!({
                    "content": [{"type": "text", "text": "file not found"}],
                    "isError": true
                }),
            ),
        ])
        .await;
        let client = McpClient::new(McpServerConfig::new("fs", url)).unwrap();
        client.initialize().await.unwrap();
        let outcome = client
            .tools_call("read", json!({"path": "/none"}))
            .await
            .unwrap();
        let failure = outcome.output.expect_err("expected ToolFailure");
        assert_eq!(failure.kind, ToolFailureKind::Runtime);
        assert!(failure.message.contains("file not found"));
    }

    #[tokio::test]
    async fn mcp_client_surfaces_jsonrpc_error_as_mcp_server_error() {
        let (url, _server) = spawn_mock_mcp_server(vec![
            jsonrpc_result(
                1,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ),
            json!({}).to_string(),
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "error": {"code": -32601, "message": "method not found"}
            })
            .to_string(),
        ])
        .await;
        let client = McpClient::new(McpServerConfig::new("fs", url)).unwrap();
        client.initialize().await.unwrap();
        let err = client.tools_list().await.unwrap_err();
        match err {
            McpError::Server { code, message } => {
                assert_eq!(code, -32601);
                assert!(message.contains("method not found"));
            }
            other => panic!("expected Server error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mcp_tool_runtime_prefixes_tool_names_and_routes_calls() {
        let (url, _server) = spawn_mock_mcp_server(vec![
            jsonrpc_result(
                1,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ),
            json!({}).to_string(),
            jsonrpc_result(
                3,
                json!({
                    "tools": [{
                        "name": "echo",
                        "description": "echo",
                        "inputSchema": {"type": "object"}
                    }]
                }),
            ),
            jsonrpc_result(
                4,
                json!({"content": [{"type": "text", "text": "routed"}], "isError": false}),
            ),
        ])
        .await;
        let rt = McpToolRuntime::discover(vec![McpServerConfig::new("fs", url)]).await;
        assert_eq!(rt.server_count(), 1);
        let specs = rt.specs();
        // Tool name carries the `fs__` prefix.
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "fs__echo");

        // invoke with prefixed name — runtime strips prefix on the wire.
        let outcome = rt
            .invoke(ToolInvocation {
                id: "tc1".into(),
                name: "fs__echo".into(),
                input: json!({"text": "x"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(outcome.output.unwrap()["content"], "routed");
    }

    #[tokio::test]
    async fn mcp_tool_runtime_unknown_tool_returns_runtime_error() {
        // No server configured — runtime is empty. invoke on any tool
        // surfaces UnknownTool, which is the contract used by
        // CompositeToolRuntime's fallback logic.
        let rt = McpToolRuntime::discover(vec![]).await;
        let err = rt
            .invoke(ToolInvocation {
                id: "tc".into(),
                name: "nope__whatever".into(),
                input: json!({}),
                raw_emitted_args: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRuntimeError::UnknownTool(ref s) if s == "nope__whatever"));
    }

    #[derive(Clone, Default)]
    struct FakeNativeRuntime {
        names: Vec<&'static str>,
    }

    #[async_trait]
    impl ToolRuntime for FakeNativeRuntime {
        fn specs(&self) -> Vec<ToolSpec> {
            self.names
                .iter()
                .map(|n| ToolSpec {
                    name: n.to_string(),
                    description: "fake".into(),
                    input_schema: json!({"type": "object"}),
                })
                .collect()
        }
        async fn invoke(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
            if self.names.contains(&inv.name.as_str()) {
                Ok(ToolOutcome {
                    output: Ok(json!({"served_by": "native", "name": inv.name})),
                    attachments: vec![],
                })
            } else {
                Err(ToolRuntimeError::UnknownTool(inv.name))
            }
        }
    }

    #[derive(Clone, Default)]
    struct FakeMcpRuntime {
        names: Vec<&'static str>,
    }

    #[async_trait]
    impl ToolRuntime for FakeMcpRuntime {
        fn specs(&self) -> Vec<ToolSpec> {
            self.names
                .iter()
                .map(|n| ToolSpec {
                    name: n.to_string(),
                    description: "mcp".into(),
                    input_schema: json!({"type": "object"}),
                })
                .collect()
        }
        async fn invoke(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
            if self.names.contains(&inv.name.as_str()) {
                Ok(ToolOutcome {
                    output: Ok(json!({"served_by": "mcp", "name": inv.name})),
                    attachments: vec![],
                })
            } else {
                Err(ToolRuntimeError::UnknownTool(inv.name))
            }
        }
    }

    #[tokio::test]
    async fn composite_runtime_merges_specs_and_falls_back_to_secondary() {
        let native = Arc::new(FakeNativeRuntime {
            names: vec!["bash", "read"],
        }) as Arc<dyn ToolRuntime>;
        let mcp = Arc::new(FakeMcpRuntime {
            names: vec!["fs__list", "git__diff"],
        }) as Arc<dyn ToolRuntime>;
        let composite = CompositeToolRuntime::new(native, mcp);

        // specs union (in primary-then-secondary order)
        let names: Vec<String> = composite.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["bash", "read", "fs__list", "git__diff"]);

        // primary serves "bash"
        let outcome = composite
            .invoke(ToolInvocation {
                id: "tc".into(),
                name: "bash".into(),
                input: json!({}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(outcome.output.unwrap()["served_by"], "native");

        // primary returns UnknownTool → fall back to secondary
        let outcome = composite
            .invoke(ToolInvocation {
                id: "tc".into(),
                name: "fs__list".into(),
                input: json!({}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(outcome.output.unwrap()["served_by"], "mcp");

        // Neither knows → final UnknownTool
        let err = composite
            .invoke(ToolInvocation {
                id: "tc".into(),
                name: "ghost".into(),
                input: json!({}),
                raw_emitted_args: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRuntimeError::UnknownTool(_)));
    }

    /// Build a tiny stdio MCP server in pure shell: stays in a loop
    /// reading newline-delimited JSON-RPC from stdin and writing
    /// pre-canned responses based on the request's `method`. Used to
    /// drive the stdio transport without a real Python/Node MCP
    /// implementation on the test host.
    ///
    /// The script handles:
    /// - `initialize` → returns `{protocolVersion, capabilities:{}}`
    /// - `tools/list` → returns one fake tool `echo`
    /// - `tools/call` → returns text content `"stdio-routed"`
    /// - `notifications/initialized` (no id) → consumed silently
    ///
    /// Returns the absolute path to the temp script file.
    fn write_mock_stdio_server(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("mock-mcp-stdio.sh");
        // The id needs to be parsed back from the inbound request so we
        // echo it in the response — use sed to extract.
        let body = r#"#!/usr/bin/env bash
set -u
while IFS= read -r line; do
  # Pull out method + id (best-effort sed; the test driver only sends
  # well-formed JSON so we don't need a real parser).
  method=$(echo "$line" | sed -n 's/.*"method"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  id=$(echo "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')
  case "$method" in
    initialize)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{}}}\n' "$id"
      ;;
    tools/list)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"d","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    tools/call)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"stdio-routed"}],"isError":false}}\n' "$id"
      ;;
    notifications/initialized)
      # No response for notifications.
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"method not found"}}\n' "$id"
      ;;
  esac
done
"#;
        std::fs::write(&path, body).unwrap();
        // chmod +x via std::fs metadata.
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[tokio::test]
    async fn mcp_stdio_initialize_lists_and_calls_a_tool() {
        // End-to-end stdio path: spawn a shell-based MCP server, run
        // initialize → tools/list → tools/call, assert routing works
        // without ever touching HTTP. This is the cheapest way to prove
        // the writer/reader/id-routing wiring is correct.
        // Use the OS temp dir directly — pulls no new dev-dep, fine for
        // a sandboxed CI worker. We pick a per-test filename so parallel
        // test runs don't clobber each other.
        let tmp = std::env::temp_dir().join(format!(
            "rd-mock-mcp-stdio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let script = write_mock_stdio_server(&tmp);

        let config =
            McpServerConfig::stdio("local-fs", script.to_string_lossy().into_owned(), vec![])
                .with_timeout(Duration::from_secs(5));
        let client = McpClient::new(config).expect("spawn");
        client.initialize().await.expect("init over stdio");
        let specs = client.tools_list().await.expect("list over stdio");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");

        let outcome = client
            .tools_call("echo", json!({"text": "hi"}))
            .await
            .expect("call over stdio");
        let v = outcome.output.expect("ok output");
        assert_eq!(v["content"], "stdio-routed");
    }

    #[tokio::test]
    async fn mcp_stdio_returns_transport_error_when_command_missing() {
        // Spawning a nonexistent command must surface a Transport
        // error eagerly from `McpClient::new`, not hang later on
        // initialize. This is the failure mode operators hit when a
        // bootstrap.yaml references an MCP server whose CLI isn't
        // installed on the RD host.
        let config = McpServerConfig::stdio(
            "nope",
            "/definitely/not/a/real/binary-xyz".to_string(),
            vec![],
        );
        let err = match McpClient::new(config) {
            Ok(_) => panic!("must fail to spawn"),
            Err(e) => e,
        };
        match err {
            McpError::Transport(msg) => {
                assert!(msg.contains("stdio spawn"), "got: {msg}");
            }
            other => panic!("expected Transport, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mcp_stdio_call_times_out_when_server_doesnt_reply() {
        // `sleep` never reads stdin and never writes stdout, so the
        // reader task can't route anything → call must hit the
        // per-request timeout cleanly (not hang).
        let config = McpServerConfig::stdio("silent", "sleep".to_string(), vec!["30".to_string()])
            .with_timeout(Duration::from_millis(250));
        let client = McpClient::new(config).expect("spawn cat");
        let err = client.initialize().await.expect_err("must time out");
        match err {
            McpError::Timeout(msg) => {
                assert!(msg.contains("timed out"), "got: {msg}");
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mcp_http_sse_response_yields_jsonrpc_result() {
        // Streamable-HTTP transport: server replies with
        // `text/event-stream`. The client must detect Content-Type and
        // drain the SSE stream looking for the id-matching envelope.
        // initialize → notifications/initialized → tools/list →
        // tools/call. Each request gets one SSE message in response
        // (notifications get an empty stream the server will just
        // close after).
        // id sequence: only `call()` calls increment `next_id`;
        // `notify()` (notifications/initialized) does not. So:
        //   initialize → 1, tools/list → 2, tools/call → 3.
        let (url, _server) = spawn_mock_mcp_sse_server(vec![
            // initialize
            vec![sse_event(&jsonrpc_result(
                1,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ))],
            // notifications/initialized — no envelope, just close
            vec![],
            // tools/list
            vec![sse_event(&jsonrpc_result(
                2,
                json!({
                    "tools": [{
                        "name": "echo",
                        "description": "d",
                        "inputSchema": {"type": "object"}
                    }]
                }),
            ))],
            // tools/call
            vec![sse_event(&jsonrpc_result(
                3,
                json!({
                    "content": [{"type": "text", "text": "sse-routed"}],
                    "isError": false
                }),
            ))],
        ])
        .await;
        let client = McpClient::new(McpServerConfig::http("sse-fs", url)).expect("build client");
        client.initialize().await.expect("init over sse");
        let specs = client.tools_list().await.expect("list over sse");
        assert_eq!(specs.len(), 1);
        let outcome = client
            .tools_call("echo", json!({}))
            .await
            .expect("call over sse");
        assert_eq!(outcome.output.unwrap()["content"], "sse-routed");
    }

    #[tokio::test]
    async fn mcp_http_sse_response_skips_server_notifications() {
        // SSE stream may interleave server-initiated notifications
        // (no `id`) before the actual response. The parser must drop
        // them and keep draining until it finds the id match.
        let server_notification =
            r#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"percent":42}}"#;
        let unrelated = r#"{"jsonrpc":"2.0","id":999,"result":{"unrelated":true}}"#;
        let (url, _server) = spawn_mock_mcp_sse_server(vec![
            // initialize — single notification + unrelated id + real
            // response. Ordering matters: parser must skip the first
            // two and consume the third.
            vec![
                sse_event(server_notification),
                sse_event(unrelated),
                sse_event(&jsonrpc_result(
                    1,
                    json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
                )),
            ],
            // notifications/initialized
            vec![],
        ])
        .await;
        let client = McpClient::new(McpServerConfig::http("sse-noisy", url)).expect("build client");
        client
            .initialize()
            .await
            .expect("must skip notifications and find id match");
    }

    #[tokio::test]
    async fn mcp_http_sse_response_propagates_jsonrpc_error() {
        // An SSE envelope with `error` field maps to McpError::Server,
        // not Transport — same as the JSON path.
        let (url, _server) = spawn_mock_mcp_sse_server(vec![vec![sse_event(
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "method not found"}
            })
            .to_string(),
        )]])
        .await;
        let client = McpClient::new(McpServerConfig::http("sse-err", url)).expect("build");
        let err = client.initialize().await.unwrap_err();
        match err {
            McpError::Server { code, message } => {
                assert_eq!(code, -32601);
                assert!(message.contains("method not found"));
            }
            other => panic!("expected Server, got {other:?}"),
        }
    }
}
