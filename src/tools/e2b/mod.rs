//! E2b sandbox tool runtime.
//!
//! Implements `ToolRuntime` backed by an e2b sandbox. Process execution goes
//! through envd's Connect Protocol (same as the official e2b SDKs); file
//! read/write uses the e2b REST API.
//!
//! # Usage
//!
//! ```ignore
//! let tools = E2bToolRuntime::connect(E2bConfig {
//!     sandbox_id: "sandbox-abc123".into(),
//!     api_key:    "e2b_sk_...".into(),
//!     api_url:    E2bConfig::DEFAULT_API_URL.into(),
//!     cwd:        "/home/user".into(),
//! }).await?;
//! let harness = AgentLoopHarness::new(model, tools);
//! ```

pub mod connect;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use prost::Message;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::tools::{
    builtin_tool_specs, ToolFailure, ToolFailureKind, ToolInvocation, ToolOutcome, ToolRuntime,
    ToolRuntimeError, ToolSpec,
};
use crate::tools::approval::{is_read_only, ApprovalGate, YoloApproval};
use crate::tools::sandbox::{ExecResult, SandboxExecutor};
use connect::{encode_data_envelope, parse_unary_error, ConnectError, Envelope, EnvelopeDecoder};

// ── prost-generated envd process proto ───────────────────────────────────────
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/process.rs"));
}

// ── Public config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct E2bConfig {
    /// E2b sandbox ID (e.g. `"abc123def456"`).
    pub sandbox_id: String,
    /// E2b API key used for REST calls and to resolve the envd access token.
    pub api_key: String,
    /// E2b control-plane URL. Defaults to `https://api.e2b.dev`.
    pub api_url: String,
    /// Working directory inside the sandbox. Defaults to `/home/user`.
    pub cwd: String,
}

impl E2bConfig {
    pub const DEFAULT_API_URL: &'static str = "https://api.e2b.dev";
    pub const DEFAULT_CWD:     &'static str = "/home/user";

    pub fn new(sandbox_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            sandbox_id: sandbox_id.into(),
            api_key:    api_key.into(),
            api_url:    Self::DEFAULT_API_URL.into(),
            cwd:        Self::DEFAULT_CWD.into(),
        }
    }
}

// ── E2bError ──────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum E2bError {
    #[error("e2b REST API error {status}: {body}")]
    Api { status: u16, body: String },

    #[error("envd transport error: {0}")]
    Transport(String),

    #[error("envd protocol error: {0}")]
    Protocol(String),

    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
}

impl From<ConnectError> for E2bError {
    fn from(e: ConnectError) -> Self {
        E2bError::Protocol(format!("{}: {}", e.code, e.message))
    }
}

// ── Sandbox detail (REST API response) ───────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SandboxDetail {
    #[serde(rename = "sandboxID")]
    sandbox_id: String,
    envd_access_token: Option<String>,
}

// ── E2bExecutor — implements SandboxExecutor ──────────────────────────────────

#[derive(Clone)]
struct E2bExecutor {
    /// `https://49983-{sandbox_id}-00000000.{domain}`
    envd_url:   String,
    /// X-Access-Token header for envd
    envd_token: String,
    /// `https://api.e2b.dev` (or custom)
    api_url:    String,
    /// X-API-Key header for REST API
    api_key:    String,
    /// Stored for reconnect / timeout-extension calls (future use).
    #[allow(dead_code)]
    sandbox_id: String,
    http:       reqwest::Client,
}

impl E2bExecutor {
    async fn resolve(config: &E2bConfig) -> Result<Self, E2bError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| E2bError::Transport(e.to_string()))?;

        // GET /sandboxes/{id} — resolve envdAccessToken + derive envd domain
        let url = format!("{}/sandboxes/{}", config.api_url, config.sandbox_id);
        let resp = http
            .get(&url)
            .header("X-API-Key", &config.api_key)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(E2bError::Api { status, body });
        }
        let detail: SandboxDetail = resp.json().await?;
        let envd_token = detail.envd_access_token.ok_or_else(|| E2bError::Api {
            status: 500,
            body: format!(
                "sandbox {} has no envdAccessToken; must be created with secure=true",
                config.sandbox_id
            ),
        })?;

        // envd domain: strip "api." prefix from api_url host
        let parsed = reqwest::Url::parse(&config.api_url)
            .map_err(|e| E2bError::Api { status: 0, body: e.to_string() })?;
        let host = parsed.host_str().unwrap_or("e2b.dev");
        let domain = host.strip_prefix("api.").unwrap_or(host);
        let envd_url = format!(
            "https://49983-{}-00000000.{}",
            config.sandbox_id, domain
        );

        tracing::debug!(
            sandbox_id = %config.sandbox_id,
            envd_url = %envd_url,
            "e2b executor resolved"
        );

        Ok(Self {
            envd_url,
            envd_token,
            api_url: config.api_url.clone(),
            api_key: config.api_key.clone(),
            sandbox_id: config.sandbox_id.clone(),
            http,
        })
    }

    fn envd_url(&self, method: &str) -> String {
        format!("{}/process.Process/{method}", self.envd_url)
    }

    fn envd_headers(&self, streaming: bool) -> Result<HeaderMap, E2bError> {
        let mut h = HeaderMap::new();
        let ct = if streaming {
            "application/connect+proto"
        } else {
            "application/proto"
        };
        h.insert("Content-Type", HeaderValue::from_static(ct));
        h.insert(
            "X-Access-Token",
            HeaderValue::from_str(&self.envd_token)
                .map_err(|e| E2bError::Transport(format!("token header: {e}")))?,
        );
        h.insert("Connect-Protocol-Version", HeaderValue::from_static("1"));
        Ok(h)
    }

    /// Unary Connect call — used for SendSignal (future kill support).
    #[allow(dead_code)]
    async fn unary<Req: Message, Resp: Message + Default>(
        &self,
        method: &str,
        req: Req,
    ) -> Result<Resp, E2bError> {
        let headers = self.envd_headers(false)?;
        let resp = self
            .http
            .post(self.envd_url(method))
            .headers(headers)
            .body(req.encode_to_vec())
            .send()
            .await?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if status.is_success() {
            Resp::decode(body).map_err(|e| E2bError::Protocol(format!("decode {method}: {e}")))
        } else if let Some(err) = parse_unary_error(&body) {
            Err(err.into())
        } else {
            Err(E2bError::Protocol(format!("{method} HTTP {status}")))
        }
    }

    /// Open a server-streaming Connect call; returns a channel of decoded messages.
    fn server_stream<Req, Resp>(
        &self,
        method: &str,
        req: Req,
    ) -> mpsc::Receiver<Result<Resp, E2bError>>
    where
        Req: Message,
        Resp: Message + Default + Send + 'static,
    {
        let url = self.envd_url(method);
        let headers = match self.envd_headers(true) {
            Ok(h) => h,
            Err(e) => {
                let (tx, rx) = mpsc::channel(1);
                tokio::spawn(async move { let _ = tx.send(Err(e)).await; });
                return rx;
            }
        };
        let body = encode_data_envelope(&req.encode_to_vec());
        let http = self.http.clone();
        let method_owned = method.to_string();
        let (tx, rx) = mpsc::channel::<Result<Resp, E2bError>>(32);

        tokio::spawn(async move {
            let resp = match http.post(&url).headers(headers).body(body).send().await {
                Ok(r) => r,
                Err(e) => { let _ = tx.send(Err(e.into())).await; return; }
            };
            let status = resp.status();
            if !status.is_success() {
                let body = resp.bytes().await.unwrap_or_default();
                let err = parse_unary_error(&body).map(E2bError::from).unwrap_or_else(|| {
                    E2bError::Protocol(format!("{method_owned} HTTP {status}"))
                });
                let _ = tx.send(Err(err)).await;
                return;
            }
            let mut dec = EnvelopeDecoder::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk { Ok(b) => b, Err(e) => { let _ = tx.send(Err(e.into())).await; return; } };
                dec.push(&chunk);
                loop {
                    match dec.try_next() {
                        Ok(None) => break,
                        Ok(Some(Envelope::Data(p))) => {
                            match Resp::decode(&p[..]) {
                                Ok(msg) => { if tx.send(Ok(msg)).await.is_err() { return; } }
                                Err(e) => {
                                    let _ = tx.send(Err(E2bError::Protocol(format!("decode {method_owned}: {e}")))).await;
                                    return;
                                }
                            }
                        }
                        Ok(Some(Envelope::EndOfStream(eos))) => {
                            if let Some(err) = eos.error { let _ = tx.send(Err(err.into())).await; }
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(E2bError::Protocol(format!("envelope: {e}")))).await;
                            return;
                        }
                    }
                }
            }
        });
        rx
    }

    /// Client-streaming Connect call — used for StreamInput (future stdin support).
    #[allow(dead_code)]
    fn client_stream<Req, Resp>(
        &self,
        method: &str,
    ) -> Result<(mpsc::Sender<Req>, tokio::task::JoinHandle<Result<Resp, E2bError>>), E2bError>
    where
        Req: Message + Send + 'static,
        Resp: Message + Default + Send + 'static,
    {
        let url = self.envd_url(method);
        let headers = self.envd_headers(true)?;
        let http = self.http.clone();
        let (req_tx, req_rx) = mpsc::channel::<Req>(16);
        let method_owned = method.to_string();

        let req_stream = ReceiverStream::new(req_rx).map(|msg| {
            let env = encode_data_envelope(&msg.encode_to_vec());
            Ok::<Bytes, std::io::Error>(Bytes::from(env))
        });
        let handle = tokio::spawn(async move {
            let resp = http.post(&url).headers(headers).body(reqwest::Body::wrap_stream(req_stream)).send().await?;
            let status = resp.status();
            let bytes = resp.bytes().await?;
            if !status.is_success() {
                if let Some(err) = parse_unary_error(&bytes) { return Err(err.into()); }
                return Err(E2bError::Protocol(format!("{method_owned} HTTP {status}")));
            }
            let mut dec = EnvelopeDecoder::new();
            dec.push(&bytes);
            loop {
                match dec.try_next() {
                    Ok(None) | Ok(Some(Envelope::EndOfStream(_))) => break,
                    Ok(Some(Envelope::Data(p))) => {
                        return Resp::decode(&p[..]).map_err(|e| E2bError::Protocol(format!("decode resp: {e}")));
                    }
                    Err(e) => return Err(E2bError::Protocol(format!("envelope: {e}"))),
                }
            }
            Err(E2bError::Protocol("no response message".into()))
        });
        Ok((req_tx, handle))
    }

    /// Core: start a process in envd, collect stdout/stderr/exit_code.
    /// When `on_line` is provided, lines are forwarded in real-time.
    async fn run_process(
        &self,
        cmd: &str,
        cwd: Option<&str>,
        timeout_ms: u64,
        on_line: Option<&(dyn Fn(String) + Send + Sync)>,
    ) -> Result<ExecResult, E2bError> {
        // Wrap in /bin/bash -l -c to source /etc/profile.d/*.sh
        let cfg = proto::ProcessConfig {
            cmd: "/bin/bash".into(),
            args: vec!["-l".into(), "-c".into(), cmd.to_string()],
            envs: HashMap::new(),
            cwd: cwd.map(str::to_string),
        };
        let start_req = proto::StartRequest {
            process: Some(cfg),
            pty: None,
            tag: None,
            stdin: Some(false),
        };
        let mut events_rx = self.server_stream::<_, proto::StartResponse>("Start", start_req);

        // Wait for the Start event to get the pid
        let _pid = loop {
            let next = events_rx.recv().await
                .ok_or_else(|| E2bError::Protocol("Start stream closed before pid".into()))??;
            let event = next.event.and_then(|e| e.event)
                .ok_or_else(|| E2bError::Protocol("missing ProcessEvent.event".into()))?;
            match event {
                proto::process_event::Event::Start(s) => break s.pid,
                proto::process_event::Event::End(_) => {
                    return Err(E2bError::Protocol("process exited before sending Start event".into()));
                }
                _ => continue,
            }
        };

        // Drain events under a timeout
        let hard = tokio::time::sleep(Duration::from_millis(timeout_ms));
        tokio::pin!(hard);

        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut exit_code = -1i32;

        loop {
            let item = tokio::select! {
                _ = &mut hard => {
                    return Err(E2bError::Transport(format!(
                        "command timed out after {timeout_ms}ms"
                    )));
                }
                item = events_rx.recv() => item,
            };
            let Some(item) = item else { break };
            let resp = item?;
            let Some(event) = resp.event.and_then(|e| e.event) else { continue };
            match event {
                proto::process_event::Event::Data(d) => {
                    let Some(out) = d.output else { continue };
                    let (line_bytes, is_stdout) = match out {
                        proto::process_event::data_event::Output::Stdout(b) => (b, true),
                        proto::process_event::data_event::Output::Stderr(b) => (b, false),
                        proto::process_event::data_event::Output::Pty(b)    => (b, true),
                    };
                    let text = String::from_utf8_lossy(&line_bytes).into_owned();
                    if let Some(cb) = on_line {
                        for line in text.lines() {
                            cb(line.to_string());
                        }
                    }
                    if is_stdout { stdout.push_str(&text); } else { stderr.push_str(&text); }
                }
                proto::process_event::Event::End(e) => {
                    exit_code = e.exit_code;
                    break;
                }
                _ => continue,
            }
        }

        Ok(ExecResult { stdout, stderr, exit_code })
    }
}

#[async_trait]
impl SandboxExecutor for E2bExecutor {
    async fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        timeout_ms: u64,
    ) -> Result<ExecResult, String> {
        self.run_process(command, cwd, timeout_ms, None)
            .await
            .map_err(|e| e.to_string())
    }

    async fn exec_streaming(
        &self,
        command: &str,
        cwd: Option<&str>,
        timeout_ms: u64,
        on_line: &(dyn Fn(String) + Send + Sync),
    ) -> Result<ExecResult, String> {
        self.run_process(command, cwd, timeout_ms, Some(on_line))
            .await
            .map_err(|e| e.to_string())
    }

    async fn read_file(&self, path: &str) -> Result<String, String> {
        // GET {api_url}/sandboxes/{id}/files?path={path}
        let url = format!("{}/sandboxes/{}/files", self.api_url, self.sandbox_id);
        let resp = self.http
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .query(&[("path", path)])
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("e2b files GET {status}: {body}"));
        }
        resp.text().await.map_err(|e| e.to_string())
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<(), String> {
        // POST {api_url}/sandboxes/{id}/files?path={path}
        // body = raw file bytes, Content-Type: application/octet-stream
        let url = format!("{}/sandboxes/{}/files", self.api_url, self.sandbox_id);
        let resp = self.http
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/octet-stream")
            .query(&[("path", path)])
            .body(content.to_string())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("e2b files POST {status}: {body}"));
        }
        Ok(())
    }
}

// ── Public E2bToolRuntime ─────────────────────────────────────────────────────

/// Tool runtime backed by an e2b sandbox.
///
/// Process execution uses Connect Protocol to envd (same as the official e2b
/// SDKs). File read/write uses the e2b REST API.
///
/// The `envd_access_token` is resolved automatically from the e2b control
/// plane on construction — callers only need `sandbox_id` + `api_key`.
#[derive(Clone)]
pub struct E2bToolRuntime {
    executor: E2bExecutor,
    cwd:      String,
    approval: Arc<dyn ApprovalGate>,
    emit:     Arc<dyn Fn(Value) + Send + Sync + 'static>,
}

impl E2bToolRuntime {
    /// Connect to an existing e2b sandbox and return a ready `E2bToolRuntime`.
    ///
    /// Makes one REST call to the e2b control plane to resolve the envd access
    /// token. Returns an error if the sandbox is not found or was not created
    /// with `secure=true`.
    pub async fn connect(config: E2bConfig) -> Result<Self, E2bError> {
        Self::connect_with(config, Arc::new(YoloApproval), Arc::new(|_| {})).await
    }

    pub async fn connect_with(
        config: E2bConfig,
        approval: Arc<dyn ApprovalGate>,
        emit:     Arc<dyn Fn(Value) + Send + Sync + 'static>,
    ) -> Result<Self, E2bError> {
        let cwd = config.cwd.clone();
        let executor = E2bExecutor::resolve(&config).await?;
        Ok(Self { executor, cwd, approval, emit })
    }
}

#[async_trait]
impl ToolRuntime for E2bToolRuntime {
    fn specs(&self) -> Vec<ToolSpec> {
        let all = builtin_tool_specs();
        if self.approval.advertise_mutating_tools() {
            all
        } else {
            all.into_iter().filter(|s| is_read_only(&s.name)).collect()
        }
    }

    async fn invoke(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        self.invoke_cancellable(inv, None).await
    }

    async fn invoke_cancellable(
        &self,
        inv: ToolInvocation,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        // Approval gate
        if !is_read_only(&inv.name) {
            let approved = if let Some(tok) = cancel {
                tokio::select! {
                    biased;
                    _ = tok.cancelled() => false,
                    r = self.approval.approve(&inv) => r,
                }
            } else {
                self.approval.approve(&inv).await
            };
            if !approved {
                return Ok(ToolOutcome {
                    output: Err(ToolFailure::new(ToolFailureKind::Denied, "操作被拒绝")),
                    attachments: vec![],
                });
            }
        }

        match inv.name.as_str() {
            "bash"  => self.e2b_bash(inv).await,
            "read"  => self.e2b_read(inv).await,
            "write" => self.e2b_write(inv).await,
            "edit"  => self.e2b_edit(inv).await,
            "glob"  => self.e2b_glob(inv).await,
            "grep"  => self.e2b_grep(inv).await,
            other   => Err(ToolRuntimeError::UnknownTool(other.into())),
        }
    }
}

// ── per-tool implementations ──────────────────────────────────────────────────

impl E2bToolRuntime {
    /// Write `content` to `/tmp/harness_out_{id}_{suffix}.txt` inside the
    /// sandbox and return a preview with the path, or return unchanged if
    /// content is within the output budget.
    async fn bound_output(&self, content: String, id: &str, suffix: &str) -> String {
        if content.len() <= crate::tools::MAX_OUTPUT_BYTES {
            return content;
        }
        let path = format!("/tmp/harness_out_{id}_{suffix}.txt");
        let _ = self.executor.write_file(&path, &content).await;
        let preview: String = content.chars().take(crate::tools::MAX_OUTPUT_BYTES / 2).collect();
        format!(
            "{preview}\n\n[{} bytes total, truncated. \
             Full output saved to {path} — use the read tool to fetch more.]",
            content.len()
        )
    }

    async fn e2b_bash(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let cmd = req_str(&inv, "command")?;
        let timeout_ms = inv.input.get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(120_000)
            .min(3_600_000);

        let emit = self.emit.clone();
        let result = self.executor
            .exec_streaming(cmd, Some(&self.cwd), timeout_ms, &move |line| {
                emit(json!({ "type": "bash_stdout_line", "line": line }));
            })
            .await;

        match result {
            Err(e) => Ok(ToolOutcome {
                output: if e.to_lowercase().contains("timed out") || e.to_lowercase().contains("timeout") {
                    Ok(json!({
                        "command": cmd,
                        "stdout": "",
                        "stderr": "",
                        "exit_code": null,
                        "success": false,
                        "timed_out": true,
                        "message": e,
                    }))
                } else {
                    Err(ToolFailure::new(ToolFailureKind::Runtime, e))
                },
                attachments: vec![],
            }),
            Ok(r) => {
                let stdout = self.bound_output(r.stdout, &inv.id, "stdout").await;
                let stderr = self.bound_output(r.stderr, &inv.id, "stderr").await;
                Ok(ToolOutcome {
                    output: Ok(json!({
                        "command": cmd,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": r.exit_code,
                        "success": r.exit_code == 0,
                    })),
                    attachments: vec![],
                })
            }
        }
    }

    async fn e2b_read(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path = req_str(&inv, "path")?;
        let abs = resolve(&self.cwd, path);
        let content = match self.executor.read_file(&abs).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
        };

        let total = content.lines().count();
        let offset = inv.input.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit  = inv.input.get("limit").and_then(Value::as_u64).map(|v| v.clamp(1, 2_000) as usize);
        let selected: Vec<&str> = match limit {
            Some(n) => content.lines().skip(offset).take(n).collect(),
            None    => content.lines().skip(offset).collect(),
        };
        let end = offset + selected.len();
        let text = if selected.is_empty() {
            String::new()
        } else {
            let mut t = selected.join("\n");
            if content.ends_with('\n') && end == total { t.push('\n'); }
            t
        };
        Ok(ToolOutcome {
            output: Ok(json!({
                "path": abs,
                "content": truncate(text),
                "offset": offset,
                "limit": limit,
                "start_line": if selected.is_empty() { Value::Null } else { json!(offset+1) },
                "end_line":   if selected.is_empty() { Value::Null } else { json!(end) },
                "total_lines": total,
                "truncated": limit.map(|n| offset+n < total).unwrap_or(false),
            })),
            attachments: vec![],
        })
    }

    async fn e2b_write(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path    = req_str(&inv, "path")?;
        let content = req_str(&inv, "content")?;
        let abs = resolve(&self.cwd, path);
        // Ensure parent directories exist
        if let Some(parent) = std::path::Path::new(&abs).parent() {
            let mkdir = format!("mkdir -p {}", shell_escape(&parent.to_string_lossy()));
            let _ = self.executor.exec(&mkdir, Some(&self.cwd), 10_000).await;
        }
        match self.executor.write_file(&abs, content).await {
            Ok(()) => Ok(ToolOutcome {
                output: Ok(json!({ "path": abs, "written": true })),
                attachments: vec![],
            }),
            Err(e) => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
        }
    }

    async fn e2b_edit(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path      = req_str(&inv, "path")?;
        let old_str   = req_str(&inv, "old_string")?;
        let new_str   = inv.input.get("new_string").and_then(Value::as_str).unwrap_or("");
        let replace_all = inv.input.get("replace_all").and_then(Value::as_bool).unwrap_or(false);
        let abs = resolve(&self.cwd, path);

        let content = match self.executor.read_file(&abs).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
        };

        let occurrences = content.matches(old_str).count();
        if occurrences == 0 {
            return Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::InvalidInput,
                    "Could not find old_string in the file. It must match exactly, including whitespace and indentation. Read the file again before retrying.".to_string())),
                attachments: vec![],
            });
        }
        if !replace_all && occurrences > 1 {
            return Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::InvalidInput,
                    format!("Found {occurrences} exact matches for old_string. Provide more surrounding context or set replace_all=true."))),
                attachments: vec![],
            });
        }
        let replaced = if replace_all { occurrences } else { 1 };
        let new_content = if replace_all {
            content.replace(old_str, new_str)
        } else {
            content.replacen(old_str, new_str, 1)
        };
        match self.executor.write_file(&abs, &new_content).await {
            Ok(()) => Ok(ToolOutcome {
                output: Ok(json!({
                    "path": abs,
                    "replaced": replaced,
                    "old_lines": old_str.lines().count(),
                    "new_lines": new_str.lines().count(),
                })),
                attachments: vec![],
            }),
            Err(e) => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
        }
    }

    async fn e2b_glob(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let pattern = req_str(&inv, "pattern")?.to_string();
        let base = inv.input.get("path").and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|p| resolve(&self.cwd, p))
            .unwrap_or_else(|| self.cwd.clone());

        // Run find inside the sandbox, excluding standard ignored dirs
        let excluded = crate::tools::FS_GLOB_IGNORED_DIRS
            .iter()
            .map(|d| format!("-not -path '*/{d}/*' -not -name '{d}'"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = format!(
            "find {} -maxdepth 20 {} 2>/dev/null | head -{}",
            shell_escape(&base),
            excluded,
            crate::tools::MAX_FS_GLOB_RESULTS + 1,
        );
        let result = self.executor.exec(&cmd, Some(&self.cwd), 30_000).await;
        match result {
            Err(e) => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
            Ok(r) => {
                let lines: Vec<String> = r.stdout.lines()
                    .filter(|l| {
                        let p = std::path::Path::new(l);
                        crate::tools::simple_glob_match(&pattern, &p.to_string_lossy())
                            || p.file_name()
                                .map(|n| crate::tools::simple_glob_match(&pattern, &n.to_string_lossy()))
                                .unwrap_or(false)
                    })
                    .take(crate::tools::MAX_FS_GLOB_RESULTS)
                    .map(String::from)
                    .collect();
                let truncated = lines.len() >= crate::tools::MAX_FS_GLOB_RESULTS;
                Ok(ToolOutcome {
                    output: Ok(json!({
                        "pattern": pattern,
                        "count": lines.len(),
                        "matches": lines,
                        "truncated": truncated,
                    })),
                    attachments: vec![],
                })
            }
        }
    }

    async fn e2b_grep(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let pattern = req_str(&inv, "pattern")?;
        let ci = inv.input.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
        let search = inv.input.get("path").and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|p| resolve(&self.cwd, p))
            .unwrap_or_else(|| self.cwd.clone());

        let ci_flag = if ci { "-i " } else { "" };
        let excluded = crate::tools::FS_GLOB_IGNORED_DIRS
            .iter()
            .map(|d| format!("--exclude-dir={d}"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = format!(
            "grep -rn {ci_flag}{excluded} -e {} -- {} 2>/dev/null",
            shell_escape(pattern),
            shell_escape(&search),
        );
        match self.executor.exec(&cmd, Some(&self.cwd), 30_000).await {
            Err(e) => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
            Ok(r) if r.exit_code >= 2 => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime,
                    truncate(format!("grep error: {}", r.stderr)))),
                attachments: vec![],
            }),
            Ok(r) => {
                let matches = self.bound_output(r.stdout, &inv.id, "matches").await;
                Ok(ToolOutcome {
                    output: Ok(json!({
                        "pattern": pattern,
                        "matches": matches,
                    })),
                    attachments: vec![],
                })
            }
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn truncate(s: String) -> String {
    if s.len() <= crate::tools::MAX_OUTPUT_BYTES { return s; }
    let kept: String = s.chars().take(crate::tools::MAX_OUTPUT_BYTES).collect();
    format!("{kept}\n\n[content truncated: use offset/limit to read more]")
}

fn resolve(cwd: &str, path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_absolute() { path.to_string() }
    else { std::path::PathBuf::from(cwd).join(p).to_string_lossy().into_owned() }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn req_str<'a>(inv: &'a ToolInvocation, key: &str) -> Result<&'a str, ToolRuntimeError> {
    inv.input.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
        .ok_or_else(|| ToolRuntimeError::InvalidInput {
            tool: inv.name.clone(),
            message: format!("missing field `{key}`"),
        })
}
