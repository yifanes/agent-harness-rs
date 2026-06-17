use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::shell_risk::{classify_shell_command, ShellRiskLevel};
use crate::tools::{
    builtin_tool_specs, fs_glob_bounded, ToolFailure, ToolFailureKind, ToolInvocation,
    ToolOutcome, ToolRuntime, ToolRuntimeError, ToolSpec,
};
use crate::tools::approval::{is_read_only, ApprovalGate};

/// Opaque event emitter: receives structured JSON events produced during tool
/// execution (e.g. `bash_stdout_line`). Use an `Arc<dyn Fn(Value) + ...>`
/// or provide a no-op with `Arc::new(|_| {})`.
pub type EmitFn = Arc<dyn Fn(Value) + Send + Sync + 'static>;

/// Configuration for `LocalToolRuntime`.
pub struct LocalToolConfig {
    /// Absolute path to the working directory for all tool operations.
    /// Relative paths inside tool arguments are resolved against this.
    /// Defaults to the process's `std::env::current_dir()` when `None`.
    pub cwd: Option<PathBuf>,
    /// Controls which tool calls are allowed and which tool specs are
    /// advertised to the model. Typically `YoloApproval`, `PlanApproval`,
    /// or a custom gate (e.g. `TauriApproval` in a desktop app).
    pub approval: Arc<dyn ApprovalGate>,
    /// Receives structured events produced during tool execution.
    /// Most callers forward these to a UI channel. Pass `Arc::new(|_| {})`
    /// to discard them.
    pub emit: EmitFn,
}

/// Tool runtime that executes bash, read, write, edit, glob and grep
/// directly on the local filesystem.
///
/// All tool output is capped at `MAX_TOOL_CHARS` characters so large
/// files and noisy commands don't blow out the model's context window.
/// Glob/grep skip dependency and build directories automatically.
#[derive(Clone)]
pub struct LocalToolRuntime {
    cwd: PathBuf,
    approval: Arc<dyn ApprovalGate>,
    emit: EmitFn,
}

impl LocalToolRuntime {
    pub fn new(config: LocalToolConfig) -> Self {
        let cwd = config.cwd
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));
        Self { cwd, approval: config.approval, emit: config.emit }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() { p.to_path_buf() } else { self.cwd.join(p) }
    }

    /// Gate that decides whether a tool invocation may proceed.
    ///
    /// * Hard-blocked bash commands are rejected in every mode.
    /// * Read-only bash / read / glob / grep pass without hitting the gate.
    /// * Everything else goes to `approval.approve()`.
    async fn gate(
        &self,
        inv: &ToolInvocation,
        cancel: Option<&CancellationToken>,
    ) -> Result<(), String> {
        if inv.name == "bash" {
            let cmd = inv.input.get("command").and_then(Value::as_str).unwrap_or("");
            let decision = classify_shell_command(cmd);
            match decision.level {
                ShellRiskLevel::Blocked => {
                    return Err(format!("命令在禁止清单上，已拒绝：{}", decision.reason));
                }
                ShellRiskLevel::SafeRead => return Ok(()),
                ShellRiskLevel::BoundedWrite
                    if self.approval.advertise_mutating_tools() =>
                {
                    return Ok(());
                }
                _ => {}
            }
        } else if is_read_only(&inv.name) {
            return Ok(());
        }

        // Non-read-only tool → delegate to the approval gate.
        let approved = if let Some(tok) = cancel {
            tokio::select! {
                biased;
                _ = tok.cancelled() => return Err("已取消".into()),
                result = self.approval.approve(inv) => result,
            }
        } else {
            self.approval.approve(inv).await
        };

        if approved { Ok(()) } else { Err("操作被拒绝".into()) }
    }
}

#[async_trait]
impl ToolRuntime for LocalToolRuntime {
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
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        if let Err(reason) = self.gate(&inv, cancel).await {
            return Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Denied, reason)),
                attachments: vec![],
            });
        }
        match inv.name.as_str() {
            "bash"  => bash_invoke(inv, cancel, &self.cwd, self.emit.clone()).await,
            "read"  => read_invoke(inv, self).await,
            "write" => write_invoke(inv, self).await,
            "edit"  => edit_invoke(inv, self).await,
            "glob"  => glob_invoke(inv, self).await,
            "grep"  => grep_invoke(inv, self).await,
            "web_fetch" => crate::tools::web_fetch::invoke(inv).await,
            other   => Err(ToolRuntimeError::UnknownTool(other.into())),
        }
    }
}

// ── bash ──────────────────────────────────────────────────────────────────────

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn bash_invoke(
    inv: ToolInvocation,
    cancel: Option<&CancellationToken>,
    cwd: &Path,
    emit: EmitFn,
) -> Result<ToolOutcome, ToolRuntimeError> {
    let command = req_str(&inv, "command")?;
    let id = &*inv.id;

    // Dual-layer timeout:
    //   soft_timeout_ms — no-output detector: if the process produces
    //     nothing for this long it is killed and the model is told to retry.
    //     Streaming output resets the clock, so a long build that prints
    //     progress will never hit this.
    //   timeout_ms — absolute hard ceiling, regardless of output activity.
    let soft_ms: u64 = inv.input.get("soft_timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(10_000);
    let hard_ms: u64 = inv.input.get("timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(120_000)
        .min(3_600_000);

    let last_out = Arc::new(AtomicU64::new(epoch_ms()));
    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let stderr_buf = Arc::new(Mutex::new(String::new()));

    let shell = if Path::new("/bin/bash").exists() { "/bin/bash" } else { "/bin/sh" };
    let mut child = Command::new(shell)
        .args(["-lc", command])
        .current_dir(cwd)
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| ToolRuntimeError::Runtime(format!("spawn failed: {e}")))?;

    let raw_stdout = child.stdout.take().expect("stdout piped");
    let raw_stderr = child.stderr.take().expect("stderr piped");

    let act1 = last_out.clone();
    let emit_out = emit.clone();
    let stdout_acc = stdout_buf.clone();
    let stdout_task = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(raw_stdout).lines();
        let mut buf = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            emit_out(json!({ "type": "bash_stdout_line", "line": line, "stream": "stdout" }));
            act1.store(epoch_ms(), Ordering::Relaxed);
            buf.push_str(&line);
            buf.push('\n');
            if let Ok(mut acc) = stdout_acc.lock() {
                acc.push_str(&line);
                acc.push('\n');
            }
        }
        buf
    });

    let act2 = last_out.clone();
    let emit_err = emit.clone();
    let stderr_acc = stderr_buf.clone();
    let stderr_task = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(raw_stderr).lines();
        let mut buf = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            emit_err(json!({ "type": "bash_stdout_line", "line": line, "stream": "stderr" }));
            act2.store(epoch_ms(), Ordering::Relaxed);
            buf.push_str(&line);
            buf.push('\n');
            if let Ok(mut acc) = stderr_acc.lock() {
                acc.push_str(&line);
                acc.push('\n');
            }
        }
        buf
    });

    let watcher_ts = last_out.clone();
    let soft_watcher = async move {
        let start = epoch_ms();
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let now = epoch_ms();
            if now.saturating_sub(start) >= soft_ms
                && now.saturating_sub(watcher_ts.load(Ordering::Relaxed)) >= soft_ms
            {
                return (now.saturating_sub(start), now.saturating_sub(watcher_ts.load(Ordering::Relaxed)));
            }
        }
    };

    let timed = async {
        let (out, err) = tokio::join!(
            async { stdout_task.await.unwrap_or_default() },
            async { stderr_task.await.unwrap_or_default() },
        );
        let status = child.wait().await;
        (out, err, status)
    };

    let hard_timer = tokio::time::sleep(Duration::from_millis(hard_ms));

    let timeout_outcome = |kind: &str, message: String| ToolOutcome {
        output: Ok(json!({
            "command": command,
            "shell": shell,
            "stdout": bound_output(stdout_buf.lock().map(|s| s.clone()).unwrap_or_default(), id, "stdout"),
            "stderr": bound_output(stderr_buf.lock().map(|s| s.clone()).unwrap_or_default(), id, "stderr"),
            "exit_code": null,
            "success": false,
            "timed_out": true,
            "timeout_kind": kind,
            "message": message,
        })),
        attachments: vec![],
    };
    let soft_err = |tot: u64, sil: u64| timeout_outcome(
        "soft",
        format!(
            "Command produced no output for {sil}ms (total {tot}ms). \
Retry with larger `soft_timeout_ms` or `timeout_ms` if it is expected to take longer."
        ),
    );
    let hard_err = || timeout_outcome(
        "hard",
        format!(
            "Command did not finish in {hard_ms}ms. Retry with a larger `timeout_ms` if it is expected to take longer."
        ),
    );

    let result: Result<(String, String, _), ToolOutcome> = if let Some(tok) = cancel {
        tokio::select! {
            v = timed => Ok(v),
            (tot, sil) = soft_watcher => Err(soft_err(tot, sil)),
            _ = hard_timer => Err(hard_err()),
            _ = tok.cancelled() => Err(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, "cancelled")),
                attachments: vec![],
            }),
        }
    } else {
        tokio::select! {
            v = timed => Ok(v),
            (tot, sil) = soft_watcher => Err(soft_err(tot, sil)),
            _ = hard_timer => Err(hard_err()),
        }
    };

    let (stdout, stderr, status_result) = match result {
        Err(outcome) => return Ok(outcome),
        Ok(v) => v,
    };

    let exit_code = status_result.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);

    Ok(ToolOutcome {
        output: Ok(json!({
            "command": command,
            "shell": shell,
            "stdout": bound_output(stdout, id, "stdout"),
            "stderr": bound_output(stderr, id, "stderr"),
            "exit_code": exit_code,
            "success": exit_code == 0,
        })),
        attachments: vec![],
    })
}

// ── read ──────────────────────────────────────────────────────────────────────

async fn read_invoke(inv: ToolInvocation, rt: &LocalToolRuntime) -> Result<ToolOutcome, ToolRuntimeError> {
    let path = req_str(&inv, "path")?;
    let resolved = rt.resolve(path);
    match tokio::fs::read_to_string(&resolved).await {
        Ok(content) => {
            let total = content.lines().count();
            let offset = inv.input.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let limit = inv.input.get("limit").and_then(Value::as_u64)
                .map(|v| v.clamp(1, 2_000) as usize);
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
                    "path": resolved.to_string_lossy(),
                    "content": truncate(text),
                    "offset": offset,
                    "limit": limit,
                    "start_line": if selected.is_empty() { Value::Null } else { json!(offset + 1) },
                    "end_line": if selected.is_empty() { Value::Null } else { json!(end) },
                    "total_lines": total,
                    "truncated": limit.map(|n| offset + n < total).unwrap_or(false),
                })),
                attachments: vec![],
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ToolOutcome {
            output: Err(ToolFailure::new(ToolFailureKind::NotFound,
                format!("file not found: {}", resolved.display()))),
            attachments: vec![],
        }),
        Err(e) => Ok(ToolOutcome {
            output: Err(ToolFailure::new(ToolFailureKind::Runtime, format!("read error: {e}"))),
            attachments: vec![],
        }),
    }
}

// ── write ─────────────────────────────────────────────────────────────────────

async fn write_invoke(inv: ToolInvocation, rt: &LocalToolRuntime) -> Result<ToolOutcome, ToolRuntimeError> {
    let path = req_str(&inv, "path")?;
    let content = req_str(&inv, "content")?;
    let resolved = rt.resolve(path);
    if let Some(parent) = resolved.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| ToolRuntimeError::Runtime(format!("mkdir: {e}")))?;
        }
    }
    tokio::fs::write(&resolved, content).await
        .map_err(|e| ToolRuntimeError::Runtime(format!("write error: {e}")))?;
    Ok(ToolOutcome {
        output: Ok(json!({ "path": resolved.to_string_lossy(), "written": true })),
        attachments: vec![],
    })
}

// ── edit ──────────────────────────────────────────────────────────────────────

async fn edit_invoke(inv: ToolInvocation, rt: &LocalToolRuntime) -> Result<ToolOutcome, ToolRuntimeError> {
    let path = req_str(&inv, "path")?;
    let old_string = req_str(&inv, "old_string")?;
    let new_string = inv.input.get("new_string").and_then(Value::as_str).unwrap_or("");
    let replace_all = inv.input.get("replace_all").and_then(Value::as_bool).unwrap_or(false);
    let resolved = rt.resolve(path);

    let content = match tokio::fs::read_to_string(&resolved).await {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ToolOutcome {
            output: Err(ToolFailure::new(ToolFailureKind::NotFound,
                format!("file not found: {}", resolved.display()))),
            attachments: vec![],
        }),
        Err(e) => return Err(ToolRuntimeError::Runtime(e.to_string())),
    };

    let occurrences = content.matches(old_string).count();
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
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };
    tokio::fs::write(&resolved, new_content).await
        .map_err(|e| ToolRuntimeError::Runtime(e.to_string()))?;
    Ok(ToolOutcome {
        output: Ok(json!({
            "path": resolved.to_string_lossy(),
            "replaced": replaced,
            "old_lines": old_string.lines().count(),
            "new_lines": new_string.lines().count(),
        })),
        attachments: vec![],
    })
}

// ── glob ──────────────────────────────────────────────────────────────────────

async fn glob_invoke(inv: ToolInvocation, rt: &LocalToolRuntime) -> Result<ToolOutcome, ToolRuntimeError> {
    let pattern = req_str(&inv, "pattern")?.to_string();
    let base = match inv.input.get("path").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        Some(p) => rt.resolve(p),
        None    => rt.cwd.clone(),
    };
    let (matches, truncated) = fs_glob_bounded(&pattern, &base);
    Ok(ToolOutcome {
        output: Ok(json!({
            "pattern": pattern,
            "count": matches.len(),
            "matches": matches,
            "truncated": truncated,
        })),
        attachments: vec![],
    })
}

// ── grep ──────────────────────────────────────────────────────────────────────

async fn grep_invoke(inv: ToolInvocation, rt: &LocalToolRuntime) -> Result<ToolOutcome, ToolRuntimeError> {
    let pattern = req_str(&inv, "pattern")?.to_string();
    let ci = inv.input.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
    let search = match inv.input.get("path").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        Some(p) => rt.resolve(p),
        None    => rt.cwd.clone(),
    };

    let mut cmd = Command::new("grep");
    cmd.args(["-rn", "-E"]);
    if ci { cmd.arg("-i"); }
    cmd.args([
        "--exclude-dir=node_modules",
        "--exclude-dir=target",
        "--exclude-dir=.git",
        "--exclude-dir=dist",
        "--exclude-dir=build",
        "--exclude-dir=__pycache__",
        "--exclude-dir=.venv",
        "--exclude-dir=vendor",
        "--exclude-dir=.next",
    ]);
    cmd.arg("-e").arg(&pattern).arg("--").arg(&search);
    cmd.current_dir(&rt.cwd);

    match tokio::time::timeout(Duration::from_secs(30), cmd.output()).await {
        Err(_) => Ok(ToolOutcome {
            output: Err(ToolFailure::new(ToolFailureKind::Timeout, "grep timed out after 30s")),
            attachments: vec![],
        }),
        Ok(Err(e)) => Err(ToolRuntimeError::Runtime(format!("grep spawn failed: {e}"))),
        Ok(Ok(out)) => {
            let code = out.status.code().unwrap_or(-1);
            if code >= 2 {
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                return Ok(ToolOutcome {
                    output: Err(ToolFailure::new(ToolFailureKind::Runtime,
                        truncate(format!("grep error: {stderr}")))),
                    attachments: vec![],
                });
            }
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            Ok(ToolOutcome {
                output: Ok(json!({
                    "pattern": pattern,
                    "matches": bound_output(stdout, &inv.id, "matches"),
                })),
                attachments: vec![],
            })
        }
    }
}

// ── output cap ────────────────────────────────────────────────────────────────

/// Write `content` to `/tmp/harness_out_{id}_{suffix}.txt` and return an
/// error-aware head+tail preview with the path. Used for bash stdout/stderr
/// and grep matches. Spills only when over budget (see `bounded_preview`).
fn bound_output(content: String, id: &str, suffix: &str) -> String {
    let path = format!("/tmp/harness_out_{id}_{suffix}.txt");
    match crate::tools::bounded_preview(&content, &path) {
        None => content,
        Some(preview) => {
            let _ = std::fs::write(&path, &content);
            preview
        }
    }
}

/// Simple safety truncation used only by the read tool as a backstop for
/// pages that exceed the output budget after pagination.
fn truncate(s: String) -> String {
    crate::tools::clip_head(s)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn req_str<'a>(inv: &'a ToolInvocation, key: &str) -> Result<&'a str, ToolRuntimeError> {
    inv.input
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ToolRuntimeError::InvalidInput {
            tool: inv.name.clone(),
            message: format!("missing field `{key}`"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::approval::YoloApproval;

    fn runtime() -> LocalToolRuntime {
        LocalToolRuntime::new(LocalToolConfig {
            cwd: Some(std::env::temp_dir()),
            approval: Arc::new(YoloApproval),
            emit: Arc::new(|_| {}),
        })
    }

    #[tokio::test]
    async fn bash_non_zero_exit_returns_structured_result() {
        let out = runtime()
            .invoke(ToolInvocation {
                id: "tc_nonzero".into(),
                name: "bash".into(),
                input: json!({"command": "printf nope >&2; exit 7"}),
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(out["exit_code"], 7);
        assert_eq!(out["success"], false);
        assert_eq!(out["stderr"], "nope\n");
    }

    #[tokio::test]
    async fn bash_timeout_returns_structured_result() {
        let out = runtime()
            .invoke(ToolInvocation {
                id: "tc_timeout".into(),
                name: "bash".into(),
                input: json!({
                    "command": "sleep 2",
                    "soft_timeout_ms": 1000,
                    "timeout_ms": 5000
                }),
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(out["success"], false);
        assert_eq!(out["timed_out"], true);
        assert_eq!(out["timeout_kind"], "soft");
    }

    #[tokio::test]
    async fn bash_tool_supports_bash_syntax_when_bash_exists() {
        if !Path::new("/bin/bash").exists() {
            return;
        }
        let out = runtime()
            .invoke(ToolInvocation {
                id: "tc_bash_syntax".into(),
                name: "bash".into(),
                input: json!({"command": "diff <(printf a) <(printf a)"}),
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(out["success"], true);
        assert_eq!(out["exit_code"], 0);
        assert_eq!(out["shell"], "/bin/bash");
    }
}
