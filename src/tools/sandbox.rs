use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::{
    builtin_tool_specs, fs_glob_bounded, ToolFailure, ToolFailureKind, ToolInvocation,
    ToolOutcome, ToolRuntime, ToolRuntimeError, ToolSpec,
};
use crate::tools::approval::{is_read_only, ApprovalGate, YoloApproval};

/// Result of executing a command in a sandbox.
#[derive(Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// A thin, dumb execution interface for any sandbox environment.
///
/// Implementors are responsible only for *running* commands and *reading/
/// writing* files inside the sandbox. All agent-level policy (output capping,
/// directory filtering for grep/glob, approval gating) is applied by
/// `SandboxToolRuntime` before calling these methods.
///
/// # Example implementors
///
/// * `E2bExecutor` (runtime-driver): wraps the e2b `EnvdProcessClient` and
///   filesystem API.
/// * A local test double that records calls or returns scripted results.
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Execute a shell command inside the sandbox.
    /// `cwd` is an absolute path inside the sandbox.
    async fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        timeout_ms: u64,
    ) -> Result<ExecResult, String>;

    /// Execute a command with live stdout streaming.
    /// Each line is passed to `on_line` as it arrives.
    async fn exec_streaming(
        &self,
        command: &str,
        cwd: Option<&str>,
        timeout_ms: u64,
        on_line: &(dyn Fn(String) + Send + Sync),
    ) -> Result<ExecResult, String>;

    /// Read a file from the sandbox filesystem.
    async fn read_file(&self, path: &str) -> Result<String, String>;

    /// Write (overwrite or create) a file in the sandbox filesystem.
    async fn write_file(&self, path: &str, content: &str) -> Result<(), String>;
}

/// Configuration for `SandboxToolRuntime`.
pub struct SandboxToolConfig<E: SandboxExecutor> {
    /// The sandbox executor — bridges harness to the actual sandbox.
    pub executor: E,
    /// Working directory inside the sandbox. Relative paths in tool arguments
    /// are resolved against this.
    pub cwd: String,
    /// Controls whether mutating tools are advertised to the model.
    /// Defaults to `YoloApproval` (all tools available, no interactive gate)
    /// since sandboxes are already isolated environments.
    pub approval: Arc<dyn ApprovalGate>,
    /// Receives structured events produced during tool execution
    /// (e.g. `bash_stdout_line`). Pass `Arc::new(|_| {})` to discard.
    pub emit: Arc<dyn Fn(Value) + Send + Sync + 'static>,
}

impl<E: SandboxExecutor> SandboxToolConfig<E> {
    pub fn new(executor: E, cwd: impl Into<String>) -> Self {
        Self {
            executor,
            cwd: cwd.into(),
            approval: Arc::new(YoloApproval),
            emit: Arc::new(|_| {}),
        }
    }
}

/// Tool runtime that delegates tool execution to any `SandboxExecutor`.
///
/// Applies the same tool policy as `LocalToolRuntime`:
/// * bash: dual-layer timeout, live stdout streaming, output cap
/// * read: pagination + output cap
/// * write/edit: standard semantics
/// * grep: excludes `node_modules`, `target`, `.git`, etc.
/// * glob: prunes dependency/build dirs via `fs_glob_bounded`
#[derive(Clone)]
pub struct SandboxToolRuntime<E: SandboxExecutor + Clone> {
    executor: E,
    cwd: String,
    approval: Arc<dyn ApprovalGate>,
    emit: Arc<dyn Fn(Value) + Send + Sync + 'static>,
}

impl<E: SandboxExecutor + Clone> SandboxToolRuntime<E> {
    pub fn new(config: SandboxToolConfig<E>) -> Self {
        Self {
            executor: config.executor,
            cwd: config.cwd,
            approval: config.approval,
            emit: config.emit,
        }
    }
}

#[async_trait]
impl<E: SandboxExecutor + Clone + Send + Sync + 'static> ToolRuntime for SandboxToolRuntime<E> {
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
        // Approval gate (same logic as LocalToolRuntime)
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
            "bash"  => self.sandbox_bash(inv).await,
            "read"  => self.sandbox_read(inv).await,
            "write" => self.sandbox_write(inv).await,
            "edit"  => self.sandbox_edit(inv).await,
            "glob"  => self.sandbox_glob(inv).await,
            "grep"  => self.sandbox_grep(inv).await,
            other   => Err(ToolRuntimeError::UnknownTool(other.into())),
        }
    }
}

// ── per-tool dispatch ──────────────────────────────────────────────────────────

impl<E: SandboxExecutor + Clone> SandboxToolRuntime<E> {
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

    async fn sandbox_bash(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let cmd = req_str(&inv, "command")?;
        let timeout_ms = inv.input.get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(120_000)
            .min(3_600_000);
        let emit = self.emit.clone();
        let result = self.executor.exec_streaming(
            cmd,
            Some(&self.cwd),
            timeout_ms,
            &move |line| emit(json!({ "type": "bash_stdout_line", "line": line })),
        ).await;

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

    async fn sandbox_read(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path = req_str(&inv, "path")?;
        let abs = resolve(&self.cwd, path);
        match self.executor.read_file(&abs).await {
            Err(e) => Ok(ToolOutcome {
                output: Err(ToolFailure::new(ToolFailureKind::Runtime, e)),
                attachments: vec![],
            }),
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
                        "path": abs,
                        "content": truncate(text),
                        "offset": offset,
                        "limit": limit,
                        "start_line": if selected.is_empty() { Value::Null } else { json!(offset+1) },
                        "end_line": if selected.is_empty() { Value::Null } else { json!(end) },
                        "total_lines": total,
                        "truncated": limit.map(|n| offset+n < total).unwrap_or(false),
                    })),
                    attachments: vec![],
                })
            }
        }
    }

    async fn sandbox_write(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path = req_str(&inv, "path")?;
        let content = req_str(&inv, "content")?;
        let abs = resolve(&self.cwd, path);
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

    async fn sandbox_edit(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        let path = req_str(&inv, "path")?;
        let old_str = req_str(&inv, "old_string")?;
        let new_str = inv.input.get("new_string").and_then(Value::as_str).unwrap_or("");
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

    async fn sandbox_glob(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        // Glob on the local filesystem is only meaningful when the harness
        // is co-located with the sandbox (e.g. volume mount). For true
        // remote sandboxes, implementors should override this via a custom
        // executor — e.g. by running `find` via exec_streaming.
        let pattern = req_str(&inv, "pattern")?.to_string();
        let base_str = inv.input.get("path").and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|p| resolve(&self.cwd, p))
            .unwrap_or_else(|| self.cwd.clone());
        let (matches, truncated) = fs_glob_bounded(&pattern, Path::new(&base_str));
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

    async fn sandbox_grep(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        // Run grep inside the sandbox via exec, with dependency-dir exclusions.
        let pattern = req_str(&inv, "pattern")?;
        let ci = inv.input.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
        let search = inv.input.get("path").and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|p| resolve(&self.cwd, p))
            .unwrap_or_else(|| self.cwd.clone());

        let mut args = vec!["grep", "-rn"];
        if ci { args.push("-i"); }
        let excludes = [
            "--exclude-dir=node_modules", "--exclude-dir=target",
            "--exclude-dir=.git", "--exclude-dir=dist", "--exclude-dir=build",
            "--exclude-dir=__pycache__", "--exclude-dir=.venv",
            "--exclude-dir=vendor", "--exclude-dir=.next",
        ];
        for e in &excludes { args.push(e); }
        let cmd = format!("{} -e {} -- {}", args.join(" "), shell_escape(pattern), shell_escape(&search));

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

fn resolve(cwd: &str, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        PathBuf::from(cwd).join(p).to_string_lossy().into_owned()
    }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn truncate(s: String) -> String {
    if s.len() <= crate::tools::MAX_OUTPUT_BYTES { return s; }
    let kept: String = s.chars().take(crate::tools::MAX_OUTPUT_BYTES).collect();
    format!("{kept}\n\n[content truncated: use offset/limit to read more]")
}

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
