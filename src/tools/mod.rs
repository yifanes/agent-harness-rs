pub mod approval;
pub mod bounded;
pub mod sandbox;
pub mod web_fetch;

#[cfg(feature = "local-tools")]
pub mod local;

#[cfg(feature = "e2b")]
pub mod e2b;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};

/// LLM-facing description of one tool. `input_schema` is a JSON Schema
/// object the model uses to generate well-formed `tool_call.arguments`.
/// Shared between providers — OpenAI wraps it in `{type:"function", function:{...}}`,
/// future Anthropic client will pass it as Messages API `input_schema` verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for `tool_call.arguments`. Must be an `object`-typed schema.
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub input: Value,
    /// Verbatim JSON argument text emitted by the model for this tool call.
    ///
    /// Populated when the provider streams raw argument deltas, such as
    /// OpenAI-compatible `function.arguments` chunks. `None` for synthetic
    /// invocations, provider-final parsed inputs, restored histories that
    /// predate this field, or repaired inputs where the literal model text no
    /// longer matches `input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_emitted_args: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    pub output: Result<Value, ToolFailure>,
    /// Structured non-text attachments returned by the tool — currently
    /// just images returned by MCP servers (e.g. a screenshot tool).
    /// Empty Vec for native tools (`SandboxToolRuntime`) — none of them
    /// produce non-text output today. Reuses `UserAttachment` rather
    /// than introducing a parallel `ToolAttachment` enum because both
    /// shapes are identical (image source); rename to `MediaAttachment`
    /// if a future tool variant differs.
    pub attachments: Vec<crate::model::UserAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolFailureKind {
    InvalidInput,
    NotFound,
    NonZeroExit,
    Timeout,
    Runtime,
    /// Rejected by policy before dispatch (e.g. a shell command on the
    /// hard-deny list). The message tells the model why so it can adjust.
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolFailure {
    pub kind: ToolFailureKind,
    pub message: String,
}

impl ToolFailure {
    pub fn new(kind: ToolFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

pub fn invalid_input_failure(
    tool: &str,
    message: impl AsRef<str>,
    input: &Value,
    schema: Option<&Value>,
) -> ToolFailure {
    ToolFailure::new(
        ToolFailureKind::InvalidInput,
        format_invalid_input_message(tool, message.as_ref(), input, schema),
    )
}

pub fn format_invalid_input_message(
    tool: &str,
    detail: &str,
    input: &Value,
    schema: Option<&Value>,
) -> String {
    let received = received_fields(input);
    let summaries = summarize_input_fields(input);
    let mut message = format!(
        "The {tool} tool was called with invalid arguments: {detail}. \
Please rewrite the input so it satisfies the expected schema."
    );
    if !received.is_empty() {
        message.push_str(&format!(" Received fields: {}.", received.join(", ")));
    }
    if !summaries.is_empty() {
        message.push_str(&format!(" Field summary: {}.", summaries.join("; ")));
    }
    // Teaching: show a minimal valid example synthesized from the schema so
    // the model sees the exact shape it should have produced.
    if let Some(schema) = schema {
        let example = crate::tool_repair::example_for_schema(schema);
        if example.as_object().is_some_and(|o| !o.is_empty()) {
            message.push_str(&format!(" Expected shape: {example}."));
        }
    }
    message
}

fn received_fields(input: &Value) -> Vec<String> {
    let Some(obj) = input.as_object() else {
        return vec![json_type(input).to_string()];
    };
    let mut keys: Vec<String> = obj.keys().cloned().collect();
    keys.sort();
    keys
}

fn summarize_input_fields(input: &Value) -> Vec<String> {
    let Some(obj) = input.as_object() else {
        return vec![format!("input: {}", summarize_value(input))];
    };
    let mut entries: Vec<_> = obj.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
        .into_iter()
        .take(12)
        .map(|(key, value)| format!("{key}: {}", summarize_value(value)))
        .collect()
}

fn summarize_value(value: &Value) -> String {
    match value {
        Value::String(s) => {
            let preview: String = s.chars().take(80).collect();
            let suffix = if s.chars().count() > 80 { "..." } else { "" };
            format!(
                "string({} chars, preview={:?}{suffix})",
                s.chars().count(),
                preview
            )
        }
        Value::Array(a) => format!("array({} items)", a.len()),
        Value::Object(o) => format!("object({} keys)", o.len()),
        Value::Bool(_) => "boolean".into(),
        Value::Number(_) => "number".into(),
        Value::Null => "null".into(),
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolRuntimeError {
    #[error("unknown tool {0}")]
    UnknownTool(String),

    #[error("invalid input for {tool}: {message}")]
    InvalidInput { tool: String, message: String },

    #[error("tool timed out: {0}")]
    Timeout(String),

    #[error("tool runtime failed: {0}")]
    Runtime(String),
}

#[async_trait]
pub trait ToolRuntime: Send + Sync {
    fn specs(&self) -> Vec<ToolSpec>;

    /// Apply schema-guided input repair in place, returning the repairs made
    /// (or `None` when the input is already clean / no schema matches).
    ///
    /// The default is a no-op; [`bounded::BoundedToolRuntime`] overrides it as
    /// the single source of truth for repair. `agent_loop` calls this BEFORE
    /// recording the invocation in history / events so the recorded arguments
    /// match what the runtime ultimately executes; the wrapper re-applies it
    /// during dispatch (idempotent) to also cover bypass callers.
    fn repair_invocation(
        &self,
        _invocation: &mut ToolInvocation,
    ) -> Option<Vec<crate::tool_repair::ToolInputRepair>> {
        None
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError>;

    /// Cancellation-aware variant of `invoke`. When `cancel` is fired
    /// the runtime SHOULD abort the in-flight tool (e.g. SIGTERM the
    /// shell subprocess for `bash`) and return `ToolRuntimeError::
    /// Runtime("cancelled")`.
    ///
    /// The default implementation races `invoke` against cancellation. This
    /// drops the invocation future when cancelled, which prevents in-process
    /// work from continuing after the turn ends. Runtimes that start work in
    /// an external system should override this and explicitly terminate it.
    async fn invoke_cancellable(
        &self,
        invocation: ToolInvocation,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        if let Some(token) = cancel {
            tokio::select! {
                biased;
                _ = token.cancelled() => {
                    Err(ToolRuntimeError::Runtime("cancelled".into()))
                }
                outcome = self.invoke(invocation) => outcome,
            }
        } else {
            self.invoke(invocation).await
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct MockToolRuntime {
    files: Arc<Mutex<HashMap<String, String>>>,
}

impl MockToolRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_file(self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.files
            .lock()
            .unwrap()
            .insert(path.into(), content.into());
        self
    }
}

#[async_trait]
impl ToolRuntime for MockToolRuntime {
    fn specs(&self) -> Vec<ToolSpec> {
        builtin_tool_specs()
    }

    async fn invoke(&self, invocation: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        match invocation.name.as_str() {
            "bash" => {
                let command = required_str(&invocation, "command")?;
                Ok(ToolOutcome {
                    output: Ok(json!({
                        "command": command,
                        "stdout": format!("mock executed: {command}\n"),
                        "stderr": "",
                        "exit_code": 0,
                    })),
                    attachments: vec![],
                })
            }
            "read" => {
                let path = required_str(&invocation, "path")?;
                let files = self.files.lock().unwrap();
                match files.get(path) {
                    Some(content) => Ok(ToolOutcome {
                        output: Ok(json!({"path": path, "content": content})),
                        attachments: vec![],
                    }),
                    None => Ok(ToolOutcome {
                        output: Err(ToolFailure::new(
                            ToolFailureKind::NotFound,
                            format!("file not found: {path}"),
                        )),
                        attachments: vec![],
                    }),
                }
            }
            "write" => {
                let path = required_str(&invocation, "path")?.to_string();
                let content = required_str(&invocation, "content")?.to_string();
                self.files.lock().unwrap().insert(path.clone(), content);
                Ok(ToolOutcome {
                    output: Ok(json!({"path": path, "written": true})),
                    attachments: vec![],
                })
            }
            "edit" => {
                let path = required_str(&invocation, "path")?.to_string();
                let old_string = required_str(&invocation, "old_string")?.to_string();
                let new_string = invocation
                    .input
                    .get("new_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let replace_all = invocation
                    .input
                    .get("replace_all")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let mut files = self.files.lock().unwrap();
                let Some(content) = files.get(&path).cloned() else {
                    return Ok(ToolOutcome {
                        output: Err(ToolFailure::new(
                            ToolFailureKind::NotFound,
                            format!("file not found: {path}"),
                        )),
                        attachments: vec![],
                    });
                };
                let resolved = match resolve_edit_search(
                    &content,
                    &old_string,
                    &new_string,
                    replace_all,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let message = match e {
                            EditSearchError::NotFound => {
                                "Could not find old_string in the file. It must match exactly, including whitespace and indentation. Read the file again before retrying.".to_string()
                            }
                            EditSearchError::EscapedNotFound =>
                                "Could not find old_string in the file, even after checking for JSON-escaped text. It must match exactly, including whitespace and indentation. Read the file again before retrying.".to_string(),
                            EditSearchError::Ambiguous { occurrences } => format!(
                                "Found {occurrences} exact matches for old_string. Provide more surrounding context or set replace_all=true."
                            ),
                            EditSearchError::EscapedAmbiguous { occurrences } => format!(
                                "old_string appears JSON-escaped and matches {occurrences} occurrences after unescaping. Provide more surrounding context or set replace_all=true."
                            ),
                        };
                        return Ok(ToolOutcome {
                            output: Err(ToolFailure::new(ToolFailureKind::InvalidInput, message)),
                            attachments: vec![],
                        });
                    }
                };
                let next = if replace_all {
                    content.replace(&resolved.old_string, &resolved.new_string)
                } else {
                    content.replacen(&resolved.old_string, &resolved.new_string, 1)
                };
                let replaced = if replace_all { resolved.occurrences } else { 1 };
                files.insert(path.clone(), next);
                // Repair is silent: a successful edit reports only the result.
                // The json-escape rescue is recorded for tracing, never
                // surfaced to the model (MiMoCode "success silent" policy).
                if let Some(repair) = resolved.repair {
                    tracing::debug!(
                        target: "harness::tool_repair",
                        tool = "edit",
                        repair,
                        "edit applied after silent json-escape repair"
                    );
                }
                Ok(ToolOutcome {
                    output: Ok(json!({"path": path, "replaced": replaced})),
                    attachments: vec![],
                })
            }
            "grep" => {
                let pattern = required_str(&invocation, "pattern")?.to_string();
                let case_insensitive = invocation
                    .input
                    .get("case_insensitive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let needle = if case_insensitive {
                    pattern.to_lowercase()
                } else {
                    pattern.clone()
                };
                let files = self.files.lock().unwrap();
                let mut matches = Vec::new();
                for (path, content) in files.iter() {
                    for (idx, line) in content.lines().enumerate() {
                        let hay = if case_insensitive {
                            line.to_lowercase()
                        } else {
                            line.to_string()
                        };
                        if hay.contains(&needle) {
                            matches.push(json!({
                                "path": path,
                                "line": idx + 1,
                                "text": line,
                            }));
                        }
                    }
                }
                Ok(ToolOutcome {
                    output: Ok(json!({"pattern": pattern, "matches": matches})),
                    attachments: vec![],
                })
            }
            "glob" => {
                let pattern = required_str(&invocation, "pattern")?.to_string();
                let files = self.files.lock().unwrap();
                let matches: Vec<&str> = files
                    .keys()
                    .filter(|k| simple_glob_match(&pattern, k))
                    .map(|k| k.as_str())
                    .collect();
                Ok(ToolOutcome {
                    output: Ok(json!({"pattern": pattern, "matches": matches})),
                    attachments: vec![],
                })
            }
            "web_fetch" => Ok(ToolOutcome {
                output: Ok(json!({
                    "url": invocation.input.get("url").and_then(Value::as_str).unwrap_or(""),
                    "final_url": invocation.input.get("url").and_then(Value::as_str).unwrap_or(""),
                    "status": 200,
                    "content_type": "text/plain",
                    "format": invocation.input.get("format").and_then(Value::as_str).unwrap_or("markdown"),
                    "content": "mock web_fetch response",
                    "truncated": false,
                })),
                attachments: vec![],
            }),
            other => Err(ToolRuntimeError::UnknownTool(other.into())),
        }
    }
}

/// Successful resolution of an edit's search/replace strings against file
/// content. `repair` is `Some("json_escape_unwrapped")` when the match
/// only succeeded after unescaping literal `\n` / `\t` / `\r` sequences —
/// weak models frequently double-escape control characters when emitting
/// `old_string` through JSON tool arguments.
#[derive(Debug)]
pub struct ResolvedEditSearch {
    pub old_string: String,
    pub new_string: String,
    pub occurrences: usize,
    pub repair: Option<&'static str>,
}

/// Why an edit search failed to resolve. Callers build the user-facing
/// message (each runtime embeds the path differently).
#[derive(Debug, PartialEq)]
pub enum EditSearchError {
    /// `old_string` not in content, no escape rescue applicable.
    NotFound,
    /// `old_string` looked JSON-escaped, but the unescaped text is not in
    /// the content either.
    EscapedNotFound,
    /// Direct match is ambiguous without `replace_all`.
    Ambiguous { occurrences: usize },
    /// Unescaped match is ambiguous without `replace_all`.
    EscapedAmbiguous { occurrences: usize },
}

/// Resolve `old_string` against `content`, falling back to unescaping
/// literal control sequences when the strict match fails. The direct
/// match keeps our existing ambiguity guard; the escape fallback mirrors
/// unescape both old and new strings, require a match, and reject
/// multi-location matches without `replace_all`.
pub fn resolve_edit_search(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<ResolvedEditSearch, EditSearchError> {
    let direct = content.matches(old_string).count();
    if direct > 0 {
        if !replace_all && direct > 1 {
            return Err(EditSearchError::Ambiguous {
                occurrences: direct,
            });
        }
        return Ok(ResolvedEditSearch {
            old_string: old_string.to_string(),
            new_string: new_string.to_string(),
            occurrences: direct,
            repair: None,
        });
    }
    if !has_literal_escaped_controls(old_string) {
        return Err(EditSearchError::NotFound);
    }
    let unescaped_old = unescape_literal_controls(old_string);
    if unescaped_old == old_string {
        return Err(EditSearchError::NotFound);
    }
    let count = content.matches(&unescaped_old).count();
    if count == 0 {
        return Err(EditSearchError::EscapedNotFound);
    }
    if !replace_all && count > 1 {
        return Err(EditSearchError::EscapedAmbiguous { occurrences: count });
    }
    Ok(ResolvedEditSearch {
        old_string: unescaped_old,
        new_string: unescape_literal_controls(new_string),
        occurrences: count,
        repair: Some("json_escape_unwrapped"),
    })
}

/// Does the string contain literal (two-character) `\n` / `\t` / `\r`
/// escape sequences?
fn has_literal_escaped_controls(s: &str) -> bool {
    s.contains("\\n") || s.contains("\\t") || s.contains("\\r")
}

/// Replace literal escape sequences with their control characters:
/// `\r\n` → newline (checked first), then
/// `\n` → newline, `\r` → CR, `\t` → tab. Anything else passes through —
/// including `\\` — so `\\n` becomes `\` + newline, matching Go.
fn unescape_literal_controls(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'r'
                && i + 3 < bytes.len()
                && bytes[i + 2] == b'\\'
                && bytes[i + 3] == b'n'
            {
                out.push(b'\n');
                i += 4;
                continue;
            }
            let replacement = match bytes[i + 1] {
                b'n' => Some(b'\n'),
                b'r' => Some(b'\r'),
                b't' => Some(b'\t'),
                _ => None,
            };
            if let Some(ch) = replacement {
                out.push(ch);
                i += 2;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Only ASCII subsequences were replaced; multi-byte UTF-8 passes
    // through verbatim, so the result is valid UTF-8.
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Glob-match a pattern against a relative path string.
/// Supports `*` (any chars except `/`), `**` (any chars including `/`), `?`,
/// and `{a,b}` brace alternation (models emit `**/*.{ts,tsx}` reflexively —
/// silently treating `{` as a literal made such patterns match nothing).
/// Used by `MockToolRuntime` and by `fs_glob` for real-filesystem walks.
pub fn simple_glob_match(pattern: &str, candidate: &str) -> bool {
    if pattern.contains('{') {
        // expand_braces returns fully-expanded patterns (no `{` groups left
        // except literal unbalanced ones), so match each directly.
        return expand_braces(pattern)
            .iter()
            .any(|p| simple_glob_match_single(p, candidate));
    }
    simple_glob_match_single(pattern, candidate)
}

/// Cap on patterns produced by brace expansion — a backstop against
/// pathological nesting like `{a,b}{c,d}{e,f}…` multiplying without bound.
const MAX_BRACE_EXPANSIONS: usize = 128;

/// Expand the first balanced `{a,b,…}` group into one pattern per
/// alternative, recursing so nested groups and multiple groups multiply
/// out. A pattern without braces (or with an unbalanced `{`, kept as a
/// literal) returns itself. Output is capped at [`MAX_BRACE_EXPANSIONS`].
fn expand_braces(pattern: &str) -> Vec<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let Some(open) = chars.iter().position(|&c| c == '{') else {
        return vec![pattern.to_string()];
    };
    // Find the matching close brace, tracking nesting depth.
    let mut depth = 0usize;
    let mut close = None;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return vec![pattern.to_string()]; // unbalanced → literal `{`
    };
    let prefix: String = chars[..open].iter().collect();
    let suffix: String = chars[close + 1..].iter().collect();
    // Split alternatives on top-level commas only (commas inside nested
    // groups belong to the inner group).
    let mut alts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut d = 0usize;
    for &c in &chars[open + 1..close] {
        match c {
            '{' => {
                d += 1;
                cur.push(c);
            }
            '}' => {
                d -= 1;
                cur.push(c);
            }
            ',' if d == 0 => alts.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    alts.push(cur);
    let mut out = Vec::new();
    for alt in alts {
        for expanded in expand_braces(&format!("{prefix}{alt}{suffix}")) {
            out.push(expanded);
            if out.len() >= MAX_BRACE_EXPANSIONS {
                return out;
            }
        }
    }
    out
}

fn simple_glob_match_single(pattern: &str, candidate: &str) -> bool {
    // Translate the glob to a regex-lite pattern by walking byte-by-byte.
    // `**` matches anything (including separators); `*` matches any chars
    // except `/`; `?` matches a single non-`/` char; everything else is
    // literal. Keep it simple: O(N*M) recursive descent.
    let pat: Vec<char> = pattern.chars().collect();
    let cand: Vec<char> = candidate.chars().collect();
    fn walk(pat: &[char], cand: &[char]) -> bool {
        let mut p = 0usize;
        let mut c = 0usize;
        while p < pat.len() {
            match pat[p] {
                '*' if pat.get(p + 1) == Some(&'*') => {
                    let rest = &pat[p + 2..];
                    for end in c..=cand.len() {
                        if walk(rest, &cand[end..]) {
                            return true;
                        }
                    }
                    return false;
                }
                '*' => {
                    let rest = &pat[p + 1..];
                    while c <= cand.len() {
                        if walk(rest, &cand[c..]) {
                            return true;
                        }
                        if c == cand.len() || cand[c] == '/' {
                            return false;
                        }
                        c += 1;
                    }
                    return false;
                }
                '?' => {
                    if c >= cand.len() || cand[c] == '/' {
                        return false;
                    }
                    c += 1;
                    p += 1;
                }
                ch => {
                    if c >= cand.len() || cand[c] != ch {
                        return false;
                    }
                    c += 1;
                    p += 1;
                }
            }
        }
        c == cand.len()
    }
    walk(&pat, &cand)
}

/// Directory names pruned before descent during a glob walk. These hold
/// build artefacts / dependency trees that a name-pattern search almost
/// never wants and that can balloon a result list into the megabytes —
/// enough to blow past the model's context window when the list rides back
/// into history. Pruned unconditionally (the walk never enters them), which
/// also keeps the walk fast on large repos.
pub const FS_GLOB_IGNORED_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    "dist",
    "build",
    "vendor",
    ".next",
    "__pycache__",
    ".venv",
];

/// Hard ceiling on paths returned by a single glob walk. Reaching it stops
/// the walk early (rather than collecting the whole tree and trimming after)
/// so a pathological pattern can't allocate an unbounded list before we cap
/// it. Callers learn whether the cap was hit via [`fs_glob_bounded`].
pub const MAX_FS_GLOB_RESULTS: usize = 2000;

/// Output size cap before spilling to a temporary file. When a tool produces
/// more bytes than this, runtimes write the full content to
/// `/tmp/harness_out_<call_id>_<suffix>.txt` and return a preview with the
/// path so the model can fetch the rest with the read tool if needed.
pub const MAX_OUTPUT_BYTES: usize = 50_000;

/// Tail bytes scanned for an error signature when bounding output. If the
/// failure the model needs is in the last chunk, head-only truncation would
/// drop it — so we detect it and keep a tail slice instead.
const TAIL_SCAN_BYTES: usize = 2048;

/// Case-insensitive substrings that mark a line worth preserving in the tail
/// of a truncated output (mirrors MiMoCode's truncation heuristic).
const ERROR_MARKERS: &[&str] = &[
    "error",
    "exception",
    "failed",
    "fatal",
    "panic",
    "traceback",
    "exit code",
];

/// Compute the preview for an over-budget tool output. Returns `None` when
/// `full` fits within [`MAX_OUTPUT_BYTES`] (caller uses it unchanged, no
/// spill). Otherwise returns a preview the caller should surface AFTER
/// writing `full` to `spill_path`:
/// * if the tail carries an error signature, keep head (70% budget) AND tail
///   (30%) so the failure survives the cut;
/// * else keep head only.
///
/// The preview ends with a hint pointing at the spilled file.
pub fn bounded_preview(full: &str, spill_path: &str) -> Option<String> {
    if full.len() <= MAX_OUTPUT_BYTES {
        return None;
    }
    Some(format!(
        "{}\n\n[{} bytes total, truncated. Full output saved to {spill_path} — \
use the read tool with offset/limit to fetch more.]",
        head_tail_body(full),
        full.len()
    ))
}

/// Error-aware head+tail clip for outputs that have NO spill file but blew
/// past a hard ceiling — the catch-all in [`bounded::BoundedToolRuntime`] for
/// tools (MCP, custom) that don't self-bound. Always returns a bounded
/// string with a note (no file reference).
pub fn clip_overflow(full: &str) -> String {
    format!(
        "{}\n\n[output clipped: {} bytes total exceeded the tool-output ceiling]",
        head_tail_body(full),
        full.len()
    )
}

/// Build the truncated body (no surrounding note): head+tail when the tail
/// carries an error signature so the failure survives, else head only.
fn head_tail_body(full: &str) -> String {
    let lines: Vec<&str> = full.split('\n').collect();
    if tail_has_error(full) {
        let head_budget = MAX_OUTPUT_BYTES * 7 / 10;
        let head = take_lines_head(&lines, head_budget);
        let tail = take_lines_tail(&lines, MAX_OUTPUT_BYTES - head_budget);
        let omitted = lines
            .len()
            .saturating_sub(head.len())
            .saturating_sub(tail.len());
        format!(
            "{}\n\n... {omitted} lines omitted — showing head and tail ...\n\n{}",
            head.join("\n"),
            tail.join("\n"),
        )
    } else {
        take_lines_head(&lines, MAX_OUTPUT_BYTES).join("\n")
    }
}

/// Head-only clip with a note, for outputs that have NO spill file (read
/// `content`, grep error text). Returns the string unchanged when within
/// budget; otherwise clips at a char boundary near [`MAX_OUTPUT_BYTES`].
pub fn clip_head(s: String) -> String {
    if s.len() <= MAX_OUTPUT_BYTES {
        return s;
    }
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[content truncated: use offset/limit to read more]",
        &s[..end]
    )
}

/// Does the last [`TAIL_SCAN_BYTES`] of `s` contain an error marker?
fn tail_has_error(s: &str) -> bool {
    let mut start = s.len().saturating_sub(TAIL_SCAN_BYTES);
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    let scan = s[start..].to_ascii_lowercase();
    ERROR_MARKERS.iter().any(|m| scan.contains(m))
}

/// Collect whole lines from the front until adding the next would exceed
/// `budget` bytes (counting the rejoining `\n`).
fn take_lines_head<'a>(lines: &[&'a str], budget: usize) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let cost = line.len() + usize::from(i > 0);
        if used + cost > budget {
            break;
        }
        out.push(*line);
        used += cost;
    }
    out
}

/// Collect whole lines from the back until adding the next would exceed
/// `budget` bytes; returned in original order.
fn take_lines_tail<'a>(lines: &[&'a str], budget: usize) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for line in lines.iter().rev() {
        let cost = line.len() + usize::from(!out.is_empty());
        if used + cost > budget {
            break;
        }
        out.push(*line);
        used += cost;
    }
    out.reverse();
    out
}

/// Walk `base_dir` recursively and return relative paths that match `pattern`.
/// Skips hidden directories (`.git`, `.DS_Store`, etc.) unless the pattern
/// explicitly starts with `.`, prunes dependency / build directories
/// ([`FS_GLOB_IGNORED_DIRS`]), and caps the result count at
/// [`MAX_FS_GLOB_RESULTS`]. Results are sorted lexicographically.
/// Intended for use by production `ToolRuntime` implementations.
///
/// When the caller needs to know whether the cap was reached (e.g. to tell
/// the model the list was truncated), use [`fs_glob_bounded`] — this wrapper
/// discards that flag for the common case.
pub fn fs_glob(pattern: &str, base_dir: &std::path::Path) -> Vec<String> {
    fs_glob_bounded(pattern, base_dir).0
}

/// Like [`fs_glob`] but also reports whether the [`MAX_FS_GLOB_RESULTS`] cap
/// was hit. A `true` second element means the returned list is a prefix of
/// the full match set and the search should be narrowed.
pub fn fs_glob_bounded(pattern: &str, base_dir: &std::path::Path) -> (Vec<String>, bool) {
    let mut matches = Vec::new();
    let mut truncated = false;
    let mut stack = vec![base_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let rel = match path.strip_prefix(base_dir) {
                Ok(r) => r.to_string_lossy().replace('\\', "/"),
                Err(_) => continue,
            };
            let first = rel.split('/').next().unwrap_or("");
            if first.starts_with('.') && !pattern.starts_with('.') {
                continue;
            }
            if !path.is_symlink() && path.is_dir() {
                // Prune dependency / build trees before descending so their
                // (often enormous) contents are never read at all.
                let name = entry.file_name();
                if FS_GLOB_IGNORED_DIRS.iter().any(|d| name.as_os_str() == *d) {
                    continue;
                }
                stack.push(path);
            } else if !path.is_dir() && simple_glob_match(pattern, &rel) {
                if matches.len() >= MAX_FS_GLOB_RESULTS {
                    truncated = true;
                    break;
                }
                matches.push(rel);
            }
        }
        if truncated {
            break;
        }
    }
    matches.sort();
    (matches, truncated)
}

/// Canonical specs for the built-in tool set (bash / read / write). Shared
/// between `MockToolRuntime` (in-process tests) and the production
/// `SandboxToolRuntime` (in `core` crate) so both report the same schema to
/// the model. Adding a new built-in tool changes one place.
///
/// Note on `additionalProperties: false`: keeps the model from inventing
/// fields. Required for some providers' strict tool-calling mode.
pub fn builtin_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command inside the sandbox working directory. \
                Returns structured command status + stdout/stderr, including non-zero \
                exits and timeouts. Bounded by `timeout_ms` \
                (default 120 000 ms, max 600 000 ms) — on timeout the process \
                is terminated and any captured output is returned. For commands \
                that may run longer than 10 min, use `nohup … &` writing to a \
                file, then poll the file with the read tool across turns."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute. Local runtimes prefer /bin/bash -lc when available and fall back to /bin/sh -lc."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds (default 120000, max 600000).",
                        "minimum": 1000,
                        "maximum": 600000
                    },
                    "soft_timeout_ms": {
                        "type": "integer",
                        "description": "Optional no-output timeout in milliseconds (default 10000). Streaming output resets this timer.",
                        "minimum": 1000,
                        "maximum": 600000
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "read".into(),
            description:
                "Read a UTF-8 file from the sandbox. Paginated by line: returns up to `limit` \
                 lines starting at `offset` (a 0-based line index). When the result is \
                 `truncated`, read the next page with the returned `next_offset`. Overlong \
                 lines are clipped."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "0-based line index to start from. Default 0.",
                        "minimum": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to return. Default 2000.",
                        "minimum": 1
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Write UTF-8 content to a file in the sandbox.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description:
                "Edit a UTF-8 file by replacing an exact substring. By default `old_string` must \
                 appear exactly once; set `replace_all=true` to substitute every occurrence."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {
                        "type": "string",
                        "description": "Substring to replace; must match verbatim including whitespace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. Empty string deletes the match."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "When true, replace every occurrence. Default false (must be unique)."
                    }
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "grep".into(),
            description:
                "Search file contents under a path with an extended-regex pattern. Returns \
                 matching lines as `path:line:text`. Uses ripgrep when available (honouring \
                 .gitignore and skipping hidden files), otherwise falls back to system \
                 `grep -rnE` with dependency/build directories (node_modules, target, …) pruned. \
                 The match count is capped — a `truncated` flag signals when to narrow the \
                 pattern or path."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for (passed to grep)."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search under. Default: current cwd."
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "When true, pass -i to grep. Default false."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "glob".into(),
            description:
                "Find files matching a shell-style glob (e.g. `*.rs`, `**/Cargo.toml`), searched \
                 recursively and honouring .gitignore/.ignore. A slash-less pattern like `*.rs` \
                 matches by file name at ANY depth; anchor with a `/`-bearing pattern (e.g. \
                 `src/*.rs`) to restrict to one directory level. Hidden files are searched only \
                 when the pattern itself starts with `.`. Returns relative file paths under the \
                 search root, one per line. The result count is capped — a `truncated` flag \
                 signals when to narrow the pattern or search a subdirectory."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Shell glob like `*.rs` or `**/Cargo.toml`."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search under. Default: current cwd."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "web_fetch".into(),
            description:
                "Fetch a known HTTP/HTTPS URL and return readable content. This is read-only \
                 and does not search the web; use it when the user supplies a URL or another \
                 tool has produced URLs. HTML can be returned as markdown, plain text, or raw HTML."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to fetch."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "text", "html"],
                        "description": "Return format. Defaults to markdown."
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters of content to return (default 50000, max 200000).",
                        "minimum": 1,
                        "maximum": 200000
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Request timeout in milliseconds (default 20000, max 60000).",
                        "minimum": 1000,
                        "maximum": 60000
                    }
                },
                "required": ["url"],
                "additionalProperties": false
            }),
        },
    ]
}

fn required_str<'a>(
    invocation: &'a ToolInvocation,
    key: &str,
) -> Result<&'a str, ToolRuntimeError> {
    invocation
        .input
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ToolRuntimeError::InvalidInput {
            tool: invocation.name.clone(),
            message: format!("missing string field {key}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_preview_none_when_within_budget() {
        assert!(bounded_preview("short output", "/tmp/x.txt").is_none());
    }

    #[test]
    fn bounded_preview_head_only_drops_tail_without_error() {
        let mut s = String::from("HEAD_MARKER\n");
        // ~60KB of innocuous lines, no error markers anywhere.
        while s.len() < MAX_OUTPUT_BYTES + 10_000 {
            s.push_str("padding line of plain text\n");
        }
        s.push_str("LAST_LINE_NO_MARKER");
        let preview = bounded_preview(&s, "/tmp/out.txt").expect("over budget");
        assert!(preview.contains("HEAD_MARKER"));
        assert!(
            !preview.contains("LAST_LINE_NO_MARKER"),
            "tail leaked in head-only mode"
        );
        assert!(preview.contains("/tmp/out.txt"));
        assert!(preview.contains("truncated"));
    }

    #[test]
    fn bounded_preview_preserves_error_in_tail() {
        let mut s = String::from("HEAD_MARKER\n");
        while s.len() < MAX_OUTPUT_BYTES + 10_000 {
            s.push_str("padding line of plain text\n");
        }
        s.push_str("ERROR: the build failed at the end");
        let preview = bounded_preview(&s, "/tmp/out.txt").expect("over budget");
        // Head+tail mode: both the head AND the trailing error survive.
        assert!(preview.contains("HEAD_MARKER"));
        assert!(preview.contains("ERROR: the build failed at the end"));
        assert!(preview.contains("omitted"));
    }

    #[test]
    fn clip_head_passes_short_strings_through() {
        assert_eq!(clip_head("hi".into()), "hi");
    }

    #[test]
    fn simple_glob_matches_star_and_doublestar() {
        assert!(simple_glob_match("*.rs", "main.rs"));
        assert!(!simple_glob_match("*.rs", "main.rs.bak"));
        assert!(!simple_glob_match("*.rs", "src/main.rs"));
        assert!(simple_glob_match("**/*.rs", "src/main.rs"));
        assert!(simple_glob_match("**/*.rs", "a/b/c.rs"));
        assert!(simple_glob_match("Cargo.toml", "Cargo.toml"));
        assert!(!simple_glob_match("Cargo.toml", "Cargo.lock"));
    }

    #[test]
    fn simple_glob_matches_brace_alternation() {
        // The exact shape models emit reflexively.
        assert!(simple_glob_match("**/*.{ts,tsx}", "src/main.ts"));
        assert!(simple_glob_match("**/*.{ts,tsx}", "src/components/App.tsx"));
        assert!(!simple_glob_match("**/*.{ts,tsx}", "src/main.rs"));
        // Multiple groups multiply out.
        assert!(simple_glob_match("{src,lib}/*.{ts,js}", "lib/util.js"));
        assert!(!simple_glob_match("{src,lib}/*.{ts,js}", "bin/util.js"));
        // Nested groups.
        assert!(simple_glob_match("*.{t{s,sx}}", "x.tsx"));
        assert!(simple_glob_match("*.{t{s,sx}}", "x.ts"));
        // Single alternative and empty alternative.
        assert!(simple_glob_match("*.{rs}", "main.rs"));
        assert!(simple_glob_match("a{,b}c", "ac"));
        assert!(simple_glob_match("a{,b}c", "abc"));
        // Unbalanced brace stays a literal.
        assert!(simple_glob_match("a{b", "a{b"));
        assert!(!simple_glob_match("a{b", "ab"));
    }

    #[test]
    fn expand_braces_caps_pathological_patterns() {
        // 4 groups × 4 alts = 256 > cap; must stop at the cap, not hang.
        let pat = "{a,b,c,d}{a,b,c,d}{a,b,c,d}{a,b,c,d}";
        assert_eq!(expand_braces(pat).len(), MAX_BRACE_EXPANSIONS);
    }

    #[tokio::test]
    async fn mock_runtime_edit_replaces_unique_substring() {
        let rt = MockToolRuntime::new().with_file("a.txt", "hello world");
        let out = rt
            .invoke(ToolInvocation {
                id: "tc_edit".into(),
                name: "edit".into(),
                input: json!({
                    "path": "a.txt",
                    "old_string": "world",
                    "new_string": "rust",
                }),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(out["replaced"], 1);
        // Confirm new contents readable.
        let after = rt
            .invoke(ToolInvocation {
                id: "tc_read".into(),
                name: "read".into(),
                input: json!({"path": "a.txt"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(after["content"], "hello rust");
    }

    #[tokio::test]
    async fn mock_runtime_edit_rejects_ambiguous_match() {
        let rt = MockToolRuntime::new().with_file("a.txt", "foo foo");
        let failure = rt
            .invoke(ToolInvocation {
                id: "tc_edit".into(),
                name: "edit".into(),
                input: json!({"path": "a.txt", "old_string": "foo", "new_string": "bar"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap_err();
        assert_eq!(failure.kind, ToolFailureKind::InvalidInput);
    }

    // ── F7: JSON-escape auto-repair ──

    #[test]
    fn unescape_literal_controls_handles_sequences() {
        assert_eq!(unescape_literal_controls(r"a\nb"), "a\nb");
        assert_eq!(unescape_literal_controls(r"a\tb"), "a\tb");
        assert_eq!(unescape_literal_controls(r"a\rb"), "a\rb");
        // \r\n collapses to a single newline (ordered before \r / \n).
        assert_eq!(unescape_literal_controls(r"a\r\nb"), "a\nb");
        // Double backslash is NOT special-cased (Go Replacer semantics):
        // `\\n` → `\` + newline.
        assert_eq!(unescape_literal_controls(r"a\\nb"), "a\\\nb");
        // No escapes → unchanged.
        assert_eq!(unescape_literal_controls("plain"), "plain");
    }

    #[test]
    fn resolve_edit_search_prefers_direct_match() {
        // Content contains the literal two-char sequence; direct match
        // wins and no repair fires.
        let r = resolve_edit_search("say \\n here", r"\n", "x", false).unwrap();
        assert!(r.repair.is_none());
        assert_eq!(r.old_string, r"\n");
    }

    #[test]
    fn resolve_edit_search_unescapes_literal_controls() {
        let r = resolve_edit_search("line1\nline2", r"line1\nline2", r"a\tb", false).unwrap();
        assert_eq!(r.repair, Some("json_escape_unwrapped"));
        assert_eq!(r.old_string, "line1\nline2");
        assert_eq!(r.new_string, "a\tb"); // new_string unescaped too
        assert_eq!(r.occurrences, 1);
    }

    #[test]
    fn resolve_edit_search_escaped_not_found() {
        assert_eq!(
            resolve_edit_search("other", r"line1\nline2", "x", false).unwrap_err(),
            EditSearchError::EscapedNotFound
        );
    }

    #[test]
    fn resolve_edit_search_escaped_ambiguous_without_replace_all() {
        let content = "a\nb a\nb";
        assert_eq!(
            resolve_edit_search(content, r"a\nb", "x", false).unwrap_err(),
            EditSearchError::EscapedAmbiguous { occurrences: 2 }
        );
        // replace_all=true accepts the multi-match.
        let r = resolve_edit_search(content, r"a\nb", "x", true).unwrap();
        assert_eq!(r.occurrences, 2);
        assert_eq!(r.repair, Some("json_escape_unwrapped"));
    }

    #[tokio::test]
    async fn mock_runtime_edit_repairs_json_escaped_old_string() {
        let rt = MockToolRuntime::new().with_file("a.txt", "line1\nline2\nline3");
        let out = rt
            .invoke(ToolInvocation {
                id: "tc_edit".into(),
                name: "edit".into(),
                // Model emitted literal \n instead of a real newline.
                input: json!({"path": "a.txt", "old_string": "line1\\nline2", "new_string": "merged"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(out["replaced"], 1);
        // Repair is silent: the success output must NOT surface a `repair`
        // field to the model (MiMoCode "success silent" policy).
        assert!(
            out.get("repair").is_none(),
            "repair leaked into output: {out}"
        );
        let after = rt
            .invoke(ToolInvocation {
                id: "tc_read".into(),
                name: "read".into(),
                input: json!({"path": "a.txt"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        assert_eq!(after["content"], "merged\nline3");
    }

    #[tokio::test]
    async fn mock_runtime_grep_finds_matches() {
        let rt = MockToolRuntime::new()
            .with_file("a.txt", "alpha\nbeta\nALPHA")
            .with_file("b.txt", "gamma");
        let out = rt
            .invoke(ToolInvocation {
                id: "tc_grep".into(),
                name: "grep".into(),
                input: json!({"pattern": "alpha", "case_insensitive": true}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[tokio::test]
    async fn mock_runtime_glob_matches_by_pattern() {
        let rt = MockToolRuntime::new()
            .with_file("src/main.rs", "")
            .with_file("src/lib.rs", "")
            .with_file("Cargo.toml", "");
        let out = rt
            .invoke(ToolInvocation {
                id: "tc_glob".into(),
                name: "glob".into(),
                input: json!({"pattern": "**/*.rs"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap()
            .output
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[tokio::test]
    async fn mock_runtime_supports_bash_read_write() {
        let rt = MockToolRuntime::new().with_file("README.md", "hello");
        let read = rt
            .invoke(ToolInvocation {
                id: "tc_read".into(),
                name: "read".into(),
                input: json!({"path":"README.md"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(read.output.unwrap()["content"], "hello");

        let write = rt
            .invoke(ToolInvocation {
                id: "tc_write".into(),
                name: "write".into(),
                input: json!({"path":"out.txt", "content":"ok"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(write.output.unwrap()["written"], true);

        let bash = rt
            .invoke(ToolInvocation {
                id: "tc_bash".into(),
                name: "bash".into(),
                input: json!({"command":"pwd"}),
                raw_emitted_args: None,
            })
            .await
            .unwrap();
        assert_eq!(bash.output.unwrap()["exit_code"], 0);
    }

    #[test]
    fn fs_glob_prunes_dependency_dirs() {
        // Regression: a wide glob must NOT descend into node_modules/target,
        // whose contents previously ballooned a result list into the
        // megabytes and blew past the model's context window.
        use std::fs;
        let root = std::env::temp_dir().join(format!("harness_fsglob_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for sub in ["src", "node_modules/dep", "target/debug"] {
            fs::create_dir_all(root.join(sub)).unwrap();
        }
        fs::write(root.join("keep.rs"), "").unwrap();
        fs::write(root.join("src/lib.rs"), "").unwrap();
        fs::write(root.join("node_modules/dep/skip.rs"), "").unwrap();
        fs::write(root.join("target/debug/skip.rs"), "").unwrap();

        let (matches, truncated) = fs_glob_bounded("**.rs", &root);
        let _ = fs::remove_dir_all(&root);

        assert!(!truncated);
        assert!(matches.iter().any(|m| m == "keep.rs"), "{matches:?}");
        assert!(matches.iter().any(|m| m == "src/lib.rs"), "{matches:?}");
        assert!(
            !matches
                .iter()
                .any(|m| m.contains("node_modules") || m.contains("target")),
            "pruned dirs leaked into results: {matches:?}"
        );
    }
}
