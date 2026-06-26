//! Static risk classification for shell commands.
//!
//! [`classify_shell_command`] inspects a bash command *before* it is
//! dispatched to the sandbox and assigns one of four risk levels:
//!
//!   * [`ShellRiskLevel::SafeRead`] — provably read-only (display/inspect
//!     commands, read-only git, pipelines/lists composed solely of such
//!     commands). Safe to auto-approve as a read.
//!   * [`ShellRiskLevel::BoundedWrite`] — writes, but only to well-known
//!     project-local artifact locations (build/test caches: `cargo test`,
//!     `go build`, `npm test`, …).
//!   * [`ShellRiskLevel::NeedsApproval`] — everything we cannot statically
//!     prove safe. This is the *default*: any parse failure, any shell
//!     metacharacter we don't model, any unknown command lands here.
//!   * [`ShellRiskLevel::Blocked`] — a short hard-deny list of operations
//!     that would destroy the sandbox session itself (recursive deletion of
//!     critical system paths, raw block-device writes, shutdown/reboot,
//!     killing PID 1, fork bombs). These are rejected before dispatch.
//!
//! Design notes:
//!
//!   * Commands run inside a disposable sandbox, so the deny list is NOT a
//!     host-security boundary (sandbox isolation is). It exists to stop a
//!     weak model from accidentally killing its own session with an
//!     unrecoverable command. It is deliberately short and precise; anything
//!     ambiguous falls to `NeedsApproval` instead of `Blocked`.
//!   * Classification is purely static and conservative. The safe-side
//!     analysis refuses anything it cannot fully parse (command
//!     substitution, globs, redirects, `||` chains — whose right side only
//!     runs on failure and therefore can't be proven read-only statically).
//!   * The deny-side analysis is the opposite: it scans best-effort through
//!     constructs the safe side refuses (subshells, `&&`/`||`/`;`/`|`
//!     chains, `sh -c '…'` wrappers) so destructive segments can't hide
//!     behind syntax that merely fails the safe parse.

/// Environment variable used by [`classify_shell_command`] to select the shell
/// risk policy.
///
/// Set to `relaxed`, `lenient`, or `permissive` to bypass conservative
/// read-only classification while still enforcing hard-deny checks. Any other
/// value, or an unset variable, uses [`ShellRiskPolicy::Strict`].
pub const SHELL_RISK_POLICY_ENV: &str = "AGENT_HARNESS_SHELL_RISK_POLICY";

/// Policy used for static shell-risk classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellRiskPolicy {
    /// Preserve the conservative static classifier.
    Strict,
    /// Temporarily bypass conservative read-only checks while keeping hard-deny
    /// protection for session-destroying commands.
    Relaxed,
}

impl ShellRiskPolicy {
    /// Resolve the policy from [`SHELL_RISK_POLICY_ENV`].
    pub fn from_env() -> Self {
        match std::env::var(SHELL_RISK_POLICY_ENV) {
            Ok(value) if shell_risk_policy_value_is_relaxed(&value) => ShellRiskPolicy::Relaxed,
            _ => ShellRiskPolicy::Strict,
        }
    }
}

/// Risk level assigned to a shell command by [`classify_shell_command`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellRiskLevel {
    /// Provably read-only; safe to auto-approve.
    SafeRead,
    /// Writes only to well-known project-local build/test artifacts.
    BoundedWrite,
    /// Cannot be statically proven safe (the default).
    NeedsApproval,
    /// On the hard-deny list; must not be dispatched.
    Blocked,
}

impl ShellRiskLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShellRiskLevel::SafeRead => "safe_read",
            ShellRiskLevel::BoundedWrite => "bounded_write",
            ShellRiskLevel::NeedsApproval => "needs_approval",
            ShellRiskLevel::Blocked => "blocked",
        }
    }
}

/// Classification result: a level plus a human-readable reason suitable for
/// logging and for surfacing to the model when a command is blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellRiskDecision {
    pub level: ShellRiskLevel,
    pub reason: String,
}

fn safe_read(reason: &str) -> ShellRiskDecision {
    ShellRiskDecision {
        level: ShellRiskLevel::SafeRead,
        reason: reason.to_string(),
    }
}

fn bounded_write(reason: String) -> ShellRiskDecision {
    ShellRiskDecision {
        level: ShellRiskLevel::BoundedWrite,
        reason,
    }
}

fn needs_approval(reason: &str) -> ShellRiskDecision {
    ShellRiskDecision {
        level: ShellRiskLevel::NeedsApproval,
        reason: reason.to_string(),
    }
}

fn blocked(reason: &str) -> ShellRiskDecision {
    ShellRiskDecision {
        level: ShellRiskLevel::Blocked,
        reason: reason.to_string(),
    }
}

/// Classify a shell command's static risk. See module docs for the levels.
pub fn classify_shell_command(command: &str) -> ShellRiskDecision {
    classify_shell_command_with_policy(command, ShellRiskPolicy::from_env())
}

/// Classify a shell command's static risk with an explicit policy.
pub fn classify_shell_command_with_policy(
    command: &str,
    policy: ShellRiskPolicy,
) -> ShellRiskDecision {
    if let Some(decision) = hard_deny(command) {
        return decision;
    }
    match policy {
        ShellRiskPolicy::Strict => classify_allowable(command),
        ShellRiskPolicy::Relaxed => safe_read("shell risk policy relaxed"),
    }
}

fn shell_risk_policy_value_is_relaxed(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "relaxed" | "lenient" | "permissive"
    )
}

// ---------------------------------------------------------------------------
// Hard deny: session-destroying operations
// ---------------------------------------------------------------------------

/// System paths whose recursive removal (or permission sweep) kills the
/// sandbox session outright.
const CRITICAL_SYSTEM_PATHS: &[&str] = &[
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/sbin", "/sys", "/usr", "/var",
];

/// Raw block/memory device prefixes a redirect or `dd of=` must never target.
const RAW_DEVICE_PREFIXES: &[&str] = &[
    "/dev/sd",
    "/dev/vd",
    "/dev/xvd",
    "/dev/hd",
    "/dev/nvme",
    "/dev/mem",
    "/dev/kmem",
    "/dev/port",
];

/// Device paths that are always fine as a write target.
const SAFE_DEVICE_SINKS: &[&str] = &["/dev/null", "/dev/stdout", "/dev/stderr"];

fn hard_deny(command: &str) -> Option<ShellRiskDecision> {
    if looks_like_fork_bomb(command) {
        return Some(blocked(
            "fork bomb pattern would exhaust sandbox PIDs and make the session unresponsive",
        ));
    }
    for segment in deny_scan_segments(command) {
        if let Some(decision) = deny_scan_segment(&segment) {
            return Some(decision);
        }
    }
    None
}

/// Best-effort split of a raw command line into candidate simple commands for
/// deny scanning. Unlike the conservative safe-side splitters, this one keeps
/// going through subshells, command substitution, `&`, and newlines: a
/// destructive segment must be found wherever it hides, even in syntax the
/// safe side refuses to parse.
fn deny_scan_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let flush = |current: &mut String, segments: &mut Vec<String>| {
        let part = current.trim();
        if !part.is_empty() {
            segments.push(part.to_string());
        }
        current.clear();
    };
    for ch in command.chars() {
        if let Some(q) = quote {
            // Inside quotes the content is preserved verbatim into the
            // current segment (the word splitter strips quotes later, so a
            // `sh -c '…'` body stays scannable).
            current.push(ch);
            if q == '"' {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    quote = None;
                }
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        if escaped {
            escaped = false;
            current.push(ch);
            continue;
        }
        match ch {
            '\\' => {
                escaped = true;
                current.push(ch);
            }
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            ';' | '|' | '&' | '\n' | '(' | ')' | '`' => flush(&mut current, &mut segments),
            _ => current.push(ch),
        }
    }
    flush(&mut current, &mut segments);
    segments
}

/// Split a deny-scan segment into words. Quote-aware (quotes are stripped,
/// content kept); unquoted `>` runs become standalone `">"` tokens so
/// redirect targets show up as the following word.
fn deny_scan_words(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let chars: Vec<char> = segment.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if let Some(q) = quote {
            if q == '"' {
                if escaped {
                    escaped = false;
                    word.push(ch);
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    quote = None;
                } else {
                    word.push(ch);
                }
            } else if ch == q {
                quote = None;
            } else {
                word.push(ch);
            }
            i += 1;
            continue;
        }
        if escaped {
            escaped = false;
            word.push(ch);
            i += 1;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '\'' | '"' => {
                quote = Some(ch);
                in_word = true;
            }
            ' ' | '\t' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '>' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
                words.push(">".to_string());
                // Collapse `>>` into a single redirect token.
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    i += 1;
                }
            }
            _ => {
                in_word = true;
                word.push(ch);
            }
        }
        i += 1;
    }
    if in_word {
        words.push(word);
    }
    words
}

/// Drop leading wrappers (`sudo`, `env VAR=… `, `nohup`, `timeout 30`, …) so
/// the real command name lands at index 0.
fn strip_command_wrappers(words: &[String]) -> Vec<String> {
    let mut rest: &[String] = words;
    loop {
        let Some(first) = rest.first() else {
            return Vec::new();
        };
        match command_basename(first).as_str() {
            "sudo" | "doas" => {
                rest = &rest[1..];
                while rest.first().is_some_and(|w| w.starts_with('-')) {
                    rest = &rest[1..];
                }
            }
            "env" => {
                rest = &rest[1..];
                while rest
                    .first()
                    .is_some_and(|w| w.contains('=') || w.starts_with('-'))
                {
                    rest = &rest[1..];
                }
            }
            "nohup" | "command" | "exec" | "time" | "nice" | "ionice" | "stdbuf" => {
                rest = &rest[1..];
                while rest.first().is_some_and(|w| w.starts_with('-')) {
                    rest = &rest[1..];
                }
            }
            "timeout" => {
                rest = &rest[1..];
                while rest.first().is_some_and(|w| w.starts_with('-')) {
                    rest = &rest[1..];
                }
                // Skip the duration operand.
                if !rest.is_empty() {
                    rest = &rest[1..];
                }
            }
            _ => return rest.to_vec(),
        }
    }
}

fn command_basename(word: &str) -> String {
    word.rsplit('/').next().unwrap_or(word).to_lowercase()
}

fn deny_scan_segment(segment: &str) -> Option<ShellRiskDecision> {
    let words = strip_command_wrappers(&deny_scan_words(segment));
    let cmd = command_basename(words.first()?);
    let args = &words[1..];

    // A `sh -c '…'` wrapper executes its quoted argument: scan it too.
    if matches!(cmd.as_str(), "sh" | "bash" | "zsh" | "dash" | "ksh") {
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if arg.starts_with('-') && arg.contains('c') {
                if let Some(script) = iter.next() {
                    if let Some(decision) = hard_deny(script) {
                        return Some(decision);
                    }
                }
                break;
            }
        }
    }

    // Redirect straight into a raw device bricks the filesystem.
    let mut expect_redirect_target = false;
    for word in &words {
        if word == ">" {
            expect_redirect_target = true;
            continue;
        }
        if std::mem::take(&mut expect_redirect_target) && is_raw_device_path(word) {
            return Some(blocked(
                "redirecting output to a raw device would corrupt the sandbox filesystem",
            ));
        }
    }

    match cmd.as_str() {
        "rm" => deny_check_rm(args),
        "chmod" | "chown" | "chgrp" => deny_check_permission_sweep(&cmd, args),
        "mkswap" | "wipefs" | "blkdiscard" => Some(blocked(
            "filesystem/block-device destruction would brick the sandbox",
        )),
        "fdisk" | "parted" | "sgdisk" => {
            let listing_only = args.iter().any(|a| a == "-l" || a == "--list");
            if listing_only {
                None
            } else {
                Some(blocked(
                    "partition-table manipulation would brick the sandbox",
                ))
            }
        }
        "dd" => {
            for arg in args {
                if let Some(target) = arg.strip_prefix("of=") {
                    if target.starts_with("/dev/") && !SAFE_DEVICE_SINKS.contains(&target) {
                        return Some(blocked(
                            "dd writing to a raw device would corrupt the sandbox filesystem",
                        ));
                    }
                }
            }
            None
        }
        "shutdown" | "reboot" | "halt" | "poweroff" | "telinit" => {
            Some(blocked("shutting down the sandbox terminates the session"))
        }
        "init" => {
            if args.iter().any(|a| a == "0" || a == "6") {
                Some(blocked(
                    "changing the runlevel to halt/reboot terminates the session",
                ))
            } else {
                None
            }
        }
        "systemctl" => {
            let sub = args.iter().find(|a| !a.starts_with('-'));
            if sub.is_some_and(|s| matches!(s.as_str(), "reboot" | "poweroff" | "halt" | "kexec")) {
                Some(blocked("shutting down the sandbox terminates the session"))
            } else {
                None
            }
        }
        "kill" => deny_check_kill(args),
        "killall5" => Some(blocked(
            "signalling every process kills the sandbox session",
        )),
        _ => {
            if cmd.starts_with("mkfs") {
                return Some(blocked(
                    "creating a filesystem over an existing device would brick the sandbox",
                ));
            }
            None
        }
    }
}

fn deny_check_rm(args: &[String]) -> Option<ShellRiskDecision> {
    let mut recursive = false;
    let mut no_preserve_root = false;
    let mut operands: Vec<&String> = Vec::new();
    let mut end_of_options = false;
    for arg in args {
        if end_of_options {
            operands.push(arg);
            continue;
        }
        if arg == "--" {
            end_of_options = true;
        } else if arg == "--recursive" {
            recursive = true;
        } else if arg == "--no-preserve-root" {
            no_preserve_root = true;
        } else if let Some(short) = arg.strip_prefix('-') {
            if !short.starts_with('-') && short.chars().any(|c| c == 'r' || c == 'R') {
                recursive = true;
            }
        } else {
            operands.push(arg);
        }
    }
    if !recursive {
        return None;
    }
    if no_preserve_root {
        return Some(blocked(
            "rm --no-preserve-root with recursion would destroy the sandbox session",
        ));
    }
    if operands.iter().any(|p| is_critical_system_path(p)) {
        return Some(blocked(
            "recursive deletion of a critical system path would destroy the sandbox session",
        ));
    }
    None
}

fn deny_check_permission_sweep(cmd: &str, args: &[String]) -> Option<ShellRiskDecision> {
    let mut recursive = false;
    let mut operands: Vec<&String> = Vec::new();
    let mut end_of_options = false;
    for arg in args {
        if end_of_options {
            operands.push(arg);
            continue;
        }
        if arg == "--" {
            end_of_options = true;
        } else if arg == "--recursive" {
            recursive = true;
        } else if let Some(short) = arg.strip_prefix('-') {
            if !short.starts_with('-') && short.chars().any(|c| c == 'R' || c == 'r') {
                recursive = true;
            }
        } else {
            operands.push(arg);
        }
    }
    if recursive && operands.iter().any(|p| is_critical_system_path(p)) {
        return Some(blocked(
            // chmod -R 000 /usr is as fatal as deleting it.
            match cmd {
                "chmod" => "recursive permission sweep over a critical system path would destroy the sandbox session",
                _ => "recursive ownership sweep over a critical system path would destroy the sandbox session",
            },
        ));
    }
    None
}

fn deny_check_kill(args: &[String]) -> Option<ShellRiskDecision> {
    let mut saw_signal = false;
    let mut end_of_options = false;
    for arg in args {
        if !end_of_options && arg == "--" {
            end_of_options = true;
            continue;
        }
        if !end_of_options && arg.starts_with('-') {
            // The first dash argument is the signal spec (`-9`, `-TERM`,
            // `-s`); a later `-1` is process group "everything".
            if saw_signal && arg == "-1" {
                return Some(blocked(
                    "kill -1 signals every process and kills the sandbox session",
                ));
            }
            saw_signal = true;
            continue;
        }
        if arg == "1" || arg == "-1" {
            return Some(blocked("killing PID 1 terminates the sandbox session"));
        }
    }
    None
}

fn is_critical_system_path(path: &str) -> bool {
    let trimmed = path.trim();
    // `/usr/`, `/usr/*` and `/usr` are all the same target.
    let stripped = trimmed.strip_suffix("/*").unwrap_or(trimmed);
    let normalized = if stripped.len() > 1 {
        stripped.trim_end_matches('/')
    } else {
        stripped
    };
    if normalized == "/" || normalized == "/*" || trimmed == "/*" {
        return true;
    }
    CRITICAL_SYSTEM_PATHS.contains(&normalized)
}

fn is_raw_device_path(path: &str) -> bool {
    RAW_DEVICE_PREFIXES
        .iter()
        .any(|prefix| path.starts_with(prefix))
}

/// Detect the classic fork-bomb shape: a function whose body pipes itself
/// into itself in the background (`:(){ :|:& };:` and renamed variants).
fn looks_like_fork_bomb(command: &str) -> bool {
    let compact: String = command.chars().filter(|c| !c.is_whitespace()).collect();
    let Some(def_at) = compact.find("(){") else {
        return false;
    };
    let name: String = compact[..def_at]
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if name.is_empty() {
        return false;
    }
    let body = &compact[def_at + 3..];
    body.contains(&format!("{name}|{name}")) && body.contains('&')
}

// ---------------------------------------------------------------------------
// Conservative safe-side classification
// ---------------------------------------------------------------------------

/// Command prefixes (matched word-by-word) that are read-only by nature.
const READ_ONLY_PREFIXES: &[&str] = &[
    "ls",
    "pwd",
    "echo",
    "cat",
    "head",
    "tail",
    "wc",
    "file",
    "tree",
    "find",
    "grep",
    "rg",
    "uptime",
    "cal",
    "free",
    "df",
    "du",
    "locale",
    "groups",
    "nproc",
    "stat",
    "strings",
    "hexdump",
    "od",
    "nl",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "cut",
    "paste",
    "tr",
    "column",
    "tac",
    "rev",
    "fold",
    "expand",
    "unexpand",
    "comm",
    "cmp",
    "numfmt",
    "true",
    "false",
    "type",
    "expr",
    "test",
    "getconf",
    "seq",
    "tsort",
    "pr",
    "go version",
    "rustc --version",
    "python --version",
    "python3 --version",
    "node --version",
    "npm --version",
    "npx --version",
    "cargo --version",
    "deno --version",
    "bun --version",
];

fn classify_allowable(command: &str) -> ShellRiskDecision {
    // A trailing stderr redirect to stdout or /dev/null doesn't change the
    // risk of the underlying command.
    if let Some(base) = strip_trailing_safe_stderr_redirect(command) {
        return classify_allowable(&base);
    }
    if let Some(parts) = split_sequence(command) {
        return classify_all_safe_read(&parts, "; list");
    }
    if let Some(parts) = split_and_list(command) {
        return classify_all_safe_read(&parts, "&& list");
    }
    if let Some(parts) = split_pipeline(command) {
        return classify_all_safe_read(&parts, "pipeline");
    }
    let Some(argv) = parse_simple_command(command) else {
        return needs_approval("command is not a simple shell command");
    };
    let lower: Vec<String> = argv.iter().map(|a| a.to_lowercase()).collect();
    if has_unsafe_args(&lower) {
        return needs_approval(
            "command contains arguments that may mutate files or execute arbitrary code",
        );
    }
    if make_bounded_target_has_extra_args(&lower) {
        return needs_approval("make bounded-write targets must not include extra targets or args");
    }
    if let Some(decision) = classify_builtin_read_only(&argv, &lower) {
        return decision;
    }
    if lower[0] == "git" {
        if git_command_read_only(&argv) {
            return safe_read("git read-only command");
        }
        return needs_approval("git command is not classified as read-only");
    }
    if let Some(decision) = classify_bounded_write(&lower) {
        return decision;
    }
    for prefix in READ_ONLY_PREFIXES {
        if argv_has_prefix(&lower, prefix) {
            return safe_read("built-in read-only command");
        }
    }
    needs_approval("command is not classified as safe read-only or bounded-write")
}

/// `;`/`&&`/`|` lists are only safe when every element independently is.
fn classify_all_safe_read(parts: &[String], kind: &str) -> ShellRiskDecision {
    for part in parts {
        let decision = classify_shell_command(part);
        if decision.level == ShellRiskLevel::Blocked {
            return decision;
        }
        if decision.level != ShellRiskLevel::SafeRead {
            return needs_approval(&format!(
                "{kind} contains a command that is not safe read-only"
            ));
        }
    }
    safe_read(&format!("{kind} of read-only commands"))
}

// --- splitters -------------------------------------------------------------

/// Shared quote-aware scanner: splits `command` at unquoted occurrences of
/// the separator. Returns `None` when the input has unbalanced quotes, a
/// dangling escape, no separator at all, or an empty element.
fn split_unquoted(command: &str, separator: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut saw_separator = false;
    let runes: Vec<char> = command.trim().chars().collect();
    let sep: Vec<char> = separator.chars().collect();
    let mut i = 0;
    while i < runes.len() {
        let r = runes[i];
        if quote == Some('\'') {
            if r == '\'' {
                quote = None;
            }
            current.push(r);
            i += 1;
            continue;
        }
        if escaped {
            escaped = false;
            current.push(r);
            i += 1;
            continue;
        }
        match r {
            '\\' => {
                if quote == Some('"') {
                    escaped = true;
                }
                current.push(r);
            }
            '"' => {
                if quote.is_none() {
                    quote = Some('"');
                } else if quote == Some('"') {
                    quote = None;
                }
                current.push(r);
            }
            '\'' => {
                if quote.is_none() {
                    quote = Some('\'');
                }
                current.push(r);
            }
            _ if r == sep[0] && quote.is_none() => {
                // Multi-char separators (`&&`) must match in full; a lone
                // `&` stays part of the element (and later fails the simple
                // parse, landing the whole command at NeedsApproval).
                if sep.len() == 2 {
                    if i + 1 >= runes.len() || runes[i + 1] != sep[1] {
                        current.push(r);
                        i += 1;
                        continue;
                    }
                    i += 1;
                }
                let part = current.trim().to_string();
                current.clear();
                if part.is_empty() {
                    return None;
                }
                parts.push(part);
                saw_separator = true;
            }
            _ => current.push(r),
        }
        i += 1;
    }
    if quote.is_some() || escaped || !saw_separator {
        return None;
    }
    let part = current.trim().to_string();
    if part.is_empty() {
        return None;
    }
    parts.push(part);
    Some(parts)
}

/// Split on `;` only.
fn split_sequence(command: &str) -> Option<Vec<String>> {
    split_unquoted(command, ";")
}

/// Split on `&&` only. `||` is intentionally not handled: its right side
/// runs only when the left side *fails*, so an `||` chain can't be proven
/// read-only statically.
fn split_and_list(command: &str) -> Option<Vec<String>> {
    split_unquoted(command, "&&")
}

/// Split on `|`. Note `a || b` splits into an empty middle element and is
/// rejected, falling through to the (failing) simple parse — by design.
fn split_pipeline(command: &str) -> Option<Vec<String>> {
    split_unquoted(command, "|")
}

// --- trailing stderr redirect ----------------------------------------------

fn strip_trailing_safe_stderr_redirect(command: &str) -> Option<String> {
    let trimmed = command.trim();
    for redirect in ["2>&1", "2>/dev/null", "2> /dev/null"] {
        if let Some(base) = strip_trailing_redirect(trimmed, redirect) {
            return Some(base);
        }
    }
    None
}

fn strip_trailing_redirect(command: &str, redirect: &str) -> Option<String> {
    let base = command.strip_suffix(redirect)?;
    if base.is_empty() || !base.ends_with([' ', '\t']) {
        return None;
    }
    if !offset_outside_quotes(command, base.len()) {
        return None;
    }
    let base = base.trim();
    if base.is_empty() {
        return None;
    }
    Some(base.to_string())
}

/// True when byte offset `offset` sits outside any quoted region of
/// `command` (double-quote backslash escapes respected).
fn offset_outside_quotes(command: &str, offset: usize) -> bool {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, r) in command.char_indices() {
        if i >= offset {
            break;
        }
        if quote == Some('\'') {
            if r == '\'' {
                quote = None;
            }
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        match r {
            '\\' => {
                if quote == Some('"') {
                    escaped = true;
                }
            }
            '"' => {
                if quote.is_none() {
                    quote = Some('"');
                } else if quote == Some('"') {
                    quote = None;
                }
            }
            '\'' if quote.is_none() => {
                quote = Some('\'');
            }
            _ => {}
        }
    }
    quote.is_none() && !escaped
}

// --- simple-command parsing -------------------------------------------------

/// Parse a command into argv, refusing anything that isn't a plain word
/// list: any unquoted shell metacharacter (globs, redirects, expansion,
/// subshells, comments) fails the parse and the command defaults to
/// NeedsApproval.
fn parse_simple_command(command: &str) -> Option<Vec<String>> {
    let mut argv = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    for r in command.trim().chars() {
        match quote {
            Some('\'') => {
                if r == '\'' {
                    quote = None;
                    continue;
                }
                word.push(r);
                continue;
            }
            Some('"') => {
                match r {
                    '"' => quote = None,
                    // Expansion inside double quotes can run arbitrary code.
                    '\\' | '$' | '`' => return None,
                    _ => word.push(r),
                }
                continue;
            }
            _ => {}
        }
        match r {
            ' ' | '\t' => {
                if in_word {
                    argv.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '\'' | '"' => {
                quote = Some(r);
                in_word = true;
            }
            _ if rejected_simple_command_char(r) => return None,
            _ => {
                in_word = true;
                word.push(r);
            }
        }
    }
    if quote.is_some() {
        return None;
    }
    if in_word {
        argv.push(word);
    }
    if argv.is_empty() {
        None
    } else {
        Some(argv)
    }
}

fn rejected_simple_command_char(r: char) -> bool {
    matches!(
        r,
        '\\' | '$'
            | '`'
            | ';'
            | '|'
            | '&'
            | '<'
            | '>'
            | '\n'
            | '\r'
            | '('
            | ')'
            | '{'
            | '}'
            | '#'
            | '*'
            | '?'
            | '['
            | ']'
    )
}

// --- unsafe argument detection ----------------------------------------------

fn has_unsafe_args(argv: &[String]) -> bool {
    for field in &argv[1..] {
        if field.contains(['$', '`', '&', '<', '>', '\n', '\r']) {
            return true;
        }
    }
    if argv_has_prefix(argv, "find") {
        for field in argv {
            match field.as_str() {
                "-delete" | "-exec" | "-execdir" | "-ok" | "-okdir" | "-fls" => return true,
                _ => {}
            }
            if field.starts_with("-fprint") {
                return true;
            }
        }
    } else if argv_has_prefix(argv, "git diff")
        || argv_has_prefix(argv, "git show")
        || argv_has_prefix(argv, "git log")
    {
        for field in argv {
            if field == "--output"
                || field.starts_with("--output=")
                || field == "--ext-diff"
                || field == "--external-diff"
                || field == "--textconv"
            {
                return true;
            }
        }
    } else if argv_has_prefix(argv, "rg") {
        for field in argv {
            if field == "--pre" || field.starts_with("--pre=") {
                return true;
            }
        }
    }
    for field in argv {
        match field.as_str() {
            "--fix" | "--write" | "--update" | "--update-snapshot" | "--updatesnapshot" => {
                return true
            }
            _ => {}
        }
        if field.starts_with("--fix=")
            || field.starts_with("--write=")
            || field.starts_with("--update=")
            || field.starts_with("--update-snapshot=")
            || field.starts_with("--updatesnapshot=")
        {
            return true;
        }
    }
    if (argv_has_prefix(argv, "npx jest") || argv_has_prefix(argv, "npx vitest"))
        && argv.iter().any(|a| a == "-u")
    {
        return true;
    }
    false
}

/// Bounded `make` targets must be invoked bare (`make test`), with no extra
/// targets or variable overrides.
const MAKE_BOUNDED_TARGETS: &[&str] =
    &["build", "test", "check", "lint", "fmt", "fmt-check", "vet"];

fn make_bounded_target_has_extra_args(argv: &[String]) -> bool {
    if argv.len() < 2 || argv[0] != "make" {
        return false;
    }
    if MAKE_BOUNDED_TARGETS.contains(&argv[1].as_str()) {
        return argv.len() != 2;
    }
    false
}

// --- builtin read-only commands with option allowlists -----------------------

fn classify_builtin_read_only(argv: &[String], lower: &[String]) -> Option<ShellRiskDecision> {
    match lower[0].as_str() {
        "date" => Some(classify_date(argv, lower)),
        "uname" => Some(classify_uname(lower)),
        "whoami" => Some(classify_whoami(lower)),
        "id" => Some(classify_id(lower)),
        "which" => Some(classify_command_lookup(&lower[1..])),
        "command" => {
            if lower.len() >= 2 && lower[1] == "-v" {
                Some(classify_command_lookup(&lower[2..]))
            } else {
                None
            }
        }
        "sed" => Some(classify_sed_read_only(argv)),
        "sort" => Some(classify_sort(argv)),
        "uniq" => Some(classify_uniq(argv)),
        "printf" => Some(classify_printf(lower)),
        _ => None,
    }
}

fn classify_date(argv: &[String], lower: &[String]) -> ShellRiskDecision {
    const FLAGS_WITH_VALUES: &[&str] = &["-d", "--date", "-r", "--reference", "--rfc-3339"];
    const SAFE_NO_VALUE_FLAGS: &[&str] = &[
        "-u",
        "--utc",
        "--universal",
        "-I",
        "-R",
        "--iso-8601",
        "--rfc-email",
        "--debug",
        "--help",
        "--version",
    ];
    let mut i = 1;
    while i < lower.len() {
        let raw = argv[i].as_str();
        let arg = lower[i].as_str();
        if raw == "-s"
            || arg == "--set"
            || arg.starts_with("--set=")
            || raw == "-f"
            || arg == "--file"
            || arg.starts_with("--file=")
        {
            return needs_approval("date can set system time or read batch dates with this option");
        }
        if raw.starts_with('+') {
            i += 1;
            continue;
        }
        if FLAGS_WITH_VALUES.contains(&raw)
            || (raw.starts_with("--") && FLAGS_WITH_VALUES.contains(&arg))
        {
            i += 1;
            if i >= lower.len() {
                return needs_approval("date flag requires a value");
            }
            i += 1;
            continue;
        }
        if arg.starts_with("--date=")
            || arg.starts_with("--reference=")
            || arg.starts_with("--iso-8601=")
            || arg.starts_with("--rfc-3339=")
        {
            i += 1;
            continue;
        }
        if SAFE_NO_VALUE_FLAGS.contains(&raw) || SAFE_NO_VALUE_FLAGS.contains(&arg) {
            i += 1;
            continue;
        }
        if raw.starts_with('-') {
            return needs_approval("date option is not on the safe display allowlist");
        }
        return needs_approval("date positional arguments can set system time");
    }
    safe_read("date display command")
}

fn classify_uname(lower: &[String]) -> ShellRiskDecision {
    const SAFE_LONG: &[&str] = &[
        "--all",
        "--kernel-name",
        "--nodename",
        "--kernel-release",
        "--kernel-version",
        "--machine",
        "--processor",
        "--hardware-platform",
        "--operating-system",
        "--help",
        "--version",
    ];
    for arg in &lower[1..] {
        if SAFE_LONG.contains(&arg.as_str()) {
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") {
            if arg[1..].chars().all(|r| "asnrvmpio".contains(r)) {
                continue;
            }
            return needs_approval("uname option is not on the safe display allowlist");
        }
        return needs_approval("uname only supports safe display flags in auto-allow");
    }
    safe_read("uname display command")
}

fn classify_whoami(lower: &[String]) -> ShellRiskDecision {
    for arg in &lower[1..] {
        if arg != "--help" && arg != "--version" {
            return needs_approval("whoami only supports help/version args in auto-allow");
        }
    }
    safe_read("whoami display command")
}

fn classify_id(lower: &[String]) -> ShellRiskDecision {
    const SAFE_LONG: &[&str] = &[
        "--user",
        "--group",
        "--groups",
        "--name",
        "--real",
        "--zero",
        "--help",
        "--version",
    ];
    for arg in &lower[1..] {
        if SAFE_LONG.contains(&arg.as_str()) || is_command_name(arg) {
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") {
            if arg[1..].chars().all(|r| "uggnrz".contains(r)) {
                continue;
            }
            return needs_approval("id option is not on the safe display allowlist");
        }
        return needs_approval("id argument is not safe for auto-allow");
    }
    safe_read("id display command")
}

fn classify_command_lookup(args: &[String]) -> ShellRiskDecision {
    if args.is_empty() {
        return needs_approval("command lookup requires at least one command name");
    }
    for arg in args {
        if !is_command_name(arg) {
            return needs_approval("command lookup operands must be simple command names");
        }
    }
    safe_read("command lookup")
}

fn classify_printf(lower: &[String]) -> ShellRiskDecision {
    for arg in &lower[1..] {
        if arg.contains('/') && arg.starts_with('-') {
            return needs_approval("printf option is not on the safe display allowlist");
        }
    }
    safe_read("printf display command")
}

fn is_command_name(v: &str) -> bool {
    let v = v.trim();
    if v.is_empty() || v.contains('/') || v.starts_with('-') {
        return false;
    }
    v.chars()
        .all(|r| r.is_alphanumeric() || matches!(r, '_' | '.' | '-' | '+'))
}

// --- sed: stream-only substitution / range print -----------------------------

fn classify_sed_read_only(argv: &[String]) -> ShellRiskDecision {
    if sed_print_range_read_only(argv) {
        return safe_read("sed range print command");
    }
    if sed_substitution_read_only(argv) {
        return safe_read("sed stream substitution command");
    }
    needs_approval("sed command is not classified as read-only")
}

fn sed_substitution_read_only(argv: &[String]) -> bool {
    if argv.len() < 2 || argv[0] != "sed" {
        return false;
    }
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-E" | "-r" | "--regexp-extended" | "-n" | "--quiet" | "--silent" => i += 1,
            "--" => {
                i += 1;
                break;
            }
            _ => break,
        }
    }
    if i >= argv.len() || !sed_substitution_script_read_only(&argv[i]) {
        return false;
    }
    i += 1;
    // Remaining operands must be plain input files, not more options.
    argv[i..].iter().all(|a| !a.starts_with('-'))
}

fn sed_substitution_script_read_only(script: &str) -> bool {
    if script.is_empty() || !script.starts_with('s') {
        return false;
    }
    let runes: Vec<char> = script.chars().collect();
    if runes.len() < 4 {
        return false;
    }
    let delim = runes[1];
    if delim == '\\' || delim == '\n' || delim == '\r' {
        return false;
    }
    let mut parts = 0;
    let mut escaped = false;
    let mut i = 2;
    while i < runes.len() {
        let r = runes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if r == '\\' {
            escaped = true;
            i += 1;
            continue;
        }
        if r == delim {
            parts += 1;
            if parts == 2 {
                let flags: String = runes[i + 1..].iter().collect();
                return sed_substitution_flags_read_only(&flags);
            }
        }
        i += 1;
    }
    false
}

fn sed_substitution_flags_read_only(flags: &str) -> bool {
    flags
        .chars()
        .all(|r| r.is_ascii_digit() || matches!(r, 'g' | 'p' | 'I' | 'i' | 'M' | 'm'))
}

fn sed_print_range_read_only(argv: &[String]) -> bool {
    if argv.len() < 3 || argv[0] != "sed" {
        return false;
    }
    let mut i = 1;
    let mut saw_quiet = false;
    while i < argv.len() {
        match argv[i].as_str() {
            "-n" | "--quiet" | "--silent" => {
                saw_quiet = true;
                i += 1;
            }
            "--" => {
                i += 1;
                break;
            }
            _ => break,
        }
    }
    if !saw_quiet || i >= argv.len() || !sed_range_print_script(&argv[i]) {
        return false;
    }
    i += 1;
    argv[i..].iter().all(|a| !a.starts_with('-'))
}

fn sed_range_print_script(script: &str) -> bool {
    let Some(addr) = script.strip_suffix('p') else {
        return false;
    };
    if script.is_empty() {
        return false;
    }
    let parts: Vec<&str> = addr.split(',').collect();
    if parts.len() > 2 {
        return false;
    }
    for part in parts {
        if part == "$" {
            continue;
        }
        if part.is_empty() || !part.chars().all(|r| r.is_ascii_digit()) {
            return false;
        }
    }
    true
}

// --- sort / uniq: display-only option allowlists -----------------------------

fn classify_sort(argv: &[String]) -> ShellRiskDecision {
    let mut end_options = false;
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        if end_options || !arg.starts_with('-') || arg == "-" {
            i += 1;
            continue;
        }
        if arg == "--" {
            end_options = true;
            i += 1;
            continue;
        }
        if arg.starts_with("--") {
            if arg == "--output" || arg.starts_with("--output=") {
                return needs_approval(
                    "sort can write to an explicit output path with this option",
                );
            }
            if arg == "--compress-program" || arg.starts_with("--compress-program=") {
                return needs_approval("sort can execute an external compressor with this option");
            }
            if arg == "--temporary-directory" || arg.starts_with("--temporary-directory=") {
                return needs_approval(
                    "sort can write temporary files outside the input stream with this option",
                );
            }
            if sort_long_option_consumes_next(arg) && !arg.contains('=') {
                i += 1;
            }
            if !sort_long_option_safe(arg) {
                return needs_approval("sort option is not on the safe display allowlist");
            }
            i += 1;
            continue;
        }
        if !sort_short_options_safe(arg) {
            return needs_approval("sort option is not on the safe display allowlist");
        }
        i += 1;
    }
    safe_read("sort display command")
}

fn sort_long_option_safe(arg: &str) -> bool {
    let name = arg.split('=').next().unwrap_or(arg);
    matches!(
        name,
        "--ignore-leading-blanks"
            | "--dictionary-order"
            | "--ignore-nonprinting"
            | "--ignore-case"
            | "--general-numeric-sort"
            | "--human-numeric-sort"
            | "--month-sort"
            | "--numeric-sort"
            | "--reverse"
            | "--unique"
            | "--stable"
            | "--version-sort"
            | "--zero-terminated"
            | "--check"
            | "--key"
            | "--field-separator"
    )
}

fn sort_long_option_consumes_next(arg: &str) -> bool {
    let name = arg.split('=').next().unwrap_or(arg);
    matches!(name, "--key" | "--field-separator")
}

/// Short sort options: display-only flags pass; `-o` (output file) and `-T`
/// (temp dir) fail; `-k`/`-t` take a key/separator value, which is harmless.
fn sort_short_options_safe(arg: &str) -> bool {
    let chars: Vec<char> = arg.chars().collect();
    for r in chars.iter().skip(1) {
        match r {
            'b' | 'c' | 'C' | 'd' | 'f' | 'g' | 'h' | 'i' | 'M' | 'm' | 'n' | 'r' | 's' | 'u'
            | 'V' | 'z' => continue,
            'k' | 't' => return true,
            _ => return false,
        }
    }
    chars.len() > 1
}

fn classify_uniq(argv: &[String]) -> ShellRiskDecision {
    let mut operands = 0;
    let mut end_options = false;
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        if end_options || !arg.starts_with('-') || arg == "-" {
            operands += 1;
            if operands > 1 {
                return needs_approval(
                    "uniq can write to an output file when given a second operand",
                );
            }
            i += 1;
            continue;
        }
        if arg == "--" {
            end_options = true;
            i += 1;
            continue;
        }
        if arg.starts_with("--") {
            if uniq_long_option_consumes_next(arg) && !arg.contains('=') {
                i += 1;
            }
            if !uniq_long_option_safe(arg) {
                return needs_approval("uniq option is not on the safe display allowlist");
            }
            i += 1;
            continue;
        }
        let Some(consumes_next) = uniq_short_options_safe(arg) else {
            return needs_approval("uniq option is not on the safe display allowlist");
        };
        if consumes_next {
            i += 1;
        }
        i += 1;
    }
    safe_read("uniq display command")
}

fn uniq_long_option_safe(arg: &str) -> bool {
    let name = arg.split('=').next().unwrap_or(arg);
    matches!(
        name,
        "--count"
            | "--repeated"
            | "--all-repeated"
            | "--unique"
            | "--ignore-case"
            | "--zero-terminated"
            | "--group"
            | "--skip-fields"
            | "--skip-chars"
            | "--check-chars"
    )
}

fn uniq_long_option_consumes_next(arg: &str) -> bool {
    let name = arg.split('=').next().unwrap_or(arg);
    matches!(name, "--skip-fields" | "--skip-chars" | "--check-chars")
}

/// Returns `Some(consumes_next)` when every flag in the cluster is safe,
/// `None` otherwise. `-f`/`-s`/`-w` take a value and must end the cluster.
fn uniq_short_options_safe(arg: &str) -> Option<bool> {
    let chars: Vec<char> = arg.chars().collect();
    for (idx, r) in chars.iter().enumerate().skip(1) {
        match r {
            'c' | 'd' | 'u' | 'i' | 'z' => continue,
            'f' | 's' | 'w' => {
                if idx == chars.len() - 1 {
                    return Some(true);
                }
                return Some(false);
            }
            _ => return None,
        }
    }
    if chars.len() > 1 {
        Some(false)
    } else {
        None
    }
}

// --- git read-only whitelist --------------------------------------------------

fn git_command_read_only(argv: &[String]) -> bool {
    if argv.len() < 2 || argv[0] != "git" {
        return false;
    }
    if argv[1..].iter().any(|f| arg_contains_unsafe_meta(f)) {
        return false;
    }
    let mut subcommand_index = 1;
    while subcommand_index < argv.len() {
        let arg = argv[subcommand_index].as_str();
        if arg == "-c" || arg == "--config-env" || arg.starts_with("--config-env=") {
            return false;
        }
        if arg == "-C" {
            if subcommand_index + 1 >= argv.len()
                || !git_relative_path_allowed(&argv[subcommand_index + 1], false)
            {
                return false;
            }
            subcommand_index += 2;
            continue;
        }
        if let Some(path) = arg.strip_prefix("-C") {
            if !git_relative_path_allowed(path, false) {
                return false;
            }
            subcommand_index += 1;
            continue;
        }
        if arg.starts_with("-c") {
            // Inline config can change command behaviour arbitrarily.
            return false;
        }
        if arg.starts_with('-') {
            return false;
        }
        break;
    }
    if subcommand_index >= argv.len() {
        return false;
    }
    let subcommand = argv[subcommand_index].as_str();
    let args = &argv[subcommand_index + 1..];
    match subcommand {
        "status" | "rev-parse" => git_args_are_read_only(args),
        "symbolic-ref" => git_args_are_read_only(args) && git_symbolic_ref_args_read_only(args),
        "branch" => git_args_are_read_only(args) && git_branch_args_read_only(args),
        "remote" => git_args_are_read_only(args) && git_remote_args_read_only(args),
        "config" => !args.is_empty() && args[0] == "--get" && git_args_are_read_only(args),
        "diff" => git_args_are_read_only(args) && git_diff_args_read_only(args),
        "show" | "log" | "shortlog" | "ls-files" => git_args_are_read_only(args),
        _ => false,
    }
}

fn git_symbolic_ref_args_read_only(args: &[String]) -> bool {
    if args.is_empty() {
        return false;
    }
    let mut refs = 0;
    for arg in args {
        match arg.as_str() {
            "--short" | "-q" | "--quiet" => continue,
            _ => {
                if arg.starts_with('-') {
                    return false;
                }
                refs += 1;
            }
        }
    }
    refs == 1
}

fn git_branch_args_read_only(args: &[String]) -> bool {
    let mut saw_list = false;
    for arg in args {
        match arg.as_str() {
            "--show-current" | "--all" | "--remotes" | "--list" | "--verbose" | "--color"
            | "--no-color" | "-a" | "-r" | "-l" | "-v" | "-vv" => {
                if arg == "--list" || arg == "-l" {
                    saw_list = true;
                }
                continue;
            }
            _ => {
                if arg.starts_with("--color=") {
                    continue;
                }
                if saw_list && !arg.starts_with('-') {
                    continue;
                }
                return false;
            }
        }
    }
    true
}

fn git_remote_args_read_only(args: &[String]) -> bool {
    if args.is_empty() {
        return true;
    }
    if args.len() == 1 && args[0] == "-v" {
        return true;
    }
    args.len() >= 2 && args[0] == "get-url"
}

fn git_args_are_read_only(args: &[String]) -> bool {
    for arg in args {
        if arg.starts_with("--output=") {
            return false;
        }
        match arg.as_str() {
            "--output" | "--ext-diff" | "--external-diff" | "--textconv" => return false,
            _ => {}
        }
    }
    true
}

fn git_diff_args_read_only(args: &[String]) -> bool {
    if !args.iter().any(|a| a == "--no-index") {
        return true;
    }
    let paths = git_diff_no_index_paths(args);
    if paths.len() != 2 {
        return false;
    }
    git_relative_path_allowed(paths[0], true) && git_relative_path_allowed(paths[1], false)
}

fn git_diff_no_index_paths(args: &[String]) -> Vec<&String> {
    let mut paths = Vec::with_capacity(2);
    let mut end_of_options = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !end_of_options && arg == "--" {
            end_of_options = true;
            i += 1;
            continue;
        }
        if !end_of_options && arg.starts_with('-') {
            if git_diff_flag_consumes_next_arg(arg) && !arg.contains('=') {
                i += 1;
            }
            i += 1;
            continue;
        }
        paths.push(arg);
        i += 1;
    }
    paths
}

fn git_diff_flag_consumes_next_arg(arg: &str) -> bool {
    // Conservative list of read-only git diff flags that consume the next
    // arg. If this list misses an option, --no-index parsing may reject the
    // command rather than accidentally treating an option value as a path.
    matches!(
        arg,
        "--relative"
            | "--diff-filter"
            | "--word-diff-regex"
            | "--color-words"
            | "--ws-error-highlight"
            | "--abbrev"
            | "--break-rewrites"
            | "--find-renames"
            | "--find-copies"
            | "--diff-algorithm"
            | "--inter-hunk-context"
            | "-S"
            | "-G"
            | "-O"
    )
}

fn git_relative_path_allowed(path: &str, allow_dev_null: bool) -> bool {
    let path = path.trim();
    if path.is_empty() {
        return false;
    }
    if allow_dev_null && path == "/dev/null" {
        return true;
    }
    if path.starts_with('/') || path.starts_with('~') || path.starts_with('-') {
        return false;
    }
    path.split('/')
        .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn arg_contains_unsafe_meta(arg: &str) -> bool {
    arg.contains(['$', '`', ';', '&', '|', '<', '>', '\n', '\r'])
}

// --- bounded-write build/test commands ----------------------------------------

fn classify_bounded_write(lower: &[String]) -> Option<ShellRiskDecision> {
    match lower[0].as_str() {
        "go" => {
            if lower.len() >= 2 {
                match lower[1].as_str() {
                    "test" => {
                        if has_any_flag_prefix(&lower[2..], &["-exec", "-toolexec"]) {
                            return Some(needs_approval(
                                "go test can run an execution wrapper with this option",
                            ));
                        }
                        if has_any_flag_prefix(&lower[2..], &["-c"]) {
                            return Some(needs_approval("go test -c emits a test binary"));
                        }
                        if has_any_flag_prefix(
                            &lower[2..],
                            &[
                                "-coverprofile",
                                "-cpuprofile",
                                "-memprofile",
                                "-blockprofile",
                                "-mutexprofile",
                                "-trace",
                                "-o",
                            ],
                        ) {
                            return Some(needs_approval(
                                "go test writes to an explicit output path with this option",
                            ));
                        }
                        return Some(bounded_write(
                            "go test may write build and test cache files".into(),
                        ));
                    }
                    "build" => {
                        if has_any_flag_prefix(&lower[2..], &["-o"]) {
                            return Some(needs_approval(
                                "go build writes to an explicit output path with this option",
                            ));
                        }
                        return Some(needs_approval("go build may emit a workspace binary"));
                    }
                    "vet" => {
                        return Some(bounded_write("go vet may write build cache files".into()))
                    }
                    _ => {}
                }
            }
        }
        "make" => {
            if lower.len() == 2 && MAKE_BOUNDED_TARGETS.contains(&lower[1].as_str()) {
                return Some(bounded_write(format!(
                    "make {} may write project-local build or test artifacts",
                    lower[1]
                )));
            }
        }
        "cargo" => {
            if lower.len() >= 2 {
                match lower[1].as_str() {
                    "build" | "test" | "check" | "clippy" | "fmt" => {
                        if has_any_flag_prefix(&lower[2..], &["--target-dir"]) {
                            return Some(needs_approval(
                                "cargo writes to an explicit target directory with this option",
                            ));
                        }
                        return Some(bounded_write(format!(
                            "cargo {} may write target build artifacts",
                            lower[1]
                        )));
                    }
                    _ => {}
                }
            }
        }
        "npm" | "pnpm" => {
            if lower.len() >= 2 {
                if lower[1] == "test" {
                    return Some(bounded_write(format!(
                        "{} test may write project-local test artifacts",
                        lower[0]
                    )));
                }
                if lower.len() >= 3 && lower[1] == "run" && npm_bounded_script(&lower[2]) {
                    return Some(bounded_write(format!(
                        "{} run {} may write project-local build or test artifacts",
                        lower[0], lower[2]
                    )));
                }
            }
        }
        "npx" => {
            if lower.len() >= 2 {
                match lower[1].as_str() {
                    "jest" | "vitest" => {
                        if has_known_test_output_flag(&lower[2..]) {
                            return Some(needs_approval(
                                "test runner writes to an explicit output path with this option",
                            ));
                        }
                        return Some(bounded_write(format!(
                            "npx {} may write project-local test artifacts",
                            lower[1]
                        )));
                    }
                    "tsc" if lower.len() >= 3 && lower[2] == "--noemit" => {
                        return Some(bounded_write(
                            "npx tsc --noEmit may write compiler cache files".into(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        "pytest" => {
            if has_known_test_output_flag(&lower[1..]) {
                return Some(needs_approval(
                    "pytest writes to an explicit output path with this option",
                ));
            }
            return Some(bounded_write(
                "pytest may write project-local test artifacts".into(),
            ));
        }
        "python" | "python3" => {
            if lower.len() >= 3 && lower[1] == "-m" && lower[2] == "pytest" {
                if has_known_test_output_flag(&lower[3..]) {
                    return Some(needs_approval(
                        "pytest writes to an explicit output path with this option",
                    ));
                }
                return Some(bounded_write(format!(
                    "{} -m pytest may write project-local test artifacts",
                    lower[0]
                )));
            }
        }
        "deno" | "bun" if lower.len() >= 2 && lower[1] == "test" => {
            return Some(bounded_write(format!(
                "{} test may write project-local test artifacts",
                lower[0]
            )));
        }
        _ => {}
    }
    None
}

fn has_known_test_output_flag(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--outputfile"
            || arg == "--output-file"
            || arg.starts_with("--outputfile=")
            || arg.starts_with("--output-file=")
            || arg == "--junitxml"
            || arg == "--junit-xml"
            || arg.starts_with("--junitxml=")
            || arg.starts_with("--junit-xml=")
            || arg == "--html"
            || arg.starts_with("--html=")
            || arg.starts_with("--cov-report=xml:")
            || arg.starts_with("--cov-report=html:")
            || arg.starts_with("--cov-report=lcov:")
            || arg.starts_with("--cov-report=json:")
    })
}

fn has_any_flag_prefix(args: &[String], prefixes: &[&str]) -> bool {
    args.iter().any(|arg| {
        prefixes
            .iter()
            .any(|prefix| arg == prefix || arg.starts_with(&format!("{prefix}=")))
    })
}

fn npm_bounded_script(script: &str) -> bool {
    matches!(script, "build" | "test" | "lint" | "typecheck")
}

fn argv_has_prefix(argv: &[String], prefix: &str) -> bool {
    let prefix_argv: Vec<&str> = prefix.split_whitespace().collect();
    if argv.len() < prefix_argv.len() {
        return false;
    }
    prefix_argv
        .iter()
        .enumerate()
        .all(|(i, want)| argv[i] == *want)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(command: &str) -> ShellRiskLevel {
        classify_shell_command(command).level
    }

    // --- safe reads ---------------------------------------------------------

    #[test]
    fn read_only_commands_are_safe() {
        for cmd in [
            "ls -la",
            "pwd",
            "cat src/main.rs",
            "grep -rn pattern src",
            "rg TODO",
            "head -n 20 file.txt",
            "wc -l file.txt",
            "which cargo",
            "uname -a",
            "whoami",
            "date -u",
            "printf hello",
            "go version",
            "rustc --version",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::SafeRead, "command: {cmd}");
        }
    }

    #[test]
    fn pipelines_of_read_only_commands_are_safe() {
        assert_eq!(
            level("cat file.txt | grep foo | wc -l"),
            ShellRiskLevel::SafeRead
        );
        assert_eq!(level("ls && pwd"), ShellRiskLevel::SafeRead);
        assert_eq!(level("pwd; ls"), ShellRiskLevel::SafeRead);
    }

    #[test]
    fn trailing_stderr_redirect_is_transparent() {
        assert_eq!(level("ls -la 2>/dev/null"), ShellRiskLevel::SafeRead);
        assert_eq!(level("cat file 2>&1"), ShellRiskLevel::SafeRead);
    }

    #[test]
    fn git_read_only_commands_are_safe() {
        for cmd in [
            "git status",
            "git log --oneline",
            "git diff HEAD~1",
            "git branch --show-current",
            "git remote -v",
            "git config --get user.name",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::SafeRead, "command: {cmd}");
        }
    }

    #[test]
    fn git_mutating_commands_need_approval() {
        for cmd in [
            "git push origin main",
            "git commit -m x",
            "git checkout -b f",
            "git diff --output=/tmp/d.patch",
            "git -c core.editor=vim log",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::NeedsApproval, "command: {cmd}");
        }
    }

    #[test]
    fn sed_stream_substitution_is_safe_but_in_place_is_not() {
        assert_eq!(level("sed s/foo/bar/g file.txt"), ShellRiskLevel::SafeRead);
        assert_eq!(level("sed -n 1,20p file.txt"), ShellRiskLevel::SafeRead);
        assert_eq!(
            level("sed -i s/foo/bar/ file.txt"),
            ShellRiskLevel::NeedsApproval
        );
    }

    #[test]
    fn sort_uniq_display_safe_output_flags_not() {
        assert_eq!(level("sort -u file.txt"), ShellRiskLevel::SafeRead);
        assert_eq!(level("uniq -c file.txt"), ShellRiskLevel::SafeRead);
        assert_eq!(
            level("sort -o out.txt file.txt"),
            ShellRiskLevel::NeedsApproval
        );
        assert_eq!(
            level("uniq file.txt out.txt"),
            ShellRiskLevel::NeedsApproval
        );
    }

    // --- bounded writes -------------------------------------------------------

    #[test]
    fn build_test_commands_are_bounded_writes() {
        for cmd in [
            "cargo test",
            "cargo check",
            "cargo clippy",
            "go test ./...",
            "go vet ./...",
            "npm test",
            "pnpm run build",
            "pytest",
            "python -m pytest tests",
            "make test",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::BoundedWrite, "command: {cmd}");
        }
    }

    #[test]
    fn bounded_write_with_explicit_output_needs_approval() {
        for cmd in [
            "go test -coverprofile=cover.out ./...",
            "cargo build --target-dir /tmp/x",
            "pytest --junitxml=report.xml",
            "make test EXTRA=1",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::NeedsApproval, "command: {cmd}");
        }
    }

    // --- needs approval (default) ---------------------------------------------

    #[test]
    fn unparseable_or_unknown_commands_need_approval() {
        for cmd in [
            "curl https://example.com -o out.html",
            "echo $(whoami)",
            "ls > listing.txt",
            "ls *.rs",
            "foo || bar",
            "rm file.txt",
            "npm install",
            "pip install requests",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::NeedsApproval, "command: {cmd}");
        }
    }

    #[test]
    fn unsafe_expansion_args_need_approval() {
        assert_eq!(level("echo `id`"), ShellRiskLevel::NeedsApproval);
        assert_eq!(
            level("find . -name x -delete"),
            ShellRiskLevel::NeedsApproval
        );
        assert_eq!(level("find . -exec rm {} +"), ShellRiskLevel::NeedsApproval);
        assert_eq!(level("rg --pre cat TODO"), ShellRiskLevel::NeedsApproval);
        assert_eq!(level("npx jest -u"), ShellRiskLevel::NeedsApproval);
    }

    #[test]
    fn pipeline_with_non_read_only_stage_needs_approval() {
        assert_eq!(
            level("cat file.txt | tee out.txt"),
            ShellRiskLevel::NeedsApproval
        );
        assert_eq!(level("ls && cargo test"), ShellRiskLevel::NeedsApproval);
    }

    #[test]
    fn relaxed_policy_bypasses_conservative_read_only_checks() {
        let command = "curl -sS https://example.com 2>&1 | head -80";
        assert_eq!(
            classify_shell_command_with_policy(command, ShellRiskPolicy::Strict).level,
            ShellRiskLevel::NeedsApproval
        );
        assert_eq!(
            classify_shell_command_with_policy(command, ShellRiskPolicy::Relaxed).level,
            ShellRiskLevel::SafeRead
        );
        assert_eq!(
            classify_shell_command_with_policy("rm -rf /", ShellRiskPolicy::Relaxed).level,
            ShellRiskLevel::Blocked
        );
    }

    // --- hard deny --------------------------------------------------------------

    #[test]
    fn recursive_delete_of_critical_paths_is_blocked() {
        for cmd in [
            "rm -rf /",
            "rm -rf /*",
            "rm -fr /usr",
            "rm -r /etc",
            "rm -rf /var/",
            "rm --recursive --force /bin",
            "rm --no-preserve-root -rf /",
            "sudo rm -rf /usr",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::Blocked, "command: {cmd}");
        }
    }

    #[test]
    fn workspace_recursive_delete_is_not_blocked() {
        for cmd in ["rm -rf target", "rm -rf ./build", "rm -rf /tmp/scratch"] {
            assert_eq!(level(cmd), ShellRiskLevel::NeedsApproval, "command: {cmd}");
        }
    }

    #[test]
    fn raw_device_writes_are_blocked() {
        for cmd in [
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "mkswap /dev/sda2",
            "wipefs -a /dev/sda",
            "echo x > /dev/sda",
            "cat data >> /dev/nvme0n1",
            "fdisk /dev/sda",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::Blocked, "command: {cmd}");
        }
    }

    #[test]
    fn benign_device_usage_is_not_blocked() {
        assert_eq!(
            level("dd if=/dev/zero of=test.img bs=1M count=10"),
            ShellRiskLevel::NeedsApproval
        );
        assert_eq!(level("ls /dev/sda"), ShellRiskLevel::SafeRead);
        assert_eq!(level("fdisk -l"), ShellRiskLevel::NeedsApproval);
    }

    #[test]
    fn system_lifecycle_commands_are_blocked() {
        for cmd in [
            "shutdown -h now",
            "reboot",
            "halt",
            "poweroff",
            "init 0",
            "telinit 6",
            "systemctl reboot",
            "systemctl poweroff",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::Blocked, "command: {cmd}");
        }
        // Non-lifecycle systemctl and init usage is not on the deny list.
        assert_eq!(
            level("systemctl status nginx"),
            ShellRiskLevel::NeedsApproval
        );
    }

    #[test]
    fn killing_pid_one_is_blocked() {
        for cmd in [
            "kill 1",
            "kill -9 1",
            "kill -TERM 1",
            "kill -9 -1",
            "killall5",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::Blocked, "command: {cmd}");
        }
        assert_eq!(level("kill -9 12345"), ShellRiskLevel::NeedsApproval);
        assert_eq!(level("kill -1 12345"), ShellRiskLevel::NeedsApproval); // SIGHUP
    }

    #[test]
    fn fork_bomb_is_blocked() {
        assert_eq!(level(":(){ :|:& };:"), ShellRiskLevel::Blocked);
        assert_eq!(level("bomb(){ bomb|bomb& };bomb"), ShellRiskLevel::Blocked);
    }

    #[test]
    fn permission_sweep_on_system_paths_is_blocked() {
        assert_eq!(level("chmod -R 777 /"), ShellRiskLevel::Blocked);
        assert_eq!(level("chmod -R 000 /usr"), ShellRiskLevel::Blocked);
        assert_eq!(level("chown -R nobody /etc"), ShellRiskLevel::Blocked);
        // Project-local sweeps are fine (well, approval-gated).
        assert_eq!(
            level("chmod -R 755 ./scripts"),
            ShellRiskLevel::NeedsApproval
        );
    }

    #[test]
    fn deny_scan_sees_through_compound_syntax() {
        for cmd in [
            "ls; rm -rf /usr",
            "true && rm -rf /etc",
            "false || rm -rf /var",
            "echo hi | tee log; reboot",
            "(rm -rf /usr)",
            "echo $(rm -rf /etc)",
            "bash -c 'rm -rf /usr'",
            "sudo sh -c \"rm -rf /etc\"",
            "env FOO=1 rm -rf /usr",
            "nohup reboot",
            "timeout 30 rm -rf /etc",
        ] {
            assert_eq!(level(cmd), ShellRiskLevel::Blocked, "command: {cmd}");
        }
    }

    #[test]
    fn quoted_destructive_text_is_not_blocked() {
        // The dangerous string is data, not a command.
        assert_eq!(level("echo 'rm -rf /usr'"), ShellRiskLevel::SafeRead);
        assert_eq!(level("grep 'rm -rf /' README.md"), ShellRiskLevel::SafeRead);
    }

    #[test]
    fn decisions_carry_reasons() {
        let decision = classify_shell_command("rm -rf /");
        assert_eq!(decision.level, ShellRiskLevel::Blocked);
        assert!(!decision.reason.is_empty());
        assert_eq!(decision.level.as_str(), "blocked");
    }
}
