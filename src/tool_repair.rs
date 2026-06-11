//! Repair of malformed tool-call arguments from weak models.
//!
//! Two layers, applied at different points:
//!
//!   1. [`repair_truncated_json`] — runs where the streamed `arguments`
//!      buffer is parsed (`consume_step_stream`); rescues JSON cut off
//!      mid-stream (unclosed braces/strings, dangling `:` or `,`) and,
//!      failing that, salvages whatever top-level pairs survive.
//!   2. [`repair_tool_input_for_spec`] — runs at dispatch time per
//!      invocation, guided by the tool's `input_schema`; fixes common
//!      shape mistakes (`"true"`→`true`, `"30"`→`30`, bare string→array,
//!      stringified array, `{}`→`[]`, optional `null` dropped, markdown
//!      autolink unwrapped on path fields).
//!
//! Safety rule: schema-guided repairs are committed ONLY if a second
//! validation pass comes back clean — a repair that leaves other issues
//! standing reverts to the original input wholesale, so we never make a
//! bad input differently-bad.
//!
//! Deferred (see phased-plan §8.1 A8): recovering tool calls from
//! reasoning text, a repeated-call breaker, and flat-argument renesting.

use serde_json::{Map, Value};

// ───────────────────────── truncated-JSON repair ─────────────────────────

/// Outcome of [`repair_truncated_json`]. `notes` carries human-readable
/// markers of what was done, for tracing.
#[derive(Debug)]
pub struct TruncationRepair {
    pub repaired: String,
    pub changed: bool,
    pub notes: Vec<&'static str>,
}

/// Best-effort rescue of a truncated / malformed tool-arguments string.
/// Valid input is returned unchanged;
/// otherwise try, in order: close open strings/brackets, complete a
/// dangling `"key":` with `null`, strip trailing commas; then shrink to
/// the longest parseable object prefix; then salvage individual top-level
/// `"key": literal` pairs; finally fall back to `{}` (empty args beat a
/// failed turn).
pub fn repair_truncated_json(raw: &str) -> TruncationRepair {
    if raw.trim().is_empty() {
        return TruncationRepair {
            repaired: "{}".into(),
            changed: true,
            notes: vec!["empty input -> {}"],
        };
    }
    if is_valid_json(raw) {
        return TruncationRepair {
            repaired: raw.into(),
            changed: false,
            notes: vec![],
        };
    }
    let mut candidate = raw.to_string();
    if let Some(closed) = close_likely_json(&candidate) {
        candidate = closed;
    }
    if ends_with_dangling_colon(&candidate) {
        candidate.push_str(" null");
    }
    candidate = strip_trailing_commas(&candidate);
    if is_valid_json(&candidate) {
        return TruncationRepair {
            repaired: candidate,
            changed: true,
            notes: vec!["truncation repaired"],
        };
    }
    // Salvage minimal object from recoverable prefix before hard fallback.
    if let Some(obj) = salvage_json_object_prefix(&candidate) {
        return TruncationRepair {
            repaired: obj,
            changed: true,
            notes: vec!["salvaged prefix object"],
        };
    }
    // If we can extract at least one k:v pair, preserve it instead of
    // dropping all args.
    if let Some(kv) = salvage_top_level_pairs(&candidate) {
        return TruncationRepair {
            repaired: kv,
            changed: true,
            notes: vec!["salvaged top-level pairs"],
        };
    }
    TruncationRepair {
        repaired: "{}".into(),
        changed: true,
        notes: vec!["fallback -> {}"],
    }
}

fn is_valid_json(s: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(s).is_ok()
}

/// String-aware bracket-stack scan: close an unterminated string and any
/// unclosed `{`/`[` in reverse order. Returns `None` when the input has a
/// mismatched closer (e.g. `{]`) — that's corruption, not truncation.
fn close_likely_json(raw: &str) -> Option<String> {
    let mut stack: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for &ch in raw.as_bytes() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            b'"' => in_string = true,
            b'{' => stack.push(b'}'),
            b'[' => stack.push(b']'),
            // Mismatched (or unopened) closer — unrepairable. `last()` keeps
            // the guard side-effect free; the matching closer is popped in
            // the arm below.
            b'}' | b']' if stack.last() != Some(&ch) => return None,
            b'}' | b']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    let mut out = String::with_capacity(raw.len() + stack.len() + 1);
    out.push_str(raw);
    if in_string {
        out.push('"');
    }
    while let Some(c) = stack.pop() {
        out.push(c as char);
    }
    Some(out)
}

/// Detect a dangling `"key":` ending: the input ends with a quoted
/// key's closing `"`, then `:`, with optional whitespace around it.
fn ends_with_dangling_colon(s: &str) -> bool {
    let trimmed_end = s.trim_end();
    let Some(before_colon) = trimmed_end.strip_suffix(':') else {
        return false;
    };
    before_colon.trim_end().ends_with('"')
}

/// Drop every comma that is
/// directly followed (modulo whitespace) by a closing brace/bracket. This is NOT string-aware — the result must still
/// pass JSON validation downstream, which bounds the damage.
fn strip_trailing_commas(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b',' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'}' || bytes[j] == b']') {
                i += 1; // skip the comma, keep the whitespace + closer
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Input was valid UTF-8 and we only removed ASCII commas.
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Shrink from the end toward the first `{` until some prefix closes into
/// valid JSON. O(n²) — tool args are bounded by model
/// output size, and this path only runs on already-broken input.
fn salvage_json_object_prefix(raw: &str) -> Option<String> {
    let start = raw.find('{')?;
    let mut end = raw.len();
    while end > start {
        if !raw.is_char_boundary(end) {
            end -= 1;
            continue;
        }
        let candidate = raw[start..end].trim();
        if candidate.starts_with('{') {
            if let Some(closed) = close_likely_json(candidate) {
                let closed = strip_trailing_commas(&closed);
                if is_valid_json(&closed) {
                    return Some(closed);
                }
            }
        }
        end -= 1;
    }
    None
}

/// Extract up to 12 top-level `"key": <literal>` pairs (string / number /
/// bool / null) from anywhere in the text and rebuild a minimal object.
fn salvage_top_level_pairs(raw: &str) -> Option<String> {
    let mut r = raw.trim();
    if r.is_empty() {
        return None;
    }
    if let Some(pos) = r.find('{') {
        r = &r[pos + 1..];
    }
    let matches = find_pair_literals(r, 12);
    if matches.is_empty() {
        return None;
    }
    let mut out = Map::new();
    for (key, lit) in matches {
        let Ok(v) = serde_json::from_str::<Value>(&lit) else {
            continue;
        };
        out.insert(key, v);
    }
    if out.is_empty() {
        return None;
    }
    serde_json::to_string(&Value::Object(out)).ok()
}

/// Hand-rolled key:literal pair matcher, equivalent to:
/// `"([A-Za-z0-9_\-\.]+)"\s*:\s*("([^"\\]|\\.)*"|-?\d+(\.\d+)?|true|false|null)`.
/// Returns `(key, literal_source)` for the first `max` matches.
fn find_pair_literals(s: &str, max: usize) -> Vec<(String, String)> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() && out.len() < max {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        match match_pair_at(bytes, i) {
            Some((key, lit, next)) => {
                out.push((key, lit));
                i = next;
            }
            None => i += 1,
        }
    }
    out
}

/// Try to match one `"key"\s*:\s*<literal>` starting at byte `start`
/// (which must be a `"`). Returns `(key, literal, index_after_match)`.
fn match_pair_at(bytes: &[u8], start: usize) -> Option<(String, String, usize)> {
    let mut i = start + 1;
    let key_start = i;
    while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-' | b'.'))
    {
        i += 1;
    }
    if i == key_start || i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    let key = String::from_utf8(bytes[key_start..i].to_vec()).ok()?;
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let (lit, next) = match_literal_at(bytes, i)?;
    Some((key, lit, next))
}

/// Match one literal: quoted string (escape-aware), number, true/false/null.
fn match_literal_at(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    if start >= bytes.len() {
        return None;
    }
    match bytes[start] {
        b'"' => {
            let mut i = start + 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i += 2,
                    b'"' => {
                        let lit = String::from_utf8(bytes[start..=i].to_vec()).ok()?;
                        return Some((lit, i + 1));
                    }
                    _ => i += 1,
                }
            }
            None
        }
        b'-' | b'0'..=b'9' => {
            let mut i = start;
            if bytes[i] == b'-' {
                i += 1;
            }
            let int_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == int_start {
                return None;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                let frac_start = i + 1;
                let mut j = frac_start;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > frac_start {
                    i = j;
                }
            }
            let lit = String::from_utf8(bytes[start..i].to_vec()).ok()?;
            Some((lit, i))
        }
        _ => {
            for word in ["true", "false", "null"] {
                if bytes[start..].starts_with(word.as_bytes()) {
                    return Some((word.to_string(), start + word.len()));
                }
            }
            None
        }
    }
}

// ──────────────────────── schema-guided repair ────────────────────────

pub const REPAIR_NULL_OPTIONAL_OMITTED: &str = "null_optional_omitted";
pub const REPAIR_STRINGIFIED_ARRAY: &str = "stringified_array";
pub const REPAIR_BARE_STRING_TO_ARRAY: &str = "bare_string_to_array";
pub const REPAIR_EMPTY_OBJECT_TO_ARRAY: &str = "empty_object_to_array";
pub const REPAIR_MARKDOWN_AUTOLINK_PATH: &str = "markdown_autolink_path";
pub const REPAIR_SEMANTIC_BOOLEAN: &str = "semantic_boolean_string";
pub const REPAIR_SEMANTIC_INTEGER: &str = "semantic_integer_string";
pub const REPAIR_FIELD_ALIAS: &str = "field_alias";

/// One applied repair, for tracing / telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInputRepair {
    pub kind: &'static str,
    pub path: String,
    pub before_type: &'static str,
    pub after_type: &'static str,
}

/// A path segment into the (cloned) working value. We navigate to the parent object by path segments rather than holding a pointer.
#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Key(String),
    Index(usize),
}

/// One schema violation found by the issue scan. `parent_loc` + `key`
/// identify the owning object field when the issue is repairable in
/// place; issues without a parent (root level, unknown fields bubbling
/// rules) are never repaired and act as "poison" that reverts the whole
/// repair in the second validation pass.
struct Issue {
    path: String,
    expected: String,
    items_type: Option<String>,
    required: bool,
    known_field: bool,
    parent_loc: Option<(Vec<Seg>, String)>,
}

/// Apply narrowly scoped, schema-guided repairs to common tool-call
/// argument shape mistakes. Valid inputs return `None` (unchanged), as do
/// inputs whose issues can't ALL be repaired under a "two passes, revert unless fully clean" contract. On success returns the repaired value plus the repair list.
pub fn repair_tool_input_for_spec(
    schema: &Value,
    input: &Value,
) -> Option<(Value, Vec<ToolInputRepair>)> {
    if !schema.is_object() || !input.is_object() {
        return None;
    }
    let mut work = input.clone();
    let mut repairs = collect_field_alias_repairs(schema, &mut work);
    repairs.extend(collect_path_string_repairs(schema, &mut work, ""));
    let issues = collect_issues(schema, &work, "", &[], true);
    if issues.is_empty() && repairs.is_empty() {
        return None;
    }
    for issue in &issues {
        if let Some(repair) = apply_issue_repair(&mut work, issue) {
            repairs.push(repair);
        }
    }
    if repairs.is_empty() {
        return None;
    }
    // Second pass: commit only if the repaired value is fully clean.
    if !collect_issues(schema, &work, "", &[], true).is_empty() {
        return None;
    }
    Some((work, repairs))
}

/// Repair common model-facing field aliases only when the target field is part
/// of the advertised schema and is currently absent. This is intentionally a
/// rename, not inference from conversation state.
fn collect_field_alias_repairs(schema: &Value, value: &mut Value) -> Vec<ToolInputRepair> {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return vec![];
    };
    let Some(obj) = value.as_object_mut() else {
        return vec![];
    };
    let aliases = [
        ("filePath", "path"),
        ("oldString", "old_string"),
        ("newString", "new_string"),
        ("replaceAll", "replace_all"),
        ("cmd", "command"),
    ];
    let mut out = Vec::new();
    for (from, to) in aliases {
        if !props.contains_key(to) || obj.contains_key(to) {
            continue;
        }
        let Some(value) = obj.remove(from) else {
            continue;
        };
        let before_type = json_type_name(&value);
        obj.insert(to.to_string(), value);
        out.push(ToolInputRepair {
            kind: REPAIR_FIELD_ALIAS,
            path: to.to_string(),
            before_type,
            after_type: before_type,
        });
    }
    out
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Walk `value` along `segs`, mutably.
fn navigate_mut<'a>(value: &'a mut Value, segs: &[Seg]) -> Option<&'a mut Value> {
    let mut cur = value;
    for seg in segs {
        cur = match seg {
            Seg::Key(k) => cur.as_object_mut()?.get_mut(k)?,
            Seg::Index(i) => cur.as_array_mut()?.get_mut(*i)?,
        };
    }
    Some(cur)
}

/// Unwrap markdown autolinks on path-ish string fields, recursively.
/// Mutates `value` in place.
fn collect_path_string_repairs(
    schema: &Value,
    value: &mut Value,
    path: &str,
) -> Vec<ToolInputRepair> {
    match schema_type(schema).as_str() {
        "object" | "" => {
            let Some(obj) = value.as_object_mut() else {
                return vec![];
            };
            let Some(props) = schema.get("properties").and_then(Value::as_object) else {
                return vec![];
            };
            let mut keys: Vec<&String> = props.keys().collect();
            keys.sort();
            let mut out = Vec::new();
            for key in keys {
                let child_schema = &props[key.as_str()];
                if !child_schema.is_object() {
                    continue;
                }
                let Some(child_value) = obj.get_mut(key.as_str()) else {
                    continue;
                };
                let child_path = join_path(path, key);
                if schema_type(child_schema) == "string" && is_path_string_field(key) {
                    if let Some(s) = child_value.as_str() {
                        if let Some(fixed) = unwrap_markdown_autolink_path(s) {
                            *child_value = Value::String(fixed);
                            out.push(ToolInputRepair {
                                kind: REPAIR_MARKDOWN_AUTOLINK_PATH,
                                path: child_path,
                                before_type: "string",
                                after_type: "string",
                            });
                        }
                    }
                    continue;
                }
                out.extend(collect_path_string_repairs(
                    child_schema,
                    child_value,
                    &child_path,
                ));
            }
            out
        }
        "array" => {
            let Some(arr) = value.as_array_mut() else {
                return vec![];
            };
            let Some(item_schema) = schema.get("items").filter(|s| s.is_object()) else {
                return vec![];
            };
            let mut out = Vec::new();
            for (i, item) in arr.iter_mut().enumerate() {
                let item_path = format!("{path}[{i}]");
                out.extend(collect_path_string_repairs(item_schema, item, &item_path));
            }
            out
        }
        _ => vec![],
    }
}

/// Scan `value` against `schema`, returning every violation. Mirrors
/// the issue scan. `loc` is the segment path to `value`
/// within the working root (used to rebuild parent locations).
fn collect_issues(
    schema: &Value,
    value: &Value,
    path: &str,
    loc: &[Seg],
    required: bool,
) -> Vec<Issue> {
    let mut expected = schema_type(schema);
    if expected.is_empty() && schema.get("properties").is_some_and(Value::is_object) {
        expected = "object".into();
    }
    let issue_here = || Issue {
        path: path.to_string(),
        expected: expected.clone(),
        items_type: schema_array_items_type(schema),
        required,
        known_field: !path.is_empty(),
        parent_loc: None,
    };
    if value.is_null() {
        if type_allows_null(schema) {
            return vec![];
        }
        return vec![issue_here()];
    }
    match expected.as_str() {
        "" => vec![],
        "object" => {
            let Some(obj) = value.as_object() else {
                return vec![issue_here()];
            };
            collect_object_issues(schema, obj, path, loc)
        }
        "array" => {
            let Some(arr) = value.as_array() else {
                return vec![issue_here()];
            };
            let mut out = Vec::new();
            if let Some(min) = schema_min_items(schema) {
                if arr.len() < min {
                    out.push(issue_here());
                }
            }
            let Some(item_schema) = schema.get("items").filter(|s| s.is_object()) else {
                return out;
            };
            for (i, item) in arr.iter().enumerate() {
                let item_path = format!("{path}[{i}]");
                let mut item_loc = loc.to_vec();
                item_loc.push(Seg::Index(i));
                out.extend(collect_issues(
                    item_schema,
                    item,
                    &item_path,
                    &item_loc,
                    true,
                ));
            }
            out
        }
        "string" if value.is_string() => vec![],
        "integer" if is_json_integer(value) => vec![],
        "number" if value.is_number() => vec![],
        "boolean" if value.is_boolean() => vec![],
        "string" | "integer" | "number" | "boolean" => vec![issue_here()],
        _ => vec![],
    }
}

fn collect_object_issues(
    schema: &Value,
    obj: &Map<String, Value>,
    path: &str,
    loc: &[Seg],
) -> Vec<Issue> {
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let req = required_set(schema.get("required"));
    let mut out = Vec::new();
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    for key in keys {
        let child_schema = &props[key.as_str()];
        if !child_schema.is_object() {
            continue;
        }
        let child_path = join_path(path, key);
        let Some(value) = obj.get(key.as_str()) else {
            if req.contains(key) {
                out.push(Issue {
                    path: child_path,
                    expected: schema_type(child_schema),
                    items_type: schema_array_items_type(child_schema),
                    required: true,
                    known_field: true,
                    parent_loc: Some((loc.to_vec(), key.clone())),
                });
            }
            continue;
        };
        let mut child_loc = loc.to_vec();
        child_loc.push(Seg::Key(key.clone()));
        let mut issues = collect_issues(
            child_schema,
            value,
            &child_path,
            &child_loc,
            req.contains(key),
        );
        for issue in &mut issues {
            issue.known_field = true;
            if issue.parent_loc.is_none() {
                issue.parent_loc = Some((loc.to_vec(), key.clone()));
            }
        }
        out.extend(issues);
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        for key in obj.keys() {
            if props.contains_key(key.as_str()) {
                continue;
            }
            out.push(Issue {
                path: join_path(path, key),
                expected: "none".into(),
                items_type: None,
                required: false,
                known_field: false,
                parent_loc: Some((loc.to_vec(), key.clone())),
            });
        }
    }
    out
}

/// Attempt one in-place repair for an issue. Only known object fields are touched,
/// only via the narrow conversions listed at the top of this module.
fn apply_issue_repair(work: &mut Value, issue: &Issue) -> Option<ToolInputRepair> {
    if !issue.known_field {
        return None;
    }
    let (parent_loc, key) = issue.parent_loc.as_ref()?;
    let parent = navigate_mut(work, parent_loc)?.as_object_mut()?;
    let value = parent.get(key.as_str())?.clone();
    if value.is_null() && !issue.required {
        parent.remove(key.as_str());
        return Some(ToolInputRepair {
            kind: REPAIR_NULL_OPTIONAL_OMITTED,
            path: issue.path.clone(),
            before_type: "null",
            after_type: "omitted",
        });
    }
    if let Some(s) = value.as_str() {
        match issue.expected.as_str() {
            "boolean" => {
                let parsed = match s {
                    "true" => true,
                    "false" => false,
                    _ => return None,
                };
                parent.insert(key.clone(), Value::Bool(parsed));
                return Some(ToolInputRepair {
                    kind: REPAIR_SEMANTIC_BOOLEAN,
                    path: issue.path.clone(),
                    before_type: "string",
                    after_type: "boolean",
                });
            }
            "integer" => {
                if !is_decimal_integer_literal(s) {
                    return None;
                }
                let n: i64 = s.parse().ok()?;
                parent.insert(key.clone(), Value::from(n));
                return Some(ToolInputRepair {
                    kind: REPAIR_SEMANTIC_INTEGER,
                    path: issue.path.clone(),
                    before_type: "string",
                    after_type: "number",
                });
            }
            _ => {}
        }
    }
    if issue.expected != "array" {
        return None;
    }
    match &value {
        Value::String(s) => {
            if let Ok(arr) = serde_json::from_str::<Vec<Value>>(s.trim()) {
                parent.insert(key.clone(), Value::Array(arr));
                return Some(ToolInputRepair {
                    kind: REPAIR_STRINGIFIED_ARRAY,
                    path: issue.path.clone(),
                    before_type: "string",
                    after_type: "array",
                });
            }
            if issue.items_type.as_deref() != Some("string") {
                return None;
            }
            parent.insert(key.clone(), Value::Array(vec![value]));
            Some(ToolInputRepair {
                kind: REPAIR_BARE_STRING_TO_ARRAY,
                path: issue.path.clone(),
                before_type: "string",
                after_type: "array",
            })
        }
        Value::Object(m) if m.is_empty() => {
            parent.insert(key.clone(), Value::Array(vec![]));
            Some(ToolInputRepair {
                kind: REPAIR_EMPTY_OBJECT_TO_ARRAY,
                path: issue.path.clone(),
                before_type: "object",
                after_type: "array",
            })
        }
        _ => None,
    }
}

fn is_decimal_integer_literal(s: &str) -> bool {
    // Bare "-" strips to an empty `rest`, so the non-empty check already
    // rejects it — no separate guard needed.
    let rest = s.strip_prefix('-').unwrap_or(s);
    !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
}

/// Resolve the schema's primary type: a plain string, or the first
/// non-"null" entry of a type array.
fn schema_type(schema: &Value) -> String {
    match schema.get("type") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .find(|s| *s != "null")
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn type_allows_null(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(s)) => s == "null",
        Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).any(|s| s == "null"),
        _ => false,
    }
}

fn schema_array_items_type(schema: &Value) -> Option<String> {
    let items = schema.get("items")?;
    items.is_object().then(|| schema_type(items))
}

fn schema_min_items(schema: &Value) -> Option<usize> {
    let v = schema.get("minItems")?.as_f64()?;
    (v >= 0.0 && v.trunc() == v).then_some(v as usize)
}

fn is_json_integer(value: &Value) -> bool {
    if value.is_i64() || value.is_u64() {
        return true;
    }
    value.as_f64().is_some_and(|f| f.trunc() == f)
}

fn required_set(required: Option<&Value>) -> Vec<String> {
    required
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn is_path_string_field(key: &str) -> bool {
    matches!(key.trim(), "file_path" | "path" | "cwd" | "directory")
}

/// Unwrap `[text](http://target)` into `text` when the link is a
/// model-mangled file path: the URL (minus protocol) must round-trip to
/// the link text or the full replacement, and any prefix must end with a
/// path separator. Normal markdown links (text ≠ target) are left alone.
fn unwrap_markdown_autolink_path(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let start = trimmed.find('[')?;
    let prefix = &trimmed[..start];
    if !prefix.is_empty() && !prefix.ends_with('/') && !prefix.ends_with('\\') {
        return None;
    }
    let rest = &trimmed[start..];
    let end_text = rest.find(']')?;
    if end_text <= 1
        || end_text + 1 >= rest.len()
        || rest.as_bytes()[end_text + 1] != b'('
        || !rest.ends_with(')')
    {
        return None;
    }
    let text = &rest[1..end_text];
    let url = &rest[end_text + 2..rest.len() - 1];
    let target = strip_http_protocol(url)?;
    let replacement = format!("{prefix}{text}");
    let normalized_target = target.trim();
    if normalized_target != text.trim() && normalized_target != replacement.trim() {
        return None;
    }
    if replacement == value {
        return None;
    }
    Some(replacement)
}

fn strip_http_protocol(value: &str) -> Option<&str> {
    let value = value.trim();
    for prefix in ["http://", "https://"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            return Some(rest.trim());
        }
    }
    None
}

fn join_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

// ───────────────────────── lightweight schema validation ─────────────────────────

/// Lightweight, best-effort validation of `input` against a tool's
/// `input_schema`. Deliberately NOT a full JSON Schema validator (no extra
/// dependency): it checks only the two failure modes weak models hit most —
/// (1) a `required` field is missing or null (when null isn't allowed), and
/// (2) a present field's top-level JSON type contradicts the schema's
/// declared `type`. Returns `Ok(())` when no such violation is found (so
/// unmodeled constraints like enums / patterns pass through), or the first
/// violation as a human-readable string for a teaching error.
///
/// Runs at dispatch time AFTER [`repair_tool_input_for_spec`], so anything
/// the repair pass can fix never reaches here. A non-object schema or input
/// is treated as un-validatable and passes.
pub fn validate_against_schema(schema: &Value, input: &Value) -> Result<(), String> {
    let (Some(_), Some(obj)) = (schema.as_object(), input.as_object()) else {
        return Ok(());
    };
    for field in required_set(schema.get("required")) {
        let prop = schema
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|p| p.get(&field));
        let allows_null = prop.is_some_and(type_allows_null);
        match obj.get(&field) {
            None => return Err(format!("missing required field `{field}`")),
            Some(Value::Null) if !allows_null => {
                return Err(format!("required field `{field}` must not be null"))
            }
            _ => {}
        }
    }
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return Ok(());
    };
    for (key, value) in obj {
        let Some(prop) = props.get(key) else { continue };
        let expected = schema_type(prop);
        if expected.is_empty() || (value.is_null() && type_allows_null(prop)) {
            continue;
        }
        if !json_value_matches_type(value, &expected) {
            return Err(format!(
                "field `{key}` should be {expected}, got {}",
                json_type_name(value)
            ));
        }
    }
    Ok(())
}

/// Does `value`'s JSON type satisfy the schema `type` string? `integer`
/// accepts whole-valued numbers; `number` accepts any number.
fn json_value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "integer" => is_json_integer(value),
        "number" => value.is_number(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Synthesize a minimal valid example object from a tool's `input_schema`:
/// each required field (or every property when none are required) mapped to
/// a type-appropriate placeholder. Used to make invalid-argument errors
/// teaching: the model sees the shape it should have produced.
pub fn example_for_schema(schema: &Value) -> Value {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return Value::Object(Map::new());
    };
    let required = required_set(schema.get("required"));
    let keys: Vec<&String> = if required.is_empty() {
        props.keys().collect()
    } else {
        required.iter().filter(|k| props.contains_key(*k)).collect()
    };
    let mut out = Map::new();
    for key in keys {
        if let Some(prop) = props.get(key) {
            out.insert(key.clone(), placeholder_for_schema(prop));
        }
    }
    Value::Object(out)
}

fn placeholder_for_schema(prop: &Value) -> Value {
    match schema_type(prop).as_str() {
        "string" => Value::String("<string>".into()),
        "integer" | "number" => Value::from(0),
        "boolean" => Value::Bool(false),
        "array" => {
            let item = schema_array_items_type(prop)
                .map(|t| placeholder_for_schema(&serde_json::json!({ "type": t })))
                .unwrap_or(Value::String("<item>".into()));
            Value::Array(vec![item])
        }
        "object" => Value::Object(Map::new()),
        _ => Value::String("<value>".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── truncation repair ──

    // ── schema validation & example synthesis ──

    fn validate_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "limit": { "type": "integer" },
                "flag": { "type": "boolean" },
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    #[test]
    fn validate_accepts_well_formed_input() {
        assert!(validate_against_schema(&validate_schema(), &json!({"path": "a", "limit": 3})).is_ok());
    }

    #[test]
    fn validate_flags_missing_required() {
        let err = validate_against_schema(&validate_schema(), &json!({"limit": 3})).unwrap_err();
        assert!(err.contains("path"), "{err}");
    }

    #[test]
    fn validate_flags_wrong_type() {
        let err =
            validate_against_schema(&validate_schema(), &json!({"path": "a", "limit": "nope"}))
                .unwrap_err();
        assert!(err.contains("limit") && err.contains("integer"), "{err}");
    }

    #[test]
    fn validate_flags_null_required() {
        let err = validate_against_schema(&validate_schema(), &json!({"path": null})).unwrap_err();
        assert!(err.contains("path") && err.contains("null"), "{err}");
    }

    #[test]
    fn example_uses_required_fields_with_typed_placeholders() {
        let ex = example_for_schema(&validate_schema());
        assert_eq!(ex["path"], json!("<string>"));
        // Only required fields appear when `required` is non-empty.
        assert!(ex.get("limit").is_none());
    }

    #[test]
    fn example_falls_back_to_all_properties_when_none_required() {
        let schema = json!({
            "type": "object",
            "properties": { "a": { "type": "integer" }, "b": { "type": "boolean" } }
        });
        let ex = example_for_schema(&schema);
        assert_eq!(ex["a"], json!(0));
        assert_eq!(ex["b"], json!(false));
    }

    #[test]
    fn truncation_closes_truncated_json() {
        let res = repair_truncated_json(r#"{"file_path":"README.md","offset":0"#);
        assert!(res.changed);
        assert_eq!(res.repaired, r#"{"file_path":"README.md","offset":0}"#);
    }

    #[test]
    fn truncation_leaves_valid_json_untouched() {
        let res = repair_truncated_json(r#"{"k":"v"}"#);
        assert!(!res.changed);
        assert_eq!(res.repaired, r#"{"k":"v"}"#);
    }

    #[test]
    fn truncation_salvages_prefix_when_value_missing() {
        let res = repair_truncated_json(r#"{"file_path":"README.md","offset":0,"limit":"#);
        assert!(res.changed);
        assert_eq!(res.repaired, r#"{"file_path":"README.md","offset":0}"#);
    }

    #[test]
    fn truncation_empty_input_becomes_empty_object() {
        let res = repair_truncated_json("   ");
        assert!(res.changed);
        assert_eq!(res.repaired, "{}");
    }

    #[test]
    fn truncation_closes_unterminated_string() {
        let res = repair_truncated_json(r#"{"file_path":"READ"#);
        assert!(res.changed);
        let v: Value = serde_json::from_str(&res.repaired).unwrap();
        assert_eq!(v, json!({"file_path": "READ"}));
    }

    #[test]
    fn truncation_strips_trailing_comma() {
        let res = repair_truncated_json(r#"{"a":1,}"#);
        assert!(res.changed);
        let v: Value = serde_json::from_str(&res.repaired).unwrap();
        assert_eq!(v, json!({"a": 1}));
    }

    #[test]
    fn truncation_salvages_top_level_pairs_from_garbage() {
        // Mismatched closer defeats close_likely_json AND prefix salvage;
        // the pair scan still rescues the flat k:v pairs.
        let res = repair_truncated_json(r#"{"cmd":"ls","count":3]"#);
        assert!(res.changed);
        let v: Value = serde_json::from_str(&res.repaired).unwrap();
        assert_eq!(v, json!({"cmd": "ls", "count": 3}));
    }

    #[test]
    fn truncation_falls_back_to_empty_object() {
        let res = repair_truncated_json("not json at all");
        assert!(res.changed);
        assert_eq!(res.repaired, "{}");
    }

    // ── schema-guided repair ──

    #[test]
    fn spec_repairs_safe_field_aliases() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"}
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": false
        });
        let input = json!({
            "filePath": "src/lib.rs",
            "oldString": "before",
            "newString": "after",
            "replaceAll": "true"
        });
        let (out, repairs) = repair_tool_input_for_spec(&schema, &input).unwrap();
        assert_eq!(out, json!({
            "path": "src/lib.rs",
            "old_string": "before",
            "new_string": "after",
            "replace_all": true
        }));
        assert!(repairs.iter().any(|r| r.kind == REPAIR_FIELD_ALIAS && r.path == "path"));
        assert!(repairs.iter().any(|r| r.kind == REPAIR_SEMANTIC_BOOLEAN && r.path == "replace_all"));
    }

    fn repair_test_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "ignore": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
                "content": {"type": "string"}
            },
            "required": ["prompts"],
            "additionalProperties": false
        })
    }

    fn grep_test_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "literal_text": {"type": "boolean"},
                "limit": {"type": "integer"}
            },
            "required": ["pattern"],
            "additionalProperties": false
        })
    }

    fn path_test_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "path": {"type": "string"},
                "cwd": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["content"],
            "additionalProperties": false
        })
    }

    #[test]
    fn spec_leaves_valid_input_unchanged() {
        let input = json!({"prompts": ["a"], "content": "[\"not an arg array\"]"});
        assert!(repair_tool_input_for_spec(&repair_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_omits_optional_null() {
        let input = json!({"prompts": ["a"], "limit": null});
        let (out, repairs) = repair_tool_input_for_spec(&repair_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_NULL_OPTIONAL_OMITTED);
        assert_eq!(repairs[0].path, "limit");
        assert!(out.get("limit").is_none());
    }

    #[test]
    fn spec_does_not_omit_required_null() {
        let input = json!({"prompts": null});
        assert!(repair_tool_input_for_spec(&repair_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_parses_stringified_array_before_wrapping() {
        let input = json!({"prompts": "[\"a\",\"b\"]"});
        let (out, repairs) = repair_tool_input_for_spec(&repair_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_STRINGIFIED_ARRAY);
        assert_eq!(repairs[0].path, "prompts");
        assert_eq!(repairs[0].before_type, "string");
        assert_eq!(repairs[0].after_type, "array");
        assert_eq!(out["prompts"], json!(["a", "b"]));
    }

    #[test]
    fn spec_wraps_bare_string_for_string_array() {
        let input = json!({"prompts": "a"});
        let (out, repairs) = repair_tool_input_for_spec(&repair_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_BARE_STRING_TO_ARRAY);
        assert_eq!(out["prompts"], json!(["a"]));
    }

    #[test]
    fn spec_rejects_empty_array_when_min_items_fails() {
        // {} → [] would repair, but the second pass sees len 0 < minItems 1
        // and reverts everything.
        let input = json!({"prompts": {}});
        assert!(repair_tool_input_for_spec(&repair_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_repairs_empty_object_to_optional_array() {
        let input = json!({"prompts": ["a"], "ignore": {}});
        let (out, repairs) = repair_tool_input_for_spec(&repair_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_EMPTY_OBJECT_TO_ARRAY);
        assert_eq!(repairs[0].path, "ignore");
        assert_eq!(out["ignore"], json!([]));
    }

    #[test]
    fn spec_leaves_unknown_field_invalid() {
        // "prompts" would be repairable, but the unknown "extra" field is
        // an unrepairable issue → whole repair reverts.
        let input = json!({"prompts": "a", "extra": true});
        assert!(repair_tool_input_for_spec(&repair_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_coerces_semantic_boolean_string() {
        let input = json!({"pattern": "needle", "literal_text": "false"});
        let (out, repairs) = repair_tool_input_for_spec(&grep_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_SEMANTIC_BOOLEAN);
        assert_eq!(repairs[0].path, "literal_text");
        assert_eq!(repairs[0].before_type, "string");
        assert_eq!(repairs[0].after_type, "boolean");
        assert_eq!(out["literal_text"], json!(false));
    }

    #[test]
    fn spec_coerces_semantic_integer_string() {
        let input = json!({"pattern": "needle", "limit": "30"});
        let (out, repairs) = repair_tool_input_for_spec(&grep_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_SEMANTIC_INTEGER);
        assert_eq!(repairs[0].path, "limit");
        assert_eq!(repairs[0].before_type, "string");
        assert_eq!(repairs[0].after_type, "number");
        assert_eq!(out["limit"], json!(30));
    }

    #[test]
    fn spec_leaves_invalid_semantic_boolean_string() {
        let input = json!({"pattern": "needle", "literal_text": "no"});
        assert!(repair_tool_input_for_spec(&grep_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_unwraps_markdown_autolink_file_path() {
        let input = json!({"file_path": "[README.md](http://README.md)", "content": "x"});
        let (out, repairs) = repair_tool_input_for_spec(&path_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_MARKDOWN_AUTOLINK_PATH);
        assert_eq!(repairs[0].path, "file_path");
        assert_eq!(repairs[0].before_type, "string");
        assert_eq!(repairs[0].after_type, "string");
        assert_eq!(out["file_path"], json!("README.md"));
    }

    #[test]
    fn spec_unwraps_markdown_autolink_path_with_prefix() {
        let input = json!({"path": "sub/[a.txt](http://a.txt)", "content": "x"});
        let (out, repairs) = repair_tool_input_for_spec(&path_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_MARKDOWN_AUTOLINK_PATH);
        assert_eq!(repairs[0].path, "path");
        assert_eq!(out["path"], json!("sub/a.txt"));
    }

    #[test]
    fn spec_unwraps_markdown_autolink_cwd() {
        let input = json!({"cwd": "[internal](http://internal)", "content": "x"});
        let (out, repairs) = repair_tool_input_for_spec(&path_test_schema(), &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_MARKDOWN_AUTOLINK_PATH);
        assert_eq!(repairs[0].path, "cwd");
        assert_eq!(out["cwd"], json!("internal"));
    }

    #[test]
    fn spec_does_not_unwrap_markdown_in_content() {
        let input = json!({"file_path": "README.md", "content": "[README.md](http://README.md)"});
        assert!(repair_tool_input_for_spec(&path_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_does_not_unwrap_normal_markdown_link() {
        let input = json!({"file_path": "[click](https://example.com)", "content": "x"});
        assert!(repair_tool_input_for_spec(&path_test_schema(), &input).is_none());
    }

    #[test]
    fn spec_repairs_nested_array_path() {
        let schema = json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "options": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "required": ["questions"],
            "additionalProperties": false
        });
        let input = json!({"questions": [{"options": "[\"yes\",\"no\"]"}]});
        let (out, repairs) = repair_tool_input_for_spec(&schema, &input).unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].kind, REPAIR_STRINGIFIED_ARRAY);
        assert_eq!(repairs[0].path, "questions[0].options");
        assert_eq!(out["questions"][0]["options"], json!(["yes", "no"]));
    }
}
