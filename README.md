# agent-harness-rs

[![Crates.io](https://img.shields.io/crates/v/agent-harness-rs.svg)](https://crates.io/crates/agent-harness-rs)
[![docs.rs](https://docs.rs/agent-harness-rs/badge.svg)](https://docs.rs/agent-harness-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Agent loop harness for building LLM-powered coding agents. Provides a complete runtime with tool execution, context management, MCP support, and e2b sandbox integration.

## Features

- **Agent loop** — OpenAI-compatible and Anthropic streaming model clients with retry, reconnect, silent-stop detection, and compaction
- **Local tools** — `bash`, `read`, `write`, `edit`, `glob`, `grep`, `web_fetch` with approval gate (`feature = "local-tools"`, default)
- **Hosted tools** — Provider-native tools that execute at the model provider; currently Anthropic native `web_search`
- **Sandbox tools** — Generic `SandboxExecutor` trait for any remote sandbox
- **E2b integration** — `E2bToolRuntime` via Connect Protocol to envd (`feature = "e2b"`)
- **Context persistence** — JSONL-based context store with incremental append and compaction rewrite
- **MCP support** — HTTP and stdio MCP server integration via `CompositeToolRuntime`
- **Model limits catalog** — async `models.dev` fetch with disk cache, backoff retry, and offline fallback for context-window / output-token resolution

## Quick start

```toml
[dependencies]
agent-harness-rs = "0.2"

# For e2b sandbox support:
agent-harness-rs = { version = "0.2", features = ["e2b"] }
```

```rust
use harness::{
    AgentLoopHarness, NativeTurnInput, OpenAiCompatibleConfig, OpenAiCompatibleModelClient,
    LocalToolRuntime, LocalToolConfig, YoloApproval,
};
use std::sync::Arc;
use std::path::PathBuf;

// Local tool runtime (runs bash/read/write on your machine)
let tools = LocalToolRuntime::new(LocalToolConfig {
    cwd: Some(PathBuf::from("/path/to/project")),
    approval: Arc::new(YoloApproval),
    emit: Arc::new(|_| {}),
});

let model = OpenAiCompatibleModelClient::new(OpenAiCompatibleConfig {
    // Full API prefix INCLUDING the version segment. The client appends only
    // `/chat/completions`. For other OpenAI-compatible providers use their own
    // prefix, e.g. GLM: "https://open.bigmodel.cn/api/paas/v4".
    base_url: "https://api.openai.com/v1".into(),
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    model: "gpt-4o".into(),
    ..Default::default()
});

let harness = AgentLoopHarness::new(model, tools);

let mut rx = harness.run_turn(NativeTurnInput {
    prompt_text: "List the Rust files in this project".into(),
    system_prompt: None,
    attachments: vec![],
    cancel_token: None,
    prior_messages: vec![],
    context_path: Some(PathBuf::from("/tmp/my-session.jsonl")),
}).await?;

while let Some(event) = rx.recv().await {
    println!("{event:?}");
}
```

## Web tools

`agent-harness-rs` intentionally separates harness-executed tools from
provider-executed hosted tools.

### `web_fetch`

`web_fetch` is a built-in read-only tool implemented by the harness. It fetches
a known HTTP/HTTPS URL and returns readable content. It does **not** search the
web or discover URLs.

Supported inputs:

| Field | Default | Purpose |
|---|---:|---|
| `url` | required | HTTP/HTTPS URL to fetch |
| `format` | `markdown` | One of `markdown`, `text`, or `html` |
| `max_length` | `50000` | Maximum returned characters, capped at `200000` |
| `timeout_ms` | `20000` | Request timeout, capped at `60000` |

The tool follows redirects, rejects non-HTTP schemes and obvious binary/image
content, caps downloaded bodies at 5 MiB, and returns JSON containing `url`,
`final_url`, `status`, `content_type`, `format`, `content`, and `truncated`.

`PlanApproval` treats `web_fetch` as read-only, so planning-mode agents can use
it alongside `read`, `glob`, and `grep`.

### Hosted `web_search`

`web_search` is modeled as a provider-hosted tool, not a normal
`ToolRuntime` tool. The provider executes it server-side and the harness only
passes the capability through the model request.

Currently supported:

- Anthropic Messages API: `HostedTool::WebSearch` is projected as
  `{"type":"web_search_20250305","name":"web_search"}`.
- OpenAI-compatible Chat Completions: explicitly unsupported. OpenAI web
  search requires a future Responses API client; this crate fails fast instead
  of pretending search is available.
- Other providers: unsupported unless their model client adds a hosted-tool
  projection.

```rust
use harness::{AgentLoopHarness, HostedTool};

// Convenience: enable Anthropic native web_search with provider defaults.
let harness = AgentLoopHarness::new(model, tools).with_web_search();

// Or set Anthropic's max_uses cap explicitly.
let harness = AgentLoopHarness::new(model, tools)
    .with_hosted_tools(vec![HostedTool::WebSearch { max_uses: Some(3) }]);
```

## Silent-stop detection

If a model step ends with no tool calls and no user-visible text, the harness no
longer treats it as a successful turn. Empty or whitespace-only final output with
`stop_reason = "end_turn"` or `"max_tokens"` returns a model error containing
`silent_stop`.

This catches provider/model failures where a turn would otherwise look
successful while delivering no answer and taking no action.

## Changelog

Every behavior change should be recorded in `CHANGELOG.md` before release. This
project uses patch-only version bumps within the current minor line, so the next
release after `0.2.5` is `0.2.6`.

### E2b sandbox

```rust
use harness::{AgentLoopHarness, E2bConfig, E2bToolRuntime, NativeTurnInput};

let tools = E2bToolRuntime::connect(E2bConfig::new(
    std::env::var("E2B_SANDBOX_ID").unwrap(),
    std::env::var("E2B_API_KEY").unwrap(),
)).await?;

let harness = AgentLoopHarness::new(model, tools);
```

## Model limits catalog

Context-window and output-token limits are resolved per model from
[`models.dev`](https://models.dev) (the public model registry opencode
also uses), with a best-effort strategy that never blocks the agent loop:

1. **In-memory table** populated by a fire-and-forget background fetch.
   On the first `resolve_limits()` call a fetch is spawned; while it is
   in flight (typically during early LLM warm-up) callers get the
   fallback value, and pick up the real value on the next turn.
2. **Disk cache** at `<cache_dir>/agent-harness-rs/models.json` (5 min
   TTL, atomic tempfile + `rename` write) so a network blip mid-session
   still serves real values.
3. **Offline fallback table** — the legacy hand-encoded claude/gpt/
   o-series/minimax/deepseek mappings, so behavior never regresses.
4. **Conservative default** `{ context: 128_000, output: 8_192 }`.

The fetch retries 3× with exponential backoff (+ jitter) and a 10 s
per-request timeout; on final failure it logs and falls back silently.

```rust
use harness::{resolve_limits, prefetch_model_limits};

// Optional: warm the cache before the first turn. Safe to skip — the
// first resolve_limits() triggers it lazily.
prefetch_model_limits();

// Fast, non-async, never blocks.
let limits = resolve_limits("claude-opus-4-7");
// limits.context  — used for compaction thresholds
// limits.output   — model's per-completion output cap
```

Configuration via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_HARNESS_MODELS_URL` | `https://models.dev/api.json` | Override the registry endpoint |
| `AGENT_HARNESS_CACHE_PATH` | `<cache_dir>/agent-harness-rs/models.json` | Relocate the disk cache |

## Approval modes

```rust
use harness::{YoloApproval, PlanApproval};

// Allow everything
Arc::new(YoloApproval)

// Read-only (hide bash/write/edit from model)
Arc::new(PlanApproval)

// Custom gate (e.g. ask user via UI)
struct MyApproval;
#[async_trait]
impl ApprovalGate for MyApproval {
    async fn approve(&self, inv: &ToolInvocation) -> bool {
        // prompt user, return true/false
    }
}
```

## License

MIT
