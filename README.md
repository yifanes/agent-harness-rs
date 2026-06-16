# agent-harness-rs

[![Crates.io](https://img.shields.io/crates/v/agent-harness-rs.svg)](https://crates.io/crates/agent-harness-rs)
[![docs.rs](https://docs.rs/agent-harness-rs/badge.svg)](https://docs.rs/agent-harness-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Agent loop harness for building LLM-powered coding agents. Provides a complete runtime with tool execution, context management, MCP support, and e2b sandbox integration.

## Features

- **Agent loop** — OpenAI-compatible streaming model client with retry, reconnect, and compaction
- **Local tools** — `bash`, `read`, `write`, `edit`, `glob`, `grep` with approval gate (`feature = "local-tools"`, default)
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
