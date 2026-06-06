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

## Quick start

```toml
[dependencies]
agent-harness-rs = "0.1"

# For e2b sandbox support:
agent-harness-rs = { version = "0.1", features = ["e2b"] }
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
    base_url: "https://api.openai.com".into(),
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
