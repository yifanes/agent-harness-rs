pub mod agent_loop;
pub mod compaction;
pub mod context;
pub mod event;
pub mod history_sanitize;
pub mod mcp;
pub mod model;
pub mod runner;
pub mod shell_risk;
pub mod tool_repair;
pub mod tools;

pub use agent_loop::{AgentLoopHarness, CompactionPolicy};
pub use compaction::{
    estimate_messages_tokens, resolve_context_window_tokens, CompactionContext, CompactionError,
    CompactionStrategy, SummarizeCompactionStrategy,
};
pub use context::jsonl::{append_context, load_context, rewrite_context};
pub use event::{HarnessInternalEvent, HarnessUsage, NativeHarnessError, NativeTurnInput};
pub use history_sanitize::{sanitize_history, SanitizeDiagnostics};
pub use mcp::{
    CompositeToolRuntime, McpClient, McpError, McpServerConfig, McpToolRuntime, McpTransport,
    MCP_PROTOCOL_VERSION,
};
pub use model::{
    collect_model_response, AnthropicConfig, AnthropicModelClient, AssistantThinking, ChatMessage,
    ImageData, ImageSource, ModelChunk, ModelClient, ModelClientError, ModelResponse,
    ModelTurnInput, OpenAiCompatibleConfig, OpenAiCompatibleModelClient, ScriptedModelClient,
    ToolChoice, UserAttachment,
};
pub use runner::{FakeNativeHarness, NativeHarness, ToolCapableFakeHarness};
pub use shell_risk::{classify_shell_command, ShellRiskDecision, ShellRiskLevel};
pub use tools::{
    builtin_tool_specs, fs_glob, fs_glob_bounded, resolve_edit_search, simple_glob_match,
    EditSearchError, MockToolRuntime, ResolvedEditSearch, ToolFailure, ToolFailureKind,
    ToolInvocation, ToolOutcome, ToolRuntime, ToolRuntimeError, ToolSpec, FS_GLOB_IGNORED_DIRS,
    MAX_FS_GLOB_RESULTS,
};

// ── New: local tool runtime ───────────────────────────────────────────────────
#[cfg(feature = "local-tools")]
pub use tools::local::{EmitFn, LocalToolConfig, LocalToolRuntime};
#[cfg(feature = "local-tools")]
pub use tools::approval::{ApprovalGate, PlanApproval, YoloApproval};

// ── New: sandbox tool runtime ─────────────────────────────────────────────────
pub use tools::sandbox::{ExecResult, SandboxExecutor, SandboxToolConfig, SandboxToolRuntime};
