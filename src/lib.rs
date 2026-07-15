pub mod agent_loop;
pub mod compaction;
pub mod context;
pub mod model_limits;
pub mod event;
pub mod history_sanitize;
pub mod mcp;
pub mod model;
pub mod runner;
pub mod shell_risk;
pub mod skills;
pub mod tool_repair;
pub mod tools;

pub use agent_loop::{AgentLoopHarness, CompactionPolicy};
pub use compaction::{
    estimate_messages_tokens, prune_tool_outputs, resolve_context_window_tokens,
    CompactionContext, CompactionError, CompactionOutcome, CompactionStrategy,
    DefaultCompactionStrategy, SummarizeCompactionStrategy, PRUNE_MIN_CONTENT_TOKENS,
    PRUNE_MINIMUM_TOKENS, PRUNE_PROTECT_TURNS, PRUNED_CONTENT_STUB,
};
pub use model_limits::{
    prefetch as prefetch_model_limits, resolve_limits, ModelLimits,
    DEFAULT_CONTEXT_TOKENS, DEFAULT_OUTPUT_TOKENS,
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
    HostedTool, ImageData, ImageSource, ModelChunk, ModelClient, ModelClientError, ModelResponse,
    ModelTurnInput, OpenAiCompatibleConfig, OpenAiCompatibleModelClient, OpenAiResponsesConfig,
    OpenAiResponsesModelClient, ScriptedModelClient, ToolChoice, UserAttachment,
};
pub use runner::{FakeNativeHarness, NativeHarness, ToolCapableFakeHarness};
pub use shell_risk::{
    classify_shell_command, classify_shell_command_with_policy, ShellRiskDecision, ShellRiskLevel,
    ShellRiskPolicy, SHELL_RISK_POLICY_ENV,
};
pub use skills::{
    parse_skill_frontmatter, LocalSkillSource, SkillError, SkillLoader, SkillMetadata,
    SkillPromptRenderer, SkillSource, SkillsManager,
};
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

// ── New: e2b sandbox ──────────────────────────────────────────────────────────
#[cfg(feature = "e2b")]
pub use tools::e2b::{E2bConfig, E2bError, E2bToolRuntime};
