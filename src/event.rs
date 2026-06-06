use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::model::{ChatMessage, UserAttachment};

/// Token / cache usage reported by the model client at the end of a turn.
///
/// Kept independent of any wire format (no `grpc::*` import) so the harness
/// crate stays a pure domain library. The `core::native_adapter` layer
/// converts this to the proto `SessionUsage` shape on the way out.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HarnessUsage {
    /// Total input tokens spent this turn (main steps + compaction
    /// summarize calls combined). Same field the wire `SessionUsage`
    /// surfaces today — keeps HR's existing dashboards correct.
    pub input_tokens: u64,
    /// Total output tokens this turn (main + compaction).
    pub output_tokens: u64,
    /// Cache-read tokens (Anthropic). 0 = unknown / not applicable.
    pub cache_read_input_tokens: u64,
    /// Cache-create tokens (Anthropic). 0 = unknown / not applicable.
    pub cache_creation_input_tokens: u64,
    /// Subset of `input_tokens` spent specifically on compaction
    /// `summarize` calls. 0 = no compaction ran this turn (or it ran
    /// but the provider didn't report usage). Surfaced in
    /// tracing for now; not yet wired to proto `SessionUsage` —
    /// when HR wants the breakdown on the wire, add fields to the
    /// proto + map in `native_adapter::harness_usage_to_proto`.
    pub compaction_input_tokens: u64,
    /// Subset of `output_tokens` spent on compaction.
    pub compaction_output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HarnessInternalEvent {
    AssistantTextChunk {
        msg_id: String,
        delta: String,
    },
    AssistantThinkingChunk {
        msg_id: String,
        delta: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        id: String,
        output: Result<Value, String>,
    },
    /// Compaction fired between steps and the running history was
    /// folded. Counts let HR estimate how aggressive compaction is at
    /// this point in the turn (and whether to raise alerts). Token
    /// counts come from the harness's own estimator — see
    /// `compaction::estimate_messages_tokens` — and are approximate.
    ///
    /// Native_adapter currently does NOT project this onto an
    /// `AdapterEvent` (no proto field reserved for it yet). It surfaces
    /// only through structured tracing on the RD side; tests can still
    /// assert on it in the harness's mpsc channel.
    CompactionApplied {
        original_message_count: usize,
        compacted_message_count: usize,
        original_tokens: u64,
        compacted_tokens: u64,
    },
    TurnEnd {
        stop_reason: String,
        usage: Option<HarnessUsage>,
        /// Final messages history at turn end (post-compaction, includes
        /// the assistant's reply and any tool round-trips). RD captures
        /// this snapshot to seed the next dispatch's `prior_messages`
        /// so multi-turn conversations in the same RD process don't
        /// re-prompt from scratch.
        ///
        /// This is **internal-only** state: it never lands on the wire.
        /// `native_adapter::HarnessEventAdapter::ingest` reads it via a
        /// side-channel (history_handle) and drops it before projecting
        /// to `AdapterEvent::TurnEnd`. Tests can still assert on the
        /// raw `HarnessInternalEvent` directly.
        final_messages: Vec<ChatMessage>,
    },
}

#[derive(Debug, Clone)]
pub struct NativeTurnInput {
    /// The user-facing prompt that triggered this turn. Lands as the first
    /// `ChatMessage::User.content` the model sees.
    pub prompt_text: String,
    /// Pre-composed system prompt (e.g. `spec_snapshot.system_prompt` +
    /// `driver.append_system_prompt`). `None` ⇒ no system message sent.
    pub system_prompt: Option<String>,
    /// Non-text attachments lifted from the inbound `UserMessagePayload`
    /// content blocks. Empty Vec ⇒ pure-text prompt. Lands on the first
    /// `ChatMessage::User.attachments` so each provider's projection
    /// can render them as image / file content blocks.
    pub attachments: Vec<UserAttachment>,
    /// Optional cancellation handle. When fired, the harness loop
    /// short-circuits at the next stream-chunk await (or before the
    /// next step starts) and emits `TurnEnd { stop_reason: "interrupt" }`.
    /// `None` ⇒ harness cannot be cancelled mid-flight (fine for tests
    /// and for fire-and-forget turns); production wires this through
    /// from RD's `active_native_cancel`.
    pub cancel_token: Option<CancellationToken>,
    /// Prior `messages` history to seed the loop with (snapshot taken
    /// at the previous `TurnEnd.final_messages`). Empty Vec = fresh
    /// conversation; the loop appends the new
    /// `ChatMessage::User { prompt_text, attachments }` after this Vec.
    /// RD owns the storage; harness just consumes and returns the new
    /// final snapshot on `TurnEnd`.
    pub prior_messages: Vec<ChatMessage>,
}

impl PartialEq for NativeTurnInput {
    fn eq(&self, other: &Self) -> bool {
        // CancellationToken doesn't implement PartialEq (it's a runtime
        // handle, not a value). Two NativeTurnInputs are considered
        // equal iff the *value-typed* fields match; the cancel handle
        // is opaque ambient state.
        self.prompt_text == other.prompt_text
            && self.system_prompt == other.system_prompt
            && self.attachments == other.attachments
            && self.prior_messages == other.prior_messages
    }
}

/// Categorised native-path failure. Each variant maps 1:1 to an
/// `acpx::NativeFaultCategory` in `core::native_adapter::native_error_to_invoker`,
/// which is in turn projected to a `RuntimeError` by `core::error::
/// native_fault_to_runtime_error`. Keeping the buckets named here lets the
/// harness crate produce structured errors without taking on any grpc /
/// proto dependency.
#[derive(Debug, thiserror::Error)]
pub enum NativeHarnessError {
    #[error("native harness failed: {0}")]
    Failed(String),

    #[error("native harness event encode failed: {0}")]
    Encode(String),

    #[error("native harness channel closed")]
    ChannelClosed,

    /// LLM provider returned 429.
    #[error("model rate limit: {0}")]
    ModelRateLimit(String),

    /// LLM provider returned 401 / 403.
    #[error("model auth error: {0}")]
    ModelAuth(String),

    /// Prompt + completion exceeded the model's context window.
    #[error("model context overflow: {0}")]
    ModelContextOverflow(String),

    /// Transport-level failure (DNS / TCP / TLS / truncated body).
    #[error("model network error: {0}")]
    ModelNetwork(String),

    /// HTTP 400 that is a config error (wrong model name, invalid params).
    /// Not retryable — the caller must fix their configuration.
    #[error("model bad request: {0}")]
    ModelBadRequest(String),

    /// HTTP 5xx or transient server error. Retryable.
    #[error("model server error: {0}")]
    ModelServerError(String),

    /// Any other model-side failure not captured above.
    #[error("model other error: {0}")]
    ModelOther(String),

    /// Sandbox / tool runtime hard error (process spawn refused, envd
    /// unreachable). NOT a `ToolFailure` — those are domain failures the
    /// model can observe and recover from; this is RD-side infrastructure
    /// breaking under the tool.
    #[error("tool runtime error: {0}")]
    ToolRuntime(String),
}
