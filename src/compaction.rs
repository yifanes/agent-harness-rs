//! Context-window-aware compaction.
//!
//! Long-running native sessions accumulate a `messages` history that
//! eventually exceeds the model's context window. Without intervention
//! the next turn either truncates server-side (silently losing context)
//! or fails with a 400 / `context_length_exceeded`. Compaction folds the
//! mid-conversation into a single `<conversation-summary>` block while
//! preserving a tail of recent messages verbatim — same idea as OMA
//! `compaction.ts` and Claude Code's `/compact`.
//!
//! Strategy contract (`CompactionStrategy`):
//!   * `should_compact` — pure boolean gate; the agent loop checks this
//!     each step BEFORE building the next model request.
//!   * `compact` — best-effort; returns a fresh `Vec<ChatMessage>` that
//!     replaces the running history. Caller emits a separate event so
//!     `native_adapter` can mark the boundary on the wire.
//!
//! Producer of compaction:
//!   * `agent_loop::run_loop` constructs `SummarizeCompactionStrategy` per
//!     turn (cheap — no internal state) and calls `should_compact` /
//!     `compact` between steps.
//!
//! The strategy reuses the session's primary `ModelClient` to produce
//! the summary; that keeps cache keys hot on the prefix shared with the
//! main agent call (Anthropic prompt cache hits straight through).

use async_trait::async_trait;
use std::sync::Arc;

use crate::model::{
    collect_model_response, ChatMessage, ModelClient, ModelClientError, ModelResponse,
    ModelTurnInput,
};
use crate::tools::ToolSpec;

/// Fraction of the model's context window above which compaction fires.
/// 0.75 mirrors OMA's `TRIGGER_FRACTION` — leaves a ~25 % headroom for
/// the in-flight turn's own tool results and the model's reply.
pub const DEFAULT_TRIGGER_FRACTION: f64 = 0.75;

/// Floor on how many of the most recent messages compaction MUST keep
/// verbatim alongside the summary. Below this, a turn can't make
/// meaningful progress (one user + one assistant minimum).
pub const DEFAULT_TAIL_MIN_MESSAGES: usize = 4;

/// Soft cap on summary length when serialised back into a `User`
/// `<conversation-summary>` block. Above this we let the model decide,
/// but pass `max_tokens` hint so it doesn't ramble. 2 000 ≈ 8 KB —
/// enough for most multi-turn dialogues; OMA uses the same default.
pub const DEFAULT_SUMMARY_MAX_TOKENS: i32 = 2_000;

/// Mixed-script token estimator:
/// ASCII runs compress at ~4 chars/token, while CJK and other non-ASCII
/// text tokenizes at ≈1 token per char on every modern BPE vocabulary. The
/// previous flat `bytes / 4` heuristic underestimated Chinese text ~3×
/// (one CJK char = 3 UTF-8 bytes → counted as 0.75 tokens instead of ~1),
/// so compaction fired far too late on Chinese-heavy conversations.
/// Whitespace-only input costs 0. Still an estimator — not for billing.
pub fn estimate_tokens(s: &str) -> u64 {
    if s.trim().is_empty() {
        return 0;
    }
    let mut ascii: u64 = 0;
    let mut non_ascii: u64 = 0;
    for c in s.chars() {
        if c.is_ascii() {
            ascii += 1;
        } else {
            non_ascii += 1;
        }
    }
    ascii.div_ceil(4) + non_ascii
}

/// Per-message estimate: `estimate_tokens` over every text part, plus
/// small fixed overheads for structural wrapping (tool-call envelope ≈ 8
/// tokens, tool-result envelope ≈ 16 — same budgets as the old byte-based
/// +32 / +64 at 4 bytes/token). Floors at 1 so empty messages still cost.
pub fn estimate_chat_message_tokens(m: &ChatMessage) -> u64 {
    let tokens = match m {
        ChatMessage::User { content, .. } => estimate_tokens(content),
        ChatMessage::Assistant {
            text,
            tool_calls,
            thinking,
        } => {
            let text_tokens = text.as_deref().map(estimate_tokens).unwrap_or(0);
            let tc_tokens: u64 = tool_calls
                .iter()
                .map(|tc| estimate_tokens(&tc.input.to_string()) + estimate_tokens(&tc.name) + 8)
                .sum();
            let thinking_tokens = thinking
                .as_ref()
                .map(|t| {
                    estimate_tokens(&t.text)
                        + t.signature.as_deref().map(estimate_tokens).unwrap_or(0)
                })
                .unwrap_or(0);
            text_tokens + tc_tokens + thinking_tokens
        }
        ChatMessage::Tool { content, .. } => estimate_tokens(content) + 16,
    };
    tokens.max(1)
}

/// Sum the per-message estimates. Stable across providers because we
/// only look at the rendered text / JSON size, not the wire shape.
pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> u64 {
    messages.iter().map(estimate_chat_message_tokens).sum()
}

/// Per-model context window in tokens. Anthropic / OpenAI don't expose
/// this through their wire APIs, so we keep a hand-encoded table and
/// fall back to a conservative 128 000 (the smallest current "modern"
/// model window — GPT-4o / Claude Haiku 3.5).
///
/// Adding a new model: extend the table. Unknown model strings fall to
/// the default and a warning would be appropriate — but compaction is
/// best-effort so we just default rather than refuse to run.
pub fn resolve_context_window_tokens(model: &str) -> u64 {
    let m = model.to_ascii_lowercase();
    // Claude 4.6 / 4.7 1M context window (Sonnet / Opus extended).
    if m.contains("opus-4-7") || m.contains("opus-4-6") || m.contains("sonnet-4-6") {
        return 1_000_000;
    }
    // Anthropic Claude 3.x / 4.x: 200K
    if m.contains("claude") {
        return 200_000;
    }
    // OpenAI GPT-4 family: 128K
    if m.contains("gpt-4") || m.contains("gpt-4o") || m.contains("gpt-4.1") {
        return 128_000;
    }
    // OpenAI o1 / o3 reasoning: 200K
    if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") {
        return 200_000;
    }
    // MiniMax / DeepSeek / Groq commonly advertise 1M.
    if m.contains("minimax") || m.contains("deepseek") {
        return 1_000_000;
    }
    // Conservative default.
    128_000
}

/// Context passed to `compact()`. Owns the model client + tools the
/// summary call will use (mirrors the main agent's call so the prefix
/// stays cache-hot).
pub struct CompactionContext {
    pub system_prompt: Option<String>,
    pub model_client: Arc<dyn ModelClient>,
    pub context_window_tokens: u64,
    pub tools: Vec<ToolSpec>,
}

/// Result of a single `CompactionStrategy::compact` call. `messages` is
/// the folded history caller installs; `usage` is the token spend for
/// the summarize round trip (provider-reported via the same path as
/// main turn calls). `usage` is `None` when the provider elides usage
/// or when the strategy short-circuited without calling the model.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionOutcome {
    pub messages: Vec<ChatMessage>,
    pub usage: Option<crate::event::HarnessUsage>,
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("compaction model call failed: {0}")]
    ModelCall(#[from] ModelClientError),
    /// The model returned an empty summary. We refuse to fold history
    /// in that case — losing N turns of conversation for zero gain is
    /// strictly worse than the original "ran out of context" failure.
    /// Caller treats this as a no-op and lets the next turn try again.
    #[error("model produced empty summary; refusing to fold history")]
    EmptySummary,
}

#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Boolean gate. Pure (no side effects). Cheap to call every step.
    fn should_compact(&self, messages: &[ChatMessage], context_window_tokens: u64) -> bool;

    /// Fold history. Caller hands over the full `messages` list and
    /// expects a shorter list back (typically `[summary, ...tail]`).
    /// Failures bubble up; agent_loop treats them as "skip this turn's
    /// compaction" rather than failing the whole turn.
    async fn compact(
        &self,
        messages: Vec<ChatMessage>,
        ctx: &CompactionContext,
    ) -> Result<CompactionOutcome, CompactionError>;
}

/// CC-style strategy: send the FULL conversation back to the model with
/// a "summarize" user message appended, parse the reply as the summary,
/// and rebuild history as `[synthetic user with summary, ...tail]`.
///
/// Why this shape:
///   * Reusing the same model client keeps the prefix cache-hot on
///     Anthropic (the messages prefix is identical to the main agent's
///     last call modulo the summarize suffix).
///   * Keeping a tail of N recent messages verbatim preserves
///     fine-grained tool-call context the summary inevitably loses.
///   * The summary lives as a regular `User` message (wrapped in
///     `<conversation-summary>` tags) so downstream wire renderers
///     don't need a special block type.
pub struct SummarizeCompactionStrategy {
    pub trigger_fraction: f64,
    pub tail_min_messages: usize,
    pub summary_max_tokens: i32,
    pub summary_prompt: String,
}

impl Default for SummarizeCompactionStrategy {
    fn default() -> Self {
        Self {
            trigger_fraction: DEFAULT_TRIGGER_FRACTION,
            tail_min_messages: DEFAULT_TAIL_MIN_MESSAGES,
            summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.into(),
        }
    }
}

impl SummarizeCompactionStrategy {
    pub fn with_trigger_fraction(mut self, fraction: f64) -> Self {
        self.trigger_fraction = fraction;
        self
    }

    pub fn with_tail_min_messages(mut self, n: usize) -> Self {
        self.tail_min_messages = n;
        self
    }

    pub fn with_summary_max_tokens(mut self, n: i32) -> Self {
        self.summary_max_tokens = n;
        self
    }
}

/// Default summarize instruction. Asks for preservation of the load-
/// bearing facts (decisions, file paths, tool outputs, in-flight tasks,
/// next steps) so the agent can resume coherently. Mirrors the spirit
/// of OMA `compaction.ts:DEFAULT_SUMMARIZE_PROMPT` but rewritten to
/// avoid the existing "<conversation-summary>" tag the agent might
/// echo back in its summary (we add the tag externally).
pub const DEFAULT_SUMMARY_PROMPT: &str = "Produce a concise summary of the conversation above. \
    Preserve: key decisions, file paths, command outputs, in-flight tasks, and explicit \
    next steps. If a prior <conversation-summary> block exists in this conversation, \
    produce an UPDATED summary that supersedes it (incorporating new activity since). \
    Output only the summary text — no preamble, no closing remarks.";

#[async_trait]
impl CompactionStrategy for SummarizeCompactionStrategy {
    fn should_compact(&self, messages: &[ChatMessage], context_window_tokens: u64) -> bool {
        // Too few messages → never compact; the summarize call would
        // cost more than the prefix it's saving.
        if messages.len() <= self.tail_min_messages {
            return false;
        }
        let tokens = estimate_messages_tokens(messages);
        let threshold = ((context_window_tokens as f64) * self.trigger_fraction).round() as u64;
        tokens > threshold
    }

    async fn compact(
        &self,
        messages: Vec<ChatMessage>,
        ctx: &CompactionContext,
    ) -> Result<CompactionOutcome, CompactionError> {
        // Hard floor — refuse to compact if we'd end up with fewer than
        // the tail count. Keeps the strategy idempotent in degenerate
        // cases. `usage: None` because no model call ran.
        if messages.len() <= self.tail_min_messages {
            return Ok(CompactionOutcome {
                messages,
                usage: None,
            });
        }

        // Build the summarize request. Same system + same tools as the
        // main agent would use, then append one User message asking
        // for the summary. We DON'T set tools: vec![] — keeping them
        // makes the prefix bytes match what the main call sent, which
        // is what Anthropic's cache compares.
        let mut summarize_messages = messages.clone();
        summarize_messages.push(ChatMessage::User {
            content: self.summary_prompt.clone(),
            attachments: vec![],
        });
        let request = ModelTurnInput {
            system_prompt: ctx.system_prompt.clone(),
            messages: summarize_messages,
            tools: ctx.tools.clone(),
            tool_choice: crate::model::ToolChoice::Auto,
            parallel_tool_calls: None,
        };

        // Drain the stream into a single response — compaction doesn't
        // care about token-level emit; it just needs the text. The
        // model client trait's default `next` does this for us, but we
        // go through stream + collect explicitly so future Anthropic-
        // path strategies can elide tool calls / thinking blocks if
        // they want to.
        let stream = ctx.model_client.stream(request).await?;
        let response = collect_model_response(stream).await?;
        let (summary_text, usage) = match response {
            ModelResponse::Message { text, usage, .. } => (text, usage),
            // Model decided to call a tool instead of answering — give
            // up on this round, history stays put.
            ModelResponse::ToolCall { .. } => return Err(CompactionError::EmptySummary),
        };
        if summary_text.trim().is_empty() {
            return Err(CompactionError::EmptySummary);
        }

        // Tail preservation: keep the last N messages verbatim. We snip
        // off the head and replace it with a synthetic User block that
        // wraps the model's summary in `<conversation-summary>` tags
        // so the model recognises it as platform-injected context.
        let total = messages.len();
        let tail_count = self.tail_min_messages.min(total);
        let mut tail_start = total - tail_count;
        // Align tail to start on a User message — Anthropic / OpenAI
        // both expect the first non-summary message to be user-role.
        while tail_start < total && !matches!(messages[tail_start], ChatMessage::User { .. }) {
            tail_start += 1;
        }
        let tail: Vec<ChatMessage> = if tail_start < total {
            messages[tail_start..].to_vec()
        } else {
            // No user message in the tail window — shouldn't happen in
            // practice (a turn always starts with user) but cheap to
            // guard. Skip compaction in this case. `usage` was already
            // spent on the (now-discarded) summary call — surface it
            // anyway so HR can see the cost.
            return Ok(CompactionOutcome { messages, usage });
        };

        let mut out = Vec::with_capacity(tail.len() + 1);
        out.push(ChatMessage::User {
            content: serialize_summary(&summary_text),
            attachments: vec![],
        });
        out.extend(tail);
        Ok(CompactionOutcome {
            messages: out,
            usage,
        })
    }
}

fn serialize_summary(summary: &str) -> String {
    // Tag-wrap so the next turn's model treats this as injected context
    // and not a fresh user instruction. Anthropic / OpenAI both train
    // on similar markers; the exact tag matches OMA's `<conversation-
    // summary>` convention so anyone reading both codebases doesn't
    // double-take.
    format!("<conversation-summary>\n{summary}\n</conversation-summary>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelChunk, ModelClient};
    use crate::tools::ToolInvocation;
    use async_trait::async_trait;
    use futures::stream::{BoxStream, StreamExt};

    /// In-process model client that returns a fixed summary string.
    /// Test fixture only — production summarisation goes through the
    /// real model client.
    #[derive(Clone)]
    struct FixedSummaryClient {
        summary: String,
    }
    #[async_trait]
    impl ModelClient for FixedSummaryClient {
        async fn stream(
            &self,
            _input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            let chunks = vec![
                Ok(ModelChunk::TextDelta {
                    msg_id: "sum".into(),
                    delta: self.summary.clone(),
                }),
                Ok(ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                }),
            ];
            Ok(futures::stream::iter(chunks).boxed())
        }
    }

    fn user(s: &str) -> ChatMessage {
        ChatMessage::User {
            content: s.into(),
            attachments: vec![],
        }
    }

    fn assistant_text(s: &str) -> ChatMessage {
        ChatMessage::Assistant {
            text: Some(s.into()),
            tool_calls: vec![],
            thinking: None,
        }
    }

    fn tool_msg(id: &str, content: &str) -> ChatMessage {
        ChatMessage::Tool {
            tool_call_id: id.into(),
            content: content.into(),
            is_error: false,
            attachments: vec![],
        }
    }

    #[test]
    fn token_estimate_grows_with_content_size() {
        let small = user("hi");
        let big = user(&"x".repeat(8000));
        assert!(estimate_chat_message_tokens(&big) > estimate_chat_message_tokens(&small));
    }

    #[test]
    fn estimate_tokens_splits_ascii_and_cjk() {
        // ASCII at 4 chars/token (ceil), non-ASCII at 1 token/char,
        // whitespace-only is free.
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("   \n"), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2); // ceil(5/4)
        assert_eq!(estimate_tokens("你好世界"), 4); // 4 CJK chars = 4 tokens
        assert_eq!(estimate_tokens("hi你好"), 3); // ceil(2/4)=1 + 2
    }

    #[test]
    fn token_estimate_counts_cjk_near_one_per_char() {
        // 1000 CJK chars ≈ 1000 tokens. The old bytes/4 heuristic said
        // ~750 (3 UTF-8 bytes / 4); the rune-aware estimator must not
        // undercount, or compaction triggers too late on Chinese text.
        let cjk = user(&"汉".repeat(1000));
        let estimate = estimate_chat_message_tokens(&cjk);
        assert!(
            estimate >= 1000,
            "CJK undercounted: got {estimate}, want >= 1000"
        );
    }

    #[test]
    fn token_estimate_includes_tool_call_input() {
        // Same text length, but assistant carrying a tool_call should
        // cost more (we count the JSON arguments).
        let bare = assistant_text("done");
        let with_tool = ChatMessage::Assistant {
            text: Some("done".into()),
            tool_calls: vec![ToolInvocation {
                id: "tc".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "echo lots of bytes here for sure"}),
            }],
            thinking: None,
        };
        assert!(estimate_chat_message_tokens(&with_tool) > estimate_chat_message_tokens(&bare));
    }

    #[test]
    fn context_window_table_known_models() {
        assert_eq!(resolve_context_window_tokens("claude-opus-4-7"), 1_000_000);
        assert_eq!(
            resolve_context_window_tokens("claude-sonnet-4-6"),
            1_000_000
        );
        assert_eq!(resolve_context_window_tokens("claude-haiku-4-5"), 200_000);
        assert_eq!(resolve_context_window_tokens("claude-3-5-sonnet"), 200_000);
        assert_eq!(resolve_context_window_tokens("gpt-4o"), 128_000);
        assert_eq!(resolve_context_window_tokens("gpt-4.1-mini"), 128_000);
        assert_eq!(resolve_context_window_tokens("o3-mini"), 200_000);
        assert_eq!(resolve_context_window_tokens("MiniMax-M2"), 1_000_000);
        // Unknown → conservative default.
        assert_eq!(resolve_context_window_tokens("unknown-model"), 128_000);
    }

    #[test]
    fn should_compact_skips_when_below_threshold() {
        let strat = SummarizeCompactionStrategy::default();
        let messages = vec![user("hello"), assistant_text("hi")];
        // 200K window, tiny conversation — never fires.
        assert!(!strat.should_compact(&messages, 200_000));
    }

    #[test]
    fn should_compact_fires_when_above_threshold() {
        let strat = SummarizeCompactionStrategy::default();
        // 5 messages * 8000 chars each ≈ 10K tokens, with a 12K window
        // that's well above 75 % (= 9K threshold).
        let messages = vec![
            user(&"x".repeat(8000)),
            assistant_text(&"y".repeat(8000)),
            user(&"x".repeat(8000)),
            assistant_text(&"y".repeat(8000)),
            user(&"x".repeat(8000)),
        ];
        assert!(strat.should_compact(&messages, 12_000));
    }

    #[test]
    fn should_compact_respects_tail_min_floor() {
        let strat = SummarizeCompactionStrategy::default();
        // Bigger than threshold but fewer than tail_min_messages — skip.
        let messages = vec![
            user(&"x".repeat(100_000)),
            assistant_text(&"y".repeat(100_000)),
        ];
        assert!(!strat.should_compact(&messages, 1_000));
    }

    #[tokio::test]
    async fn compact_folds_history_into_summary_plus_tail() {
        let strat = SummarizeCompactionStrategy::default().with_tail_min_messages(2);
        let ctx = CompactionContext {
            system_prompt: None,
            model_client: Arc::new(FixedSummaryClient {
                summary: "we ran ls and grep".into(),
            }),
            context_window_tokens: 10_000,
            tools: vec![],
        };
        let messages = vec![
            user("first user"),
            assistant_text("response 1"),
            user("second user"),
            tool_msg("tc1", "tool result"),
            user("third user"),
            assistant_text("final response"),
        ];
        let outcome = strat.compact(messages, &ctx).await.unwrap();
        let out = outcome.messages;
        // First message is the synthetic summary user message; the rest
        // is the preserved tail (last 2 messages, aligned to user start).
        assert!(
            matches!(&out[0], ChatMessage::User { content, .. } if content.contains("<conversation-summary>") && content.contains("we ran ls and grep"))
        );
        // Tail aligned to start on a User message (the "third user").
        match &out[1] {
            ChatMessage::User { content, .. } => assert_eq!(content, "third user"),
            other => panic!("expected User in tail, got {other:?}"),
        }
        // Output is much shorter than input.
        assert!(out.len() < 6);
        // FixedSummaryClient doesn't report usage → outcome.usage is None.
        assert!(outcome.usage.is_none());
    }

    #[tokio::test]
    async fn compact_returns_empty_summary_error_on_blank_response() {
        let strat = SummarizeCompactionStrategy::default().with_tail_min_messages(2);
        let ctx = CompactionContext {
            system_prompt: None,
            model_client: Arc::new(FixedSummaryClient { summary: "".into() }),
            context_window_tokens: 10_000,
            tools: vec![],
        };
        let messages = vec![
            user("a"),
            assistant_text("b"),
            user("c"),
            assistant_text("d"),
        ];
        let err = strat.compact(messages, &ctx).await.unwrap_err();
        assert!(matches!(err, CompactionError::EmptySummary));
    }

    #[tokio::test]
    async fn compact_skips_when_messages_at_or_below_tail_min() {
        let strat = SummarizeCompactionStrategy::default().with_tail_min_messages(4);
        let ctx = CompactionContext {
            system_prompt: None,
            model_client: Arc::new(FixedSummaryClient {
                summary: "irrelevant".into(),
            }),
            context_window_tokens: 1_000,
            tools: vec![],
        };
        let messages = vec![
            user("1"),
            assistant_text("2"),
            user("3"),
            assistant_text("4"),
        ];
        let outcome = strat.compact(messages.clone(), &ctx).await.unwrap();
        // Same messages back — no compaction happened, no model call.
        assert_eq!(outcome.messages, messages);
        assert!(outcome.usage.is_none());
    }
}
