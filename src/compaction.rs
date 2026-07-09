//! Context-window-aware compaction.
//!
//! Long-running native sessions accumulate a `messages` history that
//! eventually exceeds the model's context window. Without intervention
//! the next turn either truncates server-side (silently losing context)
//! or fails with a 400 / `context_length_exceeded`. Compaction folds the
//! mid-conversation into a `<conversation-summary>` checkpoint while
//! preserving all true user messages verbatim within a token budget —
//! same idea as Codex local compaction.
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

use crate::event::HarnessUsage;
use crate::model::{
    collect_model_response, ChatMessage, ModelClient, ModelClientError, ModelResponse,
    ModelTurnInput,
};
use crate::tools::ToolSpec;

/// Fraction of the model's context window above which compaction fires.
/// 0.90 leaves a ~10 % headroom for the in-flight turn's tool results and
/// the model's reply, matching Codex's default threshold.
pub const DEFAULT_TRIGGER_FRACTION: f64 = 0.90;

/// Number of recent user-turn boundaries whose tool outputs are protected
/// from the prune pass.  The last `PRUNE_PROTECT_TURNS` complete turns are
/// left untouched so the model retains full detail for the current task.
pub const PRUNE_PROTECT_TURNS: usize = 2;

/// Minimum net tokens that the prune pass must free before its mutations
/// are committed.  Below this floor the overhead is not worth the churn.
pub const PRUNE_MINIMUM_TOKENS: u64 = 2_000;

/// Per-output floor: tool results shorter than this are never pruned even
/// when they fall outside the protected window.
pub const PRUNE_MIN_CONTENT_TOKENS: u64 = 100;

/// Replacement text written into pruned tool outputs.
pub const PRUNED_CONTENT_STUB: &str =
    "[pruned — output removed to free context; re-run the tool if needed]";

/// Minimum number of messages the history must contain before compaction
/// is allowed to run. Below this floor the history is too short to benefit
/// from compaction and the call is a no-op.
pub const DEFAULT_TAIL_MIN_MESSAGES: usize = 4;

/// Soft cap on summary length when serialised back into a `User`
/// `<conversation-summary>` block. Above this we let the model decide,
/// but pass `max_tokens` hint so it doesn't ramble. 2 000 ≈ 8 KB —
/// enough for most multi-turn dialogues; OMA uses the same default.
pub const DEFAULT_SUMMARY_MAX_TOKENS: i32 = 2_000;

/// Token budget for verbatim user-message retention in the replacement
/// history. All true user messages are collected from the full history
/// (oldest to newest) and as many as fit within this budget are kept
/// verbatim — newest first. Matches Codex's `COMPACT_USER_MESSAGE_MAX_TOKENS`.
pub const DEFAULT_USER_MESSAGE_TOKEN_BUDGET: u64 = 20_000;

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
            usage: _,
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

/// Reduce a reported usage tally to a single context-size figure: the
/// tokens the provider counted as present on input for that turn. Provider
/// cache telemetry is intentionally not added here because those buckets are
/// subsets/billing details of the prompt, not extra context-window occupancy.
/// Output is excluded because it is emitted after the input window is measured
/// (the emitted output only occupies the window on the *next* turn, where it
/// appears as a later message we estimate directly).
fn usage_context_tokens(u: &HarnessUsage) -> u64 {
    // Treat `input_tokens` as the provider-reported prompt/window size.
    //
    // OpenAI-compatible usage reports `cached_tokens` as a subset of
    // `prompt_tokens`; adding cache_read again double-counts cached prompt
    // tokens and can trigger compaction far too early. Anthropic's cache
    // fields are also telemetry about how the prompt was billed/cached; the
    // value used for context-window occupancy must remain the total prompt
    // size, not prompt size plus cache sub-buckets.
    u.input_tokens
}

/// Estimate the total context size of `messages`, anchoring on the last
/// assistant turn that carried a provider-reported usage tally.
///
/// The provider's reported input count is the exact number of tokens the
/// model saw up to and including that turn — system prompt, tools, and all
/// prior messages combined. Re-estimating that prefix by character count
/// drifts (often by thousands of tokens) as a conversation grows, which
/// skews the compaction trigger. So the newest usage-bearing assistant
/// message is treated as ground truth for everything at or before it, and
/// only the messages appended after it are estimated heuristically.
///
/// With no usage anywhere (first turn, or a client that never reports it)
/// this falls back to a full heuristic sum, matching
/// [`estimate_messages_tokens`].
pub fn estimate_context_tokens_anchored(messages: &[ChatMessage]) -> u64 {
    let anchor = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, m)| match m {
            ChatMessage::Assistant { usage: Some(u), .. } => Some((idx, usage_context_tokens(u))),
            _ => None,
        });

    match anchor {
        Some((idx, measured)) => {
            let tail: u64 = messages[idx + 1..]
                .iter()
                .map(estimate_chat_message_tokens)
                .sum();
            measured + tail
        }
        None => estimate_messages_tokens(messages),
    }
}

/// Per-model context window in tokens. Delegates to
/// [`crate::model_limits::resolve_limits`] — see that module for the
/// full resolution chain (models.dev fetch + disk cache + fallback
/// table + conservative default).
///
/// Kept as a thin shim for backwards compatibility; new code should
/// call [`crate::model_limits::resolve_limits`] to also get the model's
/// output-token cap.
pub fn resolve_context_window_tokens(model: &str) -> u64 {
    crate::model_limits::resolve_context_window_tokens(model)
}

/// Context passed to [`CompactionStrategy::compact`].
///
/// The policy receives the same system prompt, model client, resolved context
/// window, and tool specs that the agent loop would use for the next model
/// step. A strategy can use these inputs to run a summarization model call,
/// make a deterministic retention decision without a model, or decline to
/// change the history by returning the original `messages`.
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
    /// Decide whether the agent loop should call [`Self::compact`] before the
    /// next model step.
    ///
    /// This method should be cheap and side-effect-free because it is checked
    /// before every step. Returning `false` leaves history unchanged.
    fn should_compact(&self, messages: &[ChatMessage], context_window_tokens: u64) -> bool;

    /// Return the message history that should replace the current history.
    ///
    /// Implementations must preserve provider invariants for any tool-call
    /// history they keep: assistant tool calls must still be followed by their
    /// matching tool results, and orphan tool results must not be introduced.
    /// If the strategy cannot safely compact, it should return the original
    /// `messages` with `usage: None` rather than fabricating an invalid
    /// history. Errors are treated by the agent loop as "skip this turn's
    /// compaction" rather than a failed user turn.
    async fn compact(
        &self,
        messages: Vec<ChatMessage>,
        ctx: &CompactionContext,
    ) -> Result<CompactionOutcome, CompactionError>;
}

/// Default compaction/retention policy.
///
/// Alias for [`SummarizeCompactionStrategy`] so consumers can refer to the
/// stable "default policy" concept while the implementation name remains
/// descriptive.
pub type DefaultCompactionStrategy = SummarizeCompactionStrategy;

/// Codex-style local compaction: send the full history to the model with a
/// handoff-summary prompt appended, collect the reply as a checkpoint, then
/// rebuild history as `[...retained user messages, summary]`.
///
/// Design rationale:
///   * User messages are preserved verbatim (up to `user_message_token_budget`)
///     because they carry precise constraints and goals that paraphrasing loses.
///   * Assistant / tool messages are folded into the summary — they are large
///     but low-density and tolerate lossy compression.
///   * The summary is placed LAST so the model reads it as the most recent
///     context rather than as background preamble.
///   * Reusing the same model client keeps the Anthropic prompt-cache prefix
///     hot (identical system + tools on every call).
pub struct SummarizeCompactionStrategy {
    pub trigger_fraction: f64,
    /// Minimum total message count below which compaction is skipped.
    /// Does not control retention; use `user_message_token_budget` for that.
    pub tail_min_messages: usize,
    pub summary_max_tokens: i32,
    pub summary_prompt: String,
    /// Token budget for verbatim user-message retention in the replacement history.
    pub user_message_token_budget: u64,
}

impl Default for SummarizeCompactionStrategy {
    fn default() -> Self {
        Self {
            trigger_fraction: DEFAULT_TRIGGER_FRACTION,
            tail_min_messages: DEFAULT_TAIL_MIN_MESSAGES,
            summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.into(),
            user_message_token_budget: DEFAULT_USER_MESSAGE_TOKEN_BUDGET,
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

    pub fn with_user_message_token_budget(mut self, budget: u64) -> Self {
        self.user_message_token_budget = budget;
        self
    }
}

/// Handoff-oriented summarise prompt. Instructs the model to produce a
/// structured checkpoint for *another agent instance* to resume from —
/// not a human-readable recap. Mirrors Codex's `SUMMARIZATION_PROMPT`.
pub const DEFAULT_SUMMARY_PROMPT: &str = "You are performing a CONTEXT CHECKPOINT COMPACTION. \
    Create a handoff summary for another agent instance that will resume this task.\n\n\
    Include:\n\
    - Current progress and key decisions made\n\
    - Important context, constraints, or user preferences that must be respected\n\
    - What remains to be done (clear next steps)\n\
    - Any critical data, file paths, command outputs, or references needed to continue\n\n\
    If a prior <conversation-summary> block exists in this conversation, produce an UPDATED \
    summary that supersedes it (incorporating all activity since). \
    Output only the summary text — no preamble, no closing remarks.";

#[async_trait]
impl CompactionStrategy for SummarizeCompactionStrategy {
    fn should_compact(&self, messages: &[ChatMessage], context_window_tokens: u64) -> bool {
        // Too few messages → never compact; the summarize call would
        // cost more than the prefix it's saving.
        if messages.len() <= self.tail_min_messages {
            return false;
        }
        let tokens = estimate_context_tokens_anchored(messages);
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

        // ── Phase 1: Prune old tool outputs (free tokens without a model call)
        //
        // Walk old turns and replace large tool outputs with a short stub.
        // If pruning alone brings the history back under the trigger threshold
        // we skip the expensive summarise round-trip entirely.
        let threshold = ((ctx.context_window_tokens as f64) * self.trigger_fraction).round() as u64;
        let (messages, freed_tokens) = prune_tool_outputs(messages, PRUNE_PROTECT_TURNS);
        if freed_tokens > 0 {
            tracing::debug!(
                target: "harness::compaction",
                freed_tokens,
                "prune pass freed tokens"
            );
            // The usage anchor was measured before pruning and pruning only
            // rewrites tool outputs that sit before it, so the anchored figure
            // still reflects the pre-prune size; subtract what pruning freed
            // to get the current size without re-summing the whole history.
            let tokens_after_prune =
                estimate_context_tokens_anchored(&messages).saturating_sub(freed_tokens);
            if tokens_after_prune <= threshold {
                tracing::info!(
                    target: "harness::compaction",
                    freed_tokens,
                    tokens_after_prune,
                    "prune sufficient — summarise skipped"
                );
                return Ok(CompactionOutcome {
                    messages,
                    usage: None,
                });
            }
        }

        // ── Phase 2: Summarise (model round-trip)
        //
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
            hosted_tools: vec![],
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

        // Collect all true user messages from history, skipping prior summary
        // messages (they are superseded by the new checkpoint we just generated).
        let user_texts = collect_user_message_texts(&messages);
        if user_texts.is_empty() {
            // No real user messages to retain — skip installing the summary.
            // Surface usage so HR can account for the (now-discarded) model call.
            return Ok(CompactionOutcome { messages, usage });
        }

        // Build replacement history: retained user messages first, summary last.
        // User messages are selected newest-first within the token budget then
        // reversed to chronological order. Placing the summary last means the
        // model reads the most recent context at the end of the prompt.
        let out =
            build_compacted_history(&user_texts, &summary_text, self.user_message_token_budget);
        Ok(CompactionOutcome {
            messages: out,
            usage,
        })
    }
}

fn serialize_summary(summary: &str) -> String {
    format!("<conversation-summary>\n{summary}\n</conversation-summary>")
}

/// Collect the text of every real `User` message in `messages`, in order,
/// filtering out prior summary messages. Prior summaries are superseded by
/// the new checkpoint and must not be recycled into the replacement history.
fn collect_user_message_texts(messages: &[ChatMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            ChatMessage::User { content, .. } if !is_summary_message(content) => {
                Some(content.clone())
            }
            _ => None,
        })
        .collect()
}

fn is_summary_message(content: &str) -> bool {
    content.trim_start().starts_with("<conversation-summary>")
}

/// Build the replacement history: retained user messages (chronological order)
/// followed by the summary as the final message.
///
/// `user_texts` is the full list of real user messages oldest→newest.
/// Messages are selected newest-first within `token_budget`; if the oldest
/// selected message only partially fits, it is truncated rather than dropped.
fn build_compacted_history(
    user_texts: &[String],
    summary_text: &str,
    token_budget: u64,
) -> Vec<ChatMessage> {
    let mut selected: Vec<String> = Vec::new();
    let mut remaining = token_budget;
    for text in user_texts.iter().rev() {
        if remaining == 0 {
            break;
        }
        let tokens = estimate_tokens(text);
        if tokens <= remaining {
            selected.push(text.clone());
            remaining -= tokens;
        } else {
            // Partially fits: truncate rather than skip so the budget is not
            // wasted and the oldest retained message still carries context.
            selected.push(truncate_to_token_budget(text, remaining));
            break;
        }
    }
    selected.reverse(); // restore chronological order
    let mut out = Vec::with_capacity(selected.len() + 1);
    for text in selected {
        out.push(ChatMessage::User {
            content: text,
            attachments: vec![],
        });
    }
    // Summary goes last — the model reads this as the most recent context.
    out.push(ChatMessage::User {
        content: serialize_summary(summary_text),
        attachments: vec![],
    });
    out
}

/// Truncate `s` to at most `budget` estimated tokens using the same
/// mixed-script estimator as `estimate_tokens`. Cuts at the last complete
/// character that keeps the running estimate within `budget`.
fn truncate_to_token_budget(s: &str, budget: u64) -> String {
    if budget == 0 {
        return String::new();
    }
    let mut ascii: u64 = 0;
    let mut non_ascii: u64 = 0;
    let mut end = 0usize;
    for (byte_pos, c) in s.char_indices() {
        let (na, nn) = if c.is_ascii() {
            (ascii + 1, non_ascii)
        } else {
            (ascii, non_ascii + 1)
        };
        if na.div_ceil(4) + nn > budget {
            break;
        }
        ascii = na;
        non_ascii = nn;
        end = byte_pos + c.len_utf8();
    }
    s[..end].to_string()
}

/// Lightweight prune pass: replace large old tool outputs with
/// [`PRUNED_CONTENT_STUB`] to free tokens before (or instead of) a full
/// summarise round-trip.
///
/// Algorithm:
///   1. Walk the message list backwards.
///   2. Skip tool outputs inside the most recent `protect_turns` user turns.
///   3. Stop at any prior `<conversation-summary>` boundary — everything
///      before it was already compacted.
///   4. Accumulate gross freed tokens for each `ChatMessage::Tool` whose
///      content exceeds [`PRUNE_MIN_CONTENT_TOKENS`].
///   5. If net freed tokens ≥ [`PRUNE_MINIMUM_TOKENS`], apply the mutations
///      and return the updated list; otherwise return the original unchanged.
///
/// Returns `(messages, net_tokens_freed)`.  `net_tokens_freed == 0` means
/// the list was not modified (either nothing qualified or the saving was
/// below the minimum).
pub fn prune_tool_outputs(
    messages: Vec<ChatMessage>,
    protect_turns: usize,
) -> (Vec<ChatMessage>, u64) {
    let stub_tokens = estimate_tokens(PRUNED_CONTENT_STUB);
    let mut user_turns_seen: usize = 0;
    // Collect (index, gross_tokens) for candidates.
    let mut candidates: Vec<(usize, u64)> = Vec::new();
    let mut gross_freed: u64 = 0;

    for (i, msg) in messages.iter().enumerate().rev() {
        match msg {
            ChatMessage::User { content, .. } => {
                if is_summary_message(content) {
                    // Prior compaction boundary — stop here.
                    break;
                }
                user_turns_seen += 1;
            }
            ChatMessage::Tool { content, .. } => {
                // Protected window: skip recent turns.
                if user_turns_seen < protect_turns {
                    continue;
                }
                let tokens = estimate_tokens(content);
                if tokens >= PRUNE_MIN_CONTENT_TOKENS {
                    candidates.push((i, tokens));
                    gross_freed += tokens;
                }
            }
            _ => {}
        }
    }

    // Net gain after we write the stub back in.
    let replacements = candidates.len() as u64;
    let net_freed = gross_freed.saturating_sub(stub_tokens * replacements);

    if net_freed < PRUNE_MINIMUM_TOKENS {
        return (messages, 0);
    }

    let mut out = messages;
    for (i, _) in &candidates {
        if let ChatMessage::Tool { content, .. } = &mut out[*i] {
            *content = PRUNED_CONTENT_STUB.to_string();
        }
    }
    (out, net_freed)
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
            usage: None,
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
                raw_emitted_args: None,
            }],
            thinking: None,
            usage: None,
        };
        assert!(estimate_chat_message_tokens(&with_tool) > estimate_chat_message_tokens(&bare));
    }

    fn assistant_with_usage(text: &str, input: u64, cache_read: u64) -> ChatMessage {
        ChatMessage::Assistant {
            text: Some(text.into()),
            tool_calls: vec![],
            thinking: None,
            usage: Some(HarnessUsage {
                input_tokens: input,
                cache_read_input_tokens: cache_read,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn anchored_estimate_uses_reported_usage_for_prefix() {
        // A short character-count prefix, but the assistant turn reports a
        // large measured input. The anchor must dominate, plus the tail.
        let msgs = vec![
            user("hi"),
            assistant_with_usage("ok", 40_000, 0),
            user(&"x".repeat(4000)), // ~1000 tokens tail
        ];
        let tail = estimate_chat_message_tokens(&msgs[2]);
        assert_eq!(estimate_context_tokens_anchored(&msgs), 40_000 + tail);
    }

    #[test]
    fn anchored_estimate_does_not_double_count_cache_reads() {
        // OpenAI-compatible cached_tokens is a subset of prompt_tokens. Cache
        // telemetry must not inflate context-window occupancy.
        let msgs = vec![assistant_with_usage("ok", 10_000, 25_000)];
        assert_eq!(estimate_context_tokens_anchored(&msgs), 10_000);
    }

    #[test]
    fn anchored_estimate_picks_newest_usage_anchor() {
        // Two usage-bearing turns; the newest one wins and everything at or
        // before it is covered by its measured count.
        let msgs = vec![
            user("a"),
            assistant_with_usage("first", 10_000, 0),
            user("b"),
            assistant_with_usage("second", 30_000, 0),
            user("tail"),
        ];
        let tail = estimate_chat_message_tokens(&msgs[4]);
        assert_eq!(estimate_context_tokens_anchored(&msgs), 30_000 + tail);
    }

    #[test]
    fn anchored_estimate_falls_back_without_usage() {
        // No usage anywhere ⇒ identical to the full heuristic sum.
        let msgs = vec![user("hi"), assistant_text("there"), user("more")];
        assert_eq!(
            estimate_context_tokens_anchored(&msgs),
            estimate_messages_tokens(&msgs)
        );
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
        // 5 messages * 8000 ASCII chars each ≈ 10K tokens.
        // With an 11K window the 90% threshold is 9 900 tokens, so
        // 10K tokens exceeds it and compaction must fire.
        let messages = vec![
            user(&"x".repeat(8000)),
            assistant_text(&"y".repeat(8000)),
            user(&"x".repeat(8000)),
            assistant_text(&"y".repeat(8000)),
            user(&"x".repeat(8000)),
        ];
        assert!(strat.should_compact(&messages, 11_000));
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
        // All three real user messages fit within the 20 000-token budget and
        // are retained verbatim. The summary is appended as the final message.
        assert_eq!(out.len(), 4, "3 user messages + 1 summary");
        match &out[0] {
            ChatMessage::User { content, .. } => assert_eq!(content, "first user"),
            other => panic!("expected User at [0], got {other:?}"),
        }
        match &out[1] {
            ChatMessage::User { content, .. } => assert_eq!(content, "second user"),
            other => panic!("expected User at [1], got {other:?}"),
        }
        match &out[2] {
            ChatMessage::User { content, .. } => assert_eq!(content, "third user"),
            other => panic!("expected User at [2], got {other:?}"),
        }
        // Summary is the last message.
        assert!(matches!(&out[3], ChatMessage::User { content, .. }
                if content.contains("<conversation-summary>") && content.contains("we ran ls and grep")));
        // Output is shorter than input (6 messages → 4).
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

    // ── prune_tool_outputs tests ─────────────────────────────────────────────

    fn big_tool(id: &str) -> ChatMessage {
        // ~2 100 tokens — above PRUNE_MIN_CONTENT_TOKENS (100) and,
        // when a single instance is pruneable, above PRUNE_MINIMUM_TOKENS (2 000)
        // net of the stub replacement (~18 tokens).
        tool_msg(id, &"x".repeat(8_400))
    }

    fn small_tool(id: &str) -> ChatMessage {
        // 10 tokens — below the per-output floor, must NOT be pruned
        tool_msg(id, &"x".repeat(40))
    }

    #[test]
    fn prune_replaces_old_large_tool_outputs() {
        // History: 3 user turns, each followed by a big tool result.
        // protect_turns=2 → last 2 turns are safe; turn 0 (oldest) is pruneable.
        let messages = vec![
            user("turn 0"),
            big_tool("t0"),
            user("turn 1"),
            big_tool("t1"),
            user("turn 2"),
            big_tool("t2"),
        ];
        let (pruned, freed) = prune_tool_outputs(messages, 2);
        // Only t0 should be pruned (oldest, outside the 2-turn window).
        assert!(freed > 0, "expected tokens to be freed");
        assert_eq!(pruned[1], tool_msg("t0", PRUNED_CONTENT_STUB));
        // t1 and t2 are inside protected window — untouched.
        assert_ne!(pruned[3], tool_msg("t1", PRUNED_CONTENT_STUB));
        assert_ne!(pruned[5], tool_msg("t2", PRUNED_CONTENT_STUB));
    }

    #[test]
    fn prune_does_not_touch_small_tool_outputs() {
        let messages = vec![
            user("turn 0"),
            small_tool("t0"),
            user("turn 1"),
            big_tool("t1"),
            user("turn 2"),
            big_tool("t2"),
        ];
        let (pruned, _freed) = prune_tool_outputs(messages.clone(), 2);
        // t0 is old but small — must stay intact.
        assert_eq!(pruned[1], messages[1]);
    }

    #[test]
    fn prune_no_op_when_savings_below_minimum() {
        // Only one small old tool output — net freed < PRUNE_MINIMUM_TOKENS.
        let messages = vec![
            user("turn 0"),
            small_tool("t0"),
            user("turn 1"),
            big_tool("t1"),
            user("turn 2"),
        ];
        // small_tool is below PRUNE_MIN_CONTENT_TOKENS → nothing qualifies.
        let (out, freed) = prune_tool_outputs(messages.clone(), 2);
        assert_eq!(freed, 0);
        assert_eq!(out, messages);
    }

    #[test]
    fn prune_stops_at_summary_boundary() {
        // Layout (chronological):
        //   turn 0 / t_before  ← before the summary; must NOT be pruned
        //   <summary>          ← compaction boundary; reverse walk stops here
        //   turn 1 / t1        ← after summary but inside protected window
        //   turn 2 / t2        ← after summary but inside protected window
        //
        // Without the summary, t_before (oldest, outside protect window)
        // WOULD be pruned.  With it, the reverse walk hits the summary and
        // breaks before reaching t_before, so freed == 0.
        let messages = vec![
            user("turn 0"),
            big_tool("t_before"),
            user("<conversation-summary>\nprevious context\n</conversation-summary>"),
            user("turn 1"),
            big_tool("t1"),
            user("turn 2"),
            big_tool("t2"),
        ];
        let (out, freed) = prune_tool_outputs(messages.clone(), 2);
        assert_eq!(
            freed, 0,
            "summary boundary must block pruning of earlier content"
        );
        assert_eq!(out, messages);
    }

    #[tokio::test]
    async fn compact_skips_summarise_when_prune_sufficient() {
        // A strategy that would summarise if called — we verify it is NOT
        // called when prune brings tokens below the threshold.
        struct PanicSummaryClient;
        #[async_trait::async_trait]
        impl ModelClient for PanicSummaryClient {
            async fn stream(
                &self,
                _: ModelTurnInput,
            ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
            {
                panic!("summarise should not be called when prune is sufficient");
            }
        }

        // Build a history that is over the threshold purely because of one
        // big tool output in the oldest turn.  After pruning it the total
        // drops below the threshold.
        let big_content = "y".repeat(200_000); // ~50k tokens
        let messages = vec![
            user("turn 0"),
            tool_msg("t0", &big_content),
            user("turn 1"),
            assistant_text("ok"),
            user("turn 2"),
            assistant_text("done"),
        ];
        let ctx = CompactionContext {
            system_prompt: None,
            model_client: Arc::new(PanicSummaryClient),
            context_window_tokens: 60_000,
            tools: vec![],
        };
        let strat = SummarizeCompactionStrategy {
            trigger_fraction: 0.9,
            tail_min_messages: 2,
            ..Default::default()
        };
        let outcome = strat.compact(messages, &ctx).await.unwrap();
        // Prune replaced the big output — usage is None (no model call).
        assert!(outcome.usage.is_none());
        // The stub is present at index 1.
        assert_eq!(outcome.messages[1], tool_msg("t0", PRUNED_CONTENT_STUB));
    }
}
