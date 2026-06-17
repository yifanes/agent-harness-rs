use async_trait::async_trait;
use futures::stream::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::compaction::{estimate_messages_tokens, CompactionContext, CompactionStrategy};
use crate::event::{HarnessInternalEvent, HarnessUsage, NativeHarnessError, NativeTurnInput};
use crate::model::{
    AssistantThinking, ChatMessage, HostedTool, ModelChunk, ModelClient, ModelClientError,
    ModelTurnInput,
};
use crate::runner::NativeHarness;
use crate::tools::{
    bounded::BoundedToolRuntime, ToolFailure, ToolFailureKind, ToolInvocation, ToolOutcome,
    ToolRuntime, ToolRuntimeError,
};

/// Optional compaction wiring: strategy + the model client used to run
/// the summarize request + the resolved context-window cap. All three
/// must travel together; without any of them the loop can't make a
/// useful compaction decision. `AgentLoopHarness::with_compaction`
/// installs it once and the per-turn loop checks it between steps.
#[derive(Clone)]
pub struct CompactionPolicy {
    pub strategy: Arc<dyn CompactionStrategy>,
    /// Model client used by `strategy.compact` to run the summarize
    /// request. Usually the same provider as the main turn model so the
    /// Anthropic cache prefix stays hot; tests may swap in a fake.
    pub model_client: Arc<dyn ModelClient>,
    pub context_window_tokens: u64,
}

/// Default mid-stream idle timeout: how long `consume_step_stream` waits
/// for the *next* model chunk before declaring the connection stalled.
/// Generous enough to cover extended-thinking pauses (a model can legitimately
/// go silent for tens of seconds while reasoning) yet bounded so a silently
/// wedged upstream (TCP open, no FIN/RST, no bytes) can't park the turn forever.
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Default stream-layer reconnect budget — how many times we re-establish the
/// SSE stream after a stall / mid-stream transport drop *before any output has
/// reached the user*. Separate from `MAX_RETRIES` (the request-establish
/// budget); stream re-establishment tends to succeed on retry since the
/// failure is usually a transient gateway / long-lived-connection hiccup, so
/// this is set higher (6).
const DEFAULT_STREAM_MAX_ATTEMPTS: u32 = 6;

#[derive(Clone)]
pub struct AgentLoopHarness<M, R> {
    model: M,
    /// Every runtime is wrapped so repair + validation + tracing + the
    /// safety-net output cap apply uniformly, regardless of which concrete
    /// runtime the caller passed to [`AgentLoopHarness::new`].
    tools: BoundedToolRuntime<R>,
    max_steps: usize,
    compaction: Option<CompactionPolicy>,
    tool_choice: crate::model::ToolChoice,
    hosted_tools: Vec<HostedTool>,
    parallel_tool_calls: Option<bool>,
    stream_idle_timeout: Duration,
    stream_max_attempts: u32,
}

impl<M, R: ToolRuntime> AgentLoopHarness<M, R> {
    pub fn new(model: M, tools: R) -> Self {
        Self {
            model,
            tools: BoundedToolRuntime::new(tools),
            max_steps: 8,
            compaction: None,
            tool_choice: crate::model::ToolChoice::Auto,
            hosted_tools: vec![],
            parallel_tool_calls: None,
            stream_idle_timeout: DEFAULT_STREAM_IDLE_TIMEOUT,
            stream_max_attempts: DEFAULT_STREAM_MAX_ATTEMPTS,
        }
    }

    /// Cap the number of LLM steps per turn. `0` means unlimited — the
    /// loop only ends when the model stops calling tools (or on
    /// cancel/error), so callers passing `0` should keep their own
    /// liveness backstop (idle/wall-clock) around the turn.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Attach a compaction policy. The loop will call
    /// `policy.strategy.should_compact` before every step and run
    /// `policy.strategy.compact` when it fires. Without a policy
    /// installed the loop never compacts — fine for short / test
    /// conversations, fatal for long production sessions.
    pub fn with_compaction(mut self, policy: CompactionPolicy) -> Self {
        self.compaction = Some(policy);
        self
    }

    /// Constrain how the model selects tools this turn.
    /// Defaults to `Auto`. See `ToolChoice` for variants.
    pub fn with_tool_choice(mut self, choice: crate::model::ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Enable provider-executed hosted tools, such as Anthropic native
    /// web_search. These tools are not part of `ToolRuntime` and are never
    /// executed by the harness itself.
    pub fn with_hosted_tools(mut self, tools: Vec<HostedTool>) -> Self {
        self.hosted_tools = tools;
        self
    }

    /// Convenience helper for provider-native web search.
    pub fn with_web_search(mut self) -> Self {
        self.hosted_tools
            .push(HostedTool::WebSearch { max_uses: None });
        self
    }

    /// OpenAI-only: whether the model may emit multiple `tool_use`
    /// blocks in one response. `None` ⇒ provider default (true on
    /// OpenAI). Ignored by Anthropic (multi tool_use is implicit).
    pub fn with_parallel_tool_calls(mut self, parallel: Option<bool>) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// Override mid-stream resilience knobs. `idle_timeout` is how long a step
    /// waits for the next model chunk before declaring a stall;
    /// `max_attempts` is the stream-layer reconnect budget (total stream
    /// attempts, so `max_attempts = 1` disables reconnection). Primarily for
    /// tests, which inject a sub-second timeout so a stall surfaces fast
    /// instead of after the 90s production default.
    pub fn with_stream_resilience(mut self, idle_timeout: Duration, max_attempts: u32) -> Self {
        self.stream_idle_timeout = idle_timeout;
        self.stream_max_attempts = max_attempts.max(1);
        self
    }
}

#[async_trait]
impl<M, R> NativeHarness for AgentLoopHarness<M, R>
where
    M: ModelClient + Clone + Send + Sync + 'static,
    R: ToolRuntime + Clone + Send + Sync + 'static,
{
    async fn run_turn(
        &self,
        input: NativeTurnInput,
    ) -> Result<mpsc::Receiver<Result<HarnessInternalEvent, NativeHarnessError>>, NativeHarnessError>
    {
        let (tx, rx) = mpsc::channel(16);
        let model = self.model.clone();
        let tools = self.tools.clone();
        let max_steps = self.max_steps;
        let compaction = self.compaction.clone();
        let tool_choice = self.tool_choice.clone();
        let hosted_tools = self.hosted_tools.clone();
        let parallel_tool_calls = self.parallel_tool_calls;
        let stream_idle_timeout = self.stream_idle_timeout;
        let stream_max_attempts = self.stream_max_attempts;

        tokio::spawn(async move {
            run_loop(
                model,
                tools,
                RunLoopConfig {
                    max_steps,
                    compaction,
                    tool_choice,
                    hosted_tools,
                    parallel_tool_calls,
                    stream_idle_timeout,
                    stream_max_attempts,
                },
                input,
                tx,
            )
            .await;
        });

        Ok(rx)
    }
}

/// Test whether the cancel token (if any) has been signalled.
fn cancel_fired(token: Option<&CancellationToken>) -> bool {
    token.is_some_and(|t| t.is_cancelled())
}

fn is_silent_stop(text: &str, stop_reason: &str) -> bool {
    text.trim().is_empty() && matches!(stop_reason, "end_turn" | "max_tokens")
}

struct RunLoopConfig {
    max_steps: usize,
    compaction: Option<CompactionPolicy>,
    tool_choice: crate::model::ToolChoice,
    hosted_tools: Vec<HostedTool>,
    parallel_tool_calls: Option<bool>,
    stream_idle_timeout: Duration,
    stream_max_attempts: u32,
}

async fn run_loop<M, R>(
    model: M,
    tools: R,
    config: RunLoopConfig,
    input: NativeTurnInput,
    tx: mpsc::Sender<Result<HarnessInternalEvent, NativeHarnessError>>,
) where
    M: ModelClient + Send + Sync,
    R: ToolRuntime + Clone + Send + Sync + 'static,
{
    let system_prompt = input.system_prompt.clone();
    let cancel_token = input.cancel_token.clone();
    let context_path = input.context_path.clone();
    // Snapshot tool specs once per turn — adding / removing tools mid-turn
    // would invalidate cached prompt prefixes on every provider that does
    // any caching, so we treat the spec list as immutable for one turn.
    let tools_snapshot = tools.specs();
    // Seed history: load from context JSONL when a path is provided
    // (persistent mode), otherwise use the in-memory prior_messages.
    let mut messages: Vec<ChatMessage> = if let Some(ref path) = context_path {
        crate::context::jsonl::load_context(path).await
    } else {
        input.prior_messages
    };
    messages.push(ChatMessage::User {
        content: input.prompt_text,
        attachments: input.attachments,
    });
    // Cursor: how many messages have been flushed to the context JSONL.
    // Set to messages.len() after the initial User flush (Some path),
    // or 0 when running in-memory (None — ctx_written is never read).
    let mut ctx_written: usize = match context_path.as_deref() {
        None => 0,
        Some(path) => {
            let start = messages.len() - 1;
            crate::context::jsonl::append_context(path, &messages[start..]).await;
            messages.len()
        }
    };
    // Per-turn accumulated token usage. Each model call may report a
    // fresh `HarnessUsage` (provider reports per-call counts, not deltas);
    // we sum them so `TurnEnd.usage` reflects what the whole turn cost.
    let mut total_usage = HarnessUsage::default();
    let mut saw_any_usage = false;

    // Fired by `cancel_token.cancel()` from RD on InterruptDispatch.
    // Emits a single TurnEnd{interrupt} and returns. We check at three
    // load-bearing points: before each step, before tool dispatch, and
    // (cheapest of all) inside `consume_step_stream`'s select! on every
    // chunk await.
    macro_rules! check_cancel {
        () => {
            if cancel_fired(cancel_token.as_ref()) {
                let _ = tx
                    .send(Ok(HarnessInternalEvent::TurnEnd {
                        stop_reason: "interrupt".into(),
                        usage: saw_any_usage.then(|| total_usage.clone()),
                        final_messages: if context_path.is_none() { messages.clone() } else { vec![] },
                    }))
                    .await;
                return;
            }
        };
    }

    for step in 0.. {
        // max_steps == 0 ⇒ unlimited: only the model finishing (or
        // cancel/error) ends the turn. Otherwise break to the trailing
        // TurnEnd{max_turns} once the cap is hit.
        if config.max_steps != 0 && step >= config.max_steps {
            break;
        }
        check_cancel!();
        // Compaction check — purely additive, never fails the turn. If
        // the strategy errors out (e.g. provider returned empty
        // summary), we leave `messages` untouched and let the next
        // step / turn try again. This keeps "context overflow" as the
        // worst case: HR sees a model error and decides how to react.
        if let Some(policy) = &config.compaction {
            if policy
                .strategy
                .should_compact(&messages, policy.context_window_tokens)
            {
                let original_count = messages.len();
                let original_tokens = estimate_messages_tokens(&messages);
                let cctx = CompactionContext {
                    system_prompt: system_prompt.clone(),
                    model_client: policy.model_client.clone(),
                    context_window_tokens: policy.context_window_tokens,
                    tools: tools_snapshot.clone(),
                };
                match policy.strategy.compact(messages.clone(), &cctx).await {
                    Ok(outcome) => {
                        let compacted_count = outcome.messages.len();
                        let compacted_tokens = estimate_messages_tokens(&outcome.messages);
                        messages = outcome.messages;
                        // Compaction summarize-call usage attributed two
                        // places: into the turn-level total (so HR sees
                        // the full cost) AND into compaction_*_tokens
                        // sub-buckets (so HR can isolate what compaction
                        // alone cost).
                        if let Some(u) = outcome.usage.as_ref() {
                            saw_any_usage = true;
                            total_usage.input_tokens += u.input_tokens;
                            total_usage.output_tokens += u.output_tokens;
                            total_usage.cache_read_input_tokens += u.cache_read_input_tokens;
                            total_usage.cache_creation_input_tokens +=
                                u.cache_creation_input_tokens;
                            total_usage.compaction_input_tokens += u.input_tokens;
                            total_usage.compaction_output_tokens += u.output_tokens;
                        }
                        // Structured tracing for operators / dashboards.
                        // Token counts are estimator output (4 chars/token),
                        // not provider-reported — labelled in field name.
                        tracing::info!(
                            target: "harness::compaction",
                            step,
                            original_message_count = original_count,
                            compacted_message_count = compacted_count,
                            original_estimated_tokens = original_tokens,
                            compacted_estimated_tokens = compacted_tokens,
                            context_window_tokens = policy.context_window_tokens,
                            "compaction applied"
                        );
                        // Rewrite the context JSONL with the compacted history.
                        if let Some(ref path) = context_path {
                            crate::context::jsonl::rewrite_context(path, &messages).await;
                            ctx_written = messages.len();
                        }
                        if tx
                            .send(Ok(HarnessInternalEvent::CompactionApplied {
                                original_message_count: original_count,
                                compacted_message_count: compacted_count,
                                original_tokens,
                                compacted_tokens,
                            }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "harness::compaction",
                            step,
                            error = %e,
                            "compaction skipped; history retained as-is, model call may now fail with context overflow"
                        );
                    }
                }
            }
        }

        // ── Model call with retry for transient errors ────────────────────────
        // Non-retryable errors (bad config, auth, context overflow) surface
        // immediately. Retryable errors (rate-limit, network, 5xx) back off
        // exponentially up to MAX_RETRIES before giving up.
        const MAX_RETRIES: u32 = 3;
        const BASE_BACKOFF_MS: u64 = 1_000;
        const MAX_BACKOFF_MS: u64 = 16_000;

        let model_input = ModelTurnInput {
            system_prompt: system_prompt.clone(),
            messages: messages.clone(),
            tools: tools_snapshot.clone(),
            hosted_tools: config.hosted_tools.clone(),
            tool_choice: config.tool_choice.clone(),
            parallel_tool_calls: config.parallel_tool_calls,
        };

        // Per-step stream lifecycle with two independent retry budgets:
        //   * establish — `model.stream()` erroring before any stream exists.
        //     Retried up to MAX_RETRIES (request-layer transient faults).
        //   * consume — a stall / drop *mid-stream*. Retried up to
        //     `stream_max_attempts`, but ONLY while `had_progress == false`:
        //     once output has reached the user, re-issuing the request would
        //     duplicate it, so a mid-stream failure becomes terminal.
        // The two are nested: each reconnect re-runs establishment (with its
        // own request-layer retry) before consuming again.
        let mut stream_attempt = 0u32;
        let outcome = 'stream: loop {
            let stream = {
                let mut attempt = 0u32;
                loop {
                    match model.stream(model_input.clone()).await {
                        Ok(s) => break s,
                        Err(e) => {
                            if e.retryable() && attempt < MAX_RETRIES {
                                let delay_ms =
                                    (BASE_BACKOFF_MS * (1 << attempt)).min(MAX_BACKOFF_MS);
                                tracing::warn!(
                                    attempt,
                                    delay_ms,
                                    error = %e,
                                    "model call failed (retryable) — backing off"
                                );
                                if !backoff_sleep(delay_ms, cancel_token.as_ref()).await {
                                    let _ = tx
                                        .send(Err(NativeHarnessError::ModelOther(
                                            "interrupted during retry backoff".into(),
                                        )))
                                        .await;
                                    return;
                                }
                                attempt += 1;
                            } else {
                                // Non-retryable (config error, auth, etc.) or retries exhausted.
                                // Surface the error immediately so the user can act on it.
                                tracing::error!(
                                    attempt,
                                    error = %e,
                                    retryable = e.retryable(),
                                    "model call failed — terminating turn"
                                );
                                let _ = tx.send(Err(model_error_to_native(e))).await;
                                return;
                            }
                        }
                    }
                }
            };

            // Consume the per-step stream: forward TextDelta chunks live
            // (token-level emit) and accumulate the tool-call state so we
            // can either dispatch a tool or finalise a message at the end.
            // The idle watchdog inside fires if the stream goes silent.
            match consume_step_stream(
                stream,
                &tx,
                step,
                cancel_token.as_ref(),
                config.stream_idle_timeout,
            )
            .await
            {
                Ok(StepDrain::Complete(o)) => break 'stream o,
                Ok(StepDrain::Cancelled) => {
                    let _ = tx
                        .send(Ok(HarnessInternalEvent::TurnEnd {
                            stop_reason: "interrupt".into(),
                            usage: saw_any_usage.then(|| total_usage.clone()),
                            final_messages: if context_path.is_none() { messages.clone() } else { vec![] },
                        }))
                        .await;
                    return;
                }
                Err(StepFailure::Model { err, had_progress }) => {
                    // Reconnect only when nothing has reached the user yet, the
                    // fault is transient, and the stream budget isn't spent.
                    if !had_progress
                        && err.retryable()
                        && stream_attempt + 1 < config.stream_max_attempts
                    {
                        let delay_ms =
                            (BASE_BACKOFF_MS * (1 << stream_attempt)).min(MAX_BACKOFF_MS);
                        tracing::warn!(
                            step,
                            stream_attempt,
                            delay_ms,
                            error = %err,
                            "model stream failed before any output — reconnecting"
                        );
                        if !backoff_sleep(delay_ms, cancel_token.as_ref()).await {
                            let _ = tx
                                .send(Err(NativeHarnessError::ModelOther(
                                    "interrupted during stream reconnect backoff".into(),
                                )))
                                .await;
                            return;
                        }
                        stream_attempt += 1;
                        continue 'stream;
                    }
                    // Terminal: output already emitted, non-retryable, or budget
                    // exhausted. Surface so the user / HR can act on it.
                    tracing::error!(
                        step,
                        stream_attempt,
                        error = %err,
                        had_progress,
                        retryable = err.retryable(),
                        "model stream failed — terminating turn"
                    );
                    let _ = tx.send(Err(model_error_to_native(err))).await;
                    return;
                }
                Err(StepFailure::ChannelClosed) => return,
                Err(StepFailure::Fatal(e)) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
        };

        if let Some(u) = outcome.usage.as_ref() {
            saw_any_usage = true;
            total_usage.input_tokens += u.input_tokens;
            total_usage.output_tokens += u.output_tokens;
            total_usage.cache_read_input_tokens += u.cache_read_input_tokens;
            total_usage.cache_creation_input_tokens += u.cache_creation_input_tokens;
        }

        match outcome.next {
            StepNext::Message { text, stop_reason } => {
                if is_silent_stop(&text, &stop_reason) {
                    let _ = tx
                        .send(Err(NativeHarnessError::ModelOther(format!(
                            "silent_stop: model returned stop_reason={stop_reason} \
                             with empty text and no tool calls"
                        ))))
                        .await;
                    return;
                }
                let assistant_text = (!text.trim().is_empty()).then_some(text);
                messages.push(ChatMessage::Assistant {
                    text: assistant_text,
                    tool_calls: vec![],
                    thinking: outcome.thinking.clone(),
                });
                // Persist the final Assistant message to context JSONL.
                if let Some(ref path) = context_path {
                    crate::context::jsonl::append_context(path, &messages[ctx_written..]).await;
                }
                // Note: AssistantTextChunk events were already emitted
                // mid-stream, so there's nothing more to send here.
                let final_msgs = if context_path.is_none() { messages.clone() } else { vec![] };
                let _ = tx
                    .send(Ok(HarnessInternalEvent::TurnEnd {
                        stop_reason,
                        usage: saw_any_usage.then(|| total_usage.clone()),
                        final_messages: final_msgs,
                    }))
                    .await;
                return;
            }
            StepNext::ToolCalls {
                preface,
                mut invocations,
            } => {
                check_cancel!();
                // Schema-guided input repair at dispatch time: fix common
                // shape mistakes from weak models before the tool sees
                // them. Runs BEFORE the history
                // push and the ToolCall events so history, wire, and the
                // actual execution all agree on the (repaired) arguments.
                // No matching spec (e.g. model hallucinated a tool name) →
                // leave the input alone; dispatch will fail it as unknown.
                for inv in &mut invocations {
                    // Repair lives in the runtime wrapper (single source of
                    // truth). Running it here — before the history push and
                    // ToolCall events below — keeps history, wire, and the
                    // wrapper's (idempotent) dispatch-time repair in agreement.
                    if let Some(repairs) = tools.repair_invocation(inv) {
                        tracing::warn!(
                            target: "harness::tool_repair",
                            tool = %inv.name,
                            id = %inv.id,
                            repairs = ?repairs,
                            "schema-guided tool input repair applied"
                        );
                    }
                }
                let preface_text = preface.filter(|s| !s.is_empty());
                // Record the assistant turn in history BEFORE executing
                // the tools. Two reasons:
                //   * the tool_use blocks live in the assistant message
                //     per the OpenAI / Anthropic protocols;
                //   * if the tool errors and the loop bails, history
                //     still reflects "model called X/Y/Z" — useful for
                //     debugging and possible retry strategies.
                messages.push(ChatMessage::Assistant {
                    text: preface_text,
                    tool_calls: invocations.clone(),
                    thinking: outcome.thinking.clone(),
                });

                // Preface AssistantTextChunk was already emitted mid-stream.

                // Emit ToolCall events in declared order so the wire
                // sees them in a stable sequence (matters for HR's
                // ordinal assignment in run_dispatch).
                for inv in &invocations {
                    if tx
                        .send(Ok(HarnessInternalEvent::ToolCall {
                            id: inv.id.clone(),
                            name: inv.name.clone(),
                            input: inv.input.clone(),
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }

                // Dispatch ALL invocations concurrently. Each tool runs
                // in its own task so InterruptDispatch can return a clean
                // TurnEnd immediately while cancellation-aware runtimes
                // (notably E2B SandboxToolRuntime) keep polling long
                // enough to SIGTERM their remote process.
                let handles = invocations.iter().cloned().map(|inv| {
                    let tools = tools.clone();
                    let cancel_for_task = cancel_token.clone();
                    let invocation_for_task = inv.clone();
                    let handle = tokio::spawn(async move {
                        tools
                            .invoke_cancellable(invocation_for_task, cancel_for_task.as_ref())
                            .await
                    });
                    (inv, handle)
                });
                let join = futures::future::join_all(handles.map(|(inv, handle)| async move {
                    let outcome = match handle.await {
                        Ok(outcome) => outcome,
                        Err(e) => Err(ToolRuntimeError::Runtime(format!("tool task failed: {e}"))),
                    };
                    (inv, outcome)
                }));

                let pairs_opt = if let Some(token) = cancel_token.as_ref() {
                    tokio::select! {
                        biased;
                        _ = token.cancelled() => None,
                        results = join => Some(results),
                    }
                } else {
                    Some(join.await)
                };
                let pairs = match pairs_opt {
                    Some(o) => o,
                    None => {
                        // Cancel won the select — emit interrupt
                        // TurnEnd and return. Tool tasks remain detached
                        // so cancellation-aware runtimes can terminate
                        // their remote process; their results aren't
                        // surfaced after the turn has ended.
                        let _ = tx
                            .send(Ok(HarnessInternalEvent::TurnEnd {
                                stop_reason: "interrupt".into(),
                                usage: saw_any_usage.then(|| total_usage.clone()),
                                final_messages: if context_path.is_none() { messages.clone() } else { vec![] },
                            }))
                            .await;
                        return;
                    }
                };

                // Walk invocations + outcomes pairwise to keep ordering
                // stable. Tool timeouts and invalid model-supplied inputs are
                // model-observable failures; infrastructure/runtime errors
                // still fail the turn.
                let mut runtime_error: Option<String> = None;
                for (inv, outcome) in pairs {
                    let id = inv.id.clone();
                    let outcome = match outcome {
                        Ok(o) => o,
                        Err(ToolRuntimeError::Timeout(message)) => ToolOutcome {
                            output: Err(ToolFailure::new(ToolFailureKind::Timeout, message)),
                            attachments: vec![],
                        },
                        Err(ToolRuntimeError::InvalidInput { tool, message }) => ToolOutcome {
                            output: Err(crate::tools::invalid_input_failure(
                                &tool,
                                message,
                                &inv.input,
                                tools_snapshot
                                    .iter()
                                    .find(|s| s.name == tool)
                                    .map(|s| &s.input_schema),
                            )),
                            attachments: vec![],
                        },
                        Err(e) => {
                            // Note: ToolRuntimeError vs ToolFailure are
                            // different beasts. ToolFailure is model-
                            // observable (file not found, exit≠0); this
                            // is sandbox / runtime infrastructure
                            // breaking and HR needs to know.
                            runtime_error = Some(e.to_string());
                            break;
                        }
                    };
                    let tool_attachments = outcome.attachments;
                    let output = outcome.output.map_err(|failure| failure.to_string());

                    // Append the tool result to history so the next model
                    // step sees it. OpenAI's `tool` role expects content as
                    // a string; we serialize successes verbatim and wrap
                    // failures into a small JSON object so the model can
                    // tell the two apart structurally.
                    let (tool_content, is_error) = match &output {
                        Ok(value) => (value.to_string(), false),
                        Err(err) => (json!({ "error": err }).to_string(), true),
                    };
                    messages.push(ChatMessage::Tool {
                        tool_call_id: id.clone(),
                        content: tool_content,
                        is_error,
                        attachments: tool_attachments,
                    });

                    if tx
                        .send(Ok(HarnessInternalEvent::ToolResult { id, output }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                if let Some(err) = runtime_error {
                    let _ = tx.send(Err(NativeHarnessError::ToolRuntime(err))).await;
                    return;
                }
                // Flush the Assistant + all Tool messages for this step.
                if let Some(ref path) = context_path {
                    crate::context::jsonl::append_context(path, &messages[ctx_written..]).await;
                    ctx_written = messages.len();
                }
                // Continue the loop — next step will see the tool
                // results in `messages` and decide what to do.
            }
        }
    }

    // max_turns reached — also flush any unflushed messages.
    if let Some(ref path) = context_path {
        crate::context::jsonl::append_context(path, &messages[ctx_written..]).await;
    }
    let final_msgs = if context_path.is_none() { messages } else { vec![] };
    let _ = tx
        .send(Ok(HarnessInternalEvent::TurnEnd {
            stop_reason: "max_turns".into(),
            usage: saw_any_usage.then(|| total_usage.clone()),
            final_messages: final_msgs,
        }))
        .await;
}

/// 1:1 lift from `ModelClientError` to `NativeHarnessError`. Two enums
/// because `ModelClient` is provider-facing (the test fixture
/// `ScriptedModelClient` exists in the same world) and shouldn't have to
/// know about the harness-runtime variants (`Encode` / `ChannelClosed`
/// don't apply to it).
/// Sleep for `delay_ms`, waking early if the cancel token fires. Returns
/// `true` if the full backoff elapsed, `false` if interrupted by cancel —
/// callers treat `false` as "abort the turn". Shared by the request-establish
/// retry and the stream-reconnect retry so both honour InterruptDispatch
/// mid-backoff.
async fn backoff_sleep(delay_ms: u64, cancel_token: Option<&CancellationToken>) -> bool {
    let sleep = tokio::time::sleep(Duration::from_millis(delay_ms));
    tokio::pin!(sleep);
    let cancelled = async {
        if let Some(t) = cancel_token {
            t.cancelled().await
        } else {
            std::future::pending().await
        }
    };
    tokio::select! {
        _ = &mut sleep => true,
        _ = cancelled => false,
    }
}

fn model_error_to_native(err: ModelClientError) -> NativeHarnessError {
    match err {
        ModelClientError::RateLimit(s) => NativeHarnessError::ModelRateLimit(s),
        ModelClientError::Auth(s) => NativeHarnessError::ModelAuth(s),
        ModelClientError::ContextOverflow(s) => NativeHarnessError::ModelContextOverflow(s),
        ModelClientError::BadRequest(s) => NativeHarnessError::ModelBadRequest(s),
        ModelClientError::ServerError(s) => NativeHarnessError::ModelServerError(s),
        ModelClientError::Network(s) => NativeHarnessError::ModelNetwork(s),
        ModelClientError::Other(s) => NativeHarnessError::ModelOther(s),
    }
}

/// Per-step accumulated state extracted while draining a `ModelChunk`
/// stream. `next` carries the "what to do next" decision (final
/// message vs tool dispatch); `usage` rides separately because it must
/// fold into the turn-level total regardless of the branch above; and
/// `thinking` carries the (text + signature) of any extended-thinking
/// block produced this step so the next turn's assistant message can
/// echo it back verbatim (Anthropic rejects modified thinking blocks).
struct StepOutcome {
    next: StepNext,
    usage: Option<HarnessUsage>,
    thinking: Option<AssistantThinking>,
}

/// Outcome of draining a single step's chunk stream. `Cancelled` is
/// distinct from `Complete` so the agent loop can emit a clean
/// `TurnEnd { interrupt }` rather than papering over the half-finished
/// state as "Message with empty text".
enum StepDrain {
    Complete(StepOutcome),
    Cancelled,
}

/// Failure modes of draining a single step's chunk stream. Split out so
/// `run_loop` can decide between reconnecting (re-establishing the stream)
/// and terminating the turn.
enum StepFailure {
    /// Stream / transport failure (chunk error, premature close, or idle
    /// stall). `err` keeps the original `ModelClientError` so the caller can
    /// consult `retryable()`; `had_progress` records whether any model output
    /// already reached the user this step. A reconnect is only safe when
    /// `!had_progress` — re-issuing the request after partial output would
    /// duplicate what the user has already seen.
    Model {
        err: ModelClientError,
        had_progress: bool,
    },
    /// Downstream event channel closed — RD dropped the receiver. Nothing left
    /// to send to; never retryable.
    ChannelClosed,
    /// Non-retryable processing error (e.g. tool-argument JSON decode failure).
    /// Surfaced to the user as-is.
    Fatal(NativeHarnessError),
}

enum StepNext {
    Message {
        text: String,
        stop_reason: String,
    },
    /// Model returned one or more `tool_use` blocks. Multi-element
    /// arrays come from providers that ship `parallel_tool_calls`
    /// (OpenAI default) or models that emit multiple tool_use
    /// blocks in a single Anthropic message. agent_loop dispatches
    /// them concurrently via `join_all`.
    ToolCalls {
        preface: Option<String>,
        invocations: Vec<ToolInvocation>,
    },
}

/// Drain one model step's chunk stream. Forwards `TextDelta` chunks to
/// the harness output channel live (token-by-token), accumulates the
/// tool call (if any), and returns once `ModelChunk::Done` lands. All
/// emitted `AssistantTextChunk` events share `msg_id = "msg_native_<step>"`
/// so `native_adapter::TextAccumulator` collapses them into a single
/// `AdapterEvent::AgentMessage` on the wire.
/// Build the `StepFailure` for an idle-watchdog timeout. Classified as
/// `ModelClientError::Network` so `retryable()` is true (a stall is a
/// transport-level fault, like a dropped connection); whether it actually
/// gets retried is gated by `had_progress` in `run_loop`.
fn stall_failure(idle_timeout: Duration, had_progress: bool) -> StepFailure {
    StepFailure::Model {
        err: ModelClientError::Network(format!(
            "model stream stalled: no output for {}s (connection open but idle)",
            idle_timeout.as_secs()
        )),
        had_progress,
    }
}

async fn consume_step_stream(
    mut stream: futures::stream::BoxStream<'static, Result<ModelChunk, ModelClientError>>,
    tx: &mpsc::Sender<Result<HarnessInternalEvent, NativeHarnessError>>,
    step: usize,
    cancel_token: Option<&CancellationToken>,
    idle_timeout: Duration,
) -> Result<StepDrain, StepFailure> {
    let emit_msg_id = format!("msg_native_{step}");
    let emit_thinking_id = format!("thinking_native_{step}");
    let mut text_buf = String::new();
    let mut text_stream_started = false;
    let mut thinking_buf = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut saw_thinking = false;
    let mut tool_states: Vec<ToolBuf> = Vec::new();
    let mut stop_reason = "end_turn".to_string();
    let mut usage: Option<HarnessUsage> = None;
    // Whether any real model output has reached the user this step. Gates
    // whether a later stall / drop is safe to retry (see `StepFailure::Model`).
    let mut had_progress = false;

    loop {
        // Mid-stream idle watchdog: a freshly-armed timer each iteration means
        // it measures the gap since the *last* chunk, i.e. it resets on every
        // chunk we receive. Keepalive / ping frames are dropped by the SSE
        // layer before they ever become a `ModelChunk`, so "received a chunk"
        // is exactly "the model made progress" — the timer only survives a
        // genuine silence, never a heartbeat-only lull.
        let idle = tokio::time::sleep(idle_timeout);
        tokio::pin!(idle);

        // select! arms: cancellation (priority via `biased`), the idle
        // watchdog, and the next stream chunk. Without `biased`, tokio's
        // randomised polling can starve cancel checks under heavy
        // chunk throughput. With it, an InterruptDispatch fires
        // exactly one stream poll later — typically <100 µs.
        let item = if let Some(token) = cancel_token {
            tokio::select! {
                biased;
                _ = token.cancelled() => {
                    return Ok(StepDrain::Cancelled);
                }
                _ = &mut idle => return Err(stall_failure(idle_timeout, had_progress)),
                next = stream.next() => next,
            }
        } else {
            tokio::select! {
                _ = &mut idle => return Err(stall_failure(idle_timeout, had_progress)),
                next = stream.next() => next,
            }
        };
        let Some(item) = item else { break };
        let chunk = match item {
            Ok(c) => c,
            Err(e) => {
                return Err(StepFailure::Model {
                    err: e,
                    had_progress,
                })
            }
        };
        match chunk {
            ModelChunk::TextDelta { msg_id: _, delta } => {
                if delta.is_empty() {
                    continue;
                }
                text_buf.push_str(&delta);
                let mut flush_delta = delta;
                if !text_stream_started {
                    if text_buf.trim().is_empty() {
                        continue;
                    }
                    text_stream_started = true;
                    // First visible chunk: flush any leading whitespace we
                    // held back while deciding whether the step is silent.
                    flush_delta = text_buf.clone();
                }
                // A non-empty text delta is model output the user is about to
                // see — past this point a stall is no longer safe to retry.
                had_progress = true;
                // Forward live to harness output. We rewrite msg_id to
                // the per-step canonical form so native_adapter groups
                // every chunk of this step into one AdapterEvent.
                if tx
                    .send(Ok(HarnessInternalEvent::AssistantTextChunk {
                        msg_id: emit_msg_id.clone(),
                        delta: flush_delta,
                    }))
                    .await
                    .is_err()
                {
                    return Err(StepFailure::ChannelClosed);
                }
            }
            ModelChunk::ThinkingDelta {
                thinking_id: _,
                delta,
                signature,
            } => {
                // Signature chunks usually arrive without text and vice
                // versa; we accept both shapes and latch whichever the
                // provider sends. The text part feeds the live
                // AssistantThinkingChunk emit; the signature rides on
                // the final ChatMessage::Assistant.thinking so the next
                // turn can re-send the block verbatim.
                if let Some(sig) = signature {
                    if !sig.is_empty() {
                        thinking_signature = Some(sig);
                    }
                }
                if !delta.is_empty() {
                    saw_thinking = true;
                    had_progress = true;
                    thinking_buf.push_str(&delta);
                    if tx
                        .send(Ok(HarnessInternalEvent::AssistantThinkingChunk {
                            msg_id: emit_thinking_id.clone(),
                            delta,
                        }))
                        .await
                        .is_err()
                    {
                        return Err(StepFailure::ChannelClosed);
                    }
                }
            }
            ModelChunk::ToolCallStart { id, name } => {
                // A tool call is committed model output. Even though we buffer
                // tool args rather than forwarding them live, treat any
                // tool-call activity as progress: re-issuing the request after
                // the model has started emitting a tool_use risks a divergent
                // / duplicated call.
                had_progress = true;
                tool_states.push(ToolBuf {
                    id,
                    name,
                    args_buf: String::new(),
                    early_input: None,
                });
            }
            ModelChunk::ToolCallInputDelta { id, delta } => {
                if let Some(s) = tool_states.iter_mut().find(|s| s.id == id) {
                    s.args_buf.push_str(&delta);
                }
            }
            ModelChunk::ToolCallEnd { id, input } => {
                if let Some(s) = tool_states.iter_mut().find(|s| s.id == id) {
                    s.early_input = input;
                }
            }
            ModelChunk::Done {
                stop_reason: sr,
                usage: u,
            } => {
                stop_reason = sr;
                usage = u;
            }
        }
    }

    // Finalised thinking block — `saw_thinking` covers the (rare) case
    // where the provider sent only signature + empty text. We only build
    // the AssistantThinking if at least one of the two parts landed.
    let thinking = if saw_thinking || thinking_signature.is_some() {
        Some(AssistantThinking {
            text: thinking_buf,
            signature: thinking_signature,
        })
    } else {
        None
    };

    // Tool call takes precedence — see collect_model_response in model.rs
    // for the same rule; the model deferred the final answer until the
    // tool runs, so we dispatch the tool(s) instead of emitting TurnEnd.
    // Multiple tool_use blocks land here when the provider runs
    // parallel_tool_calls — we forward all of them to run_loop.
    if !tool_states.is_empty() {
        let mut invocations = Vec::with_capacity(tool_states.len());
        for state in tool_states {
            let parsed_input = match state.early_input {
                Some(v) => v,
                None => {
                    let trimmed = state.args_buf.trim();
                    if trimmed.is_empty() {
                        // Some providers ship the final tool_call with
                        // no args delta (e.g. zero-arg tools); treat
                        // empty buffer as an empty object.
                        Value::Object(serde_json::Map::new())
                    } else {
                        match serde_json::from_str(trimmed) {
                            Ok(v) => v,
                            Err(e) => {
                                // Weak models truncate / malform streamed
                                // arguments; run the repair chain before
                                // failing the turn. A rescued (possibly
                                // partial) input the tool can reject is
                                // strictly better than a dead turn.
                                let res = crate::tool_repair::repair_truncated_json(trimmed);
                                match serde_json::from_str(&res.repaired) {
                                    Ok(v) if res.changed => {
                                        tracing::warn!(
                                            target: "harness::tool_repair",
                                            tool = %state.name,
                                            id = %state.id,
                                            notes = ?res.notes,
                                            "repaired malformed tool arguments"
                                        );
                                        v
                                    }
                                    _ => {
                                        return Err(StepFailure::Fatal(
                                            NativeHarnessError::ModelOther(format!(
                                                "decode tool arguments for {id}: {e}",
                                                id = state.id
                                            )),
                                        ))
                                    }
                                }
                            }
                        }
                    }
                }
            };
            invocations.push(ToolInvocation {
                id: state.id,
                name: state.name,
                input: parsed_input,
            });
        }
        return Ok(StepDrain::Complete(StepOutcome {
            next: StepNext::ToolCalls {
                preface: (!text_buf.is_empty()).then_some(text_buf),
                invocations,
            },
            usage,
            thinking,
        }));
    }

    Ok(StepDrain::Complete(StepOutcome {
        next: StepNext::Message {
            text: text_buf,
            stop_reason,
        },
        usage,
        thinking,
    }))
}

struct ToolBuf {
    id: String,
    name: String,
    args_buf: String,
    early_input: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::{CompactionContext, CompactionError, CompactionStrategy};
    use crate::model::{ModelChunk, ModelClient, ModelClientError, ModelResponse};
    use crate::tools::{ToolInvocation, ToolOutcome};
    use crate::{HarnessInternalEvent, MockToolRuntime, ScriptedModelClient};
    use async_trait::async_trait;
    use futures::stream::{BoxStream, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Test-only model client that returns a scripted sequence of responses.
    /// Each `next()` pops the front of the queue. Used to assert how
    /// `AgentLoopHarness` folds per-call usage into the turn total.
    #[derive(Clone)]
    struct QueueModelClient {
        queue: Arc<Mutex<Vec<ModelResponse>>>,
    }

    impl QueueModelClient {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                queue: Arc::new(Mutex::new(responses)),
            }
        }
    }

    #[async_trait]
    impl ModelClient for QueueModelClient {
        async fn stream(
            &self,
            _input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            let mut q = self.queue.lock().unwrap();
            if q.is_empty() {
                return Err(ModelClientError::Other("queue exhausted".into()));
            }
            let response = q.remove(0);
            let chunks = response_to_chunks(response);
            Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed())
        }
    }

    /// Render a synthetic `ModelResponse` as the `ModelChunk` sequence the
    /// streaming impl would have emitted. Lets QueueModelClient assert
    /// agent-loop behaviour without doing real SSE in tests.
    fn response_to_chunks(response: ModelResponse) -> Vec<ModelChunk> {
        match response {
            ModelResponse::Message {
                text,
                stop_reason,
                usage,
            } => {
                let mut out = Vec::new();
                if !text.is_empty() {
                    out.push(ModelChunk::TextDelta {
                        msg_id: "queue_msg".into(),
                        delta: text,
                    });
                }
                out.push(ModelChunk::Done { stop_reason, usage });
                out
            }
            ModelResponse::ToolCall {
                preface,
                invocation,
                usage,
            } => {
                let mut out = Vec::new();
                if let Some(p) = preface {
                    if !p.is_empty() {
                        out.push(ModelChunk::TextDelta {
                            msg_id: "queue_msg".into(),
                            delta: p,
                        });
                    }
                }
                out.push(ModelChunk::ToolCallStart {
                    id: invocation.id.clone(),
                    name: invocation.name.clone(),
                });
                out.push(ModelChunk::ToolCallEnd {
                    id: invocation.id.clone(),
                    input: Some(invocation.input.clone()),
                });
                out.push(ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage,
                });
                out
            }
        }
    }

    fn usage(input: u64, output: u64, cache_read: u64) -> HarnessUsage {
        HarnessUsage {
            input_tokens: input,
            output_tokens: output,
            cache_read_input_tokens: cache_read,
            cache_creation_input_tokens: 0,
            compaction_input_tokens: 0,
            compaction_output_tokens: 0,
        }
    }

    #[tokio::test]
    async fn agent_loop_accumulates_usage_across_steps() {
        // 2 steps: tool call (10/5 tokens) then final message (20/15 tokens).
        let model = QueueModelClient::new(vec![
            ModelResponse::ToolCall {
                preface: None,
                invocation: ToolInvocation {
                    id: "tc_1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "pwd"}),
                },
                usage: Some(usage(10, 5, 0)),
            },
            ModelResponse::Message {
                text: "done".into(),
                stop_reason: "end_turn".into(),
                usage: Some(usage(20, 15, 4)),
            },
        ]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "pwd".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        // Drain until TurnEnd and inspect usage.
        let mut final_usage = None;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd { usage: u, .. } = item.unwrap() {
                final_usage = u;
                break;
            }
        }
        let u = final_usage.expect("TurnEnd carried usage");
        assert_eq!(u.input_tokens, 30);
        assert_eq!(u.output_tokens, 20);
        assert_eq!(u.cache_read_input_tokens, 4);
    }

    #[tokio::test]
    async fn agent_loop_errors_on_silent_stop() {
        let model = QueueModelClient::new(vec![ModelResponse::Message {
            text: "   \n".into(),
            stop_reason: "end_turn".into(),
            usage: None,
        }]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "say something".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        match rx.recv().await.unwrap() {
            Err(NativeHarnessError::ModelOther(msg)) => {
                assert!(msg.contains("silent_stop"));
                assert!(msg.contains("stop_reason=end_turn"));
            }
            other => panic!("expected silent_stop model error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn agent_loop_errors_on_empty_max_tokens_stop() {
        let model = QueueModelClient::new(vec![ModelResponse::Message {
            text: "".into(),
            stop_reason: "max_tokens".into(),
            usage: None,
        }]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "think".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        match rx.recv().await.unwrap() {
            Err(NativeHarnessError::ModelOther(msg)) => {
                assert!(msg.contains("silent_stop"));
                assert!(msg.contains("stop_reason=max_tokens"));
            }
            other => panic!("expected silent_stop model error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn agent_loop_turn_end_usage_is_none_when_no_step_reported() {
        // Provider reports no usage on either step.
        let model = QueueModelClient::new(vec![ModelResponse::Message {
            text: "ok".into(),
            stop_reason: "end_turn".into(),
            usage: None,
        }]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "noop".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();
        let mut saw_usage = None;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd { usage, .. } = item.unwrap() {
                saw_usage = Some(usage);
                break;
            }
        }
        assert_eq!(saw_usage.unwrap(), None);
    }

    /// Streaming-aware fake client. Emits a pre-computed `ModelChunk`
    /// sequence per call — distinct from `QueueModelClient` which uses
    /// the `ModelResponse → chunks` translation. Tests that need
    /// token-level chunking go through this one.
    #[derive(Clone)]
    struct StreamingFakeClient {
        chunks_per_call: Arc<Mutex<Vec<Vec<ModelChunk>>>>,
    }

    impl StreamingFakeClient {
        fn new(per_call: Vec<Vec<ModelChunk>>) -> Self {
            Self {
                chunks_per_call: Arc::new(Mutex::new(per_call)),
            }
        }
    }

    #[async_trait]
    impl ModelClient for StreamingFakeClient {
        async fn stream(
            &self,
            _input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            let mut bucket = self.chunks_per_call.lock().unwrap();
            if bucket.is_empty() {
                return Err(ModelClientError::Other("queue exhausted".into()));
            }
            let chunks = bucket.remove(0);
            Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed())
        }
    }

    #[tokio::test]
    async fn agent_loop_forwards_token_chunks_to_harness_output() {
        let model = StreamingFakeClient::new(vec![vec![
            ModelChunk::TextDelta {
                msg_id: "remote_msg".into(),
                delta: "Hel".into(),
            },
            ModelChunk::TextDelta {
                msg_id: "remote_msg".into(),
                delta: "lo ".into(),
            },
            ModelChunk::TextDelta {
                msg_id: "remote_msg".into(),
                delta: "world".into(),
            },
            ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: None,
            },
        ]]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut deltas: Vec<String> = Vec::new();
        let mut saw_end = false;
        while let Some(item) = rx.recv().await {
            match item.unwrap() {
                HarnessInternalEvent::AssistantTextChunk { msg_id, delta } => {
                    // The harness rewrites msg_id to the step-local form
                    // so native_adapter accumulates everything from one
                    // step into a single AgentMessage frame.
                    assert_eq!(msg_id, "msg_native_0");
                    deltas.push(delta);
                }
                HarnessInternalEvent::TurnEnd { stop_reason, .. } => {
                    assert_eq!(stop_reason, "end_turn");
                    saw_end = true;
                    break;
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert_eq!(deltas, vec!["Hel", "lo ", "world"]);
        assert!(saw_end);
    }

    #[tokio::test]
    async fn agent_loop_streaming_tool_call_then_summary() {
        // Two scripted streams: first dispatches a tool with streamed
        // arguments; second returns a final message after the tool
        // result. Tests that the agent loop:
        //  * accumulates streamed JSON arguments correctly
        //  * runs the tool with the parsed value
        //  * feeds the tool result back into the next stream's input
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r1".into(),
                    delta: "running ".into(),
                },
                ModelChunk::ToolCallStart {
                    id: "tc_1".into(),
                    name: "bash".into(),
                },
                ModelChunk::ToolCallInputDelta {
                    id: "tc_1".into(),
                    delta: "{\"command\":".into(),
                },
                ModelChunk::ToolCallInputDelta {
                    id: "tc_1".into(),
                    delta: "\"pwd\"}".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_1".into(),
                    input: None,
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r2".into(),
                    delta: "done".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "pwd".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        // Expected event sequence:
        //   AssistantTextChunk("running ")
        //   ToolCall{ name=bash, input={"command":"pwd"} }
        //   ToolResult{ ok }
        //   AssistantTextChunk("done")
        //   TurnEnd
        let ev = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            ev,
            HarnessInternalEvent::AssistantTextChunk { ref delta, .. } if delta == "running "
        ));
        let ev = rx.recv().await.unwrap().unwrap();
        let HarnessInternalEvent::ToolCall { name, input, .. } = ev else {
            panic!("expected ToolCall");
        };
        assert_eq!(name, "bash");
        assert_eq!(input["command"], "pwd");
        let ev = rx.recv().await.unwrap().unwrap();
        assert!(matches!(ev, HarnessInternalEvent::ToolResult { .. }));
        let ev = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            ev,
            HarnessInternalEvent::AssistantTextChunk { ref delta, .. } if delta == "done"
        ));
        let ev = rx.recv().await.unwrap().unwrap();
        assert!(matches!(ev, HarnessInternalEvent::TurnEnd { .. }));
    }

    #[tokio::test]
    async fn agent_loop_repairs_truncated_tool_arguments() {
        // OpenAI-style streamed arguments cut off mid-object (missing the
        // closing brace). Without the repair chain this was a fatal
        // ModelOther; with it the args close cleanly and the tool runs.
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::ToolCallStart {
                    id: "tc_trunc".into(),
                    name: "bash".into(),
                },
                ModelChunk::ToolCallInputDelta {
                    id: "tc_trunc".into(),
                    delta: r#"{"command":"pwd""#.into(), // truncated
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_trunc".into(),
                    input: None,
                },
                ModelChunk::Done {
                    stop_reason: "tool_use".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r2".into(),
                    delta: "done".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "pwd".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut saw_tool_call = false;
        let mut saw_turn_end = false;
        while let Some(item) = rx.recv().await {
            match item.expect("turn must not fail on truncated args") {
                HarnessInternalEvent::ToolCall { name, input, .. } => {
                    assert_eq!(name, "bash");
                    assert_eq!(input["command"], "pwd", "repaired args reach the wire");
                    saw_tool_call = true;
                }
                HarnessInternalEvent::TurnEnd { .. } => {
                    saw_turn_end = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_tool_call, "expected ToolCall with repaired input");
        assert!(saw_turn_end);
    }

    /// Records the invocation input the runtime actually received, so a
    /// test can assert dispatch saw the schema-repaired arguments.
    #[derive(Clone)]
    struct ProbeToolRuntime {
        seen_input: Arc<Mutex<Option<Value>>>,
    }

    #[async_trait]
    impl ToolRuntime for ProbeToolRuntime {
        fn specs(&self) -> Vec<crate::tools::ToolSpec> {
            vec![crate::tools::ToolSpec {
                name: "probe".into(),
                description: "records its input".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "literal": {"type": "boolean"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["pattern"]
                }),
            }]
        }

        async fn invoke(
            &self,
            invocation: ToolInvocation,
        ) -> Result<ToolOutcome, ToolRuntimeError> {
            *self.seen_input.lock().unwrap() = Some(invocation.input);
            Ok(ToolOutcome {
                output: Ok(r#"{"ok":true}"#.into()),
                attachments: vec![],
            })
        }
    }

    #[tokio::test]
    async fn agent_loop_applies_schema_repair_before_dispatch() {
        // Weak-model shape mistakes — "true" for a boolean, "30" for an
        // integer — are coerced against the tool's input_schema before the
        // runtime sees them.
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::ToolCallStart {
                    id: "tc_shape".into(),
                    name: "probe".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_shape".into(),
                    input: Some(json!({"pattern": "x", "literal": "true", "limit": "30"})),
                },
                ModelChunk::Done {
                    stop_reason: "tool_use".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r2".into(),
                    delta: "done".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let seen_input = Arc::new(Mutex::new(None));
        let tools = ProbeToolRuntime {
            seen_input: seen_input.clone(),
        };
        let harness = AgentLoopHarness::new(model, tools);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "go".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut wire_input: Option<Value> = None;
        let mut history: Option<Vec<ChatMessage>> = None;
        while let Some(item) = rx.recv().await {
            match item.unwrap() {
                HarnessInternalEvent::ToolCall { input, .. } => wire_input = Some(input),
                HarnessInternalEvent::TurnEnd { final_messages, .. } => {
                    history = Some(final_messages);
                    break;
                }
                _ => {}
            }
        }
        let repaired = json!({"pattern": "x", "literal": true, "limit": 30});
        // Runtime, wire event, and history all agree on the repaired input.
        assert_eq!(seen_input.lock().unwrap().clone().unwrap(), repaired);
        assert_eq!(wire_input.unwrap(), repaired);
        let history = history.unwrap();
        let assistant_tool_calls = history
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    Some(tool_calls.clone())
                }
                _ => None,
            })
            .expect("assistant message with tool_calls in history");
        assert_eq!(assistant_tool_calls[0].input, repaired);
    }

    #[derive(Clone)]
    struct TimeoutToolRuntime;

    #[async_trait]
    impl ToolRuntime for TimeoutToolRuntime {
        fn specs(&self) -> Vec<crate::tools::ToolSpec> {
            vec![crate::tools::ToolSpec {
                name: "slow".into(),
                description: "always times out".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }]
        }

        async fn invoke(
            &self,
            _invocation: ToolInvocation,
        ) -> Result<ToolOutcome, ToolRuntimeError> {
            Err(ToolRuntimeError::Timeout("tool timed out after 1s".into()))
        }
    }

    #[tokio::test]
    async fn agent_loop_tool_timeout_is_model_observable_result() {
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::ToolCallStart {
                    id: "tc_timeout".into(),
                    name: "slow".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_timeout".into(),
                    input: Some(json!({})),
                },
                ModelChunk::Done {
                    stop_reason: "tool_use".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r2".into(),
                    delta: "recovered".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let harness = AgentLoopHarness::new(model, TimeoutToolRuntime);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "run slow".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::ToolCall { .. }
        ));
        match rx.recv().await.unwrap().unwrap() {
            HarnessInternalEvent::ToolResult { output, .. } => {
                let err = output.unwrap_err();
                assert!(err.contains("Timeout"));
                assert!(err.contains("tool timed out"));
            }
            other => panic!("expected timeout ToolResult, got {other:?}"),
        }
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::AssistantTextChunk { ref delta, .. } if delta == "recovered"
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::TurnEnd { ref stop_reason, .. } if stop_reason == "end_turn"
        ));
    }

    #[tokio::test]
    async fn agent_loop_invalid_tool_input_is_model_observable_and_bounded() {
        let huge_content = "x".repeat(20_000);
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::ToolCallStart {
                    id: "tc_bad_write".into(),
                    name: "write".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_bad_write".into(),
                    input: Some(json!({"content": huge_content})),
                },
                ModelChunk::Done {
                    stop_reason: "tool_use".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "r2".into(),
                    delta: "recovered".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "write file".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::ToolCall { .. }
        ));
        match rx.recv().await.unwrap().unwrap() {
            HarnessInternalEvent::ToolResult { output, .. } => {
                let err = output.unwrap_err();
                // Now caught by the wrapper's schema validation BEFORE the
                // inner runtime, with a teaching example appended.
                assert!(err.contains("The write tool was called with invalid arguments"));
                assert!(err.contains("missing required field `path`"), "{err}");
                assert!(err.contains("Received fields: content"));
                assert!(err.contains("string(20000 chars"));
                assert!(
                    err.contains("Expected shape"),
                    "teaching example missing: {err}"
                );
                assert!(
                    !err.contains(&"x".repeat(2000)),
                    "error should not echo full content"
                );
            }
            other => panic!("expected invalid-input ToolResult, got {other:?}"),
        }
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::AssistantTextChunk { ref delta, .. } if delta == "recovered"
        ));
    }

    /// Spy strategy that always fires and records the call count. Lets
    /// us verify agent_loop actually consults the compaction policy
    /// between steps without depending on a real summarize round trip.
    struct CountingCompactionStrategy {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl CompactionStrategy for CountingCompactionStrategy {
        fn should_compact(&self, _messages: &[ChatMessage], _context_window_tokens: u64) -> bool {
            true
        }

        async fn compact(
            &self,
            _messages: Vec<ChatMessage>,
            _ctx: &CompactionContext,
        ) -> Result<crate::compaction::CompactionOutcome, CompactionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // Replace history with a single synthetic user message — the
            // test asserts on the call count, not the content shape.
            Ok(crate::compaction::CompactionOutcome {
                messages: vec![ChatMessage::User {
                    content: "<conversation-summary>FOLDED</conversation-summary>".into(),
                    attachments: vec![],
                }],
                usage: None,
            })
        }
    }

    /// Spy strategy that reports a fixed `HarnessUsage` from its compact
    /// call. Lets us assert that agent_loop forwards compaction usage
    /// into the turn-level total + the `compaction_*` sub-buckets.
    struct UsageReportingCompactionStrategy {
        invoked: Arc<AtomicUsize>,
        per_call_usage: HarnessUsage,
    }

    #[async_trait]
    impl CompactionStrategy for UsageReportingCompactionStrategy {
        fn should_compact(&self, _: &[ChatMessage], _: u64) -> bool {
            // Fire once per step. Since the model fixture below ends the
            // turn after one step, this triggers exactly once per
            // run_turn.
            self.invoked.load(Ordering::SeqCst) == 0
        }
        async fn compact(
            &self,
            messages: Vec<ChatMessage>,
            _ctx: &CompactionContext,
        ) -> Result<crate::compaction::CompactionOutcome, CompactionError> {
            self.invoked.fetch_add(1, Ordering::SeqCst);
            Ok(crate::compaction::CompactionOutcome {
                messages,
                usage: Some(self.per_call_usage.clone()),
            })
        }
    }

    #[tokio::test]
    async fn agent_loop_attributes_compaction_usage_to_subbucket_and_total() {
        // Compaction reports 50/20 tokens; main step reports 100/30.
        // TurnEnd.usage should sum into 150/50, with compaction_*
        // sub-buckets showing the 50/20 isolated.
        let model = StreamingFakeClient::new(vec![vec![
            ModelChunk::TextDelta {
                msg_id: "m".into(),
                delta: "done".into(),
            },
            ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: Some(usage(100, 30, 0)),
            },
        ]]);
        let invoked = Arc::new(AtomicUsize::new(0));
        let strategy = UsageReportingCompactionStrategy {
            invoked: invoked.clone(),
            per_call_usage: usage(50, 20, 0),
        };
        let policy = CompactionPolicy {
            strategy: Arc::new(strategy),
            model_client: Arc::new(ScriptedModelClient),
            context_window_tokens: 1, // forces should_compact's true branch
        };
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new()).with_compaction(policy);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();
        let mut final_usage = None;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd { usage, .. } = item.unwrap() {
                final_usage = usage;
                break;
            }
        }
        assert_eq!(invoked.load(Ordering::SeqCst), 1);
        let u = final_usage.expect("TurnEnd carried usage");
        // Main step (100, 30) + compaction (50, 20) = total (150, 50).
        assert_eq!(u.input_tokens, 150);
        assert_eq!(u.output_tokens, 50);
        // Compaction sub-bucket isolates the (50, 20) portion.
        assert_eq!(u.compaction_input_tokens, 50);
        assert_eq!(u.compaction_output_tokens, 20);
    }

    /// Stub client that records every `ModelTurnInput.messages` it was
    /// asked to stream. Lets the test assert that the compaction-replaced
    /// messages are what reaches the model on the next step.
    #[derive(Clone)]
    struct RecordingFakeClient {
        last_messages: Arc<Mutex<Option<Vec<ChatMessage>>>>,
        chunks: Vec<ModelChunk>,
    }

    #[async_trait]
    impl ModelClient for RecordingFakeClient {
        async fn stream(
            &self,
            input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            *self.last_messages.lock().unwrap() = Some(input.messages);
            Ok(futures::stream::iter(self.chunks.clone().into_iter().map(Ok)).boxed())
        }
    }

    #[tokio::test]
    async fn agent_loop_invokes_compaction_between_steps() {
        let calls = Arc::new(AtomicUsize::new(0));
        let last_messages = Arc::new(Mutex::new(None::<Vec<ChatMessage>>));
        let model = RecordingFakeClient {
            last_messages: last_messages.clone(),
            chunks: vec![
                ModelChunk::TextDelta {
                    msg_id: "m".into(),
                    delta: "done".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        };
        // Summary client used by the spy strategy's ctx — not actually
        // called because our strategy short-circuits, but we satisfy
        // the policy contract.
        let summary_client: Arc<dyn ModelClient> = Arc::new(ScriptedModelClient);
        let policy = CompactionPolicy {
            strategy: Arc::new(CountingCompactionStrategy {
                calls: calls.clone(),
            }),
            model_client: summary_client,
            context_window_tokens: 1, // forces should_compact to fire
        };

        let harness = AgentLoopHarness::new(model, MockToolRuntime::new()).with_compaction(policy);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hello".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();
        let mut compaction_event: Option<(usize, usize)> = None;
        while let Some(item) = rx.recv().await {
            match item.unwrap() {
                HarnessInternalEvent::CompactionApplied {
                    original_message_count,
                    compacted_message_count,
                    ..
                } => {
                    compaction_event = Some((original_message_count, compacted_message_count));
                }
                HarnessInternalEvent::TurnEnd { .. } => break,
                _ => {}
            }
        }
        // Compaction ran exactly once before the single model step.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // CompactionApplied event surfaced with sensible counts.
        let (orig, comp) = compaction_event.expect("CompactionApplied event emitted");
        assert_eq!(orig, 1, "started with 1 message ([User \"hello\"])");
        assert_eq!(comp, 1, "spy strategy folded to single User message");
        // The model saw the FOLDED messages, not the original
        // [User "hello"] prefix.
        let observed = last_messages.lock().unwrap().clone().expect("model called");
        assert_eq!(observed.len(), 1);
        match &observed[0] {
            ChatMessage::User { content, .. } => {
                assert!(content.contains("FOLDED"), "got {content:?}");
            }
            other => panic!("expected User, got {other:?}"),
        }
    }

    /// Model client whose stream blocks indefinitely until cancelled.
    /// Lets the test prove that cancel_token.cancelled() races
    /// stream.next() and wins.
    #[derive(Clone)]
    struct HangingModelClient {
        started: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl ModelClient for HangingModelClient {
        async fn stream(
            &self,
            _input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            // Channel-backed stream whose sender never sends and never
            // drops — `rx.next().await` parks forever. Mirrors a real
            // LLM that opened the SSE response but hasn't shipped a
            // chunk yet (slow first-token time).
            let (tx, rx) = mpsc::channel::<Result<ModelChunk, ModelClientError>>(1);
            let started = self.started.clone();
            tokio::spawn(async move {
                // Hold the sender alive for the test's lifetime. Notify
                // the test that the stream is "started" so it knows
                // when to fire cancel — proves the cancel races a
                // pending stream.next(), not the pre-step check.
                started.notify_one();
                let _retain = tx; // suppress drop warning
                let () = std::future::pending().await;
            });
            Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
        }
    }

    #[tokio::test]
    async fn agent_loop_cancellation_interrupts_in_flight_stream() {
        // Without cancel: agent_loop would hang forever waiting for the
        // first chunk. With cancel fired *after* the stream began, the
        // select! arm in consume_step_stream wins and we get TurnEnd
        // with stop_reason "interrupt" within milliseconds.
        let started = Arc::new(tokio::sync::Notify::new());
        let model = HangingModelClient {
            started: started.clone(),
        };
        let cancel = CancellationToken::new();
        let cancel_for_outside = cancel.clone();

        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: Some(cancel),
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        // Wait for the model to actually start streaming, then cancel.
        // (Cancelling before the stream begins would short-circuit at
        // the pre-step check_cancel! macro — also correct, but a
        // different code path. We want to exercise the select!.)
        started.notified().await;
        cancel_for_outside.cancel();

        // Within a small window we should observe a TurnEnd{interrupt}.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut saw_interrupt = false;
        while tokio::time::Instant::now() < deadline {
            tokio::select! {
                item = rx.recv() => {
                    match item {
                        Some(Ok(HarnessInternalEvent::TurnEnd { stop_reason, .. })) => {
                            assert_eq!(stop_reason, "interrupt");
                            saw_interrupt = true;
                            break;
                        }
                        Some(_) => continue,
                        None => break,
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
            }
        }
        assert!(saw_interrupt, "expected TurnEnd{{interrupt}} after cancel");
    }

    /// Per-`stream()`-call scripted client for the mid-stream idle-timeout
    /// tests. Each behavior either streams chunks to completion (stream
    /// closes), or emits an optional prefix then parks forever without
    /// closing — simulating a silently wedged upstream (TCP open, no FIN/RST,
    /// no further bytes). `calls` counts establishments so tests can assert
    /// whether a reconnect happened.
    enum StallBehavior {
        /// Stream these chunks, then close (clean end).
        Complete(Vec<ModelChunk>),
        /// Emit these chunks (possibly none), then hang forever.
        EmitThenHang(Vec<ModelChunk>),
    }

    #[derive(Clone)]
    struct StallingModelClient {
        behaviors: Arc<Mutex<Vec<StallBehavior>>>,
        calls: Arc<AtomicUsize>,
    }

    impl StallingModelClient {
        fn new(behaviors: Vec<StallBehavior>) -> Self {
            Self {
                behaviors: Arc::new(Mutex::new(behaviors)),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl ModelClient for StallingModelClient {
        async fn stream(
            &self,
            _input: ModelTurnInput,
        ) -> Result<BoxStream<'static, Result<ModelChunk, ModelClientError>>, ModelClientError>
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // Pop the next scripted behavior; once the script is exhausted,
            // default to hanging (covers "every attempt stalls" tests).
            let behavior = {
                let mut b = self.behaviors.lock().unwrap();
                if b.is_empty() {
                    StallBehavior::EmitThenHang(vec![])
                } else {
                    b.remove(0)
                }
            };
            let (tx, rx) = mpsc::channel::<Result<ModelChunk, ModelClientError>>(8);
            tokio::spawn(async move {
                match behavior {
                    StallBehavior::Complete(chunks) => {
                        for c in chunks {
                            if tx.send(Ok(c)).await.is_err() {
                                return;
                            }
                        }
                        // tx dropped here → stream ends cleanly.
                    }
                    StallBehavior::EmitThenHang(chunks) => {
                        for c in chunks {
                            if tx.send(Ok(c)).await.is_err() {
                                return;
                            }
                        }
                        let _retain = tx; // hold sender open so rx parks
                        let () = std::future::pending().await;
                    }
                }
            });
            Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn agent_loop_reconnects_after_stall_before_any_output() {
        // First establishment opens the stream then goes silent → idle
        // watchdog fires → no output yet, so it's safe to reconnect. Second
        // establishment streams a full response. We should see the text
        // exactly once and a clean end_turn, with two establishments total.
        let model = StallingModelClient::new(vec![
            StallBehavior::EmitThenHang(vec![]),
            StallBehavior::Complete(vec![
                ModelChunk::TextDelta {
                    msg_id: "m".into(),
                    delta: "ok".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ]),
        ]);
        let calls = model.calls.clone();
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new())
            .with_stream_resilience(Duration::from_millis(50), 3);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut text = String::new();
        let mut stop = None;
        while let Some(item) = rx.recv().await {
            match item.expect("no error expected") {
                HarnessInternalEvent::AssistantTextChunk { delta, .. } => text.push_str(&delta),
                HarnessInternalEvent::TurnEnd { stop_reason, .. } => {
                    stop = Some(stop_reason);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(stop.as_deref(), Some("end_turn"));
        assert_eq!(text, "ok", "text delivered exactly once, no duplication");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "stream established twice (one reconnect)"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn agent_loop_surfaces_error_when_reconnect_budget_exhausted() {
        // Every establishment stalls. With max_attempts = 2 we get one
        // reconnect, then the second stall is terminal → ModelNetwork error.
        let model = StallingModelClient::new(vec![]); // all default to hang
        let calls = model.calls.clone();
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new())
            .with_stream_resilience(Duration::from_millis(50), 2);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut saw_error = false;
        while let Some(item) = rx.recv().await {
            match item {
                Err(NativeHarnessError::ModelNetwork(msg)) => {
                    assert!(msg.contains("stalled"), "got {msg:?}");
                    saw_error = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
                Ok(_) => {}
            }
        }
        assert!(
            saw_error,
            "expected ModelNetwork stall error after budget exhausted"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "two establishments (initial + one reconnect)"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn agent_loop_does_not_reconnect_after_stall_with_partial_output() {
        // Stream emits text (the user now sees it) then stalls. Even though
        // the reconnect budget is generous, a stall *after* output is
        // terminal — reconnecting would re-issue the request and duplicate
        // what was already shown. Expect: text once, then a ModelNetwork
        // error, and exactly one establishment (no reconnect).
        let model = StallingModelClient::new(vec![StallBehavior::EmitThenHang(vec![
            ModelChunk::TextDelta {
                msg_id: "m".into(),
                delta: "partial".into(),
            },
        ])]);
        let calls = model.calls.clone();
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new())
            .with_stream_resilience(Duration::from_millis(50), 5);
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut text = String::new();
        let mut saw_error = false;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(HarnessInternalEvent::AssistantTextChunk { delta, .. }) => text.push_str(&delta),
                Err(NativeHarnessError::ModelNetwork(_)) => {
                    saw_error = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
                Ok(_) => {}
            }
        }
        assert!(saw_error, "expected terminal ModelNetwork error");
        assert_eq!(
            text, "partial",
            "partial output delivered once, not replayed"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "no reconnect once output has reached the user"
        );
    }

    #[tokio::test]
    async fn agent_loop_accumulates_thinking_chunks_and_signature() {
        // Anthropic-style step: thinking deltas + signature, then text,
        // then Done. Asserts that:
        //   * each ThinkingDelta with non-empty text emits an
        //     AssistantThinkingChunk;
        //   * the signature latches and ends up on
        //     ChatMessage::Assistant.thinking;
        //   * an empty-text ThinkingDelta carrying a signature does NOT
        //     emit a chunk (signature-only chunks are silent).
        let model = StreamingFakeClient::new(vec![vec![
            ModelChunk::ThinkingDelta {
                thinking_id: "th_1".into(),
                delta: "let me think...".into(),
                signature: None,
            },
            ModelChunk::ThinkingDelta {
                thinking_id: "th_1".into(),
                delta: "".into(),
                signature: Some("sig_abc".into()),
            },
            ModelChunk::TextDelta {
                msg_id: "m1".into(),
                delta: "ok".into(),
            },
            ModelChunk::Done {
                stop_reason: "end_turn".into(),
                usage: None,
            },
        ]]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "hi".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        let mut thinking_chunks: Vec<String> = Vec::new();
        let mut text_chunks: Vec<String> = Vec::new();
        let mut saw_end = false;
        while let Some(item) = rx.recv().await {
            match item.unwrap() {
                HarnessInternalEvent::AssistantThinkingChunk { msg_id, delta } => {
                    assert_eq!(msg_id, "thinking_native_0");
                    thinking_chunks.push(delta);
                }
                HarnessInternalEvent::AssistantTextChunk { msg_id, delta } => {
                    assert_eq!(msg_id, "msg_native_0");
                    text_chunks.push(delta);
                }
                HarnessInternalEvent::TurnEnd { .. } => {
                    saw_end = true;
                    break;
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        // Only the non-empty thinking delta emits a chunk; signature-only
        // chunk is silent.
        assert_eq!(thinking_chunks, vec!["let me think..."]);
        assert_eq!(text_chunks, vec!["ok"]);
        assert!(saw_end);
    }

    #[tokio::test]
    async fn agent_loop_runs_tool_then_final_message() {
        let harness = AgentLoopHarness::new(
            ScriptedModelClient,
            MockToolRuntime::new().with_file("README.md", "hello"),
        );
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "read README.md".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::AssistantTextChunk { .. }
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::ToolCall { ref name, .. } if name == "read"
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::ToolResult { .. }
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::AssistantTextChunk { .. }
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            HarnessInternalEvent::TurnEnd { .. }
        ));
        assert!(rx.recv().await.is_none());
    }

    /// `TurnEnd.final_messages` must reflect the whole conversation:
    /// every `prior_messages` entry RD seeded the turn with, plus the
    /// new user prompt, plus the assistant's reply. This is the
    /// contract RD's `native_history` slot depends on for multi-turn
    /// replay — if it ever shrinks (e.g. we accidentally clone before
    /// the final push), same-process multi-turn loses history.
    #[tokio::test]
    async fn agent_loop_turn_end_carries_full_message_history() {
        let model = QueueModelClient::new(vec![ModelResponse::Message {
            text: "second reply".into(),
            stop_reason: "end_turn".into(),
            usage: None,
        }]);
        let harness = AgentLoopHarness::new(model, MockToolRuntime::new());
        // Simulate "RD captured this from a previous turn".
        let prior = vec![
            ChatMessage::User {
                content: "first prompt".into(),
                attachments: vec![],
            },
            ChatMessage::Assistant {
                text: Some("first reply".into()),
                tool_calls: vec![],
                thinking: None,
            },
        ];
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "second prompt".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: prior,
                context_path: None,
            })
            .await
            .unwrap();
        let mut final_messages: Option<Vec<ChatMessage>> = None;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd {
                final_messages: m, ..
            } = item.unwrap()
            {
                final_messages = Some(m);
                break;
            }
        }
        let msgs = final_messages.expect("TurnEnd carried final_messages");
        // [user-1, assistant-1, user-2, assistant-2] — 4 entries.
        assert_eq!(msgs.len(), 4, "got {msgs:?}");
        match &msgs[0] {
            ChatMessage::User { content, .. } => assert_eq!(content, "first prompt"),
            other => panic!("msgs[0] not user-1: {other:?}"),
        }
        match &msgs[1] {
            ChatMessage::Assistant { text, .. } => {
                assert_eq!(text.as_deref(), Some("first reply"));
            }
            other => panic!("msgs[1] not assistant-1: {other:?}"),
        }
        match &msgs[2] {
            ChatMessage::User { content, .. } => assert_eq!(content, "second prompt"),
            other => panic!("msgs[2] not user-2: {other:?}"),
        }
        match &msgs[3] {
            ChatMessage::Assistant { text, .. } => {
                assert_eq!(text.as_deref(), Some("second reply"));
            }
            other => panic!("msgs[3] not assistant-2: {other:?}"),
        }
    }

    /// Tool runtime that sleeps for a configurable duration before
    /// returning. Records the actual concurrency observed (max number
    /// of in-flight invocations at any point) so we can assert the
    /// agent loop is truly running them in parallel, not interleaving.
    #[derive(Clone)]
    struct ConcurrencyProbeRuntime {
        sleep_for: std::time::Duration,
        in_flight: Arc<AtomicUsize>,
        max_concurrency: Arc<AtomicUsize>,
        call_order: Arc<Mutex<Vec<String>>>,
        cancelled: Arc<AtomicUsize>,
    }

    impl ConcurrencyProbeRuntime {
        fn new(sleep_for: std::time::Duration) -> Self {
            Self {
                sleep_for,
                in_flight: Arc::new(AtomicUsize::new(0)),
                max_concurrency: Arc::new(AtomicUsize::new(0)),
                call_order: Arc::new(Mutex::new(Vec::new())),
                cancelled: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl ToolRuntime for ConcurrencyProbeRuntime {
        fn specs(&self) -> Vec<crate::tools::ToolSpec> {
            vec![crate::tools::ToolSpec {
                name: "slow".into(),
                description: "sleeps".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }]
        }

        async fn invoke(
            &self,
            invocation: ToolInvocation,
        ) -> Result<ToolOutcome, ToolRuntimeError> {
            self.call_order.lock().unwrap().push(invocation.id.clone());
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut prev = self.max_concurrency.load(Ordering::SeqCst);
            while now > prev {
                match self.max_concurrency.compare_exchange(
                    prev,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
            tokio::time::sleep(self.sleep_for).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolOutcome {
                output: Ok(serde_json::json!({"slept": true, "id": invocation.id})),
                attachments: vec![],
            })
        }

        async fn invoke_cancellable(
            &self,
            invocation: ToolInvocation,
            cancel: Option<&CancellationToken>,
        ) -> Result<ToolOutcome, ToolRuntimeError> {
            self.call_order.lock().unwrap().push(invocation.id.clone());
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut prev = self.max_concurrency.load(Ordering::SeqCst);
            while now > prev {
                match self.max_concurrency.compare_exchange(
                    prev,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
            if let Some(token) = cancel {
                tokio::select! {
                    _ = token.cancelled() => {
                        self.cancelled.fetch_add(1, Ordering::SeqCst);
                        self.in_flight.fetch_sub(1, Ordering::SeqCst);
                        Err(ToolRuntimeError::Runtime("cancelled".into()))
                    }
                    _ = tokio::time::sleep(self.sleep_for) => {
                        self.in_flight.fetch_sub(1, Ordering::SeqCst);
                        Ok(ToolOutcome {
                            output: Ok(serde_json::json!({"slept": true, "id": invocation.id})),
                            attachments: vec![],
                        })
                    }
                }
            } else {
                tokio::time::sleep(self.sleep_for).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(ToolOutcome {
                    output: Ok(serde_json::json!({"slept": true, "id": invocation.id})),
                    attachments: vec![],
                })
            }
        }
    }

    /// When the model returns multiple `tool_use` blocks in a single
    /// step (parallel_tool_calls on OpenAI / multi tool_use on
    /// Anthropic), the agent loop MUST dispatch them concurrently —
    /// not sequentially. Before the F3 fix, only the first one was
    /// invoked and the rest were silently dropped.
    #[tokio::test]
    async fn agent_loop_runs_multi_tool_calls_concurrently() {
        // One step that emits 3 tool_use blocks back-to-back, then a
        // second step that returns a final message.
        let model = StreamingFakeClient::new(vec![
            vec![
                ModelChunk::ToolCallStart {
                    id: "tc_a".into(),
                    name: "slow".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_a".into(),
                    input: Some(json!({})),
                },
                ModelChunk::ToolCallStart {
                    id: "tc_b".into(),
                    name: "slow".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_b".into(),
                    input: Some(json!({})),
                },
                ModelChunk::ToolCallStart {
                    id: "tc_c".into(),
                    name: "slow".into(),
                },
                ModelChunk::ToolCallEnd {
                    id: "tc_c".into(),
                    input: Some(json!({})),
                },
                ModelChunk::Done {
                    stop_reason: "tool_use".into(),
                    usage: None,
                },
            ],
            vec![
                ModelChunk::TextDelta {
                    msg_id: "remote".into(),
                    delta: "done".into(),
                },
                ModelChunk::Done {
                    stop_reason: "end_turn".into(),
                    usage: None,
                },
            ],
        ]);
        let probe = ConcurrencyProbeRuntime::new(std::time::Duration::from_millis(80));
        let max_concurrency = probe.max_concurrency.clone();
        let harness = AgentLoopHarness::new(model, probe);

        let start = std::time::Instant::now();
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "go".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: None,
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();
        let mut tool_results = 0;
        while let Some(item) = rx.recv().await {
            match item.unwrap() {
                HarnessInternalEvent::ToolResult { .. } => tool_results += 1,
                HarnessInternalEvent::TurnEnd { .. } => break,
                _ => {}
            }
        }
        let elapsed = start.elapsed();
        // All 3 tools surfaced results — none were silently dropped.
        assert_eq!(
            tool_results, 3,
            "expected 3 tool results, got {tool_results}"
        );
        // Concurrency probe saw all 3 in flight simultaneously.
        assert_eq!(
            max_concurrency.load(Ordering::SeqCst),
            3,
            "expected max concurrency 3 (parallel dispatch), got {}",
            max_concurrency.load(Ordering::SeqCst)
        );
        // Wall clock < 3× sleep duration confirms parallelism (3 × 80ms
        // = 240ms sequential; parallel should be ~80ms + scheduler
        // overhead, allow up to 200ms for slow CI).
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "elapsed {elapsed:?} suggests sequential execution"
        );
    }

    /// When the cancel token fires while tool invocations are in
    /// flight, the agent loop must emit a clean `TurnEnd { interrupt }`
    /// and stop — not wait for the tools to drain naturally.
    #[tokio::test]
    async fn agent_loop_cancels_in_flight_tool_calls() {
        let model = StreamingFakeClient::new(vec![vec![
            ModelChunk::ToolCallStart {
                id: "tc_slow".into(),
                name: "slow".into(),
            },
            ModelChunk::ToolCallEnd {
                id: "tc_slow".into(),
                input: Some(json!({})),
            },
            ModelChunk::Done {
                stop_reason: "tool_use".into(),
                usage: None,
            },
        ]]);
        // 5-second sleep — if cancel didn't propagate, the test would
        // take 5s. We assert it returns in < 200ms.
        let probe = ConcurrencyProbeRuntime::new(std::time::Duration::from_secs(5));
        let cancelled_count = probe.cancelled.clone();
        let harness = AgentLoopHarness::new(model, probe);

        let cancel = CancellationToken::new();
        let cancel_for_input = cancel.clone();
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "go".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: Some(cancel_for_input),
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        // Let the tool spin up briefly, then cancel.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cancel.cancel();

        let start = std::time::Instant::now();
        let mut saw_interrupt = false;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd { stop_reason, .. } = item.unwrap() {
                assert_eq!(stop_reason, "interrupt");
                saw_interrupt = true;
                break;
            }
        }
        let elapsed = start.elapsed();
        assert!(saw_interrupt, "must see interrupt TurnEnd");
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "cancel propagation took too long: {elapsed:?}"
        );
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
        while cancelled_count.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            cancelled_count.load(Ordering::SeqCst),
            1,
            "tool runtime must observe the cancellation token"
        );
    }

    /// Cancellation while the tool is in flight must put the harness
    /// into a clean state: TurnEnd.final_messages carries the
    /// assistant tool_use blocks but no synthetic tool_result rows
    /// (since the tool never finished).
    #[tokio::test]
    async fn agent_loop_cancel_during_tools_yields_clean_history() {
        let model = StreamingFakeClient::new(vec![vec![
            ModelChunk::ToolCallStart {
                id: "tc_a".into(),
                name: "slow".into(),
            },
            ModelChunk::ToolCallEnd {
                id: "tc_a".into(),
                input: Some(json!({})),
            },
            ModelChunk::ToolCallStart {
                id: "tc_b".into(),
                name: "slow".into(),
            },
            ModelChunk::ToolCallEnd {
                id: "tc_b".into(),
                input: Some(json!({})),
            },
            ModelChunk::Done {
                stop_reason: "tool_use".into(),
                usage: None,
            },
        ]]);
        let probe = ConcurrencyProbeRuntime::new(std::time::Duration::from_secs(3));
        let harness = AgentLoopHarness::new(model, probe);

        let cancel = CancellationToken::new();
        let cancel_for_input = cancel.clone();
        let mut rx = harness
            .run_turn(NativeTurnInput {
                prompt_text: "go".into(),
                system_prompt: None,
                attachments: vec![],
                cancel_token: Some(cancel_for_input),
                prior_messages: vec![],
                context_path: None,
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cancel.cancel();

        let mut final_msgs = None;
        while let Some(item) = rx.recv().await {
            if let HarnessInternalEvent::TurnEnd { final_messages, .. } = item.unwrap() {
                final_msgs = Some(final_messages);
                break;
            }
        }
        let msgs = final_msgs.expect("interrupt TurnEnd");
        // History: [user, assistant(tool_use a + b)]
        // — no tool_result rows because the tools never finished.
        assert_eq!(msgs.len(), 2, "expected 2 messages, got {msgs:?}");
        match &msgs[1] {
            ChatMessage::Assistant { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 2);
                assert_eq!(tool_calls[0].id, "tc_a");
                assert_eq!(tool_calls[1].id, "tc_b");
            }
            other => panic!("msgs[1] not assistant: {other:?}"),
        }
    }
}
