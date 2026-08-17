//! Live correctness harness for `OpenAiResponsesModelClient`.
//!
//! Talks to the official OpenAI Responses endpoint by default. Set your key
//! and run:
//!   OPENAI_API_KEY=sk-... cargo run --example responses_smoke
//!
//! Override the target only when you deliberately want a different backend:
//!   RESPONSES_BASE_URL=https://my-gateway.example/v1   # must be one you trust
//!   RESPONSES_MODEL=gpt-5.5
//!   RESPONSES_EFFORT=high
//! The key is sent as a Bearer token to whatever `RESPONSES_BASE_URL` names,
//! so point it only at an endpoint you trust (prefer HTTPS).
//!
//! Exercises the projection + SSE parsing + tool loop + reasoning round-trip
//! end to end against the real API. Not a `#[test]` — it needs network + a key.

use std::collections::HashMap;

use harness::{
    AssistantThinking, ChatMessage, ModelCatalog, ModelChunk, ModelClient, ModelClientError,
    ModelRequestConfig, ModelTurnInput, OpenAiResponsesConfig, OpenAiResponsesModelClient,
    ReasoningConfig, ReasoningMode, ToolChoice, ToolInvocation, ToolSpec, WireProtocol,
};
use serde_json::json;
use tokio_stream::StreamExt;

/// Folded result of one streamed model step, mirroring what
/// `agent_loop::consume_step_stream` builds (text + tool calls + latched
/// thinking + stop_reason). Used to drive multi-turn loops by hand so we can
/// round-trip reasoning back into the next request.
#[derive(Debug, Default)]
struct Step {
    text: String,
    thinking_text: String,
    thinking_signature: Option<String>,
    tool_calls: Vec<ToolInvocation>,
    stop_reason: String,
    saw_reasoning: bool,
}

impl Step {
    fn thinking(&self) -> Option<AssistantThinking> {
        if self.saw_reasoning || self.thinking_signature.is_some() {
            Some(AssistantThinking {
                text: self.thinking_text.clone(),
                signature: self.thinking_signature.clone(),
            })
        } else {
            None
        }
    }

    /// Project this step into the `ChatMessage::Assistant` the harness would
    /// append to history (so the next turn re-sends reasoning + tool calls).
    fn as_assistant(&self) -> ChatMessage {
        ChatMessage::Assistant {
            text: (!self.text.is_empty()).then(|| self.text.clone()),
            tool_calls: self.tool_calls.clone(),
            thinking: self.thinking(),
            usage: None,
        }
    }
}

/// Consume a `ModelChunk` stream into a `Step`, replicating the harness's
/// accumulation rules (signature latch, per-id argument buffering).
async fn fold_step(
    mut stream: futures::stream::BoxStream<'static, Result<ModelChunk, ModelClientError>>,
) -> Result<Step, ModelClientError> {
    let mut step = Step {
        stop_reason: "end_turn".into(),
        ..Default::default()
    };
    // id -> (name, args buffer, early parsed input)
    let mut tools: Vec<(String, String, String, Option<serde_json::Value>)> = Vec::new();

    while let Some(item) = stream.next().await {
        match item? {
            ModelChunk::TextDelta { delta, .. } => step.text.push_str(&delta),
            ModelChunk::ThinkingDelta {
                delta, signature, ..
            } => {
                if let Some(sig) = signature {
                    if !sig.is_empty() {
                        step.thinking_signature = Some(sig);
                    }
                }
                if !delta.is_empty() {
                    step.saw_reasoning = true;
                    step.thinking_text.push_str(&delta);
                }
            }
            ModelChunk::ToolCallStart { id, name } => {
                tools.push((id, name, String::new(), None));
            }
            ModelChunk::ToolCallInputDelta { id, delta } => {
                if let Some(t) = tools.iter_mut().find(|t| t.0 == id) {
                    t.2.push_str(&delta);
                }
            }
            ModelChunk::ToolCallEnd { id, input } => {
                if let Some(t) = tools.iter_mut().find(|t| t.0 == id) {
                    t.3 = input;
                }
            }
            ModelChunk::Done { stop_reason, .. } => step.stop_reason = stop_reason,
        }
    }

    for (id, name, buf, early) in tools {
        let input = match early {
            Some(v) => v,
            None => {
                let t = buf.trim();
                if t.is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(t).unwrap_or(json!({}))
                }
            }
        };
        step.tool_calls.push(ToolInvocation {
            id,
            name,
            input,
            raw_emitted_args: None,
        });
    }
    Ok(step)
}

fn bash_tool() -> ToolSpec {
    ToolSpec {
        name: "bash".into(),
        description: "Run a shell command and return its stdout.".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "command": { "type": "string", "description": "the shell command" } },
            "required": ["command"],
            "additionalProperties": false
        }),
    }
}

fn get_weather_tool() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Get the current temperature in Celsius for a city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    }
}

fn user(text: &str) -> ChatMessage {
    ChatMessage::User {
        content: text.into(),
        attachments: vec![],
    }
}

/// Decode the `item_id\nencrypted_content` packing used by the Responses
/// client's reasoning round-trip. Returns `(item_id, encrypted_len)`.
fn harness_decode(sig: &str) -> Option<(String, usize)> {
    let (id, enc) = sig.split_once('\n')?;
    if id.is_empty() || enc.is_empty() {
        return None;
    }
    Some((id.to_string(), enc.len()))
}

struct Ctx {
    client: OpenAiResponsesModelClient,
    system: String,
}

impl Ctx {
    fn turn(&self, messages: Vec<ChatMessage>, tools: Vec<ToolSpec>) -> ModelTurnInput {
        ModelTurnInput {
            system_prompt: Some(self.system.clone()),
            messages,
            tools,
            hosted_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }
    }

    async fn step(&self, input: ModelTurnInput) -> Result<Step, ModelClientError> {
        let stream = self.client.stream(input).await?;
        fold_step(stream).await
    }
}

#[tokio::main]
async fn main() {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    // Default to the official OpenAI endpoint. The key is Bearer-sent to this
    // URL, so a plaintext-HTTP or third-party default would leak it — require
    // an explicit opt-in for any non-default backend.
    let base_url = std::env::var("RESPONSES_BASE_URL")
        .unwrap_or_else(|_| OpenAiResponsesConfig::DEFAULT_BASE_URL.into());
    let model = std::env::var("RESPONSES_MODEL").unwrap_or_else(|_| "gpt-5.5".into());
    let effort = std::env::var("RESPONSES_EFFORT").unwrap_or_else(|_| "high".into());

    println!("== config == base_url={base_url} model={model} effort={effort}\n");

    let model = ModelCatalog::initialize()
        .await
        .expect("initialize models.dev catalog")
        .resolve(
            ModelRequestConfig {
                model,
                max_output_tokens: 4_096,
                temperature: None,
                reasoning: ReasoningConfig {
                    mode: ReasoningMode::Enabled,
                    effort: Some(effort),
                    budget_tokens: None,
                },
            },
            WireProtocol::OpenAiResponses,
        )
        .await
        .expect("resolve model capabilities");

    let client = OpenAiResponsesModelClient::new(OpenAiResponsesConfig {
        base_url,
        api_key: key,
        model,
        reasoning_summary: Some("auto".into()),
    });
    let ctx = Ctx {
        client,
        system: "You are a terse test assistant. When a tool is available and \
                 relevant, call it rather than guessing."
            .into(),
    };

    let mut pass = 0usize;
    let mut fail = 0usize;
    macro_rules! check {
        ($name:expr, $cond:expr, $detail:expr) => {{
            if $cond {
                pass += 1;
                println!("  \u{2713} {}", $name);
            } else {
                fail += 1;
                println!("  \u{2717} {}  --  {}", $name, $detail);
            }
        }};
    }

    // ── Scenario 1: plain text, no tools ────────────────────────────────
    println!("[1] plain text (no tools)");
    match ctx
        .step(ctx.turn(vec![user("Reply with exactly one word: pong")], vec![]))
        .await
    {
        Ok(s) => {
            println!("    text={:?} stop={}", s.text.trim(), s.stop_reason);
            check!(
                "got non-empty text",
                !s.text.trim().is_empty(),
                "empty text"
            );
            check!(
                "no spurious tool calls",
                s.tool_calls.is_empty(),
                format!("{} tool calls", s.tool_calls.len())
            );
        }
        Err(e) => check!("scenario completed", false, format!("error: {e}")),
    }

    // ── Scenario 2: single tool call ────────────────────────────────────
    println!("\n[2] single tool call");
    let s2 = ctx
        .step(ctx.turn(
            vec![user(
                "Run the shell command that prints the current working directory.",
            )],
            vec![bash_tool()],
        ))
        .await;
    let mut assistant_tool_call: Option<ToolInvocation> = None;
    match &s2 {
        Ok(s) => {
            println!(
                "    tool_calls={:?} reasoning={}",
                s.tool_calls
                    .iter()
                    .map(|t| (&t.name, &t.input))
                    .collect::<Vec<_>>(),
                s.saw_reasoning
            );
            check!(
                "model called a tool",
                !s.tool_calls.is_empty(),
                "no tool call"
            );
            if let Some(tc) = s.tool_calls.first() {
                check!("tool name is bash", tc.name == "bash", tc.name.clone());
                check!(
                    "args parsed as object with `command`",
                    tc.input.get("command").and_then(|v| v.as_str()).is_some(),
                    format!("input={}", tc.input)
                );
                assistant_tool_call = Some(tc.clone());
            }
        }
        Err(e) => check!("scenario completed", false, format!("error: {e}")),
    }

    // ── Scenario 3: reasoning round-trip (encrypted_content, store:false) ─
    // A reasoning-heavy NO-tool prompt reliably emits a `reasoning` item with
    // encrypted_content; trivial tool calls often don't. We capture that item,
    // then re-send the assistant turn (carrying the reasoning item, encoded
    // into thinking.signature) plus a follow-up question. Under store:false a
    // malformed reasoning round-trip 400s, so acceptance proves the full path:
    // capture encrypted_content → encode signature → project reasoning item →
    // API accepts.
    println!("\n[3] reasoning round-trip (store:false)");
    {
        let q = "Work it out step by step: a store had 3 boxes of 24 apples, \
                 sold 17 apples, then received 2 more full boxes. How many \
                 apples now? Show your reasoning, then give the number.";
        match ctx.step(ctx.turn(vec![user(q)], vec![])).await {
            Ok(step1) => {
                let sig = step1.thinking().and_then(|t| t.signature).clone();
                let decoded = sig.as_deref().and_then(harness_decode);
                match &decoded {
                    Some((id, enc_len)) => println!(
                        "    captured reasoning item_id={id} encrypted_content_len={enc_len}"
                    ),
                    None => println!("    (no encrypted reasoning captured this run)"),
                }
                check!(
                    "captured an encrypted reasoning item",
                    decoded.is_some(),
                    "model emitted no reasoning item with encrypted_content"
                );
                check!(
                    "first-turn answer is correct (103)",
                    step1.text.contains("103"),
                    format!("text={:?}", step1.text)
                );
                // Re-send the reasoning-bearing assistant turn + a follow-up.
                let messages = vec![
                    user(q),
                    step1.as_assistant(),
                    user("Now multiply that result by 10 and give only the number."),
                ];
                match ctx.step(ctx.turn(messages, vec![])).await {
                    Ok(step2) => {
                        println!("    follow-up text={:?}", step2.text.trim());
                        check!(
                            "reasoning round-trip accepted (no 400)",
                            true,
                            "unreachable"
                        );
                        check!(
                            "follow-up answer is correct (1030)",
                            step2.text.contains("1030"),
                            format!("text={:?}", step2.text)
                        );
                    }
                    Err(e) => {
                        check!(
                            "reasoning round-trip accepted (no 400)",
                            false,
                            format!("error: {e}")
                        );
                    }
                }
            }
            Err(e) => check!("scenario completed", false, format!("error: {e}")),
        }
    }
    let _ = assistant_tool_call;

    // ── Scenario 4: two-round reasoning chain ───────────────────────────
    // A second tool round on the SAME conversation, so we re-send TWO
    // assistant turns each potentially carrying reasoning items. Stresses
    // the encrypted_content round-trip across multiple accumulated items.
    println!("\n[4] two-round reasoning chain");
    {
        let mut messages = vec![user(
            "Use the get_weather tool for Tokyo, then for Paris, one at a time. \
             After you know both, tell me which is warmer.",
        )];
        let tools = vec![get_weather_tool()];
        let fake: HashMap<&str, &str> = HashMap::from([
            ("Tokyo", "22"),
            ("Paris", "15"),
            ("tokyo", "22"),
            ("paris", "15"),
        ]);
        let mut ok = true;
        let mut rounds = 0;
        let mut final_text = String::new();
        loop {
            rounds += 1;
            if rounds > 5 {
                break;
            }
            let step = match ctx.step(ctx.turn(messages.clone(), tools.clone())).await {
                Ok(s) => s,
                Err(e) => {
                    ok = false;
                    println!("    round {rounds} error: {e}");
                    break;
                }
            };
            messages.push(step.as_assistant());
            if step.tool_calls.is_empty() {
                final_text = step.text.clone();
                println!("    round {rounds}: final answer");
                break;
            }
            for tc in &step.tool_calls {
                let city = tc.input.get("city").and_then(|v| v.as_str()).unwrap_or("");
                let temp = fake.get(city).copied().unwrap_or("20");
                println!("    round {rounds}: {}({}) -> {}C", tc.name, city, temp);
                messages.push(ChatMessage::Tool {
                    tool_call_id: tc.id.clone(),
                    content: json!({ "temp_c": temp }).to_string(),
                    is_error: false,
                    attachments: vec![],
                });
            }
        }
        check!("multi-round chain accepted (no 400)", ok, "a round errored");
        check!(
            "reached a final answer",
            !final_text.trim().is_empty(),
            "no final text"
        );
        check!(
            "final answer identifies Tokyo as warmer",
            final_text.to_lowercase().contains("tokyo"),
            format!("text={:?}", final_text)
        );
    }

    // ── Scenario 5: parallel tool calls ─────────────────────────────────
    println!("\n[5] parallel tool calls (parallel_tool_calls=true)");
    {
        let mut input = ctx.turn(
            vec![user(
                "Call get_weather for BOTH Tokyo and Paris now, in parallel, in a single turn.",
            )],
            vec![get_weather_tool()],
        );
        input.parallel_tool_calls = Some(true);
        match ctx.step(input).await {
            Ok(s) => {
                let cities: Vec<String> = s
                    .tool_calls
                    .iter()
                    .filter_map(|t| t.input.get("city").and_then(|v| v.as_str()))
                    .map(|c| c.to_string())
                    .collect();
                println!("    {} tool call(s): {:?}", s.tool_calls.len(), cities);
                check!("at least one tool call", !s.tool_calls.is_empty(), "none");
                // Not all models emit two in one turn; report but don't hard-fail.
                if s.tool_calls.len() >= 2 {
                    println!("    (emitted {} parallel calls)", s.tool_calls.len());
                }
            }
            Err(e) => check!("scenario completed", false, format!("error: {e}")),
        }
    }

    println!("\n== summary == {pass} passed, {fail} failed");
    if fail > 0 {
        std::process::exit(1);
    }
}
