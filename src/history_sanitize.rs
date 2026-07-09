//! Structural sanitization of a restored message history.
//!
//! A history reloaded from sandbox-persisted `messages.jsonl` (cross-process
//! resume) can be structurally malformed — most often because a weak model's
//! turn was interrupted, or earlier persistence wrote a half-finished turn.
//! Both Anthropic and OpenAI reject a request whose `tool_use` blocks lack a
//! matching `tool_result` (or vice-versa) with a 400, so a single bad resume
//! would wedge every subsequent turn of that session.
//!
//! [`sanitize_history`] enforces the one invariant both providers share:
//! every assistant `tool_call` is followed by exactly one matching tool
//! result, and no tool result is orphaned. It does this with three narrow,
//! provider-agnostic repairs:
//!
//!   1. **Missing tool_call id** — an assistant tool call with a blank id
//!      gets a synthetic stable id so the result can be paired.
//!   2. **Missing tool result** — an expected tool result that isn't present
//!      is synthesized as an error placeholder (the tool never completed, e.g.
//!      the turn was interrupted; the model should see "this didn't run", not
//!      a 400).
//!   3. **Stray tool result** — a tool message with no preceding assistant
//!      tool call to pair with is dropped.
//!
//! It deliberately does NOT touch message *content*, thinking blocks, or
//! anything provider-specific — those are handled at projection time. The
//! returned [`SanitizeDiagnostics`] records what was changed so the caller
//! can log it (and surface it on a subsequent 400).

use crate::model::ChatMessage;

/// Tally of repairs applied by [`sanitize_history`]. All-zero means the
/// history was already well-formed and was returned untouched.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SanitizeDiagnostics {
    /// Assistant tool calls whose blank `id` was replaced with a synthetic one.
    pub repaired_missing_tool_call_id: usize,
    /// Expected tool results that were absent and got an error placeholder.
    pub synthetic_tool_results: usize,
    /// Orphan tool messages (no matching preceding tool call) that were dropped.
    pub dropped_stray_tools: usize,
}

impl SanitizeDiagnostics {
    /// True when no repair was needed.
    pub fn is_clean(&self) -> bool {
        self.repaired_missing_tool_call_id == 0
            && self.synthetic_tool_results == 0
            && self.dropped_stray_tools == 0
    }
}

/// Content of a synthesized placeholder for a tool result that the persisted
/// history is missing (the tool never produced one — typically an interrupted
/// turn). Marked `is_error` so the model treats it as a failure, not data.
const MISSING_TOOL_RESULT_CONTENT: &str = r#"{"error":"tool result missing from restored history (turn likely interrupted)","code":"missing_tool_result_recovered"}"#;

/// Repair structural tool_call/tool_result pairing in a restored history.
/// Returns the cleaned message list plus a tally of what changed. A
/// well-formed history is returned unchanged with all-zero diagnostics.
pub fn sanitize_history(messages: Vec<ChatMessage>) -> (Vec<ChatMessage>, SanitizeDiagnostics) {
    let mut diag = SanitizeDiagnostics::default();
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        match &messages[i] {
            ChatMessage::Tool { .. } => {
                // A tool message reaching the top level was never claimed by a
                // preceding assistant's tool_calls lookahead → orphan. Drop it.
                diag.dropped_stray_tools += 1;
                i += 1;
            }
            ChatMessage::Assistant {
                tool_calls,
                text,
                thinking,
                usage,
            } if !tool_calls.is_empty() => {
                // Repair blank ids, then push the assistant message.
                let mut repaired_calls = tool_calls.clone();
                for (call_idx, call) in repaired_calls.iter_mut().enumerate() {
                    if call.id.trim().is_empty() {
                        call.id = format!("synthetic_call_{i}_{call_idx}");
                        diag.repaired_missing_tool_call_id += 1;
                    }
                }
                let expected_ids: Vec<String> =
                    repaired_calls.iter().map(|c| c.id.clone()).collect();
                out.push(ChatMessage::Assistant {
                    text: text.clone(),
                    tool_calls: repaired_calls,
                    thinking: thinking.clone(),
                    usage: usage.clone(),
                });

                // Pair each expected id with a following tool message (in
                // order). A matching id is consumed as-is; a blank-id tool
                // message is adopted for the expected id; anything else means
                // the result is missing → synthesize a placeholder.
                let mut next = i + 1;
                for id in &expected_ids {
                    if let Some(ChatMessage::Tool {
                        tool_call_id,
                        content,
                        is_error,
                        attachments,
                    }) = messages.get(next)
                    {
                        let tid = tool_call_id.trim();
                        if tid == id {
                            out.push(messages[next].clone());
                            next += 1;
                            continue;
                        }
                        if tid.is_empty() {
                            out.push(ChatMessage::Tool {
                                tool_call_id: id.clone(),
                                content: content.clone(),
                                is_error: *is_error,
                                attachments: attachments.clone(),
                            });
                            diag.repaired_missing_tool_call_id += 1;
                            next += 1;
                            continue;
                        }
                    }
                    out.push(synthetic_tool_result(id));
                    diag.synthetic_tool_results += 1;
                }

                // Any remaining tool messages directly after this block are
                // strays (more results than calls) → drop them.
                while let Some(ChatMessage::Tool { .. }) = messages.get(next) {
                    diag.dropped_stray_tools += 1;
                    next += 1;
                }
                i = next;
            }
            _ => {
                // User, or assistant without tool calls — pass through.
                out.push(messages[i].clone());
                i += 1;
            }
        }
    }
    (out, diag)
}

fn synthetic_tool_result(id: &str) -> ChatMessage {
    ChatMessage::Tool {
        tool_call_id: id.to_string(),
        content: MISSING_TOOL_RESULT_CONTENT.to_string(),
        is_error: true,
        attachments: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolInvocation;

    fn user(s: &str) -> ChatMessage {
        ChatMessage::User {
            content: s.into(),
            attachments: vec![],
        }
    }

    fn assistant_calls(ids: &[&str]) -> ChatMessage {
        ChatMessage::Assistant {
            text: None,
            tool_calls: ids
                .iter()
                .map(|id| ToolInvocation {
                    id: (*id).into(),
                    name: "bash".into(),
                    input: serde_json::json!({}),
                    raw_emitted_args: None,
                })
                .collect(),
            thinking: None,
            usage: None,
        }
    }

    fn tool(id: &str) -> ChatMessage {
        ChatMessage::Tool {
            tool_call_id: id.into(),
            content: "ok".into(),
            is_error: false,
            attachments: vec![],
        }
    }

    fn ids_of(m: &ChatMessage) -> Vec<String> {
        match m {
            ChatMessage::Assistant { tool_calls, .. } => {
                tool_calls.iter().map(|c| c.id.clone()).collect()
            }
            _ => vec![],
        }
    }

    #[test]
    fn well_formed_history_is_untouched() {
        let msgs = vec![
            user("hi"),
            assistant_calls(&["c1"]),
            tool("c1"),
            ChatMessage::Assistant {
                text: Some("done".into()),
                tool_calls: vec![],
                thinking: None,
                usage: None,
            },
        ];
        let (out, diag) = sanitize_history(msgs.clone());
        assert!(diag.is_clean());
        assert_eq!(out, msgs);
    }

    #[test]
    fn synthesizes_missing_tool_result() {
        // Assistant called two tools but only the first result was persisted
        // (turn interrupted before the second ran).
        let msgs = vec![user("hi"), assistant_calls(&["c1", "c2"]), tool("c1")];
        let (out, diag) = sanitize_history(msgs);
        assert_eq!(diag.synthetic_tool_results, 1);
        assert_eq!(diag.dropped_stray_tools, 0);
        // c1 real result kept, c2 synthesized as error placeholder.
        assert_eq!(out.len(), 4);
        match &out[3] {
            ChatMessage::Tool {
                tool_call_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_call_id, "c2");
                assert!(is_error);
            }
            other => panic!("expected synthetic Tool, got {other:?}"),
        }
    }

    #[test]
    fn missing_assistant_means_all_results_synthesized() {
        // Assistant tool_calls is the last message (turn cut before any tool
        // ran) → every expected result is synthesized.
        let msgs = vec![user("hi"), assistant_calls(&["c1", "c2"])];
        let (out, diag) = sanitize_history(msgs);
        assert_eq!(diag.synthetic_tool_results, 2);
        assert_eq!(out.len(), 4); // user + assistant + 2 placeholders
    }

    #[test]
    fn drops_stray_tool_without_preceding_call() {
        let msgs = vec![user("hi"), tool("orphan"), user("again")];
        let (out, diag) = sanitize_history(msgs);
        assert_eq!(diag.dropped_stray_tools, 1);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[1], ChatMessage::User { .. }));
    }

    #[test]
    fn drops_extra_results_beyond_calls() {
        // One call, two tool messages following → second is a stray.
        let msgs = vec![assistant_calls(&["c1"]), tool("c1"), tool("c1")];
        let (out, diag) = sanitize_history(msgs);
        assert_eq!(diag.dropped_stray_tools, 1);
        assert_eq!(diag.synthetic_tool_results, 0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn repairs_blank_tool_call_id() {
        let msgs = vec![assistant_calls(&[""]), tool("")];
        let (out, diag) = sanitize_history(msgs);
        assert_eq!(diag.repaired_missing_tool_call_id, 2); // call id + adopted tool id
        let call_ids = ids_of(&out[0]);
        assert_eq!(call_ids.len(), 1);
        assert!(!call_ids[0].is_empty());
        // The blank-id tool message was adopted for the synthesized call id.
        match &out[1] {
            ChatMessage::Tool { tool_call_id, .. } => assert_eq!(tool_call_id, &call_ids[0]),
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn assistant_without_tool_calls_passes_through_with_thinking() {
        let msg = ChatMessage::Assistant {
            text: Some("final".into()),
            tool_calls: vec![],
            thinking: Some(crate::model::AssistantThinking {
                text: "ponder".into(),
                signature: Some("sig".into()),
            }),
            usage: None,
        };
        let (out, diag) = sanitize_history(vec![msg.clone()]);
        assert!(diag.is_clean());
        assert_eq!(out, vec![msg]);
    }
}
