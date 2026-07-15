# Harness Capability Map

This document describes the current harness boundary: what the model sees, what the runtime executes, and how recoverable failures are returned so the model can continue the turn.

## Turn Loop

- The model streams assistant text, thinking, and tool calls.
- Tool calls are normalized before dispatch, then written to conversation history before execution.
- Tool results are appended after execution. A failed tool call is still a valid tool result when the failure is model-correctable.
- Infrastructure failures stop the turn; model-correctable tool failures do not.

## Model Clients

The crate exposes three streaming model-client families:

- `OpenAiCompatibleModelClient` for Chat Completions-compatible APIs.
- `OpenAiResponsesModelClient` for OpenAI Responses API requests.
- `AnthropicModelClient` for Anthropic Messages API requests.

`OpenAiResponsesModelClient` uses stateless Responses requests:

- request route: `POST {base_url}/responses`
- request body includes `store:false`
- request body includes `include:["reasoning.encrypted_content"]`
- local history is replayed in full by the harness
- encrypted reasoning is packed into `AssistantThinking.signature` and replayed
  as a Responses `reasoning` input item on the next turn

The Responses client projects function tools as flat
`{"type":"function","name":...}` tool definitions. Hosted `web_search` is
projected as `{"type":"web_search"}` and is executed by the provider, not by
the harness runtime.

## Tool Boundary

All built-in tools share one boundary contract:

1. Advertise a JSON Schema to the model.
2. Repair only narrow, schema-safe argument shape mistakes.
3. Execute the tool only after required arguments are present.
4. Return structured success data or a model-visible `ToolFailure`.
5. Keep large raw inputs out of model-visible error messages.

## Input Repair Policy

Safe repairs are allowed only when they are deterministic and do not infer user intent from history.

Allowed examples:

- `filePath` -> `path`
- `oldString` -> `old_string`
- `newString` -> `new_string`
- `replaceAll` -> `replace_all`
- `cmd` -> `command`
- string booleans/integers where the schema requires boolean/integer
- markdown autolink cleanup on path-like fields

Disallowed examples:

- Inferring a missing `path` from a previous `read` or `edit`
- Guessing a write target from content
- Repairing an argument when unknown fields or other unresolved schema issues remain

If required arguments are still missing after safe repairs, the tool is not executed.

## Model-Visible Failures

Recoverable tool failures are returned as tool results, not turn-ending runtime errors. The message should be short, actionable, and bounded.

Current recoverable classes:

- `InvalidInput`: malformed or incomplete model-supplied tool arguments
- `NotFound`: requested file or resource does not exist
- `Denied`: policy rejected the tool call
- `Timeout`: non-bash tools that time out before producing a normal settlement
- `NonZeroExit`: retained for non-shell runtimes and compatibility; local shell now reports non-zero exits as structured command results

`InvalidInput` messages include:

- the tool name
- the validation problem
- received field names
- short per-field summaries, including string length and a small preview

They do not include full large fields such as `content`.

## Shell Tool

The `bash` tool returns structured command status for normal exits, non-zero exits, and local timeouts.

Local runtime behavior:

- prefers `/bin/bash -lc` when available
- falls back to `/bin/sh -lc`
- streams stdout/stderr lines to the event channel
- returns JSON with `command`, `shell`, `stdout`, `stderr`, `exit_code`, and `success`
- returns `timed_out: true` plus `timeout_kind` and `message` when a soft or hard timeout fires

Non-zero exit codes are command results, not harness runtime failures. The model can inspect `success: false`, `exit_code`, stdout, and stderr and decide how to recover.

## Edit Tool

The edit tool is exact-match by default:

- `old_string` must match file content exactly, including whitespace and indentation.
- Multiple matches require `replace_all=true` or more context.
- Not-found and ambiguous-match failures are model-visible `InvalidInput` results.
- JSON-escaped `old_string` can be repaired when it produces a unique match.

## History and Replay

The assistant tool call is recorded before tool execution. This ensures history remains well-formed even if execution fails.

Required invariant:

- every tool result must have a preceding assistant tool call with the same `tool_call_id`

Persistent context mode appends new messages to JSONL as the turn progresses. In-memory mode returns the full final message list in `TurnEnd.final_messages`.

## Operational Failures

The following still stop the turn because they indicate harness/runtime infrastructure trouble rather than a model-correctable tool result:

- tool task panics or join failures
- runtime transport failures not mapped to a tool settlement
- provider stream failures after retry budget is exhausted
- cancellation interrupts
