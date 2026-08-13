# Changelog

All notable changes to this crate must be documented in this file before each
release. The project uses patch-only version bumps within the current minor
line.

## [0.2.12] - 2026-08-13

### Added

- Added `WebSearchMode::{Off, Auto, Native, Managed}` and deterministic
  native/managed search arbitration. Search now defaults to `Off`.
- Added `HostedCapability`, `CapabilitySupport`, and
  `ModelClient::hosted_capability` so search support is determined by the
  concrete API client and endpoint instead of model-name allowlists.
- Added the pluggable `WebSearchProvider` contract, `WebSearchToolRuntime`, a
  normalized managed-search result contract, and a Brave Search adapter.

### Changed

- Replaced the ambiguous no-argument `AgentLoopHarness::with_web_search()`
  with `with_web_search(WebSearchMode)`.
- Removed direct public hosted-tool configuration from `AgentLoopHarness` and
  removed the Anthropic-specific `max_uses` field from the common
  `HostedTool::WebSearch` representation.
- `ModelClient` implementations must now explicitly implement
  `hosted_capability`; the trait no longer supplies a compatibility default.
- Web-search routing now guarantees that provider-hosted and managed
  `web_search` are never advertised together.

## [0.2.11] - 2026-07-15

### Added

- Added `OpenAiResponsesModelClient` and `OpenAiResponsesConfig` for the
  OpenAI Responses API (`POST /v1/responses`), exported from the crate root.
- Added Responses request projection for typed `input[]` history, flat
  function tools, hosted `web_search`, image inputs, structured tool outputs,
  `max_output_tokens`, reasoning effort, and reasoning summary configuration.
- Added stateless Responses reasoning replay with `store:false`,
  `include:["reasoning.encrypted_content"]`, and encrypted reasoning state
  round-tripped through `AssistantThinking.signature`.
- Added a live `examples/responses_smoke.rs` harness for end-to-end Responses
  API smoke testing with text, tools, and reasoning replay.

### Fixed

- Ensured `ToolChoice::None` suppresses both client-side function tools and
  provider-hosted tools for Responses turns.
- Ensured hosted-only `ToolChoice::Required` sends
  `"tool_choice":"required"` instead of silently degrading to provider default
  auto.
- Avoided emitting empty reasoning-summary separators from structural
  `reasoning_summary_part.added` events; separators are now attached only to
  real summary text deltas.
- Accepted both `response.reasoning_summary_text.delta` and
  `response.reasoning_summary.delta` SSE event names for Responses-compatible
  gateways.

### Documentation

- Documented how to construct and use `OpenAiResponsesModelClient`, including
  the stateless reasoning replay contract and hosted web-search behavior.

## [0.2.10] - 2026-07-09

### Fixed

- Prevented cached prompt-token telemetry from being double-counted in anchored
  compaction estimates.
- Scoped dynamic `models.dev` limits to one trusted provider, defaulting to
  `opencode`, so provider-local model ID collisions do not overwrite context
  windows with unrelated gateway values.

## [0.2.9] - 2026-07-09

### Added

- Added `ToolInvocation::raw_emitted_args`, an optional verbatim copy of
  streamed model-emitted tool-call argument JSON when it matches the parsed
  `input`.
- Added raw tool-argument replay for OpenAI-compatible history projection, so
  preserved argument bytes are used instead of re-serializing the parsed JSON
  when safe.
- Added `CompactionPolicy::new` for injecting custom `CompactionStrategy`
  implementations.
- Added `CompactionPolicy::summarizing` and `DefaultCompactionStrategy` as
  explicit public entry points for the existing default compaction behavior.

### Documentation

- Documented pluggable compaction policy inputs, output invariants, and README
  setup examples.
- Documented when `ToolInvocation::raw_emitted_args` is populated and when it
  is cleared or omitted.

### Tests

- Added coverage for preserving streamed raw tool arguments, using them in
  OpenAI-compatible projection, and ignoring stale raw arguments.
- Added coverage for the public compaction policy constructors.

## [0.2.7] - 2026-06-26

### Added

- Added `AGENT_HARNESS_SHELL_RISK_POLICY=relaxed` to temporarily bypass
  conservative bash read-only classification while preserving hard-deny checks.

## [0.2.6] - 2026-06-17

### Added

- Added `web_fetch` as a built-in read-only tool across mock, local, sandbox,
  and e2b runtimes.
- Added `web_fetch` JSON schema with `url`, `format`, `max_length`, and
  `timeout_ms` inputs.
- Added HTTP/HTTPS validation, redirects, timeout handling, 5 MiB response cap,
  textual content-type checks, simple HTML-to-text/markdown conversion, and
  structured JSON output for `web_fetch`.
- Added hosted-tool abstraction via `HostedTool`.
- Added Anthropic native hosted `web_search` support by projecting
  `HostedTool::WebSearch` to Anthropic's `web_search_20250305` server tool.
- Added `AgentLoopHarness::with_hosted_tools(...)` and
  `AgentLoopHarness::with_web_search()` convenience APIs.
- Added fast-fail behavior when hosted tools are requested through the
  OpenAI-compatible Chat Completions client, because OpenAI web search requires
  a future Responses API client.
- Added silent-stop detection for empty or whitespace-only model turns with no
  tool calls and `stop_reason` of `end_turn` or `max_tokens`.

### Changed

- `PlanApproval` now treats `web_fetch` as read-only, so it remains available
  in read-only planning mode.
- Anthropic streaming now explicitly ignores provider-hosted web-search blocks
  such as `server_tool_use` and `web_search_tool_result` instead of treating
  unknown content blocks as text-like placeholders.
- Leading whitespace-only assistant chunks are buffered until visible text
  arrives, so silent-stop turns do not emit meaningless whitespace chunks.

### Tests

- Added coverage for silent-stop detection on `end_turn` and `max_tokens`.
- Added local HTTP-server tests for `web_fetch` HTML extraction, truncation, and
  invalid URL rejection.
- Added coverage for Anthropic hosted web-search projection and OpenAI-compatible
  hosted-tool rejection.

### Fixed

- Prevented fetched or cached model-limit entries from reducing known offline
  fallback limits; known larger context/output caps now win over stale registry
  data.
