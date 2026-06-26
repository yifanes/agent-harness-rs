# Changelog

All notable changes to this crate must be documented in this file before each
release. The project uses patch-only version bumps within the current minor
line.

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
