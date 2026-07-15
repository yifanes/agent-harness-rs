//! Cross-cutting wrapper applied to every tool runtime.
//!
//! Mirrors the concerns MiMoCode bundles into its `Tool.wrap()`:
//!
//!   1. **Repair** — schema-guided input repair is the single source of
//!      truth here (`repair_invocation`). `agent_loop` calls it BEFORE
//!      pushing history / emitting `ToolCall` events so the recorded args
//!      match what the inner runtime executes; the same call inside
//!      `invoke_cancellable` is then idempotent (already-repaired input
//!      yields `None`) and also covers bypass callers (`runner`, `mcp`).
//!   2. **Validate** — lightweight schema validation; a violation returns a
//!      teaching [`crate::tools::ToolFailure`] (with an `Expected shape` example) WITHOUT
//!      reaching the inner runtime.
//!   3. **Span** — one `tracing` span per invocation.
//!   4. **Safety-net bound** — a success output whose serialized form blows
//!      past a hard ceiling is clipped (error-aware head+tail), covering
//!      tools that don't self-bound their output (MCP, custom plugins).

use async_trait::async_trait;
use serde_json::{json, Value};
use tracing::Instrument;

use crate::tool_repair::{self, ToolInputRepair};
use crate::tools::{
    clip_overflow, invalid_input_failure, ToolInvocation, ToolOutcome, ToolRuntime,
    ToolRuntimeError, ToolSpec, MAX_OUTPUT_BYTES,
};

/// Hard ceiling on a single tool's serialized success output before the
/// catch-all clip fires. Set well above [`MAX_OUTPUT_BYTES`] so that the
/// per-tool field bounding (which already caps stdout/stderr/content near
/// `MAX_OUTPUT_BYTES`) is never clobbered — this only rescues genuinely
/// unbounded outputs from tools that forgot to self-limit.
const CATCH_ALL_CEILING: usize = 4 * MAX_OUTPUT_BYTES;

/// Decorates any [`ToolRuntime`] with repair + validation + tracing + a
/// safety-net output cap. Construct via [`BoundedToolRuntime::new`].
#[derive(Clone)]
pub struct BoundedToolRuntime<R> {
    inner: R,
    /// Specs captured at construction, used only for schema lookup during
    /// repair / validation. A tool's schema is stable for the life of the
    /// runtime, so caching avoids re-running `inner.specs()` on the hot path.
    specs: Vec<ToolSpec>,
}

impl<R: ToolRuntime> BoundedToolRuntime<R> {
    pub fn new(inner: R) -> Self {
        let specs = inner.specs();
        Self { inner, specs }
    }

    /// Borrow the inner runtime (e.g. for downcasting / direct access in
    /// call sites that need the concrete type).
    pub fn inner(&self) -> &R {
        &self.inner
    }

    fn schema_for(&self, name: &str) -> Option<&Value> {
        self.specs
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.input_schema)
    }
}

#[async_trait]
impl<R: ToolRuntime> ToolRuntime for BoundedToolRuntime<R> {
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner.specs()
    }

    /// Schema-guided repair, the single source of truth. Idempotent:
    /// re-running on already-clean input returns `None`.
    fn repair_invocation(&self, inv: &mut ToolInvocation) -> Option<Vec<ToolInputRepair>> {
        let schema = self.schema_for(&inv.name)?;
        let (fixed, repairs) = tool_repair::repair_tool_input_for_spec(schema, &inv.input)?;
        inv.input = fixed;
        Some(repairs)
    }

    async fn invoke(&self, inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
        self.invoke_cancellable(inv, None).await
    }

    async fn invoke_cancellable(
        &self,
        mut inv: ToolInvocation,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<ToolOutcome, ToolRuntimeError> {
        let span = tracing::info_span!("tool.invoke", tool = %inv.name, id = %inv.id);
        async move {
            // 1. Repair (idempotent — a no-op when agent_loop already ran it).
            if let Some(repairs) = self.repair_invocation(&mut inv) {
                tracing::warn!(
                    target: "harness::tool_repair",
                    tool = %inv.name,
                    id = %inv.id,
                    repairs = ?repairs,
                    "schema-guided tool input repair applied"
                );
            }

            // 2. Validate. A violation is a model-observable teaching failure
            //    that never reaches the inner runtime.
            if let Some(schema) = self.schema_for(&inv.name) {
                if let Err(detail) = tool_repair::validate_against_schema(schema, &inv.input) {
                    return Ok(ToolOutcome {
                        output: Err(invalid_input_failure(
                            &inv.name,
                            detail,
                            &inv.input,
                            Some(schema),
                        )),
                        attachments: vec![],
                    });
                }
            }

            // 3. Dispatch to the inner runtime.
            let call_id = inv.id.clone();
            let mut outcome = self.inner.invoke_cancellable(inv, cancel).await?;

            // 4. Safety-net bound for tools that don't self-limit.
            if let Ok(value) = &outcome.output {
                let serialized = value.to_string();
                if serialized.len() > CATCH_ALL_CEILING {
                    tracing::warn!(
                        target: "harness::tool_bound",
                        id = %call_id,
                        bytes = serialized.len(),
                        "tool output exceeded catch-all ceiling; clipping"
                    );
                    outcome.output = Ok(json!({
                        "tool_output_clipped": true,
                        "preview": clip_overflow(&serialized),
                    }));
                }
            }
            Ok(outcome)
        }
        .instrument(span)
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ToolFailure, ToolFailureKind};

    /// Inner runtime returning a fixed oversized success output to exercise
    /// the catch-all clip.
    struct BigOutput;

    #[async_trait]
    impl ToolRuntime for BigOutput {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![ToolSpec {
                name: "big".into(),
                description: "returns a huge blob".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "n": { "type": "integer" } },
                    "required": ["n"],
                    "additionalProperties": false
                }),
            }]
        }

        async fn invoke(&self, _inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
            let blob = "x".repeat(CATCH_ALL_CEILING + 1_000);
            Ok(ToolOutcome {
                output: Ok(json!({ "blob": blob })),
                attachments: vec![],
            })
        }
    }

    fn inv(name: &str, input: Value) -> ToolInvocation {
        ToolInvocation {
            id: "tc_1".into(),
            name: name.into(),
            input,
            raw_emitted_args: None,
        }
    }

    #[tokio::test]
    async fn invalid_input_returns_teaching_failure_with_example() {
        let rt = BoundedToolRuntime::new(BigOutput);
        // Missing required `n`.
        let out = rt.invoke(inv("big", json!({}))).await.unwrap();
        let ToolFailure { kind, message } = out.output.unwrap_err();
        assert_eq!(kind, ToolFailureKind::InvalidInput);
        assert!(message.contains("Expected shape"), "msg: {message}");
        assert!(message.contains("\"n\""), "msg: {message}");
    }

    #[tokio::test]
    async fn oversized_success_output_is_clipped() {
        let rt = BoundedToolRuntime::new(BigOutput);
        let out = rt.invoke(inv("big", json!({ "n": 1 }))).await.unwrap();
        let value = out.output.unwrap();
        assert_eq!(value["tool_output_clipped"], true);
        let preview = value["preview"].as_str().unwrap();
        assert!(preview.len() < CATCH_ALL_CEILING, "preview not clipped");
        assert!(preview.contains("output clipped"));
    }

    #[tokio::test]
    async fn repair_invocation_is_idempotent() {
        let rt = BoundedToolRuntime::new(BigOutput);
        // `n` arrives as a stringified integer — repairable to a number.
        let mut i = inv("big", json!({ "n": "5" }));
        let first = rt.repair_invocation(&mut i);
        assert!(first.is_some(), "expected a repair on first pass");
        assert_eq!(i.input["n"], json!(5));
        // Second pass over the now-clean input is a no-op.
        assert!(rt.repair_invocation(&mut i).is_none());
    }
}
