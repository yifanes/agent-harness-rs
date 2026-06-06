use async_trait::async_trait;
use crate::tools::ToolInvocation;

/// Decides whether a tool call may proceed.
///
/// Implement this trait to plug in any approval strategy — interactive UI,
/// policy engine, or a simple always-allow/deny rule.
#[async_trait]
pub trait ApprovalGate: Send + Sync {
    /// Return `true` to allow the tool call, `false` to deny it.
    async fn approve(&self, inv: &ToolInvocation) -> bool;

    /// Whether mutating tools (bash/write/edit) should be included in the
    /// spec list advertised to the model.
    ///
    /// `PlanApproval` returns `false` so the model never even sees mutating
    /// tool schemas. All other built-in gates return `true`.
    fn advertise_mutating_tools(&self) -> bool {
        true
    }
}

/// Allow every tool call immediately — no gate, no prompt.
#[derive(Debug, Clone, Default)]
pub struct YoloApproval;

#[async_trait]
impl ApprovalGate for YoloApproval {
    async fn approve(&self, _inv: &ToolInvocation) -> bool {
        true
    }
}

/// Read-only planning mode: hides and denies all mutating tools.
///
/// The model never receives bash/write/edit schemas so it cannot call them.
/// If one slips through (e.g. via prompt injection), `approve` still denies it.
#[derive(Debug, Clone, Default)]
pub struct PlanApproval;

#[async_trait]
impl ApprovalGate for PlanApproval {
    async fn approve(&self, inv: &ToolInvocation) -> bool {
        is_read_only(&inv.name)
    }

    fn advertise_mutating_tools(&self) -> bool {
        false
    }
}

pub(crate) fn is_read_only(name: &str) -> bool {
    matches!(name, "read" | "glob" | "grep")
}
