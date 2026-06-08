use async_trait::async_trait;
use tokio::sync::mpsc;

use super::event::{HarnessInternalEvent, NativeHarnessError, NativeTurnInput};
use super::tools::{MockToolRuntime, ToolInvocation, ToolRuntime};

#[async_trait]
pub trait NativeHarness: Send + Sync {
    async fn run_turn(
        &self,
        input: NativeTurnInput,
    ) -> Result<mpsc::Receiver<Result<HarnessInternalEvent, NativeHarnessError>>, NativeHarnessError>;
}

#[derive(Debug, Clone)]
pub struct FakeNativeHarness {
    events: Vec<HarnessInternalEvent>,
}

impl FakeNativeHarness {
    pub fn new(events: Vec<HarnessInternalEvent>) -> Self {
        Self { events }
    }
}

#[async_trait]
impl NativeHarness for FakeNativeHarness {
    async fn run_turn(
        &self,
        _input: NativeTurnInput,
    ) -> Result<mpsc::Receiver<Result<HarnessInternalEvent, NativeHarnessError>>, NativeHarnessError>
    {
        let (tx, rx) = mpsc::channel(self.events.len().max(1));
        for ev in self.events.clone() {
            tx.send(Ok(ev))
                .await
                .map_err(|_| NativeHarnessError::ChannelClosed)?;
        }
        drop(tx);
        Ok(rx)
    }
}

pub struct ToolCapableFakeHarness<R = MockToolRuntime> {
    runtime: R,
}

impl ToolCapableFakeHarness<MockToolRuntime> {
    pub fn mock() -> Self {
        Self {
            runtime: MockToolRuntime::new().with_file("README.md", "mock readme"),
        }
    }
}

impl<R> ToolCapableFakeHarness<R> {
    pub fn new(runtime: R) -> Self {
        Self { runtime }
    }
}

#[async_trait]
impl<R> NativeHarness for ToolCapableFakeHarness<R>
where
    R: ToolRuntime + Send + Sync,
{
    async fn run_turn(
        &self,
        input: NativeTurnInput,
    ) -> Result<mpsc::Receiver<Result<HarnessInternalEvent, NativeHarnessError>>, NativeHarnessError>
    {
        let (tx, rx) = mpsc::channel(8);
        let tool_input = tool_invocation_from_prompt(&input.prompt_text);
        let tool_name = tool_input.name.clone();
        let tool_id = tool_input.id.clone();
        let raw_input = tool_input.input.clone();
        tx.send(Ok(HarnessInternalEvent::AssistantTextChunk {
            msg_id: "msg_native_1".into(),
            delta: format!("native harness executing tool: {tool_name}"),
        }))
        .await
        .map_err(|_| NativeHarnessError::ChannelClosed)?;
        tx.send(Ok(HarnessInternalEvent::ToolCall {
            id: tool_id.clone(),
            name: tool_name,
            input: raw_input,
        }))
        .await
        .map_err(|_| NativeHarnessError::ChannelClosed)?;

        let outcome = self.runtime.invoke(tool_input).await.map_err(|e| {
            NativeHarnessError::Failed(format!("tool runtime invocation failed: {e}"))
        })?;
        tx.send(Ok(HarnessInternalEvent::ToolResult {
            id: tool_id,
            output: outcome.output.map_err(|failure| failure.to_string()),
        }))
        .await
        .map_err(|_| NativeHarnessError::ChannelClosed)?;
        tx.send(Ok(HarnessInternalEvent::TurnEnd {
            stop_reason: "end_turn".into(),
            usage: None,
            // ScriptedNativeRunner is a single-shot stub kept around for
            // smoke tests; multi-turn replay isn't a use case here, so
            // we emit an empty history snapshot.
            final_messages: vec![],
        }))
        .await
        .map_err(|_| NativeHarnessError::ChannelClosed)?;
        drop(tx);
        Ok(rx)
    }
}

fn tool_invocation_from_prompt(prompt: &str) -> ToolInvocation {
    let trimmed = prompt.trim();
    if let Some(path) = trimmed.strip_prefix("read ") {
        ToolInvocation {
            id: "tc_read_1".into(),
            name: "read".into(),
            input: serde_json::json!({"path": path.trim()}),
        }
    } else if let Some(rest) = trimmed.strip_prefix("write ") {
        let (path, content) = rest.split_once(' ').unwrap_or((rest, ""));
        ToolInvocation {
            id: "tc_write_1".into(),
            name: "write".into(),
            input: serde_json::json!({"path": path.trim(), "content": content}),
        }
    } else {
        ToolInvocation {
            id: "tc_bash_1".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": trimmed}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HarnessInternalEvent;

    #[tokio::test]
    async fn tool_capable_fake_harness_emits_tool_lifecycle() {
        let harness = ToolCapableFakeHarness::mock();
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
            HarnessInternalEvent::TurnEnd { .. }
        ));
        assert!(rx.recv().await.is_none());
    }
}
