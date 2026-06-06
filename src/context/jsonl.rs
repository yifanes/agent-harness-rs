use std::path::Path;
use tokio::io::AsyncWriteExt;

use crate::model::ChatMessage;

/// Load context messages from a JSONL file.
///
/// Each line is a `serde_json`-serialised `ChatMessage`. Lines that fail to
/// parse are skipped with a warning so a single corrupted entry doesn't
/// prevent the session from resuming.
///
/// Returns an empty `Vec` when the file does not exist — callers treat that
/// as the start of a new session.
pub async fn load_context(path: &Path) -> Vec<ChatMessage> {
    let content = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return vec![],
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "context load failed");
            return vec![];
        }
    };
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            match serde_json::from_str::<ChatMessage>(line) {
                Ok(msg) => Some(msg),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed context line");
                    None
                }
            }
        })
        .collect()
}

/// Append new messages to the context JSONL, creating the file and any
/// parent directories if needed.
///
/// This is called incrementally during a turn (after each User/Assistant/Tool
/// message is committed) to provide crash resilience.
pub async fn append_context(path: &Path, messages: &[ChatMessage]) {
    if messages.is_empty() {
        return;
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
    }
    let mut file = match tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(f) => f,
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "context append open failed");
            return;
        }
    };
    for msg in messages {
        match serde_json::to_string(msg) {
            Ok(line) => {
                let _ = file.write_all(line.as_bytes()).await;
                let _ = file.write_all(b"\n").await;
            }
            Err(e) => tracing::warn!(error = %e, "context message serialize failed; skipping"),
        }
    }
}

/// Rewrite the entire context JSONL with the given messages.
///
/// Called after compaction fires, replacing the previous (possibly long)
/// history with a compacted snapshot.
pub async fn rewrite_context(path: &Path, messages: &[ChatMessage]) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
    }
    let mut content = String::with_capacity(messages.len() * 128);
    for msg in messages {
        match serde_json::to_string(msg) {
            Ok(line) => {
                content.push_str(&line);
                content.push('\n');
            }
            Err(e) => tracing::warn!(error = %e, "context message serialize failed; skipping"),
        }
    }
    if let Err(e) = tokio::fs::write(path, &content).await {
        tracing::error!(path = %path.display(), error = %e, "context rewrite failed");
    }
}
