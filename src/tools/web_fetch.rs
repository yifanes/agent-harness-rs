use futures::StreamExt;
use serde_json::{json, Value};
use std::time::Duration;

use crate::tools::{
    invalid_input_failure, ToolFailure, ToolFailureKind, ToolInvocation, ToolOutcome,
    ToolRuntimeError,
};

const DEFAULT_MAX_LENGTH: usize = 50_000;
const MAX_LENGTH: usize = 200_000;
const MAX_RESPONSE_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS: u64 = 20_000;
const MAX_TIMEOUT_MS: u64 = 60_000;

pub async fn invoke(inv: ToolInvocation) -> Result<ToolOutcome, ToolRuntimeError> {
    match fetch(inv).await {
        Ok(value) => Ok(ToolOutcome {
            output: Ok(value),
            attachments: vec![],
        }),
        Err(failure) => Ok(ToolOutcome {
            output: Err(failure),
            attachments: vec![],
        }),
    }
}

async fn fetch(inv: ToolInvocation) -> Result<Value, ToolFailure> {
    let Some(url) = inv.input.get("url").and_then(Value::as_str) else {
        return Err(invalid(&inv, "missing required string field `url`"));
    };
    let parsed =
        reqwest::Url::parse(url).map_err(|e| invalid(&inv, format!("invalid URL: {e}")))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(invalid(&inv, "URL must use http:// or https://"));
    }

    let format = inv
        .input
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("markdown");
    if !matches!(format, "markdown" | "text" | "html") {
        return Err(invalid(&inv, "format must be one of: markdown, text, html"));
    }
    let max_length = inv
        .input
        .get("max_length")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_MAX_LENGTH)
        .min(MAX_LENGTH);
    let timeout_ms = inv
        .input
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TIMEOUT_MS)
        .min(MAX_TIMEOUT_MS);

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_millis(timeout_ms))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| runtime(format!("create HTTP client: {e}")))?;

    let resp = client
        .get(parsed)
        .header(
            reqwest::header::USER_AGENT,
            "agent-harness-rs/0.1 (+web_fetch)",
        )
        .header(reqwest::header::ACCEPT, accept_header(format))
        .send()
        .await
        .map_err(|e| runtime(format!("fetch URL: {e}")))?;

    let status = resp.status();
    let final_url = resp.url().to_string();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let mime = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if is_image_mime(&mime) {
        return Err(invalid(
            &inv,
            format!("unsupported image content type: {mime}"),
        ));
    }
    if !is_textual_mime(&mime) {
        return Err(invalid(
            &inv,
            format!("unsupported non-text content type: {mime}"),
        ));
    }
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_RESPONSE_BYTES {
            return Err(runtime(format!(
                "response too large: content-length {len} exceeds {MAX_RESPONSE_BYTES} bytes"
            )));
        }
    }

    let body = collect_limited(resp).await?;
    let raw = String::from_utf8_lossy(&body).into_owned();
    let converted = convert_content(&raw, &mime, format);
    let (content, truncated) = truncate_chars(&converted, max_length);

    Ok(json!({
        "url": url,
        "final_url": final_url,
        "status": status.as_u16(),
        "content_type": content_type,
        "format": format,
        "content": content,
        "truncated": truncated,
    }))
}

fn invalid(inv: &ToolInvocation, message: impl AsRef<str>) -> ToolFailure {
    ToolFailure::new(
        ToolFailureKind::InvalidInput,
        invalid_input_failure("web_fetch", message, &inv.input, None).message,
    )
}

fn runtime(message: impl Into<String>) -> ToolFailure {
    ToolFailure::new(ToolFailureKind::Runtime, message.into())
}

async fn collect_limited(resp: reqwest::Response) -> Result<Vec<u8>, ToolFailure> {
    let mut stream = resp.bytes_stream();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| runtime(format!("read response body: {e}")))?;
        if out.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(runtime(format!(
                "response too large: exceeds {MAX_RESPONSE_BYTES} bytes"
            )));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

fn accept_header(format: &str) -> &'static str {
    match format {
        "html" => "text/html, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
        "text" => "text/plain, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
        _ => "text/markdown, text/plain;q=0.9, text/html;q=0.8, */*;q=0.1",
    }
}

fn is_image_mime(mime: &str) -> bool {
    mime.starts_with("image/") && mime != "image/svg+xml"
}

fn is_textual_mime(mime: &str) -> bool {
    mime.is_empty()
        || mime.starts_with("text/")
        || mime == "application/json"
        || mime.ends_with("+json")
        || mime == "application/xml"
        || mime.ends_with("+xml")
        || mime == "application/javascript"
        || mime == "application/x-javascript"
}

fn convert_content(raw: &str, mime: &str, format: &str) -> String {
    if !mime.contains("html") && mime != "image/svg+xml" {
        return raw.to_string();
    }
    match format {
        "html" => raw.to_string(),
        "text" => html_to_text(raw),
        _ => html_to_markdown(raw),
    }
}

fn html_to_markdown(html: &str) -> String {
    // First pass keeps useful line breaks/headings without pulling in a heavy parser.
    let mut s = html.to_string();
    for (from, to) in [
        ("</h1>", "\n\n"),
        ("</h2>", "\n\n"),
        ("</h3>", "\n\n"),
        ("</p>", "\n\n"),
        ("<br>", "\n"),
        ("<br/>", "\n"),
        ("<br />", "\n"),
        ("</li>", "\n"),
    ] {
        s = s.replace(from, to);
    }
    html_to_text(&s)
}

fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag_buf = String::new();
    let mut skip: Option<String> = None;
    let mut entity = String::new();
    let mut in_entity = false;

    for ch in html.chars() {
        if in_entity {
            if ch == ';' {
                out.push_str(decode_entity(&entity).unwrap_or(""));
                entity.clear();
                in_entity = false;
            } else if entity.len() < 16 {
                entity.push(ch);
            } else {
                out.push('&');
                out.push_str(&entity);
                entity.clear();
                in_entity = false;
            }
            continue;
        }
        if in_tag {
            if ch == '>' {
                let tag = tag_buf
                    .trim()
                    .trim_start_matches('/')
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let closing = tag_buf.trim_start().starts_with('/');
                if let Some(skip_tag) = skip.as_deref() {
                    if closing && tag == skip_tag {
                        skip = None;
                    }
                } else if !closing
                    && matches!(
                        tag.as_str(),
                        "script" | "style" | "noscript" | "iframe" | "object" | "embed" | "svg"
                    )
                {
                    skip = Some(tag);
                } else if matches!(
                    tag.as_str(),
                    "p" | "div" | "br" | "li" | "tr" | "h1" | "h2" | "h3"
                ) {
                    out.push('\n');
                }
                tag_buf.clear();
                in_tag = false;
            } else {
                tag_buf.push(ch);
            }
            continue;
        }
        if ch == '<' {
            in_tag = true;
            continue;
        }
        if skip.is_some() {
            continue;
        }
        if ch == '&' {
            in_entity = true;
            continue;
        }
        out.push(ch);
    }
    normalize_ws(&out)
}

fn decode_entity(entity: &str) -> Option<&'static str> {
    match entity {
        "amp" => Some("&"),
        "lt" => Some("<"),
        "gt" => Some(">"),
        "quot" => Some("\""),
        "apos" | "#39" => Some("'"),
        "nbsp" => Some(" "),
        _ => None,
    }
}

fn normalize_ws(s: &str) -> String {
    let mut out = String::new();
    let mut blank_lines = 0usize;
    for line in s.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank_lines += 1;
            if blank_lines <= 1 && !out.is_empty() {
                out.push('\n');
            }
        } else {
            blank_lines = 0;
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

fn truncate_chars(s: &str, max_chars: usize) -> (String, bool) {
    let mut iter = s.chars();
    let content: String = iter.by_ref().take(max_chars).collect();
    let truncated = iter.next().is_some();
    if truncated {
        (
            format!("{content}\n\n[truncated to {max_chars} chars]"),
            true,
        )
    } else {
        (content, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve_once(body: &'static str, content_type: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/page")
    }

    #[tokio::test]
    async fn fetches_html_as_markdown_like_text() {
        let url = serve_once(
            "<html><body><h1>Hello</h1><script>bad()</script><p>World &amp; friends</p></body></html>",
            "text/html; charset=utf-8",
        )
        .await;
        let outcome = invoke(ToolInvocation {
            id: "wf1".into(),
            name: "web_fetch".into(),
            input: json!({"url": url}),
            raw_emitted_args: None,
        })
        .await
        .unwrap();
        let value = outcome.output.unwrap();
        let content = value["content"].as_str().unwrap();
        assert!(content.contains("Hello"));
        assert!(content.contains("World & friends"));
        assert!(!content.contains("bad()"));
        assert_eq!(value["format"], "markdown");
        assert_eq!(value["truncated"], false);
    }

    #[tokio::test]
    async fn truncates_to_max_length() {
        let url = serve_once("abcdef", "text/plain").await;
        let outcome = invoke(ToolInvocation {
            id: "wf2".into(),
            name: "web_fetch".into(),
            input: json!({"url": url, "max_length": 3}),
            raw_emitted_args: None,
        })
        .await
        .unwrap();
        let value = outcome.output.unwrap();
        assert_eq!(value["truncated"], true);
        assert!(value["content"].as_str().unwrap().starts_with("abc"));
    }

    #[tokio::test]
    async fn rejects_non_http_urls() {
        let outcome = invoke(ToolInvocation {
            id: "wf3".into(),
            name: "web_fetch".into(),
            input: json!({"url": "file:///etc/passwd"}),
            raw_emitted_args: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            outcome.output,
            Err(ToolFailure {
                kind: ToolFailureKind::InvalidInput,
                ..
            })
        ));
    }
}
