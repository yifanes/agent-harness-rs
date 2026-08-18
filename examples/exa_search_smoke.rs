//! Live smoke harness for `ExaSearchProvider` behind `WebSearchToolRuntime`.
//!
//! Set your key and run:
//!   EXA_API_KEY=... cargo run --example exa_search_smoke
//!
//! Goes through the full managed-tool path (spec advertisement, request
//! parse, provider call, result normalization, untrusted-content wrapping),
//! then exercises the graceful-degradation path with a bad key. Not a
//! `#[test]` — it needs network + a key.

use harness::{
    ExaSearchConfig, ExaSearchProvider, ToolInvocation, ToolRuntime, WebSearchToolRuntime,
};
use serde_json::json;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("EXA_API_KEY").expect("set EXA_API_KEY");
    let runtime = WebSearchToolRuntime::from_provider(ExaSearchProvider::new(
        ExaSearchConfig::new(api_key),
    ));

    assert_eq!(runtime.specs()[0].name, "web_search");

    let outcome = runtime
        .invoke(ToolInvocation {
            id: "smoke-1".into(),
            name: "web_search".into(),
            input: json!({"query": "latest Anthropic Claude model", "count": 3}),
            raw_emitted_args: None,
        })
        .await
        .expect("invoke must not runtime-error");
    let output = outcome.output.expect("search should succeed with a valid key");
    println!("provider: {}", output["provider"]);
    println!("count: {}", output["count"]);
    for result in output["results"].as_array().unwrap() {
        println!(
            "- {} | {}",
            result["title"].as_str().unwrap_or("?").chars().take(80).collect::<String>(),
            result["url"]
        );
    }
    assert_eq!(output["provider"], "exa");
    assert!(output["count"].as_u64().unwrap() > 0);

    // Degradation path: a bad key must surface as a tool failure, never as a
    // runtime error that would kill the agent loop.
    let bad = WebSearchToolRuntime::from_provider(ExaSearchProvider::new(
        ExaSearchConfig::new("sk-definitely-invalid"),
    ));
    let outcome = bad
        .invoke(ToolInvocation {
            id: "smoke-2".into(),
            name: "web_search".into(),
            input: json!({"query": "anything"}),
            raw_emitted_args: None,
        })
        .await
        .expect("bad key must still not runtime-error");
    let failure = outcome.output.expect_err("bad key must be a tool failure");
    println!("degradation ok: kind={:?} message={}", failure.kind, failure.message);

    println!("exa_search_smoke: OK");
}
