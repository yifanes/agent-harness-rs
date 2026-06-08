# Project Memory

## Release Versioning

For this crate, releases follow patch-only increments within the current minor line:

- Current rule: `0.1.N -> 0.1.{N+1}`
- Do not skip patch versions unless the user explicitly asks.
- Before publishing, run the test suite and publish the next patch version.
