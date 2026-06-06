#[cfg(feature = "e2b")]
fn main() {
    prost_build::compile_protos(
        &["proto/envd/process/v1/process.proto"],
        &["proto"],
    )
    .expect("prost_build: failed to compile envd process proto");
}

#[cfg(not(feature = "e2b"))]
fn main() {}
