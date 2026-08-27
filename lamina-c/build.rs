// build.rs — lamina-c
//
// Passes the lamina compiler version to the crate via LAMINA_COMPILER_VERSION
// so lia_compiler_version() can return it without a runtime dep on a non-const fn.
//
// cbindgen 0.27 does not yet recognise #[unsafe(no_mangle)] (Rust 2024 edition),
// so lamina.h is maintained by hand until upstream support lands.

use std::fs;
use std::path::Path;

fn main() {
    // Read lamina's version from its Cargo.toml at build time.
    let manifest = match Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
        Some(parent) => parent.join("Cargo.toml"),
        None => panic!("lamina-c has no parent directory"),
    };

    let content = fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", manifest.display(), e));

    let version = content
        .lines()
        .find(|l| l.starts_with("version"))
        .and_then(|l| l.split('"').nth(1))
        .unwrap_or("unknown");

    println!("cargo:rustc-env=LAMINA_COMPILER_VERSION={}", version);
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
