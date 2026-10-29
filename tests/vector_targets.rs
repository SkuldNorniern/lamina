#![allow(clippy::expect_used)]

//! A vector that no register on the target can hold, and a target with no vector unit,
//! are both refused by name rather than split or scalarized behind the programmer's back.

mod common;

use common::TempDir;
use std::{fs, process::Command};

const PROGRAM: &str = "fn @main() -> i64 {\nentry:\n  %a = splat.<4 x i32> 3\n  %b = add.<4 x i32> %a, %a\n  %x = extractlane.<4 x i32> %b, 2\n  %r = extend.i32.i64 %x\n  ret.i64 %r\n}\n";

const WIDE: &str = "fn @main() -> i64 {\nentry:\n  %a = splat.<4 x i64> 3\n  %b = add.<4 x i64> %a, %a\n  %x = extractlane.<4 x i64> %b, 2\n  ret.i64 %x\n}\n";

fn compile(source: &str, target: &str) -> (bool, String) {
    let guard = TempDir::new("lamina-vector-target");
    let input = guard.path().join("vector.lamina");
    let asm = guard.path().join("vector.s");
    fs::write(&input, source).expect("write program");
    let out = Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&input)
        .args(["--target", target, "--emit-asm", "-o"])
        .arg(&asm)
        .output()
        .expect("run lamina");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn targets_with_a_vector_unit_accept_the_surface() {
    for target in ["x86_64_linux", "aarch64_linux", "wasm32_wasi"] {
        let (ok, text) = compile(PROGRAM, target);
        assert!(ok, "{target} refused a 128-bit vector:\n{text}");
    }
}

#[test]
fn targets_without_a_vector_unit_refuse_by_name() {
    for (target, name) in [
        ("s390x_linux", "s390x vector operations"),
        ("mips64_linux", "mips64 vector operations"),
        ("loongarch64_linux", "LoongArch64 vector operations"),
        (
            "armv7_linux",
            "armv7 backend does not implement vector operations",
        ),
        ("ppc64le_linux", "PowerPC64 vector operations"),
    ] {
        let (ok, text) = compile(PROGRAM, target);
        assert!(!ok, "{target} accepted a vector:\n{text}");
        assert!(text.contains(name), "{target} did not name itself:\n{text}");
    }
}

#[test]
fn a_vector_wider_than_the_target_register_is_refused_by_name() {
    let (ok, text) = compile(WIDE, "x86_64_linux");
    assert!(!ok, "x86_64 accepted a 256-bit vector:\n{text}");
    assert!(text.contains("x86_64 V256 requires AVX"), "{text}");

    let (ok, text) = compile(PROGRAM, "riscv64_linux");
    assert!(!ok, "rv64imafd accepted a vector:\n{text}");
    assert!(
        text.contains("V extension is required for vectors"),
        "{text}"
    );
}
