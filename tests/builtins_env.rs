#![allow(clippy::expect_used)]

mod common;

use common::TempDir;
use std::{fs, path::PathBuf, process, process::Command};

const PROGRAM: &str = r#"
@extern
fn @malloc(i64 %n) -> ptr
@extern
fn @free(ptr %p) -> void
@extern
fn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr
@extern
fn @memmove(ptr %d, ptr %s, i64 %n) -> ptr
@extern
fn @memset(ptr %d, i32 %v, i64 %n) -> ptr

fn @main() -> i32 {
  entry:
    %p = call @malloc(8)
    %src = call @malloc(8)
    call @memcpy(%p, %src, 8)
    call @memmove(%p, %src, 8)
    call @memset(%p, 0, 8)
    %w = writebyte 55
    call @free(%p)
    ret.i32 0
}
"#;

const OVERRIDES: &str = "malloc=k_alloc,dealloc=k_free,memcpy=k_cpy,memmove=k_mv,memset=k_set";

fn run_compile(
    tag: &str,
    target: &str,
    builtins: Option<&str>,
) -> (process::Output, PathBuf, TempDir) {
    let guard = TempDir::new("lamina-builtins");
    let dir = guard.path();
    let stem = format!("lamina-builtins-{}-{tag}", process::id());
    let input = dir.join(format!("{stem}.lamina"));
    let output = dir.join(format!("{stem}.s"));
    fs::write(&input, PROGRAM).expect("write program");

    let mut command = Command::new(env!("CARGO_BIN_EXE_lamina"));
    command
        .arg(&input)
        .args(["--emit-asm", "--target", target, "-o"])
        .arg(&output)
        .env_remove("LAMINA_BUILTINS");
    if let Some(value) = builtins {
        command.env("LAMINA_BUILTINS", value);
    }
    let run = command.output().expect("run lamina");
    (run, output, guard)
}

fn compile_asm(tag: &str, target: &str, builtins: Option<&str>) -> String {
    let (run, output, _guard) = run_compile(tag, target, builtins);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    fs::read_to_string(&output).expect("read asm")
}

fn refusal(tag: &str, builtins: &str) -> String {
    let (run, _output, _guard) = run_compile(tag, "x86_64_linux", Some(builtins));
    assert!(
        !run.status.success(),
        "an override the compiler cannot apply must not compile"
    );
    String::from_utf8_lossy(&run.stderr).into_owned()
}

#[test]
fn default_output_uses_c_runtime_names() {
    let asm = compile_asm("default-linux", "x86_64_linux", None);
    for name in ["malloc", "memcpy", "memmove", "memset", "free"] {
        assert!(asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
    }
    assert!(!asm.contains("k_"), "{asm}");
}

#[test]
fn lamina_builtins_retargets_every_runtime_call_x86_64() {
    let asm = compile_asm("override-linux", "x86_64_linux", Some(OVERRIDES));
    for name in ["k_alloc", "k_free", "k_cpy", "k_mv", "k_set"] {
        assert!(asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
    }
    for name in ["malloc", "memcpy", "memmove", "memset", "free"] {
        assert!(!asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
    }
}

#[test]
fn lamina_builtins_retargets_every_runtime_call_aarch64_macos() {
    let asm = compile_asm("override-aarch64-macos", "aarch64_macos", Some(OVERRIDES));
    for name in ["_k_alloc", "_k_free", "_k_cpy", "_k_mv", "_k_set"] {
        assert!(asm.contains(&format!("{name}\n")), "{name}:\n{asm}");
    }
    assert!(!asm.contains("_malloc"), "{asm}");
}

#[test]
fn an_unknown_builtin_name_is_refused_not_dropped() {
    let message = refusal("unknown-name", "malloc=k_alloc,mallocc=k_alloc");
    assert!(message.contains("mallocc"), "{message}");
    assert!(message.contains("malloc"), "{message}");
}

#[test]
fn a_malformed_builtins_entry_is_refused() {
    let message = refusal("malformed", "malloc");
    assert!(message.contains("name=symbol"), "{message}");

    let empty = refusal("empty-side", "malloc=");
    assert!(empty.contains("empty"), "{empty}");
}

#[test]
fn lamina_builtins_retargets_a_compiler_runtime_libcall() {
    const DIVIDE: &str = r#"
fn @divide(i64 %a, i64 %b) -> i64 {
  entry:
    %q = div.i64 %a, %b
    ret.i64 %q
}
"#;
    let guard = TempDir::new("lamina-builtins-libcall");
    let dir = guard.path();
    let input = dir.join("libcall.lamina");
    let output = dir.join("libcall.s");
    fs::write(&input, DIVIDE).expect("write program");

    let run = Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&input)
        .args(["--emit-asm", "--target", "i686_linux", "-o"])
        .arg(&output)
        .env("LAMINA_BUILTINS", "__divdi3=k_divdi3")
        .output()
        .expect("run lamina");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let asm = fs::read_to_string(&output).expect("read asm");
    assert!(asm.contains("k_divdi3"), "{asm}");
    assert!(!asm.contains("__divdi3"), "{asm}");
}
