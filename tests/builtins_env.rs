#![allow(clippy::expect_used)]

use std::{env, fs, process, process::Command};

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

fn compile_asm(tag: &str, target: &str, builtins: Option<&str>) -> String {
    let dir = env::temp_dir();
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
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let asm = fs::read_to_string(&output).expect("read asm");
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);
    asm
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
