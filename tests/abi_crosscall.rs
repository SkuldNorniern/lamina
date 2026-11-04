//! Cross-toolchain calls: a C file built by gcc calls a Lamina function and is called
//! back by one. A suite that compiles Lamina and checks Lamina's answer cannot see an
//! ABI disagreement, because both sides of it are ours.
#![allow(clippy::expect_used)]

mod common;

use common::TempDir;
use std::{
    env, fs,
    path::Path,
    process::{Command, Stdio},
};

const LAMINA: &str = r#"
type !P5 = packed struct { a: i8, b: i32 }
type !P10 = packed struct { x: i32, s: i16, y: i32 }
type !P9 = packed struct { a: i64, b: i8 }
type !Plain = struct { a: i32, b: i32 }

@extern
fn @c_p5(!P5 %p) -> i64

@extern
fn @c_p10(!P10 %p) -> i64

@export
fn @take_p5(!P5 %p) -> i64 {
  entry:
    %ap = getfieldptr.!P5 %p, 0
    %bp = getfieldptr.!P5 %p, 1
    %a = load.i8 %ap
    %b = load.i32 %bp
    %aw = extend.i8.i64 %a
    %bw = extend.i32.i64 %b
    %s = add.i64 %aw, %bw
    ret.i64 %s
}

@export
fn @take_p10(!P10 %p) -> i64 {
  entry:
    %xp = getfieldptr.!P10 %p, 0
    %yp = getfieldptr.!P10 %p, 2
    %x = load.i32 %xp
    %y = load.i32 %yp
    %xw = extend.i32.i64 %x
    %yw = extend.i32.i64 %y
    %s = add.i64 %xw, %yw
    ret.i64 %s
}

@export
fn @take_p9(!P9 %p) -> i64 {
  entry:
    %ap = getfieldptr.!P9 %p, 0
    %bp = getfieldptr.!P9 %p, 1
    %a = load.i64 %ap
    %b = load.i8 %bp
    %bw = extend.i8.i64 %b
    %s = add.i64 %a, %bw
    ret.i64 %s
}

@export
fn @take_plain(!Plain %p) -> i64 {
  entry:
    %ap = getfieldptr.!Plain %p, 0
    %bp = getfieldptr.!Plain %p, 1
    %a = load.i32 %ap
    %b = load.i32 %bp
    %aw = extend.i32.i64 %a
    %bw = extend.i32.i64 %b
    %s = add.i64 %aw, %bw
    ret.i64 %s
}

@export
fn @spilled(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, !P5 %p, i64 %g) -> i64 {
  entry:
    %bp = getfieldptr.!P5 %p, 1
    %bv = load.i32 %bp
    %bw = extend.i32.i64 %bv
    %t = add.i64 %a, %bw
    %u = add.i64 %t, %g
    ret.i64 %u
}

@export
fn @make_p5(i32 %v) -> !P5 {
  entry:
    %s = stackalloc.!P5
    %ap = getfieldptr.!P5 %s, 0
    %bp = getfieldptr.!P5 %s, 1
    store.i8 %ap, 3
    store.i32 %bp, %v
    ret.!P5 %s
}

@export
fn @calls_back() -> i64 {
  entry:
    %p = stackalloc.!P5
    %pa = getfieldptr.!P5 %p, 0
    %pb = getfieldptr.!P5 %p, 1
    store.i8 %pa, 7
    store.i32 %pb, 305419896
    %r1 = call @c_p5(!P5 %p)
    %q = stackalloc.!P10
    %qx = getfieldptr.!P10 %q, 0
    %qs = getfieldptr.!P10 %q, 1
    %qy = getfieldptr.!P10 %q, 2
    store.i32 %qx, 1000
    store.i16 %qs, 200
    store.i32 %qy, 30000
    %r2 = call @c_p10(!P10 %q)
    %s = add.i64 %r1, %r2
    ret.i64 %s
}
"#;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdint.h>
struct __attribute__((packed)) P5 { int8_t a; int32_t b; };
struct __attribute__((packed)) P10 { int32_t x; int16_t s; int32_t y; };
struct __attribute__((packed)) P9 { int64_t a; int8_t b; };
struct Plain { int32_t a, b; };

long take_p5(struct P5 p);
long take_p10(struct P10 p);
long take_p9(struct P9 p);
long take_plain(struct Plain p);
long spilled(long a, long b, long c, long d, long e, long f, struct P5 p, long g);
struct P5 make_p5(int32_t v);
long calls_back(void);

long c_p5(struct P5 p) { return (long)p.a + p.b; }
long c_p10(struct P10 p) { return (long)p.x + p.s + p.y; }

static int fails = 0;
static void eq(const char *n, long got, long want) {
    if (got != want) { printf("FAIL %s: got %ld want %ld\n", n, got, want); fails++; }
}
int main(void) {
    struct P5 p5 = {7, 305419896};
    struct P10 p10 = {1000, 200, 30000};
    struct P9 p9 = {123456789, 9};
    struct Plain pl = {5, 6};
    eq("take_p5", take_p5(p5), 7L + 305419896L);
    eq("take_p10", take_p10(p10), 1000L + 30000);
    eq("take_p9", take_p9(p9), 123456789L + 9);
    eq("take_plain", take_plain(pl), 11L);
    eq("spilled", spilled(5, 0, 0, 0, 0, 0, p5, 7), 5L + 305419896L + 7);
    struct P5 made = make_p5(4242);
    eq("make_p5.a", made.a, 3);
    eq("make_p5.b", made.b, 4242);
    eq("calls_back", calls_back(), (7L + 305419896L) + (1000L + 200 + 30000));
    return fails != 0;
}
"#;

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Skip, unless this run is the one that promises the cross toolchain.
///
/// `gcc-15-aarch64-linux-gnu` conflicts with `gcc-multilib` on Ubuntu 26, so the
/// job that tests i686 cannot also host the aarch64 cross compiler. The cross
/// half runs in a job of its own, which sets `LAMINA_REQUIRE_CROSS_TOOLS`.
fn skip(reason: &str) -> bool {
    let required = env::var_os("LAMINA_REQUIRE_CROSS_TOOLS").is_some_and(|value| value == "1");
    assert!(!required, "{reason} (LAMINA_REQUIRE_CROSS_TOOLS=1)");
    eprintln!("skipped: {reason}");
    true
}

fn run(command: &mut Command) {
    let output = command.output().expect("run tool");
    assert!(
        output.status.success(),
        "{command:?}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn cross_call(label: &str, target: &str, cc: &str, extra: &[&str], runner: Option<&str>) {
    let guard = TempDir::new(label);
    let dir: &Path = guard.path();
    let source = dir.join("abi.lamina");
    let asm = dir.join("abi.s");
    let driver = dir.join("driver.c");
    let binary = dir.join("abi");
    fs::write(&source, LAMINA).expect("write lamina");
    fs::write(&driver, DRIVER).expect("write driver");
    run(Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&source)
        .args(["--target", target, "--emit-asm", "-o"])
        .arg(&asm));
    run(Command::new(cc)
        .args(extra)
        .arg("-O1")
        .arg("-o")
        .arg(&binary)
        .arg(&driver)
        .arg(&asm));
    let mut call = match runner {
        Some(runner) => {
            let mut command = Command::new(runner);
            command.arg(&binary);
            command
        }
        None => Command::new(&binary),
    };
    let output = call.output().expect("run cross-toolchain program");
    assert!(
        output.status.success(),
        "{target}:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn x86_64_agrees_with_gcc_on_packed_aggregate_arguments() {
    if cfg!(not(target_arch = "x86_64")) || cfg!(not(target_os = "linux")) {
        eprintln!("skipped: the host does not run x86_64 Linux binaries");
        return;
    }
    if !have("gcc") && skip("gcc is not installed") {
        return;
    }
    cross_call("lamina-crosscall-x86_64", "x86_64_linux", "gcc", &[], None);
}

#[test]
fn aarch64_agrees_with_gcc_on_packed_aggregate_arguments() {
    if !have("aarch64-linux-gnu-gcc") && skip("aarch64-linux-gnu-gcc is not installed") {
        return;
    }
    if !have("qemu-aarch64-static") && skip("qemu-aarch64-static is not installed") {
        return;
    }
    cross_call(
        "lamina-crosscall-aarch64",
        "aarch64_linux",
        "aarch64-linux-gnu-gcc",
        &["-static"],
        Some("qemu-aarch64-static"),
    );
}
