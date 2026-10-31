#![allow(clippy::expect_used)]

mod common;

use common::TempDir;
use std::{fs, process, process::Command};

const HELPERS: &str = r#"
fn @put_u64(u64 %n) -> void {
entry:
  %q = div.u64 %n, 10
  %r = rem.u64 %n, 10
  %d = add.u64 %r, 48
  %z = eq.u64 %q, 0
  br %z, last, more
more:
  call @put_u64(%q)
  jmp last
last:
  %w = writebyte %d
  ret.void
}

fn @print_i64(i64 %n) -> void {
entry:
  %neg = lt.i64 %n, 0
  %m = sub.i64 0, %n
  %s = select.i64 %neg, %m, %n
  %u = bitcast.i64.u64 %s
  br %neg, minus, digits
minus:
  %c = writebyte 45
  jmp digits
digits:
  call @put_u64(%u)
  %nl = writebyte 10
  ret.void
}

fn @show(i128 %v) -> i64 {
entry:
  %lo = trunc.i128.i64 %v
  call @print_i64(%lo)
  %hs = shr.i128 %v, 64
  %hi = trunc.i128.i64 %hs
  call @print_i64(%hi)
  ret.i64 0
}
"#;

const WIDE_DIVISION: &str = r#"
fn @main() -> i32 {
entry:
  %big = shl.i128 1, 100
  %d = add.i128 0, 1000000007
  %q = div.i128 %big, %d
  call @show(%q)
  %r = rem.i128 %big, %d
  call @show(%r)
  %ub0 = shl.i128 1, 127
  %ub = bitcast.i128.u128 %ub0
  %uq = div.u128 %ub, 3
  %uqs = bitcast.u128.i128 %uq
  call @show(%uqs)
  %ur = rem.u128 %ub, 3
  %urs = bitcast.u128.i128 %ur
  call @show(%urs)
  %nb = sub.i128 0, %big
  %sq = div.i128 %nb, %d
  call @show(%sq)
  %sr = rem.i128 %nb, %d
  call @show(%sr)
  ret.i32 0
}
"#;

/// The corners where lamina's own routines and the host compiler runtime are known to
/// part company: lamina saturates and returns zero for NaN, libgcc leaves them undefined.
const FLOAT_CORNERS: &str = r#"
fn @main() -> i32 {
entry:
  %inf_b = add.u64 0, 9218868437227405312
  %inf = bitcast.u64.f64 %inf_b
  %nan_b = add.u64 0, 9221120237041090560
  %nan = bitcast.u64.f64 %nan_b
  %a = fptoint.f64.i128 %nan
  call @show(%a)
  %b = fptoint.f64.i128 %inf
  call @show(%b)
  %c = fptoint.f64.i128 1e40
  call @show(%c)
  %d = fptoint.f64.u128 -5.0
  %ds = bitcast.u128.i128 %d
  call @show(%ds)
  %e = fptoint.f64.u128 %nan
  %es = bitcast.u128.i128 %e
  call @show(%es)
  ret.i32 0
}
"#;

fn outputs(tag: &str, program: &str) -> (String, String) {
    let guard = TempDir::new("lamina-jit-agreement");
    let dir = guard.path();
    let stem = format!("lamina-jit-{}-{tag}", process::id());
    let input = dir.join(format!("{stem}.lamina"));
    let exe = dir.join(&stem);
    fs::write(&input, format!("{program}{HELPERS}")).expect("write program");

    let build = Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&input)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run lamina");
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let aot = Command::new(&exe).output().expect("run compiled program");
    assert!(aot.status.success(), "compiled program failed: {aot:?}");

    let jit = Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&input)
        .arg("--jit")
        .output()
        .expect("run lamina --jit");
    assert!(
        jit.status.success(),
        "{}",
        String::from_utf8_lossy(&jit.stderr)
    );

    (
        String::from_utf8_lossy(&aot.stdout).into_owned(),
        String::from_utf8_lossy(&jit.stdout).into_owned(),
    )
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn jit_and_aot_agree_on_wide_division() {
    let (aot, jit) = outputs("div", WIDE_DIVISION);
    assert_eq!(aot, jit);
    assert!(!aot.trim().is_empty());
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn jit_and_aot_agree_on_float_to_wide_corners() {
    let (aot, jit) = outputs("corners", FLOAT_CORNERS);
    assert_eq!(aot, jit);
    let lines: Vec<&str> = aot.lines().collect();
    assert_eq!(
        lines,
        [
            "0",
            "0",
            "-1",
            "9223372036854775807",
            "-1",
            "9223372036854775807",
            "0",
            "0",
            "0",
            "0"
        ]
    );
}
