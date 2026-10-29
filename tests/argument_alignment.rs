#![allow(clippy::expect_used)]

mod common;

use common::TempDir;
use std::{fs, process::Command};

const PAIR_THEN_I64: &str = "fn @pair(i64 %a, i128 %b) -> i64 {\nentry:\n  %x = trunc.i128.i64 %b\n  %r = add.i64 %a, %x\n  ret.i64 %r\n}\n";

const I128_THEN_I64: &str = "fn @skip(i64 %a, i128 %b, i64 %c) -> i64 {\nentry:\n  %x = trunc.i128.i64 %b\n  %r = add.i64 %a, %x\n  %r2 = add.i64 %r, %c\n  ret.i64 %r2\n}\n";

fn assemble(target: &str, source: &str) -> String {
    let guard = TempDir::new("lamina-argument-alignment");
    let input = guard.path().join("probe.lamina");
    let output = guard.path().join("probe.s");
    fs::write(&input, source).expect("write program");
    let run = Command::new(env!("CARGO_BIN_EXE_lamina"))
        .arg(&input)
        .args(["--target", target, "--emit-asm", "-o"])
        .arg(&output)
        .output()
        .expect("run lamina");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    fs::read_to_string(&output).expect("read assembly")
}

fn spills(asm: &str, prefix: &str) -> Vec<String> {
    asm.lines()
        .map(str::trim)
        .filter(|line| line.starts_with(prefix))
        .map(str::to_string)
        .collect()
}

#[test]
fn aarch64_i128_takes_even_register_pair() {
    let asm = assemble("aarch64_linux", PAIR_THEN_I64);
    let stores = spills(&asm, "str x");
    assert!(
        stores.iter().any(|l| l.starts_with("str x2, ")),
        "{stores:?}"
    );
    assert!(
        stores.iter().any(|l| l.starts_with("str x3, ")),
        "{stores:?}"
    );
    assert!(
        !stores.iter().any(|l| l.starts_with("str x1, ")),
        "{stores:?}"
    );
}

#[test]
fn aarch64_skipped_register_is_not_backfilled() {
    let asm = assemble("aarch64_linux", I128_THEN_I64);
    let stores = spills(&asm, "str x");
    assert!(
        stores.iter().any(|l| l.starts_with("str x4, ")),
        "{stores:?}"
    );
    assert!(
        !stores.iter().any(|l| l.starts_with("str x1, ")),
        "{stores:?}"
    );
}

#[test]
fn riscv64_i128_stays_in_next_pair_without_skip() {
    let asm = assemble("riscv64_linux", PAIR_THEN_I64);
    let stores = spills(&asm, "sd a");
    assert!(stores.iter().any(|l| l.starts_with("sd a1,")), "{stores:?}");
    assert!(stores.iter().any(|l| l.starts_with("sd a2,")), "{stores:?}");
    assert!(
        !stores.iter().any(|l| l.starts_with("sd a3,")),
        "{stores:?}"
    );
}

#[test]
fn ppc64le_i128_stays_in_next_pair_without_skip() {
    let asm = assemble("ppc64le_linux", PAIR_THEN_I64);
    let stores = spills(&asm, "std ");
    assert!(stores.iter().any(|l| l.starts_with("std 4,")), "{stores:?}");
    assert!(stores.iter().any(|l| l.starts_with("std 5,")), "{stores:?}");
    assert!(
        !stores.iter().any(|l| l.starts_with("std 6,")),
        "{stores:?}"
    );
}
