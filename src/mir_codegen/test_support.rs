use crate::cli::{compile::compile as run_compiler, options::parse_args_from};
use std::{error::Error, fs, path::Path, process::Command};

pub(crate) fn compile(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    flags: &[&str],
) -> Result<(), Box<dyn Error>> {
    let args = [
        "lamina".to_owned(),
        source.as_ref().to_string_lossy().into_owned(),
        "-o".to_owned(),
        output.as_ref().to_string_lossy().into_owned(),
    ]
    .into_iter()
    .chain(flags.iter().map(|flag| (*flag).to_owned()))
    .collect();
    run_compiler(parse_args_from(args)?)
}

/// Whether the host clang accepts `args` and llvm-objdump-18 exists. Hosts without the
/// cross target skip the assembly tests instead of failing them.
pub(crate) fn cross_assembler_ready(dir: &Path, args: &[&str]) -> bool {
    let source = dir.join("lamina-probe.s");
    let object = dir.join("lamina-probe.o");
    if fs::write(&source, "").is_err() {
        return false;
    }
    let assembled = Command::new("clang")
        .args(args)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output();
    if !matches!(&assembled, Ok(out) if out.status.success()) {
        return false;
    }
    matches!(Command::new("llvm-objdump-18").arg("--version").output(), Ok(out) if out.status.success())
}

/// Decimal printing written in Lamina, so a test needs no libc. Append it to a module and
/// call `@print_i64` or `@print_u64`; each writes its digits and a newline.
pub(crate) const PRINT_I64_HELPERS: &str = r#"
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

fn @print_u64(u64 %n) -> void {
entry:
  call @put_u64(%n)
  %nl = writebyte 10
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
"#;
