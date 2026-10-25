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
