#![allow(clippy::expect_used)]

use std::env::consts::EXE_SUFFIX;
use std::path::PathBuf;
use std::process::Command;

pub fn run_lamina_benchmark(
    bench_dir: &str,
    source_file: &str,
    output_stem: &str,
    opt: u8,
) -> String {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lamina_bin = option_env!("CARGO_BIN_EXE_lamina")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repo_root
                .join("target")
                .join("debug")
                .join(format!("lamina{}", EXE_SUFFIX))
        });
    let source = repo_root
        .join("benchmarks")
        .join(bench_dir)
        .join(source_file);
    let opt_level = opt.to_string();

    let status = Command::new(&lamina_bin)
        .current_dir(&repo_root)
        .arg("--verbose")
        .arg(&source)
        .arg("--opt-level")
        .arg(&opt_level)
        .status()
        .expect("failed to run lamina");
    assert!(status.success(), "lamina failed at -O{opt}");

    let output_exe = repo_root.join(format!("{output_stem}{}", EXE_SUFFIX));
    let output = Command::new(&output_exe)
        .current_dir(&repo_root)
        .output()
        .expect("failed to run produced executable");
    assert!(output.status.success(), "{output_stem} run failed");

    String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n")
}
