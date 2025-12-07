use std::process::Command;
use std::str;

fn run_lamina(opt: u8) -> String {
    let repo_root = env!("CARGO_MANIFEST_DIR");
    let lamina_bin = format!("{}/target/release/lamina", repo_root);
    let bench_dir = format!("{}/benchmarks/factorial", repo_root);
    let src = format!("{}/factorial.lamina", bench_dir);

    // Build lamina release if needed: recommend running `cargo build --release` before
    let status = Command::new(&lamina_bin)
        .current_dir(repo_root)
        .args([
            "--emit-mir-asm",
            "--verbose",
            &src,
            "--opt-level",
            &opt.to_string(),
        ])
        .status()
        .expect("failed to run lamina");
    assert!(status.success(), "lamina failed at -O{}", opt);

    // Run produced executable
    let exe = format!("{}/factorial", repo_root);
    let out = Command::new(&exe)
        .current_dir(repo_root)
        .output()
        .expect("failed to run produced executable");
    assert!(out.status.success(), "factorial run failed");
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn expected_output() -> &'static str {
    // Baseline expected output for the benchmark (O0)
    // Ensure trailing newlines formatting matches the program output
    "123456789\n3628800\n479001600\n1307674368000\n6402373705728000\n987654321\n"
}

#[test]
fn bench_factorial_outputs_match_all_opt_levels() {
    // O0 baseline
    let o0 = run_lamina(0);
    assert_eq!(o0, expected_output(), "O0 output mismatch");

    // O1 should match baseline
    let o1 = run_lamina(1);
    assert_eq!(o1, expected_output(), "O1 output mismatch");

    // O2 should match baseline
    let o2 = run_lamina(2);
    assert_eq!(o2, expected_output(), "O2 output mismatch");

    // O3 should match baseline
    let o3 = run_lamina(3);
    assert_eq!(o3, expected_output(), "O3 output mismatch");
}
