mod common;

fn run_lamina(opt: u8) -> String {
    common::run_lamina_benchmark("2Dmatmul", "2Dmatmul.lamina", "2Dmatmul", opt)
}

fn expected_output() -> &'static str {
    // Baseline expected output for the benchmark (O0)
    // Ensure trailing newlines formatting matches the program output
    "123456789\n256\n256\n256\n5923659543740416\n33554432\n987654323\n"
}

#[test]
fn bench_2dmatmul_outputs_match_all_opt_levels() {
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
