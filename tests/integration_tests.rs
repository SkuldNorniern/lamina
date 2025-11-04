use std::process::Command;

#[test]
fn test_basic_functionality() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("simple_const.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "simple_const test failed");
}

#[test]
fn test_arithmetic() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("arithmetic.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "arithmetic test failed");
}

#[test]
fn test_variables() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("variables.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "variables test failed");
}

#[test]
fn test_conditionals() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("conditionals.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "conditionals test failed");
}

#[test]
fn test_functions() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("functions.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "functions test failed");
}

#[test]
fn test_complex_arithmetic() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("complex_arithmetic.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "complex_arithmetic test failed");
}

#[test]
fn test_nested_calls() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("nested_calls.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "nested_calls test failed");
}

#[test]
fn test_large_constants() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .arg("large_constants.lamina")
        .output()
        .expect("Failed to execute test runner");

    assert!(output.status.success(), "large_constants test failed");
}

#[test]
fn test_all_lamina_tests() {
    let output = Command::new("python3")
        .arg("run_tests.py")
        .output()
        .expect("Failed to execute test runner");

    if !output.status.success() {
        println!("STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
        println!("STDERR:\n{}", String::from_utf8_lossy(&output.stderr));
        panic!("Some lamina tests failed");
    }
}
