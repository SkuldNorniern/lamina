#![allow(clippy::expect_used)]

use super::parse_module;

fn body(line: &str) -> String {
    format!("fn @main(ptr %p, bool %c, i64 %x) -> i64 {{\nentry:\n    {line}\n    ret.i64 0\n}}\n")
}

fn rejected(source: &str) -> String {
    parse_module(source)
        .expect_err("source should be rejected")
        .to_string()
}

fn accepted(source: &str) {
    parse_module(source).expect("source should parse");
}

fn with_callee(callee_param: &str, call: &str) -> String {
    format!(
        "fn @g({callee_param}) -> i64 {{\nentry:\n    ret.i64 0\n}}\n\
         fn @main(ptr %p) -> i64 {{\nentry:\n    {call}\n    ret.i64 0\n}}\n"
    )
}

#[test]
fn out_of_range_literal_is_rejected_in_every_typed_position() {
    for line in [
        "store.i8 %p, 300",
        "store.u8 %p, -1",
        "%r = select.i8 %c, 300, 0",
        "%r = select.i8 %c, 0, 300",
        "%r = extend.i8.i64 300",
    ] {
        let err = rejected(&body(line));
        assert!(err.contains("out of range for"), "{line}: {err}");
    }
}

#[test]
fn in_range_literal_is_accepted_in_every_typed_position() {
    accepted(&body("store.i8 %p, 127"));
    accepted(&body("store.u8 %p, 255"));
    accepted(&body("%r = select.i8 %c, -128, 0"));
    accepted(&body("%r = extend.i8.i64 -128"));
}

#[test]
fn typed_call_argument_literal_is_range_checked() {
    let err = rejected(&with_callee("i8 %a", "%r = call @g(i8 300)"));
    assert!(err.contains("out of range for i8"), "{err}");
}

#[test]
fn untyped_call_argument_literal_is_range_checked_against_parameter() {
    let err = rejected(&with_callee("i8 %a", "%r = call @g(300)"));
    assert!(err.contains("out of range for i8"), "{err}");

    let err = rejected(&with_callee(
        "i64 %a",
        "%r = call @g(3000000000000000000000)",
    ));
    assert!(err.contains("Integer literal"), "{err}");

    accepted(&with_callee("i8 %a", "%r = call @g(-128)"));
    accepted(&with_callee("i64 %a", "%r = call @g(300)"));
}

#[test]
fn call_to_undefined_function_is_a_compile_error() {
    let err = rejected(&body("%r = call @nosuch(i64 1)"));
    assert!(err.contains("undefined function @nosuch"), "{err}");
    assert!(err.contains("at line 3"), "{err}");
}

#[test]
fn call_to_declared_extern_is_accepted() {
    accepted(
        "@extern\nfn @nosuch(i64 %a) -> i64\n\
         fn @main() -> i64 {\nentry:\n    %r = call @nosuch(i64 1)\n    ret.i64 %r\n}\n",
    );
}

#[test]
fn fence_without_ordering_does_not_read_the_next_line() {
    let err = rejected(&body("fence"));
    assert!(err.contains("fence needs a memory ordering"), "{err}");
    assert!(err.contains("at line 3"), "{err}");
}

#[test]
fn fence_with_ordering_is_accepted() {
    accepted(&body("fence seqcst"));
    accepted(&body("fence acqrel # barrier"));
}

#[test]
fn write_without_result_name_does_not_read_the_next_line() {
    let err = rejected(&body("write %p, 8"));
    assert!(err.contains("write needs a result name"), "{err}");
    assert!(err.contains("at line 3"), "{err}");
}

#[test]
fn writebyte_and_readbyte_without_result_name_do_not_read_the_next_line() {
    let err = rejected(&body("writebyte 10"));
    assert!(err.contains("writebyte needs a result name"), "{err}");
    assert!(err.contains("at line 3"), "{err}");

    let err = rejected(&body("readbyte"));
    assert!(err.contains("readbyte needs a result name"), "{err}");
    assert!(err.contains("at line 3"), "{err}");
}

#[test]
fn integer_literal_is_rejected_in_every_bool_position() {
    for line in [
        "store.bool %p, 1",
        "%r = add.bool 1, 0",
        "%r = select.bool %c, 1, false",
        "%r = eq.bool %c, 0",
    ] {
        let err = rejected(&body(line));
        assert!(err.contains("bool type"), "{line}: {err}");
    }
    let err = rejected(&with_callee("bool %a", "%r = call @g(1)"));
    assert!(err.contains("bool parameter"), "{err}");

    accepted(&body("store.bool %p, true"));
    accepted(&with_callee("bool %a", "%r = call @g(true)"));
}

#[test]
fn gpu_intrinsic_calls_are_not_undefined_functions() {
    accepted(&body("%r = call @__lamina_gpu_lane_id()"));
}
