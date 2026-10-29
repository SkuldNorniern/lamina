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
    assert!(err.contains("a bool takes 'true' or 'false'"), "{err}");

    accepted(&body("store.bool %p, true"));
    accepted(&with_callee("bool %a", "%r = call @g(true)"));
}

#[test]
fn gpu_intrinsic_calls_are_not_undefined_functions() {
    accepted(&body("%r = call @__lamina_gpu_lane_id()"));
}

fn aggregate_module(call: &str) -> String {
    format!(
        "type !Pair = struct {{ a: i64, b: i64 }}\n\
         type !Trio = struct {{ a: i64, b: i64, c: i64 }}\n\
         fn @take_pair(!Pair %p) -> i64 {{\nentry:\n    ret.i64 0\n}}\n\
         fn @take_ptr(ptr %p) -> i64 {{\nentry:\n    ret.i64 0\n}}\n\
         fn @main() -> i64 {{\nentry:\n    \
         %q = stackalloc.!Pair\n    %t = stackalloc.!Trio\n    {call}\n    ret.i64 0\n}}\n"
    )
}

#[test]
fn pointer_passed_where_an_aggregate_is_declared_is_rejected() {
    let err = rejected(&aggregate_module("%r = call @take_pair(%q)"));
    assert!(err.contains("must be !Pair for parameter '%p'"), "{err}");
    assert!(err.contains("%q is a plain ptr"), "{err}");
    assert!(err.contains("write '!Pair %q'"), "{err}");
}

#[test]
fn aggregate_argument_that_names_its_type_is_accepted() {
    accepted(&aggregate_module("%r = call @take_pair(!Pair %q)"));
}

#[test]
fn aggregate_argument_of_the_wrong_type_is_rejected() {
    let err = rejected(&aggregate_module("%r = call @take_pair(!Trio %t)"));
    assert!(
        err.contains("is declared !Trio, but parameter '%p' of '@take_pair' has type !Pair"),
        "{err}"
    );
}

#[test]
fn aggregate_passed_where_a_pointer_is_declared_is_rejected() {
    let err = rejected(&aggregate_module("%r = call @take_ptr(!Pair %q)"));
    assert!(
        err.contains("is declared !Pair, but parameter '%p' of '@take_ptr' has type ptr"),
        "{err}"
    );
}

#[test]
fn narrow_integer_passed_where_a_wider_one_is_declared_is_rejected() {
    let source = "fn @g(i64 %a) -> i64 {\nentry:\n    ret.i64 0\n}\n\
                  fn @main(i32 %n) -> i64 {\nentry:\n    %r = call @g(%n)\n    ret.i64 0\n}\n";
    let err = rejected(source);
    assert!(err.contains("must be i64 for parameter '%a'"), "{err}");
    assert!(err.contains("%n has type i32"), "{err}");
}

#[test]
fn float_passed_where_an_integer_is_declared_is_rejected() {
    let err = rejected(&with_callee("i64 %a", "%r = call @g(1.5)"));
    assert!(err.contains("1.5 is a float literal"), "{err}");

    let source = "fn @g(i64 %a) -> i64 {\nentry:\n    ret.i64 0\n}\n\
                  fn @main(f64 %d) -> i64 {\nentry:\n    %r = call @g(%d)\n    ret.i64 0\n}\n";
    let err = rejected(source);
    assert!(err.contains("%d has type f64"), "{err}");
}

#[test]
fn a_variadic_call_checks_its_fixed_arguments_and_leaves_the_tail_free() {
    let source = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\
                  fn @main(i64 %x) -> i64 {\nentry:\n    %r = call @printf(%x, 1, 2.5)\n    ret.i64 0\n}\n";
    let err = rejected(source);
    assert!(err.contains("must be ptr for parameter '%fmt'"), "{err}");

    accepted(
        "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\
         fn @main(ptr %fmt) -> i64 {\nentry:\n    %r = call @printf(%fmt, 1, 2.5)\n    ret.i64 0\n}\n",
    );
}

#[test]
fn an_indirect_call_checks_each_argument_against_the_type_it_states() {
    let source = "fn @main(i64 %x) -> i64 {\nentry:\n    \
                  %f = funcaddr @main\n    %r = callptr.i64 %f(f64 %x)\n    ret.i64 0\n}\n";
    let err = rejected(source);
    assert!(
        err.contains("argument 1 of the indirect call in '@main'"),
        "{err}"
    );
    assert!(
        err.contains("is declared f64, but %x has type i64"),
        "{err}"
    );

    accepted(
        "fn @main(i64 %x) -> i64 {\nentry:\n    \
         %f = funcaddr @main\n    %r = callptr.i64 %f(i64 %x)\n    ret.i64 0\n}\n",
    );
}

#[test]
fn a_call_with_the_wrong_argument_count_names_the_signature() {
    let err = rejected(&with_callee("i64 %a", "%r = call @g(1, 2)"));
    assert!(
        err.contains("passes 2 arguments, but '@g' declares 1"),
        "{err}"
    );
    assert!(
        err.contains("Match the signature '(i64 %a) -> i64'"),
        "{err}"
    );
}

fn return_module(signature: &str, ret: &str) -> String {
    format!(
        "type !Pair = struct {{ a: i64, b: i64 }}\n\
         fn @make{signature} {{\nentry:\n    %q = stackalloc.!Pair\n    %n = add.i64 1, 2\n    {ret}\n}}\n"
    )
}

#[test]
fn an_aggregate_return_must_name_its_type() {
    let err = rejected(&return_module("() -> !Pair", "ret.ptr %q"));
    assert!(
        err.contains("is written ret.ptr, but '@make' declares !Pair"),
        "{err}"
    );
    assert!(err.contains("write 'ret.!Pair <value>'"), "{err}");

    accepted(&return_module("() -> !Pair", "ret.!Pair %q"));
}

#[test]
fn a_pointer_returned_where_a_scalar_is_declared_is_rejected() {
    let err = rejected(&return_module("() -> i64", "ret.ptr %q"));
    assert!(
        err.contains("is written ret.ptr, but '@make' declares i64"),
        "{err}"
    );
}

#[test]
fn a_returned_value_of_the_wrong_type_is_rejected() {
    let err = rejected(&return_module("() -> ptr", "ret.ptr %n"));
    assert!(err.contains("must be ptr, but %n has type i64"), "{err}");
}

#[test]
fn a_return_resolves_the_name_on_either_side() {
    let source = "type !Pair = struct { a: i64, b: i64 }\n\
                  fn @make() -> !Pair {\nentry:\n    %q = stackalloc.struct { a: i64, b: i64 }\n    \
                  ret.struct { a: i64, b: i64 } %q\n}\n";
    accepted(source);
}

#[test]
fn a_scalar_returned_where_an_aggregate_is_declared_is_rejected() {
    let err = rejected(&return_module("() -> !Pair", "ret.!Pair %n"));
    assert!(err.contains("must be !Pair, but %n has type i64"), "{err}");
}

#[test]
fn a_void_return_from_a_value_returning_function_is_rejected() {
    let err = rejected(&return_module("() -> i64", "ret.void"));
    assert!(
        err.contains("is written ret.void, but '@make' declares i64"),
        "{err}"
    );
}
