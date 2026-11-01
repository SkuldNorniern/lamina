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
fn load_and_store_accept_a_stated_alignment() {
    for line in [
        "%r = load.i32 %p align 1",
        "%r = load.i32 %p align 16",
        "store.i32 %p, 1 align 1",
        "volatile store.i32 %p, 1 align 2",
        "%r = volatile load.i32 %p align 4",
    ] {
        accepted(&body(line));
    }
}

#[test]
fn a_stated_alignment_is_printed_back() {
    let source = body("%r = load.i32 %p align 1");
    let text = parse_module(&source).expect("parse").to_string();
    assert!(text.contains("load.i32 %p align 1"), "{text}");
    let source = body("store.i32 %p, 1 align 2");
    let text = parse_module(&source).expect("parse").to_string();
    assert!(text.contains("store.i32 %p, 1 align 2"), "{text}");
}

#[test]
fn a_non_power_of_two_access_alignment_is_rejected() {
    let err = rejected(&body("%r = load.i32 %p align 3"));
    assert!(err.contains("Invalid alignment: align 3"), "{err}");
}

#[test]
fn overflow_flags_parse_for_add_sub_and_mul() {
    for line in [
        "%r = add.ovf.i64 %x, 1",
        "%r = sub.ovf.i64 %x, 1",
        "%r = mul.ovf.i64 %x, 1",
        "%r = add.ovf.u8 1, 2",
    ] {
        accepted(&body(line));
    }
}

#[test]
fn an_overflow_flag_is_printed_back() {
    let source = body("%r = mul.ovf.i64 %x, 3");
    let text = parse_module(&source).expect("parse").to_string();
    assert!(text.contains("%r = mul.ovf.i64 %x, 3"), "{text}");
}

#[test]
fn overflow_flags_are_refused_where_they_have_no_meaning() {
    let err = rejected(&body("%r = div.ovf.i64 %x, 1"));
    assert!(
        err.contains("'.ovf' applies to add, sub and mul only"),
        "{err}"
    );
    let err = rejected(&body("%r = add.ovf.f64 1.0, 1.0"));
    assert!(err.contains("takes an integer type"), "{err}");
    let err = rejected(&body("%r = add.ovf.i128 1, 1"));
    assert!(
        err.contains("128-bit arithmetic is lowered to word pairs"),
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

#[test]
fn thread_local_globals_parse_in_both_models() {
    for annotation in [
        "@thread_local",
        "@thread_local(local_exec)",
        "@thread_local(initial_exec)",
    ] {
        accepted(&format!(
            "{annotation} global $t: i64 = 0\nfn @main() -> i64 {{\nentry:\n    ret.i64 0\n}}\n"
        ));
    }
}

#[test]
fn the_dynamic_tls_models_are_refused_by_name() {
    for model in ["general_dynamic", "local_dynamic"] {
        let err = rejected(&format!(
            "@thread_local({model}) global $t: i64 = 0\nfn @main() -> i64 {{\nentry:\n    ret.i64 0\n}}\n"
        ));
        assert!(err.contains("__tls_get_addr"), "{model}: {err}");
        assert!(err.contains(model), "{model}: {err}");
    }
}

#[test]
fn an_unknown_tls_model_is_named() {
    let err = rejected(
        "@thread_local(global_dynamic) global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n",
    );
    assert!(err.contains("Unknown TLS model 'global_dynamic'"), "{err}");
}

#[test]
fn a_duplicate_thread_local_annotation_is_rejected() {
    let err = rejected(
        "@thread_local @thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n",
    );
    assert!(err.contains("Duplicate annotation: @thread_local"), "{err}");
}

#[test]
fn an_operand_of_another_width_is_named_with_both_types() {
    for (line, found, stated) in [
        (
            "%c32 = add.i32 1, 2\n    %r = shl.i64 1, %c32",
            "i32",
            "i64",
        ),
        (
            "%c32 = add.i32 1, 2\n    %r = add.i64 %c32, 1",
            "i32",
            "i64",
        ),
        ("%c32 = add.i32 1, 2\n    %r = lt.i64 1, %c32", "i32", "i64"),
        (
            "%f = add.f32 1.0, 2.0\n    %r = add.f64 %f, 1.0",
            "f32",
            "f64",
        ),
    ] {
        let err = rejected(&body(line));
        assert!(err.contains("Validation Error"), "{line}: {err}");
        assert!(
            err.contains(&format!("which has type {found}")),
            "{line}: {err}"
        );
        assert!(
            err.contains(&format!("the instruction states {stated}")),
            "{line}: {err}"
        );
    }
}

#[test]
fn signedness_alone_leaves_an_operand_usable() {
    for line in [
        "%u = bitcast.i64.u64 %x\n    %r = add.i64 %u, 1",
        "%u = bitcast.i64.u64 %x\n    %r = lt.u64 %u, 1",
    ] {
        accepted(&body(line));
    }
}

/// Every spelling the parser removed, with the replacement its own error names.
const REMOVED_SPELLINGS: &[&str] = &[
    "zext",
    "sext",
    "sitofp",
    "uitofp",
    "fptosi",
    "fptoui",
    "fpext",
    "fptrunc",
    "heapalloc",
    "alloc",
    "getfield",
    "getelementptr",
    "tuple",
    "extract",
    "phi",
    "writeptr",
    "heapfree",
    "dealloc",
    "print",
    "memcpy",
    "memmove",
    "memset",
];

#[test]
fn unknown_opcode_hints_never_name_a_removed_spelling() {
    let hints = [
        rejected(&body("%r = zzqqxxvv.i64 1, 2")),
        rejected(&body("zzqqxxvv %p, 1")),
    ];
    for hint in &hints {
        for removed in REMOVED_SPELLINGS {
            assert!(
                !hint
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .any(|word| word == *removed),
                "the fallback hint names the removed spelling '{removed}': {hint}"
            );
        }
    }
}

#[test]
fn opcode_suggestion_lists_hold_no_removed_spelling() {
    let lists = [
        super::get_assignment_opcode_names(),
        super::get_non_assignment_opcode_names(),
    ];
    for list in lists {
        for removed in REMOVED_SPELLINGS {
            assert!(
                !list.contains(removed),
                "a suggestion list offers the removed spelling '{removed}'"
            );
        }
    }
}

#[test]
fn a_vector_operand_of_the_wrong_shape_is_rejected() {
    for (line, found, stated) in [
        (
            "%a = splat.<2 x i64> 1\n    %b = splat.<4 x i32> 2\n    %c = add.<2 x i64> %a, %b",
            "<4 x i32>",
            "<2 x i64>",
        ),
        (
            "%a = splat.<2 x i64> 1\n    %c = add.<2 x i64> %a, %x",
            "i64",
            "<2 x i64>",
        ),
        (
            "%a = splat.<4 x i32> 1\n    %b = eq.i32 %a, 1",
            "<4 x i32>",
            "i32",
        ),
        (
            "%a = splat.<2 x i64> 1\n    %b = splat.<4 x i32> 2\n    %c = shuffle.<2 x i64> %a, %b, [0, 1]",
            "<4 x i32>",
            "<2 x i64>",
        ),
        (
            "%a = splat.<4 x i32> 1\n    %b = insertlane.<4 x i32> %a, %x, 0",
            "i64",
            "i32",
        ),
    ] {
        let err = rejected(&body(line));
        assert!(
            err.contains(&format!("which has type {found}")),
            "{line}: {err}"
        );
        assert!(
            err.contains(&format!("the instruction states {stated}")),
            "{line}: {err}"
        );
        assert!(
            !err.contains("lowering produced invalid MIR"),
            "{line}: {err}"
        );
    }
}

#[test]
fn a_named_value_of_the_wrong_type_is_rejected_beyond_binary_and_compare() {
    for (line, side) in [
        ("%r = neg.i32 %x", "the operand"),
        ("store.i32 %p, %x", "the stored value"),
        ("%r = select.i32 %c, %x, 0", "the true value"),
        ("%r = atomic_add.i32 %p, %x, seqcst", "the stored value"),
        (
            "%r = atomic_cmpxchg.i32 %p, %x, 1, seqcst, seqcst",
            "the expected value",
        ),
    ] {
        let err = rejected(&body(line));
        assert!(err.contains(side), "{line}: {err}");
        assert!(err.contains("which has type i64"), "{line}: {err}");
        assert!(err.contains("the instruction states i32"), "{line}: {err}");
    }
}

#[test]
fn a_vector_shift_still_takes_a_scalar_count() {
    accepted(&body(
        "%a = splat.<4 x i32> 1\n    %n = add.i32 2, 0\n    %s = shl.<4 x i32> %a, %n",
    ));
}

#[test]
fn a_float_literal_echoes_as_a_float() {
    for (line, echoed) in [
        ("%r = add.f64 %x, 1.0", "1.0"),
        ("%r = add.f32 %x, 0.1", "0.1"),
        ("%r = add.f64 %x, -0.0", "-0.0"),
        ("%r = add.f32 %x, 1e30", "1000000000000000000000000000000.0"),
        ("%r = lt.f64 %x, 2.0", "2.0"),
    ] {
        let err = rejected(&body(line));
        assert!(err.contains(&format!("%x, {echoed}'")), "{line}: {err}");
    }
}
