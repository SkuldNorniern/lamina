#![allow(clippy::expect_used)]

use super::parse_module;

fn rejected(line: &str) -> String {
    let source = format!("fn @main(i64 %a) -> i64 {{\nentry:\n    {line}\n    ret.i64 0\n}}\n");
    parse_module(&source)
        .expect_err("asm should be rejected")
        .to_string()
}

#[test]
fn template_operand_index_must_be_covered_by_operands() {
    let message = rejected("%v = asm.i64 \"movq $1, $0\", out(rax)");
    assert!(message.contains("uses $1"), "{message}");
}

#[test]
fn stray_dollar_in_template_is_rejected() {
    let message = rejected("%v = asm.i64 \"movq $x, $0\", out(rax)");
    assert!(message.contains("stray '$'"), "{message}");
}

#[test]
fn operand_and_clobber_on_one_register_is_rejected() {
    let message = rejected("asm.void \"nop\", in(rcx) %a, clobber(rcx)");
    assert!(
        message.contains("both an asm operand and a clobber"),
        "{message}"
    );
}

#[test]
fn operand_and_clobber_match_across_register_widths() {
    let message = rejected("asm.void \"nop\", in(rcx) %a, clobber(ecx)");
    assert!(
        message.contains("both an asm operand and a clobber"),
        "{message}"
    );
}

#[test]
fn clobbering_the_stack_pointer_is_rejected() {
    let message = rejected("asm.void \"nop\", clobber(rsp)");
    assert!(
        message.contains("cannot clobber the stack or frame pointer"),
        "{message}"
    );
}

#[test]
fn clobbering_the_frame_pointer_is_rejected() {
    let message = rejected("asm.void \"nop\", clobber(rbp)");
    assert!(
        message.contains("cannot clobber the stack or frame pointer"),
        "{message}"
    );
}

#[test]
fn frame_pointer_as_operand_is_rejected() {
    let message = rejected("%v = asm.i64 \"nop\", out(rsp)");
    assert!(message.contains("as an asm operand"), "{message}");
}

#[test]
fn void_asm_takes_no_out_operand() {
    let message = rejected("asm.void \"nop\", out(rax)");
    assert!(message.contains("takes no out"), "{message}");
}

#[test]
fn result_asm_needs_exactly_one_out_operand() {
    let missing = rejected("%v = asm.i64 \"nop\", in(rcx) %a");
    assert!(missing.contains("found 0"), "{missing}");
    let two = rejected("%v = asm.i64 \"nop\", out(rax), out(rcx)");
    assert!(two.contains("found 2"), "{two}");
}

#[test]
fn void_asm_cannot_bind_a_result() {
    let message = rejected("%v = asm.void \"nop\"");
    assert!(message.contains("cannot be bound"), "{message}");
}

#[test]
fn float_asm_result_is_rejected() {
    let message = rejected("%v = asm.f64 \"nop\", out(rax)");
    assert!(message.contains("integer or ptr"), "{message}");
}

#[test]
fn operand_location_used_twice_is_rejected() {
    let message = rejected("%v = asm.i64 \"nop\", in(rcx) %a, in(ecx) %a, out(rax)");
    assert!(message.contains("used by two asm operands"), "{message}");
}

#[test]
fn literal_asm_input_is_rejected() {
    let message = rejected("%v = asm.i64 \"nop\", in(rcx) 5, out(rax)");
    assert!(message.contains("must be values"), "{message}");
}

#[test]
fn well_formed_asm_parses() {
    let source = "fn @main(i64 %a) -> i64 {\nentry:\n    %v = asm.i64 \"movq $$7, $0\\nmovq $0, $0\", in(rcx) %a, out(rax), clobber(rdx, memory, flags)\n    asm.void \"nop\", clobber(rbx)\n    ret.i64 %v\n}\n";
    parse_module(source).expect("well-formed asm should parse");
}
