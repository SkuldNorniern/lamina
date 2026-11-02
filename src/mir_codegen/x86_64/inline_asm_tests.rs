#![allow(clippy::expect_used)]

use super::*;
use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

fn assembly_for(source: &str, os: TargetOperatingSystem) -> Result<String, String> {
    let ir = parse_module(source).expect("parse inline asm");
    let mir = from_ir_x86_64(&ir, "asm").expect("lower inline asm");
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, os).map_err(|error| error.to_string())?;
    Ok(String::from_utf8(output).expect("utf8 assembly"))
}

fn linux(source: &str) -> String {
    assembly_for(source, TargetOperatingSystem::Linux).expect("inline asm should emit")
}

#[test]
fn template_placeholders_expand_to_their_registers() {
    let asm = linux(
        "fn @f(i64 %a) -> i64 {\nentry:\n    %v = asm.i64 \"leaq ($0,$1), $$2 $2\", in(rcx) %a, in(rdx) %a, out(rax)\n    ret.i64 %v\n}\n",
    );
    assert!(asm.contains("leaq (%rcx,%rdx), $2 %rax"), "{asm}");
}

#[test]
fn explicit_32_bit_register_keeps_its_spelling_in_the_template() {
    let asm = linux(
        "fn @f() -> i64 {\nentry:\n    %v = asm.i64 \"movl $$9, $0\", out(eax)\n    ret.i64 %v\n}\n",
    );
    assert!(asm.contains("movl $9, %eax"), "{asm}");
}

#[test]
fn unknown_register_is_rejected_with_the_valid_set() {
    let error = assembly_for(
        "fn @f() -> i64 {\nentry:\n    %v = asm.i64 \"nop\", out(xyz)\n    ret.i64 %v\n}\n",
        TargetOperatingSystem::Linux,
    )
    .expect_err("unknown register must be rejected");
    assert!(error.contains("unknown register 'xyz'"), "{error}");
    assert!(error.contains("valid names are: reg, rax"), "{error}");
}

#[test]
fn inline_asm_is_rejected_off_x86_64_linux_and_windows() {
    let error = assembly_for(
        "fn @f() -> i64 {\nentry:\n    asm.void \"nop\"\n    ret.i64 0\n}\n",
        TargetOperatingSystem::MacOS,
    )
    .expect_err("inline asm must be rejected on macOS");
    assert!(
        error.contains("inline assembly for Linux and Windows only"),
        "{error}"
    );
}

#[test]
fn win64_callee_saved_registers_clobbered_by_inline_asm_are_saved() {
    let asm = assembly_for(
        "fn @f() -> i64 {\nentry:\n    asm.void \"nop\", clobber(rsi, rdi)\n    ret.i64 0\n}\n",
        TargetOperatingSystem::Windows,
    )
    .expect("inline asm should emit on Windows");
    assert!(asm.contains("movq %rsi, "), "{asm}");
    assert!(asm.contains("movq %rdi, "), "{asm}");
}

#[test]
fn clobbered_callee_saved_register_is_saved_in_the_prologue() {
    let asm = linux(
        "fn @f() -> i64 {\nentry:\n    asm.void \"movq $$1, %rbx\", clobber(rbx)\n    ret.i64 0\n}\n",
    );
    assert!(asm.contains("movq %rbx, "), "{asm}");
    assert!(
        asm.contains("movq -") || asm.contains("(%rbp), %rbx"),
        "{asm}"
    );
}

#[test]
fn asm_result_is_read_from_its_output_register() {
    let asm = linux(
        "fn @f() -> i64 {\nentry:\n    %v = asm.i64 \"movq $$42, $0\", out(rcx)\n    ret.i64 %v\n}\n",
    );
    assert!(asm.contains("movq $42, %rcx"), "{asm}");
}
