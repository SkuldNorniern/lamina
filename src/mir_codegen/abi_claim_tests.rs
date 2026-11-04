#![allow(clippy::expect_used)]

use super::generate_mir_to_target;
use crate::mir::codegen::from_ir_for_target;
use crate::parser::parse_module;
use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};

fn emit(source: &str, arch: TargetArchitecture, os: TargetOperatingSystem) -> Result<(), String> {
    let target = Target::new(arch, os);
    let ir = parse_module(source).expect("parse abi claim source");
    let mir = from_ir_for_target(&ir, "claims", target.data_layout()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    generate_mir_to_target(&mir, &mut out, &target, 1).map_err(|e| e.to_string())
}

fn refusal(source: &str, arch: TargetArchitecture, os: TargetOperatingSystem) -> String {
    emit(source, arch, os).expect_err("expected the stated placement to be refused")
}

const BIG: &str = "type !Big = struct { a: i64, b: i64, c: i64 }\n";
const PAIR: &str = "type !Pair = struct { a: i64, b: i64 }\n";
const WORD: &str = "type !Word = struct { a: i64 }\n";

fn body(signature: &str) -> String {
    format!("{signature} {{\nentry:\nret.i64 0\n}}\n")
}

#[test]
fn sysv_honours_byval_on_an_aggregate_it_copies_to_the_stack() {
    let source = format!("{BIG}{}", body("fn @take(!Big byval(24, 8) %p) -> i64"));
    emit(
        &source,
        TargetArchitecture::X86_64,
        TargetOperatingSystem::Linux,
    )
    .expect("a 24-byte integer aggregate is a by-value stack copy under System V");
}

#[test]
fn sysv_refuses_byval_on_an_aggregate_it_puts_in_registers() {
    let source = format!("{PAIR}{}", body("fn @take(!Pair byval(16, 8) %p) -> i64"));
    assert_eq!(
        refusal(
            &source,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ),
        "Codegen Error: Invalid codegen options: %p of @take states byval, but x86_64_linux passes it in registers"
    );
}

#[test]
fn sysv_honours_sret_on_a_return_it_writes_through_a_hidden_pointer() {
    let source = "type !Quad = struct { a: i64, b: i64, c: i64, d: i64 }\n\
                  fn @make() -> sret !Quad {\nentry:\n%q = stackalloc.!Quad\nret.!Quad %q\n}\n";
    emit(
        source,
        TargetArchitecture::X86_64,
        TargetOperatingSystem::Linux,
    )
    .expect("a 32-byte return travels through a hidden pointer under System V");
}

#[test]
fn sysv_refuses_sret_on_a_return_it_puts_in_registers() {
    let source = format!(
        "{PAIR}fn @make() -> sret !Pair {{\nentry:\n%q = stackalloc.!Pair\nret.!Pair %q\n}}\n"
    );
    assert_eq!(
        refusal(
            &source,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ),
        "Codegen Error: Invalid codegen options: @make states sret on its return, but x86_64_linux returns that type in registers, not through a hidden pointer"
    );
}

#[test]
fn win64_refuses_byval_on_an_aggregate_it_passes_by_reference() {
    let source = format!("{PAIR}{}", body("fn @take(!Pair byval(16, 8) %p) -> i64"));
    assert_eq!(
        refusal(
            &source,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows
        ),
        "Codegen Error: Invalid codegen options: %p of @take states byval, but x86_64_windows passes it in a pointer to a copy the caller owns, in a register"
    );
}

#[test]
fn win64_honours_byval_on_a_register_sized_aggregate_that_spills() {
    let source = format!(
        "{WORD}{}",
        body("fn @take(i64 %a, i64 %b, i64 %c, i64 %d, !Word byval(8, 8) %p) -> i64")
    );
    emit(
        &source,
        TargetArchitecture::X86_64,
        TargetOperatingSystem::Windows,
    )
    .expect("the fifth argument is a by-value stack copy under Win64");
}

#[test]
fn aapcs64_refuses_byval_on_an_aggregate_it_passes_by_reference() {
    let source = format!("{BIG}{}", body("fn @take(!Big byval(24, 8) %p) -> i64"));
    assert_eq!(
        refusal(
            &source,
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux
        ),
        "Codegen Error: Invalid codegen options: %p of @take states byval, but aarch64_linux passes it in a pointer to a copy the caller owns, in a register"
    );
}

#[test]
fn aapcs64_honours_byval_on_an_aggregate_that_runs_out_of_registers() {
    let source = format!(
        "{PAIR}{}",
        body(
            "fn @take(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, \
             !Pair byval(16, 8) %p) -> i64"
        )
    );
    emit(
        &source,
        TargetArchitecture::Aarch64,
        TargetOperatingSystem::Linux,
    )
    .expect("an aggregate past the eight argument registers is a by-value stack copy");
}

#[test]
fn aapcs64_honours_sret_on_a_return_it_writes_through_x8() {
    let source =
        format!("{BIG}fn @make() -> sret !Big {{\nentry:\n%q = stackalloc.!Big\nret.!Big %q\n}}\n");
    emit(
        &source,
        TargetArchitecture::Aarch64,
        TargetOperatingSystem::Linux,
    )
    .expect("a 24-byte return travels through x8 under AAPCS64");
}

#[test]
fn a_backend_that_does_not_check_placement_refuses_the_attribute_by_name() {
    let source = format!("{BIG}{}", body("fn @take(!Big byval(24, 8) %p) -> i64"));
    assert_eq!(
        refusal(
            &source,
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Linux
        ),
        "Codegen Error: Unsupported feature: @take states byval on %p, and the riscv64_linux backend does not check stated argument placement; remove the attribute or build for x86_64 or aarch64"
    );
    let returns =
        format!("{BIG}fn @make() -> sret !Big {{\nentry:\n%q = stackalloc.!Big\nret.!Big %q\n}}\n");
    assert_eq!(
        refusal(
            &returns,
            TargetArchitecture::S390x,
            TargetOperatingSystem::Linux
        ),
        "Codegen Error: Unsupported feature: @make states sret on its return, and the s390x_linux backend does not check stated argument placement; remove the attribute or build for x86_64 or aarch64"
    );
}

/// A stated attribute is a claim Lamina checks, never one it acts on, so accepting it
/// must leave the emitted code exactly as it was.
#[test]
fn an_accepted_claim_does_not_move_the_generated_code() {
    let spilled = format!(
        "{PAIR}{BIG}fn @take(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, \
         !Pair byval(16, 8) %p) -> sret !Big {{\n\
         entry:\n%q = stackalloc.!Big\nret.!Big %q\n}}\n"
    );
    let win64 = format!(
        "{WORD}{BIG}fn @take(i64 %a, i64 %b, i64 %c, i64 %d, !Word byval(8, 8) %p) \
         -> sret !Big {{\n\
         entry:\n%q = stackalloc.!Big\nret.!Big %q\n}}\n"
    );
    for (arch, os, stated) in [
        (
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            &spilled,
        ),
        (
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
            &win64,
        ),
        (
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux,
            &spilled,
        ),
    ] {
        let silent = stated
            .replace(" byval(16, 8)", "")
            .replace(" byval(8, 8)", "")
            .replace("-> sret ", "-> ");
        let target = Target::new(arch, os);
        let mut with = Vec::new();
        let mut without = Vec::new();
        for (source, out) in [
            (stated.as_str(), &mut with),
            (silent.as_str(), &mut without),
        ] {
            let ir = parse_module(source).expect("parse claim comparison");
            let mir = from_ir_for_target(&ir, "claims", target.data_layout()).expect("lower");
            generate_mir_to_target(&mir, out, &target, 1).expect("emit");
        }
        assert_eq!(
            String::from_utf8_lossy(&with),
            String::from_utf8_lossy(&without),
            "{arch}_{os}"
        );
    }
}
