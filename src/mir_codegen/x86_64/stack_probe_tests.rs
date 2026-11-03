use super::generate_mir_x86_64;
use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
use lamina_platform::TargetOperatingSystem;

fn assembly_with_frame(slots: &str, os: TargetOperatingSystem) -> String {
    let source =
        format!("fn @f() -> i64 {{\nentry:\n    %p = stackalloc.{slots}\n    ret.i64 0\n}}\n");
    let ir = parse_module(&source).expect("parse function");
    let mir = from_ir_x86_64(&ir, "probe").expect("lower function");
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, os).expect("frame function should emit");
    String::from_utf8(output).expect("utf8 assembly")
}

#[test]
fn frame_larger_than_a_page_is_probed_page_by_page_on_linux() {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::Linux);
    let expected = "    leaq -65552(%rsp), %r11\n    subq $16, %rsp\n    orq $0, (%rsp)\n1:\n    subq $4096, %rsp\n    orq $0, (%rsp)\n    cmpq %r11, %rsp\n    ja 1b\n";
    assert!(asm.contains(expected), "{asm}");
    assert!(!asm.contains("__chkstk"), "{asm}");
}

#[test]
fn probed_frame_touches_the_partial_page_before_stepping_a_whole_page() {
    let asm = assembly_with_frame("[4112 x i8]", TargetOperatingSystem::MacOS);
    let expected = "    leaq -4128(%rsp), %r11\n    subq $32, %rsp\n    orq $0, (%rsp)\n1:\n    subq $4096, %rsp\n";
    assert!(asm.contains(expected), "{asm}");
}

#[test]
fn frame_of_exactly_one_page_is_not_probed() {
    let asm = assembly_with_frame("[4088 x i8]", TargetOperatingSystem::Linux);
    assert!(asm.contains("    subq $4096, %rsp\n"), "{asm}");
    assert!(!asm.contains("orq $0, (%rsp)"), "{asm}");
}

#[test]
fn small_frame_gets_no_probe() {
    let asm = assembly_with_frame("[24 x i8]", TargetOperatingSystem::Linux);
    assert!(!asm.contains("orq $0, (%rsp)"), "{asm}");
    assert!(!asm.contains("%r11"), "{asm}");
    assert!(!asm.contains("__chkstk"), "{asm}");
}

#[test]
fn windows_large_frame_probes_without_calling_out() {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::Windows);
    let expected = "    leaq -65552(%rsp), %r11\n    subq $16, %rsp\n    orq $0, (%rsp)\n1:\n    subq $4096, %rsp\n    orq $0, (%rsp)\n    cmpq %r11, %rsp\n    ja 1b\n";
    assert!(asm.contains(expected), "{asm}");
    assert!(!asm.contains("__chkstk"), "{asm}");
}

#[test]
fn windows_small_frame_gets_no_chkstk() {
    let asm = assembly_with_frame("[24 x i8]", TargetOperatingSystem::Windows);
    assert!(!asm.contains("__chkstk"), "{asm}");
}

#[test]
fn operating_systems_without_a_probe_keep_their_frame() {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::FreeBSD);
    assert!(asm.contains("    subq $65552, %rsp\n"), "{asm}");
    assert!(!asm.contains("orq $0, (%rsp)"), "{asm}");
    assert!(!asm.contains("__chkstk"), "{asm}");
}
