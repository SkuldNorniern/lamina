use super::*;
use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
use std::error::Error;

const PROBE_LOOP: &str =
    "1:\n    sub sp, sp, x17\n    str xzr, [sp]\n    cmp sp, x16\n    b.hi 1b\n";

fn assembly_with_frame(slots: &str, os: TargetOperatingSystem) -> Result<String, Box<dyn Error>> {
    let source =
        format!("fn @f() -> i64 {{\nentry:\n    %p = stackalloc.{slots}\n    ret.i64 0\n}}\n");
    let ir = parse_module(&source)?;
    let mir = from_ir_x86_64(&ir, "probe").expect("lower function");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, os, PointerWidth::Bits64)?;
    Ok(String::from_utf8(output)?)
}

#[test]
fn frame_larger_than_a_page_is_probed_on_linux() -> Result<(), Box<dyn Error>> {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::Linux)?;
    assert!(
        asm.contains("    sub x16, sp, x16\n") && asm.contains(PROBE_LOOP),
        "{asm}"
    );
    Ok(())
}

#[test]
fn frame_larger_than_a_page_is_probed_on_macos() -> Result<(), Box<dyn Error>> {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::MacOS)?;
    assert!(asm.contains(PROBE_LOOP), "{asm}");
    Ok(())
}

#[test]
fn small_frame_gets_no_probe() -> Result<(), Box<dyn Error>> {
    let asm = assembly_with_frame("[24 x i8]", TargetOperatingSystem::Linux)?;
    assert!(!asm.contains("b.hi 1b"), "{asm}");
    assert!(!asm.contains("x16"), "{asm}");
    Ok(())
}

#[test]
fn frame_of_exactly_one_page_is_not_probed() -> Result<(), Box<dyn Error>> {
    let asm = assembly_with_frame("[4088 x i8]", TargetOperatingSystem::Linux)?;
    assert!(!asm.contains("b.hi 1b"), "{asm}");
    Ok(())
}

#[test]
fn windows_keeps_its_unprobed_frame() -> Result<(), Box<dyn Error>> {
    let asm = assembly_with_frame("[65536 x i8]", TargetOperatingSystem::Windows)?;
    assert!(!asm.contains("b.hi 1b"), "{asm}");
    Ok(())
}
