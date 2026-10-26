use super::*;
use crate::mir::{
    FunctionBuilder,
    instruction::{Instruction, IntBinOp, Operand},
};
use std::error::Error;

fn make_module_with_empty_func() -> MirModule {
    let mut m = MirModule::new("test_ppc64");
    let f = FunctionBuilder::new("test_fn")
        .returns(MirType::Scalar(ScalarType::I64))
        .block("entry")
        .instr(Instruction::Ret { value: None })
        .build();
    m.add_function(f);
    m
}

/// Replays the constant loads that build the frame size and the first address built
/// from `r1`. Returns the frame size and that address, both as offsets from `r1`.
fn ppc64_frame_arithmetic(asm: &str) -> (i64, Option<i64>) {
    let mut regs: HashMap<&str, i64> = HashMap::new();
    let (mut frame, mut address) = (0i64, None);
    for line in asm.lines().map(str::trim) {
        let (op, args) = line.split_once(' ').unwrap_or((line, ""));
        let parts: Vec<&str> = args.split(", ").collect();
        match (op, parts.as_slice()) {
            ("li", [r, v]) => {
                regs.insert(r, v.parse().unwrap_or(0));
            }
            ("lis", [r, v]) => {
                regs.insert(r, v.parse::<i64>().unwrap_or(0) << 16);
            }
            ("ori", [r, _, v]) => {
                *regs.entry(r).or_insert(0) |= v.parse::<i64>().unwrap_or(0);
            }
            ("addis", [r, base, v]) => {
                let base = if *base == "0" {
                    0
                } else {
                    regs.get(base).copied().unwrap_or(0)
                };
                regs.insert(r, base + (v.parse::<i64>().unwrap_or(0) << 16));
            }
            ("stdux", _) => frame = -regs.get("12").copied().unwrap_or(0),
            ("stdu", [_, v]) => {
                frame = -v.trim_end_matches("(1)").parse::<i64>().unwrap_or(0);
            }
            ("addi", ["3", "1", v]) => {
                let v = v.parse().unwrap_or(0);
                regs.insert("3", v);
                address.get_or_insert(v);
            }
            ("add", ["3", "1", "3"]) => {
                address.get_or_insert(regs.get("3").copied().unwrap_or(0));
            }
            _ => {}
        }
    }
    (frame, address)
}

#[test]
fn lea_offset_beyond_the_immediate_range_is_added_through_r0() -> Result<(), Box<dyn Error>> {
    let slot = VirtualReg::gpr(1);
    let base = Register::Virtual(VirtualReg::gpr(0));
    let mut function = FunctionBuilder::new("element")
        .returns(MirType::ptr())
        .block("entry")
        .instr(MirInst::StackAddr {
            dst: base.clone(),
            slot,
        })
        .instr(MirInst::Lea {
            dst: Register::Virtual(VirtualReg::gpr(2)),
            base,
            offset: 39999,
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(2)))),
        })
        .build();
    function.stack_reservations.insert(slot, 4999);
    let mut module = MirModule::new("element");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_ppc64_with_units_and_settings(
        &module,
        &mut output,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains("addi 3, 3, 39999"), "{assembly}");
    assert!(assembly.contains("    add 3, 3, 0\n"), "{assembly}");
    Ok(())
}

#[test]
fn stack_reservation_is_reserved_in_the_prologue() -> Result<(), Box<dyn Error>> {
    let slot = VirtualReg::gpr(1);
    let mut function = FunctionBuilder::new("reserved")
        .returns(MirType::ptr())
        .block("entry")
        .instr(MirInst::StackAddr {
            dst: Register::Virtual(VirtualReg::gpr(0)),
            slot,
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(0)))),
        })
        .build();
    function.stack_reservations.insert(slot, 4999);
    let mut module = MirModule::new("reserved");
    module.add_function(function);
    for regalloc in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_ppc64_with_units_and_settings(
            &module,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )?;
        let assembly = String::from_utf8(output)?;
        let (frame, address) = ppc64_frame_arithmetic(&assembly);
        let address = address.unwrap_or(i64::MIN);
        assert!(
            frame >= address + 40000,
            "{regalloc:?}: frame {frame} {assembly}"
        );
        assert!(address >= 32, "{regalloc:?}: address {address} {assembly}");
    }
    Ok(())
}

#[test]
fn stack_address_reserves_slot_without_register_use() {
    let slot = VirtualReg::gpr(10);
    let address = Register::Virtual(VirtualReg::gpr(0));
    let stack_addr = MirInst::StackAddr {
        dst: address.clone(),
        slot,
    };
    assert!(stack_addr.use_regs().is_empty());
    let mut module = MirModule::new("stack_address");
    module.add_function(
        FunctionBuilder::new("address")
            .returns(MirType::ptr())
            .block("entry")
            .instr(stack_addr)
            .instr(MirInst::Ret {
                value: Some(Operand::Register(address)),
            })
            .build(),
    );
    for regalloc in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_ppc64_with_units_and_settings(
            &module,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )
        .unwrap();
        let assembly = String::from_utf8(output).unwrap();
        assert!(
            assembly.contains("    stdu 1, -48(1)"),
            "{regalloc:?}: {assembly}"
        );
        assert!(
            assembly.contains("    addi 3, 1, 40"),
            "{regalloc:?}: {assembly}"
        );
        assert!(
            !assembly.contains("    ld 3, 40(1)"),
            "{regalloc:?}: {assembly}"
        );
    }
}

#[test]
fn narrow_parameters_are_sign_extended_on_entry() {
    for (ty, extend) in [
        (MirType::Scalar(ScalarType::I8), "extsb 3, 3"),
        (MirType::Scalar(ScalarType::I16), "extsh 3, 3"),
        (MirType::Scalar(ScalarType::I32), "extsw 3, 3"),
    ] {
        let reg = Register::Virtual(VirtualReg::gpr(0));
        let mut module = MirModule::new("narrow_entry");
        module.add_function(
            FunctionBuilder::new("narrow")
                .param(reg.clone(), ty)
                .returns(ty)
                .block("entry")
                .instr(MirInst::Ret {
                    value: Some(Operand::Register(reg)),
                })
                .build(),
        );
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains(&format!("    {extend}\n    std 3, ")), "{s}");
    }
}

#[test]
fn bool_parameters_are_zero_extended_on_entry() {
    let reg = Register::Virtual(VirtualReg::gpr(0));
    let ty = MirType::Scalar(ScalarType::I1);
    let mut module = MirModule::new("flag_entry");
    module.add_function(
        FunctionBuilder::new("flag")
            .param(reg.clone(), ty)
            .returns(ty)
            .block("entry")
            .instr(MirInst::Ret {
                value: Some(Operand::Register(reg)),
            })
            .build(),
    );
    let mut output = Vec::new();
    generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
    let s = String::from_utf8(output).unwrap();
    assert!(s.contains("    clrldi 3, 3, 56\n    std 3, "), "{s}");
}

#[test]
fn float_memory_transfers_use_fpr_bits() {
    for (ty, load, store) in [
        (MirType::Scalar(ScalarType::F32), "lfs", "stfs"),
        (MirType::Scalar(ScalarType::F64), "lfd", "stfd"),
    ] {
        let base = Register::Virtual(VirtualReg::gpr(0));
        let value = Register::Virtual(VirtualReg::fpr(1));
        let mut module = MirModule::new("float_memory");
        module.add_function(
            FunctionBuilder::new("transfer")
                .param(base.clone(), MirType::ptr())
                .block("entry")
                .instr(MirInst::Load {
                    ty,
                    dst: value.clone(),
                    addr: AddressMode::BaseOffset {
                        base: base.clone(),
                        offset: 0,
                        address_space: Default::default(),
                    },
                    attrs: Default::default(),
                })
                .instr(MirInst::Store {
                    ty,
                    src: Operand::Register(value),
                    addr: AddressMode::BaseOffset {
                        base,
                        offset: 8,
                        address_space: Default::default(),
                    },
                    attrs: Default::default(),
                })
                .instr(MirInst::Ret { value: None })
                .build(),
        );
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let assembly = String::from_utf8(output).unwrap();
        let lines: Vec<_> = assembly.lines().collect();
        let start = lines
            .iter()
            .position(|line| line.contains(&format!("{load} 3, 0(5)")))
            .unwrap();
        assert!(
            lines[start + 1].contains(&format!("{store} 3,")),
            "{assembly}"
        );
        assert!(lines[start + 1].ends_with("(1)"), "{assembly}");
        let end = lines
            .iter()
            .position(|line| line.contains(&format!("{store} 3, 8(5)")))
            .unwrap();
        assert!(
            lines[start + 2..end]
                .iter()
                .any(|line| line.contains(&format!("{load} 3,")) && line.ends_with("(1)")),
            "{assembly}"
        );
    }
}

#[test]
fn stack_arguments_sit_in_the_parameter_save_area() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @ten(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j) -> i64 {\nentry:\n  %s = add.i64 %i, %j\n  ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n  %r = call @ten(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "stack_args").expect("lower stack arguments");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // The ninth and tenth parameters sit at 32 + index * 8 of the caller's frame,
    // which `stdu` claims while writing the back chain.
    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    assert!(caller.contains("    stdu 1, -112(1)\n"), "{assembly}");
    assert!(caller.contains("    std 11, 96(1)\n"), "{assembly}");
    assert!(caller.contains("    std 11, 104(1)\n"), "{assembly}");
    assert!(caller.contains("    addi 1, 1, 112\n"), "{assembly}");

    let callee = assembly.split("\nten:\n").nth(1).ok_or("missing ten")?;
    let frame = callee
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("stdu 1, -")?;
            rest.strip_suffix("(1)")?.parse::<i32>().ok()
        })
        .ok_or("missing frame")?;
    assert!(
        callee.contains(&format!("    ld 0, {}(1)\n", frame + 96)),
        "{assembly}"
    );
    assert!(
        callee.contains(&format!("    ld 0, {}(1)\n", frame + 104)),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn byte_io_goes_through_putchar_and_getchar() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %w = writebyte 72\n  %r = readbyte\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "byte_io").expect("lower byte io");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    bl putchar\n"), "{assembly}");
    assert!(assembly.contains("    bl getchar\n"), "{assembly}");
    assert!(!assembly.contains("bl writebyte"), "{assembly}");
    assert!(!assembly.contains("bl readbyte"), "{assembly}");
    Ok(())
}

#[test]
fn writeptr_sends_the_pointee_bytes_through_write() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i32 {\nentry:\n  %p = alloc.stack i32\n  store.i32 %p, 72\n  %n = writeptr %p\n  ret.i32 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "writeptr").expect("lower writeptr");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    li 5, 4\n"), "{assembly}");
    assert!(assembly.contains("    li 3, 1\n"), "{assembly}");
    assert!(assembly.contains("    bl write\n"), "{assembly}");
    Ok(())
}

#[test]
fn floats_cross_the_boundary_in_f_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @add(i64 %n, f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  %w = sitofp.i64.f64 %n\n  %t = add.f64 %s, %w\n  ret.f64 %t\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(7, 1.5, 2.5)\n  %i = fptosi.f64.i64 %r\n  ret.i64 %i\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "floats").expect("lower floats");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // The integer takes r3 by its parameter position, while the two floats take f1
    // and f2 and leave r4 and r5 unused.
    let callee = assembly.split("\nadd:\n").nth(1).ok_or("missing add")?;
    let prologue = callee
        .split(".L_add.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(prologue.contains("    std 3, "), "{assembly}");
    assert!(prologue.contains("    stfd 1, "), "{assembly}");
    assert!(prologue.contains("    stfd 2, "), "{assembly}");
    assert!(!prologue.contains("    std 4, "), "{assembly}");
    assert!(callee.contains("    lfd 1, "), "{assembly}");

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("bl add").next().ok_or("missing call")?;
    assert!(call.contains("    lfd 1, "), "{assembly}");
    assert!(call.contains("    lfd 2, "), "{assembly}");
    Ok(())
}

#[test]
fn conversions_saturate_clamp_and_zero_nan() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %f = add.f64 1.5, 0.0\n  %wide = fptosi.f64.i64 %f\n  %narrow = fptosi.f64.i8 %f\n  %unsigned = fptoui.f64.u32 %f\n  %back = sitofp.i64.f64 %wide\n  %from_u = uitofp.u32.f64 %unsigned\n  ret.i64 %wide\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "conversions").expect("lower conversions");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // The unordered bit of a self-comparison selects zero for NaN, and a narrow
    // destination is clamped to its own range before it is extended.
    assert!(assembly.contains("    fcmpu 0, 1, 1\n"), "{assembly}");
    assert!(assembly.contains("    fctidz 1, 1\n"), "{assembly}");
    assert!(assembly.contains("    fctiduz 1, 1\n"), "{assembly}");
    assert!(assembly.contains("    isel 3, 4, 3, 3\n"), "{assembly}");
    assert!(assembly.contains("    cmpld 0, 3, 4\n"), "{assembly}");
    assert!(assembly.contains("    extsb 3, 3\n"), "{assembly}");
    assert!(assembly.contains("    fcfid 1, 1\n"), "{assembly}");
    assert!(assembly.contains("    fcfidu 1, 1\n"), "{assembly}");
    Ok(())
}

#[test]
fn module_globals_are_addressed_through_toc_relocations() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "global @counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 @counter\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "globals").expect("lower global module");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(
        assembly.contains("    addis 3, 2, counter@toc@ha\n"),
        "{assembly}"
    );
    assert!(
        assembly.contains("    addi 3, 3, counter@toc@l\n"),
        "{assembly}"
    );
    assert!(
        assembly.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn test_generate_empty_function() {
    let module = make_module_with_empty_func();
    let mut output = Vec::new();
    generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
    let s = String::from_utf8(output).unwrap();
    assert!(s.contains("test_fn:"), "Expected function label: {s}");
    assert!(s.contains("blr"), "Expected blr in output: {s}");
}

#[test]
fn test_generate_integer_add() {
    let mut m = MirModule::new("test_ppc64_add");
    let v0 = VirtualReg::gpr(0);
    let v2 = VirtualReg::gpr(2);
    let f = FunctionBuilder::new("add_fn")
        .param(Register::Virtual(v0), MirType::Scalar(ScalarType::I64))
        .returns(MirType::Scalar(ScalarType::I64))
        .block("entry")
        .instr(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        })
        .instr(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(v2))),
        })
        .build();
    m.add_function(f);
    let mut output = Vec::new();
    generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
    let s = String::from_utf8(output).unwrap();
    assert!(s.contains("add 3, 3, 4"), "Expected add instruction: {s}");
}

#[test]
fn test_generate_jmp_br() {
    let mut m = MirModule::new("test_ppc64_cf");
    let cond = VirtualReg::gpr(0);
    let f = FunctionBuilder::new("cf_fn")
        .param(Register::Virtual(cond), MirType::Scalar(ScalarType::I1))
        .returns(MirType::Scalar(ScalarType::I64))
        .block("entry")
        .instr(Instruction::Br {
            cond: Register::Virtual(cond),
            true_target: "then".to_string(),
            false_target: "else".to_string(),
        })
        .block("then")
        .instr(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(1))),
        })
        .block("else")
        .instr(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        })
        .build();
    m.add_function(f);
    let mut output = Vec::new();
    generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
    let s = String::from_utf8(output).unwrap();
    assert!(s.contains("bne"), "Expected bne in output: {s}");
    assert!(
        s.contains(".L_cf_fn.then:"),
        "Expected function-qualified .L_cf_fn.then in output: {s}"
    );
    assert!(
        !s.contains("\n.L_then:"),
        "block labels must be qualified by function: {s}"
    );
}

#[test]
fn float_copy_emits_bit_move_not_add() {
    let ty = MirType::Scalar(ScalarType::F64);
    let src = Register::Virtual(VirtualReg::fpr(0));
    let dst = Register::Virtual(VirtualReg::fpr(1));
    let function = FunctionBuilder::new("copy_float")
        .param(src.clone(), ty)
        .returns(ty)
        .block("entry")
        .instr(Instruction::Copy {
            ty,
            dst: dst.clone(),
            src: Operand::Register(src),
        })
        .instr(Instruction::Ret {
            value: Some(Operand::Register(dst)),
        })
        .build();
    let mut module = MirModule::new("copy_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
    let assembly = String::from_utf8(output).unwrap();

    assert!(assembly.contains("ld 3"), "expected bit load: {assembly}");
    assert!(assembly.contains("std 3"), "expected bit store: {assembly}");
    assert!(
        !assembly.contains("fadd"),
        "unexpected float add: {assembly}"
    );
}
