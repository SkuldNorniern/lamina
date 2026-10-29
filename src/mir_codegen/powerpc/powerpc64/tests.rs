use super::*;
use crate::mir::{
    FunctionBuilder,
    instruction::{Instruction, IntBinOp, Operand},
};
use lamina_codegen::powerpc::frame::LOCALS_BASE;
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
        let slot = LOCALS_BASE + 8;
        assert!(
            assembly.contains(&format!("    stdu 1, -{}(1)", slot + 8)),
            "{regalloc:?}: {assembly}"
        );
        assert!(
            assembly.contains(&format!("    addi 3, 1, {slot}")),
            "{regalloc:?}: {assembly}"
        );
        assert!(
            !assembly.contains(&format!("    ld 3, {slot}(1)")),
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
fn byte_io_goes_through_write_and_read_without_stdio() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %w = writebyte 72\n  %r = readbyte\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "byte_io").expect("lower byte io");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    stb 3, 0(1)\n"), "{assembly}");
    assert!(assembly.contains("    bl write\n"), "{assembly}");
    assert!(assembly.contains("    bl read\n"), "{assembly}");
    assert!(assembly.contains("    lbz 4, 0(1)\n"), "{assembly}");
    assert!(!assembly.contains("putchar"), "{assembly}");
    assert!(!assembly.contains("getchar"), "{assembly}");
    assert!(!assembly.contains("bl writebyte"), "{assembly}");
    assert!(!assembly.contains("bl readbyte"), "{assembly}");
    Ok(())
}

#[test]
fn floats_cross_the_boundary_in_f_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @add(i64 %n, f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  %w = inttofp.i64.f64 %n\n  %t = add.f64 %s, %w\n  ret.f64 %t\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(7, 1.5, 2.5)\n  %i = fptoint.f64.i64 %r\n  ret.i64 %i\n}\n";
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
    let ir = "fn @main() -> i64 {\nentry:\n  %f = add.f64 1.5, 0.0\n  %wide = fptoint.f64.i64 %f\n  %narrow = fptoint.f64.i8 %f\n  %unsigned = fptoint.f64.u32 %f\n  %back = inttofp.i64.f64 %wide\n  %from_u = inttofp.u32.f64 %unsigned\n  ret.i64 %wide\n}\n";
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
    let ir = "global $counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 $counter\n  ret.i64 %a\n}\n";
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
    assert!(s.contains("beq 1f"), "Expected beq 1f in output: {s}");
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

#[test]
fn global_allocation_spills_values_that_outlive_a_call() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @id(i64 %x) -> i64 {\nentry:\n  %r = mul.i64 %x, 1\n  ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n  %a = add.i64 7, 0\n  %b = mul.i64 %a, 3\n  %c = call @id(5)\n  %d = add.i64 %b, %c\n  ret.i64 %d\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "across_call").expect("lower call");
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_ppc64_with_units_and_settings(
            &mir,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )?;
        let assembly = String::from_utf8(output)?;
        let body = assembly
            .split("\n.L_main.entry:")
            .nth(1)
            .ok_or("missing main body")?;
        let mut sides = body.split("    bl id");
        let before = sides.next().ok_or("missing call")?;
        let after = sides.next().ok_or("missing call")?;
        let operands = |line: &str| -> Vec<String> {
            line.trim()
                .split_once(' ')
                .map_or_else(Vec::new, |(_, rest)| {
                    rest.split(',').map(|o| o.trim().to_string()).collect()
                })
        };
        let volatile = ["6", "7", "8", "9", "10", "11", "12"];
        let mut live: Vec<String> = Vec::new();
        for line in before.lines() {
            let ops = operands(line);
            if let Some(dst) = ops.first()
                && volatile.contains(&dst.as_str())
                && !live.contains(dst)
            {
                live.push(dst.clone());
            }
        }
        for line in after.lines().take_while(|l| !l.trim().starts_with("blr")) {
            let ops = operands(line);
            for source in ops.iter().skip(1) {
                assert!(
                    !live.contains(source),
                    "{regalloc:?} reads r{source} across the call: {assembly}"
                );
            }
            if let Some(dst) = ops.first() {
                live.retain(|reg| reg != dst);
            }
        }
    }
    Ok(())
}

#[test]
fn variadic_callee_receives_floats_in_both_banks() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nglobal $fmt: [5 x i8] = \"%g\\n\"\n\nfn @sink(ptr %fmt, f64 %x) -> i32 {\nentry:\n  ret.i32 0\n}\n\nfn @main() -> i64 {\nentry:\n  %p = getelem.i8 $fmt, 0\n  %c = call @printf(%p, 2.5)\n  %d = call @sink(%p, 2.5)\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "variadic").expect("lower extern call");
    let mut output = Vec::new();
    generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    let before = |call: &str| -> String {
        assembly
            .split(call)
            .next()
            .unwrap_or_default()
            .lines()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        before("    bl printf").contains("ld 4, -8(1)"),
        "{assembly}"
    );
    assert!(!before("    bl sink").contains("ld 4, -8(1)"), "{assembly}");
    Ok(())
}

fn emit_aggregate_module(ir: &str, name: &str) -> Result<String, Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let module = parse_module(ir)?;
    let mir = from_ir(&module, name).expect("lower aggregates");
    let mut output = Vec::new();
    emit_ppc64_module(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )?;
    Ok(String::from_utf8(output)?)
}

fn function_body<'a>(assembly: &'a str, name: &str) -> Result<&'a str, Box<dyn Error>> {
    Ok(assembly
        .split(&format!("\n{name}:\n"))
        .nth(1)
        .ok_or_else(|| format!("missing {name}"))?)
}

fn store_offset(text: &str, store: &str) -> Result<i64, Box<dyn Error>> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(store))
        .ok_or_else(|| format!("missing `{store}`"))?;
    let offset = line
        .split(", ")
        .nth(1)
        .and_then(|rest| rest.strip_suffix("(1)"))
        .ok_or("store is not r1-relative")?;
    Ok(offset.parse()?)
}

#[test]
fn aggregate_argument_of_two_words_takes_two_registers() -> Result<(), Box<dyn Error>> {
    let ir = "fn @pick(i64 %x, struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  %pb = getfieldptr.struct { a: i64, b: i64 } %p, 1\n  %b = load.i64 %pb\n  ret.i64 %b\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.struct { a: i64, b: i64 }\n  %r = call @pick(1, %q)\n  ret.i64 %r\n}\n";
    let assembly = emit_aggregate_module(ir, "pair")?;

    let caller = function_body(&assembly, "main")?;
    let call = caller.split("bl pick").next().ok_or("missing call")?;
    assert!(call.contains("    ld 4, 0(11)\n"), "{assembly}");
    assert!(call.contains("    ld 5, 8(11)\n"), "{assembly}");
    assert!(!call.contains("    std 4, "), "{assembly}");

    let callee = function_body(&assembly, "pick")?;
    let prologue = callee
        .split(".L_pick.entry:")
        .next()
        .ok_or("missing entry")?;
    let first = store_offset(prologue, "std 4, ")?;
    let second = store_offset(prologue, "std 5, ")?;
    assert_eq!(second, first + 8, "{assembly}");
    Ok(())
}

#[test]
fn aggregate_argument_of_four_words_fills_r3_through_r6() -> Result<(), Box<dyn Error>> {
    let ir = "fn @sum(struct { a: i64, b: i64, c: i64, d: i64 } %p) -> i64 {\nentry:\n  %pd = getfieldptr.struct { a: i64, b: i64, c: i64, d: i64 } %p, 3\n  %d = load.i64 %pd\n  ret.i64 %d\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.struct { a: i64, b: i64, c: i64, d: i64 }\n  %r = call @sum(%q)\n  ret.i64 %r\n}\n";
    let assembly = emit_aggregate_module(ir, "quad")?;

    let caller = function_body(&assembly, "main")?;
    let call = caller.split("bl sum").next().ok_or("missing call")?;
    for (reg, offset) in [("3", 0), ("4", 8), ("5", 16), ("6", 24)] {
        assert!(
            call.contains(&format!("    ld {reg}, {offset}(11)\n")),
            "{assembly}"
        );
    }

    let callee = function_body(&assembly, "sum")?;
    let prologue = callee
        .split(".L_sum.entry:")
        .next()
        .ok_or("missing entry")?;
    let first = store_offset(prologue, "std 3, ")?;
    for (reg, step) in [("4", 1), ("5", 2), ("6", 3)] {
        assert_eq!(
            store_offset(prologue, &format!("std {reg}, "))?,
            first + 8 * step,
            "{assembly}"
        );
    }
    Ok(())
}

#[test]
fn aggregate_split_between_last_register_and_save_area() -> Result<(), Box<dyn Error>> {
    let ir = "fn @wide(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, struct { a: i64, b: i64 } %p, i64 %z) -> i64 {\nentry:\n  %pb = getfieldptr.struct { a: i64, b: i64 } %p, 1\n  %b = load.i64 %pb\n  %s = add.i64 %b, %z\n  ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.struct { a: i64, b: i64 }\n  %r = call @wide(1, 2, 3, 4, 5, 6, 7, %q, 8)\n  ret.i64 %r\n}\n";
    let assembly = emit_aggregate_module(ir, "split_pair")?;

    let caller = function_body(&assembly, "main")?;
    let call = caller.split("bl wide").next().ok_or("missing call")?;
    assert!(call.contains("    ld 10, 0(11)\n"), "{assembly}");
    assert!(
        call.contains("    ld 12, 8(11)\n    std 12, 96(1)\n"),
        "{assembly}"
    );
    assert!(call.contains("    std 11, 104(1)\n"), "{assembly}");
    assert!(call.contains("    stdu 1, -112(1)\n"), "{assembly}");

    let callee = function_body(&assembly, "wide")?;
    let prologue = callee
        .split(".L_wide.entry:")
        .next()
        .ok_or("missing entry")?;
    let frame = ppc64_frame_arithmetic(prologue).0;
    let low = store_offset(prologue, "std 10, ")?;
    assert!(
        prologue.contains(&format!(
            "    ld 0, {}(1)\n    std 0, {}(1)\n",
            frame + 96,
            low + 8
        )),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn aggregate_result_past_two_words_uses_hidden_pointer() -> Result<(), Box<dyn Error>> {
    let ir = "fn @make() -> struct { a: i64, b: i64, c: i64, d: i64 } {\nentry:\n  %p = stackalloc.struct { a: i64, b: i64, c: i64, d: i64 }\n  %pa = getfieldptr.struct { a: i64, b: i64, c: i64, d: i64 } %p, 0\n  store.i64 %pa, 7\n  ret.ptr %p\n}\nfn @main() -> i64 {\nentry:\n  %r = call @make()\n  %ra = getfieldptr.struct { a: i64, b: i64, c: i64, d: i64 } %r, 0\n  %a = load.i64 %ra\n  ret.i64 %a\n}\n";
    let assembly = emit_aggregate_module(ir, "hidden_result")?;

    let callee = function_body(&assembly, "make")?;
    let prologue = callee
        .split(".L_make.entry:")
        .next()
        .ok_or("missing entry")?;
    let frame = ppc64_frame_arithmetic(prologue).0;
    assert!(
        prologue.contains(&format!("    std 3, {}(1)\n", frame + 32)),
        "{assembly}"
    );
    let body = callee
        .split(".L_make.entry:")
        .nth(1)
        .ok_or("missing entry")?;
    assert!(
        body.contains(&format!("    ld 4, {}(1)\n", frame + 32)),
        "{assembly}"
    );
    assert!(body.contains("    std 12, 24(4)\n"), "{assembly}");
    assert!(body.contains("    mr 3, 4\n"), "{assembly}");

    let caller = function_body(&assembly, "main")?;
    let call = caller.split("bl make").next().ok_or("missing call")?;
    assert!(call.contains("    addi 3, 1, "), "{assembly}");
    Ok(())
}

#[test]
fn aggregate_result_of_two_words_comes_back_in_r3_and_r4() -> Result<(), Box<dyn Error>> {
    let ir = "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n  %p = stackalloc.struct { a: i64, b: i64 }\n  %pa = getfieldptr.struct { a: i64, b: i64 } %p, 0\n  store.i64 %pa, 7\n  ret.ptr %p\n}\nfn @main() -> i64 {\nentry:\n  %r = call @make()\n  %ra = getfieldptr.struct { a: i64, b: i64 } %r, 0\n  %a = load.i64 %ra\n  ret.i64 %a\n}\n";
    let assembly = emit_aggregate_module(ir, "pair_result")?;

    let callee = function_body(&assembly, "make")?;
    assert!(callee.contains("    ld 3, 0(11)\n"), "{assembly}");
    assert!(callee.contains("    ld 4, 8(11)\n"), "{assembly}");

    let caller = function_body(&assembly, "main")?;
    let after = caller.split("bl make").nth(1).ok_or("missing call")?;
    let low = store_offset(after, "std 3, ")?;
    assert_eq!(store_offset(after, "std 4, ")?, low + 8, "{assembly}");
    Ok(())
}
