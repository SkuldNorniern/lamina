use super::*;
use crate::mir::{FunctionBuilder, Immediate, ScalarType};
use lamina_platform::PointerWidth;
use std::error::Error;

#[test]
fn global_parameters_are_initialized_before_entry() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    let reg = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("identity")
            .with_params(vec![Parameter::new(
                reg.clone(),
                MirType::Scalar(ScalarType::I64),
            )])
            .with_return(MirType::Scalar(ScalarType::I64)),
    );
    let mut block = Block::new("entry");
    block.push(MirInst::Ret {
        value: Some(Operand::Register(reg)),
    });
    func.add_block(block);
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let asm = String::from_utf8(
            compile_single_function_aarch64(
                "identity",
                &func,
                TargetOperatingSystem::Linux,
                &settings,
                &AggregateSignatures::empty(PointerWidth::Bits64),
            )
            .map_err(|error| error.to_string())?,
        )?;
        let (prologue, body) = asm
            .split_once(".L_identity.entry:")
            .ok_or("missing entry")?;
        let reg = body
            .lines()
            .find_map(|line| line.strip_prefix("    mov x0, "))
            .ok_or("missing return move")?;
        assert!(
            prologue.contains(&format!("ldur {reg}, [x29, #-8]")),
            "{asm}"
        );
    }
    Ok(())
}

#[test]
fn narrow_parameters_are_sign_extended_on_entry() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    for (ty, extend) in [
        (MirType::Scalar(ScalarType::I8), "sxtb x0, w0"),
        (MirType::Scalar(ScalarType::I16), "sxth x0, w0"),
        (MirType::Scalar(ScalarType::I32), "sxtw x0, w0"),
    ] {
        let reg = Register::Virtual(VirtualReg::gpr(0));
        let mut func = Function::new(
            Signature::new("narrow")
                .with_params(vec![Parameter::new(reg.clone(), ty)])
                .with_return(ty),
        );
        let mut block = Block::new("entry");
        block.push(MirInst::Ret {
            value: Some(Operand::Register(reg)),
        });
        func.add_block(block);
        let asm = String::from_utf8(
            compile_single_function_aarch64(
                "narrow",
                &func,
                TargetOperatingSystem::Linux,
                &MirCodegenSettings::default(),
                &AggregateSignatures::empty(PointerWidth::Bits64),
            )
            .map_err(|error| error.to_string())?,
        )?;
        let (prologue, _) = asm.split_once(".L_narrow.entry:").ok_or("missing entry")?;
        assert!(prologue.contains(&format!("    {extend}\n")), "{asm}");
    }
    Ok(())
}

#[test]
fn bool_parameters_are_zero_extended_on_entry() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    let ty = MirType::Scalar(ScalarType::I1);
    let reg = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("flag")
            .with_params(vec![Parameter::new(reg.clone(), ty)])
            .with_return(ty),
    );
    let mut block = Block::new("entry");
    block.push(MirInst::Ret {
        value: Some(Operand::Register(reg)),
    });
    func.add_block(block);
    let asm = String::from_utf8(
        compile_single_function_aarch64(
            "flag",
            &func,
            TargetOperatingSystem::Linux,
            &MirCodegenSettings::default(),
            &AggregateSignatures::empty(PointerWidth::Bits64),
        )
        .map_err(|error| error.to_string())?,
    )?;
    let (prologue, _) = asm.split_once(".L_flag.entry:").ok_or("missing entry")?;
    assert!(prologue.contains("    uxtb x0, w0\n"), "{asm}");
    Ok(())
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
        .instr(MirInst::Copy {
            ty,
            dst: dst.clone(),
            src: Operand::Register(src),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(dst)),
        })
        .build();
    let mut module = MirModule::new("copy_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_aarch64(
        &module,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )
    .expect("AArch64 codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(assembly.contains("mov "), "expected move: {assembly}");
    assert!(
        !assembly.contains("fadd"),
        "unexpected float add: {assembly}"
    );
}

#[test]
fn mach_o_targets_call_underscored_libc_symbols() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "@extern\nfn @labs(i64 %x) -> i64\n\nfn @main() -> i64 {\nentry:\n  %v = add.i64 1, 2\n  %r = call @labs(%v)\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "libc_test").expect("lower libc call module");
    for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::IOS] {
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, os, PointerWidth::Bits64)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("bl _labs\n"), "{os:?}: {assembly}");
        assert!(assembly.contains("_main:\n"), "{os:?}: {assembly}");
        assert!(!assembly.contains("bl labs\n"), "{os:?}: {assembly}");
    }
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("bl labs\n"), "{assembly}");
    Ok(())
}

/// Scan straight-line assembly and report a caller-saved pool register that is read
/// after a call without having been written since: that value did not survive the call.
#[cfg(test)]
fn caller_saved_read_back_after_a_call(asm: &str) -> Option<String> {
    const STORES: &[&str] = &[
        "str", "stur", "stp", "stnp", "strb", "strh", "sturb", "sturh",
    ];
    const READ_ONLY: &[&str] = &["cmp", "cmn", "tst", "ret", "cbz", "cbnz", "tbz", "tbnz"];
    let mut written: HashSet<&str> = HashSet::new();
    let mut called = false;
    for line in asm.lines() {
        let line = line.trim();
        let Some((mnemonic, operands)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        if mnemonic == "bl" || mnemonic == "blr" {
            written.clear();
            called = true;
            continue;
        }
        let operands: Vec<&str> = operands
            .split(',')
            .map(|operand| operand.trim().trim_matches(['[', ']', '!']))
            .collect();
        let defines = !STORES.contains(&mnemonic)
            && !READ_ONLY.contains(&mnemonic)
            && !mnemonic.starts_with('b');
        for (index, operand) in operands.iter().enumerate() {
            if !matches!(*operand, "x13" | "x14" | "x15") {
                continue;
            }
            if defines && index == 0 {
                written.insert(operand);
            } else if called && !written.contains(operand) {
                return Some(format!("{operand} is read back by `{line}`"));
            }
        }
    }
    None
}

#[test]
fn global_allocation_keeps_values_out_of_caller_saved_pool_across_calls()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @id(i64 %x) -> i64 {\nentry:\n  %r = mul.i64 %x, 1\n  ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n  %a = add.i64 7, 0\n  %b = mul.i64 %a, 3\n  %c = call @id(5)\n  %d = add.i64 %b, %c\n  ret.i64 %d\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "call_crossing").expect("lower call module");
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_aarch64_with_units_and_settings(
            &mir,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &settings,
            PointerWidth::Bits64,
        )?;
        let assembly = String::from_utf8(output)?;
        let (_, main) = assembly.split_once("\nmain:\n").ok_or("missing main")?;
        let main = main.split_once(".Lret_main:").ok_or("missing epilogue")?.0;
        assert!(main.contains("bl id"), "{assembly}");
        if let Some(reason) = caller_saved_read_back_after_a_call(main) {
            panic!("{regalloc:?}: {reason}: {assembly}");
        }
    }
    Ok(())
}

#[test]
fn tail_call_releases_the_frame_the_prologue_reserved() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    let x = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("f")
            .with_params(vec![Parameter::new(
                x.clone(),
                MirType::Scalar(ScalarType::I64),
            )])
            .with_return(MirType::Scalar(ScalarType::I64)),
    );
    let mut block = Block::new("entry");
    let mut prev = Operand::Register(x);
    for id in 1..=101 {
        let dst = Register::Virtual(VirtualReg::gpr(id));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: prev,
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        prev = Operand::Register(dst);
    }
    block.push(MirInst::TailCall {
        name: "g".into(),
        args: vec![prev],
    });
    func.add_block(block);
    let asm = String::from_utf8(
        compile_single_function_aarch64(
            "f",
            &func,
            TargetOperatingSystem::Linux,
            &MirCodegenSettings::default(),
            &AggregateSignatures::empty(PointerWidth::Bits64),
        )
        .map_err(|error| error.to_string())?,
    )?;
    let reserved = asm
        .lines()
        .find_map(|line| line.strip_prefix("    sub sp, sp, #"))
        .ok_or("missing frame reservation")?;
    assert!(
        asm.contains(&format!(
            "    .cfi_remember_state\n    add sp, sp, #{reserved}\n    ldp x29, x30, [sp], #16\n    .cfi_def_cfa sp, 0\n    .cfi_restore 29\n    .cfi_restore 30\n    b g\n    .cfi_restore_state\n"
        )),
        "{asm}"
    );
    Ok(())
}

#[test]
fn elf_frames_carry_cfi_and_mach_o_frames_do_not() -> Result<(), Box<dyn Error>> {
    let mut func = Function::new(Signature::new("f").with_return(MirType::Scalar(ScalarType::I64)));
    use crate::mir::Block;
    let mut block = Block::new("entry");
    block.push(MirInst::Ret {
        value: Some(Operand::Immediate(Immediate::I64(0))),
    });
    func.add_block(block);
    let settings = MirCodegenSettings::default();
    let signatures = AggregateSignatures::empty(PointerWidth::Bits64);
    let emit = |os| -> Result<String, Box<dyn Error>> {
        let bytes = compile_single_function_aarch64("f", &func, os, &settings, &signatures)
            .map_err(|error| error.to_string())?;
        Ok(String::from_utf8(bytes)?)
    };
    let elf = emit(TargetOperatingSystem::Linux)?;
    assert!(
        elf.starts_with(
            "f:\n    .cfi_startproc\n    stp x29, x30, [sp, #-16]!\n    .cfi_def_cfa_offset 16\n    .cfi_offset 29, -16\n    .cfi_offset 30, -8\n    mov x29, sp\n    .cfi_def_cfa_register 29\n"
        ),
        "{elf}"
    );
    assert!(
        elf.ends_with("    ret\n    .cfi_restore_state\n    .cfi_endproc\n"),
        "{elf}"
    );
    let macho = emit(TargetOperatingSystem::MacOS)?;
    assert!(!macho.contains(".cfi_"), "{macho}");
    Ok(())
}

#[test]
fn long_conditional_branches_are_inverted_over_unconditional_branches() -> Result<(), Box<dyn Error>>
{
    let mut direct = Vec::new();
    emit_branch_if(&mut direct, "lt", ".L_f.far", false)?;
    assert_eq!(String::from_utf8(direct)?, "    b.lt .L_f.far\n");
    let mut long = Vec::new();
    emit_branch_if(&mut long, "lt", ".L_f.far", true)?;
    assert_eq!(
        String::from_utf8(long)?,
        "    b.ge 9f\n    b .L_f.far\n9:\n"
    );
    Ok(())
}

#[test]
fn switch_case_constants_are_materialized_without_immediate_range_limits()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %x = add.i64 100000, 0\n  switch.i64 %x, default, [100000, hit], [-5, neg]\nhit:\n  ret.i64 1\nneg:\n  ret.i64 2\ndefault:\n  ret.i64 3\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "switch_range").expect("lower switch module");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains("mov x9, #100000"), "{assembly}");
    assert!(assembly.contains("movz x9, #34464, lsl #0"), "{assembly}");
    Ok(())
}

#[test]
fn integer_float_conversions_select_width_and_saturation_forms() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %a = inttofp.i8.f32 -3\n  %b = fptoint.f64.i8 2.5\n  %c = fptoint.f64.u64 1.0\n  %d = inttofp.u64.f64 7\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "conversions").expect("lower conversion module");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("sbfx x"), "{assembly}");
    assert!(assembly.contains("scvtf s0, x"), "{assembly}");
    assert!(assembly.contains("fcvtzs w"), "{assembly}");
    assert!(assembly.contains("csel w"), "{assembly}");
    assert!(assembly.contains("fcvtzu x"), "{assembly}");
    assert!(assembly.contains("ucvtf d0, x"), "{assembly}");
    Ok(())
}

#[test]
fn module_globals_are_addressed_through_page_relocations() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "global $counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 $counter\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "globals").expect("lower global module");
    let mut linux = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut linux,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let linux = String::from_utf8(linux)?;
    assert!(linux.contains("adrp x9, counter\n"), "{linux}");
    assert!(linux.contains("add x9, x9, :lo12:counter\n"), "{linux}");
    assert!(
        linux.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
        "{linux}"
    );
    let mut macho = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut macho,
        TargetOperatingSystem::MacOS,
        PointerWidth::Bits64,
    )?;
    let macho = String::from_utf8(macho)?;
    assert!(macho.contains("adrp x9, _counter@PAGE\n"), "{macho}");
    assert!(macho.contains("add x9, x9, _counter@PAGEOFF\n"), "{macho}");
    Ok(())
}

#[test]
fn frames_beyond_twelve_bit_immediates_are_addressed_through_a_register()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %arr = stackalloc.[5000 x i8]\n  %p = getelem.i8 %arr, 4999\n  store.i8 %p, 42\n  %v = load.i8 %p\n  %v_u = bitcast.i8.u8 %v\n  %w = extend.u8.i64 %v_u\n  ret.i64 %w\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "big_frame").expect("lower big frame module");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Unknown,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    for line in assembly.lines() {
        if line.contains("sp, sp, #") || line.contains("x29, #") || line.contains("x29, x29, #") {
            let imm: i64 = line
                .rsplit_once('#')
                .and_then(|(_, n)| n.trim().parse().ok())
                .unwrap_or(0);
            assert!(
                imm.abs() <= 4095,
                "immediate out of range in: {line}\n{assembly}"
            );
        }
    }
    assert!(assembly.contains("sub sp, sp, x16"), "{assembly}");
    Ok(())
}

#[test]
fn integer_aggregates_use_consecutive_x_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @sum(struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  %ap = getfieldptr.struct { a: i64, b: i64 } %p, 0\n  %a = load.i64 %ap\n  ret.i64 %a\n}\nfn @wide(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  ret.i64 %a\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.struct { a: i64, b: i64 }\n  %r = call @sum(struct { a: i64, b: i64 } %q)\n  %s = call @wide(1, 2, 3, 4, 5, 6, 7, struct { a: i64, b: i64 } %q)\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "aggregates").expect("lower aggregates");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;

    let prologue = assembly
        .split("\nsum:\n")
        .nth(1)
        .and_then(|body| body.split(".L_sum.entry:").next())
        .ok_or("missing sum prologue")?;
    let home = prologue
        .lines()
        .filter(|line| line.contains(", x29, #"))
        .find_map(|line| line.trim().strip_prefix("sub ")?.split(',').next())
        .ok_or("missing frame address")?
        .to_string();
    assert!(
        prologue.contains(&format!("str x0, [{home}, #0]")),
        "{assembly}"
    );
    assert!(
        prologue.contains(&format!("str x1, [{home}, #8]")),
        "{assembly}"
    );

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("bl sum").next().ok_or("missing call")?;
    assert!(call.contains("ldr x0, ["), "{assembly}");
    assert!(call.contains("ldr x1, ["), "{assembly}");

    // Seven scalars leave one register, which is too few for two words, so the
    // aggregate goes to the stack whole and x7 stays unused.
    let wide = assembly
        .split("\nwide:\n")
        .nth(1)
        .and_then(|body| body.split(".L_wide.entry:").next())
        .ok_or("missing wide prologue")?;
    assert!(!wide.contains("str x7, "), "{assembly}");
    let loaded: Vec<&str> = wide
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("ldr ") && line.contains("[x29, #"))
        .collect();
    assert!(
        loaded.iter().any(|line| line.ends_with("[x29, #16]")),
        "{assembly}"
    );
    assert!(
        loaded.iter().any(|line| line.ends_with("[x29, #24]")),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn floats_cross_the_boundary_in_v_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @add(f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  ret.f64 %s\n}\nfn @addf(f32 %a, i64 %n) -> f32 {\nentry:\n  %s = add.f32 %a, 1.0\n  ret.f32 %s\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(1.5, 2.5)\n  %i = fptoint.f64.i64 %r\n  ret.i64 %i\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "floats").expect("lower floats");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;

    // Both arguments arrive in the float bank, and the result leaves in d0.
    let callee = assembly.split("\nadd:\n").nth(1).ok_or("missing add")?;
    let prologue = callee
        .split(".L_add.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(prologue.contains("fmov x"), "{assembly}");
    assert!(prologue.contains(", d0\n"), "{assembly}");
    assert!(prologue.contains(", d1\n"), "{assembly}");
    assert!(callee.contains("    fmov d0, x"), "{assembly}");

    // A float and an integer draw from separate banks, so the integer still takes x0.
    let mixed = assembly.split("\naddf:\n").nth(1).ok_or("missing addf")?;
    let mixed_prologue = mixed
        .split(".L_addf.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(mixed_prologue.contains(", s0\n"), "{assembly}");
    assert!(mixed_prologue.contains("    str x0, "), "{assembly}");

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("bl add").next().ok_or("missing call")?;
    assert!(call.contains("    fmov d0, x"), "{assembly}");
    assert!(call.contains("    fmov d1, x"), "{assembly}");
    assert!(caller.contains("    fmov x0, d0\n"), "{assembly}");
    Ok(())
}

#[test]
fn homogeneous_float_aggregates_use_v_registers_and_fall_back_to_the_stack()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_for_target, parser::parse_module};
    use lamina_platform::TargetArchitecture;
    let ir = "type !F2 = struct { a: f32, b: f32 }\ntype !D3 = struct { a: f64, b: f64, c: f64 }\nfn @pass(!F2 %p) -> !F2 {\nentry:\n  ret.!F2 %p\n}\nfn @spill(f64 %a, f64 %b, f64 %c, f64 %d, f64 %e, f64 %f, !D3 %p, f64 %t) -> !D3 {\nentry:\n  %r = stackalloc.!D3\n  ret.!D3 %r\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.!F2\n  %r = call @pass(!F2 %q)\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_for_target(
        &module,
        "homogeneous",
        TargetArchitecture::Aarch64.data_layout(),
    )
    .expect("lower homogeneous aggregates");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;

    let callee = assembly.split("\npass:\n").nth(1).ok_or("missing pass")?;
    let prologue = callee
        .split(".L_pass.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(prologue.contains("    str s0, ["), "{assembly}");
    assert!(prologue.contains("    str s1, ["), "{assembly}");
    assert!(!prologue.contains("str x0, ["), "{assembly}");
    let epilogue = callee.split(".Lret_pass:").next().ok_or("missing return")?;
    assert!(epilogue.contains("    ldr s0, ["), "{assembly}");
    assert!(epilogue.contains("    ldr s1, ["), "{assembly}");

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("bl pass").next().ok_or("missing call")?;
    assert!(call.contains("    ldr s0, ["), "{assembly}");
    assert!(call.contains("    ldr s1, ["), "{assembly}");
    assert!(!call.contains("    ldr x0, ["), "{assembly}");

    // Six doubles leave two v registers, too few for the three-member aggregate, which
    // goes to the stack whole; the scalar after it takes no register either.
    let spill = assembly.split("\nspill:\n").nth(1).ok_or("missing spill")?;
    let spill_prologue = spill
        .split(".L_spill.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(spill_prologue.contains(", d5\n"), "{assembly}");
    assert!(!spill_prologue.contains(", d6\n"), "{assembly}");
    assert!(!spill_prologue.contains(", d7\n"), "{assembly}");
    Ok(())
}

#[test]
fn integer_aggregate_returns_come_back_in_x0_and_x1() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n  %p = stackalloc.struct { a: i64, b: i64 }\n  %pa = getfieldptr.struct { a: i64, b: i64 } %p, 0\n  store.i64 %pa, 7\n  ret.struct { a: i64, b: i64 } %p\n}\nfn @main() -> i64 {\nentry:\n  %r = call @make()\n  %ra = getfieldptr.struct { a: i64, b: i64 } %r, 0\n  %a = load.i64 %ra\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "returns").expect("lower aggregate return");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;

    let callee = assembly.split("\nmake:\n").nth(1).ok_or("missing make")?;
    let epilogue = callee.split(".Lret_make:").next().ok_or("missing return")?;
    let address = epilogue
        .lines()
        .rev()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("ldr x0, [")?;
            rest.strip_suffix(']')
        })
        .ok_or("missing result load")?
        .to_string();
    assert!(
        epilogue.contains(&format!("ldr x1, [{address}, #8]")),
        "{assembly}"
    );

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let after = caller.split("bl make").nth(1).ok_or("missing call")?;
    let home = after
        .lines()
        .find_map(|line| line.trim().strip_prefix("sub ")?.split(',').next())
        .ok_or("missing result slot")?
        .to_string();
    assert!(after.contains(&format!("str x0, [{home}]")), "{assembly}");
    assert!(
        after.contains(&format!("str x1, [{home}, #8]")),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn stack_aggregate_prologue_keeps_off_callee_saved_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @big(i64 %p0, i64 %p1, i64 %p2, i64 %p3, i64 %p4, i64 %p5, i64 %p6, i64 %p7, struct { a: i64, b: i64, c: i64, d: i64 } %q) -> i64 {\nentry:\n  %qa = getfieldptr.struct { a: i64, b: i64, c: i64, d: i64 } %q, 0\n  %a = load.i64 %qa\n  ret.i64 %a\n}\nfn @main() -> i64 {\nentry:\n  %q = stackalloc.struct { a: i64, b: i64, c: i64, d: i64 }\n  %r = call @big(1, 2, 3, 4, 5, 6, 7, 8, struct { a: i64, b: i64, c: i64, d: i64 } %q)\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "stack_aggregate").expect("lower aggregate");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;

    for index in 19..=28 {
        assert!(
            !assembly.contains(&format!("x{index},")) && !assembly.contains(&format!("x{index}]")),
            "x{index} used without a save: {assembly}"
        );
    }
    Ok(())
}

#[test]
fn darwin_packs_narrow_stack_arguments_at_natural_size() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "fn @wide(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i32 %i, i32 %j) -> i64 {\nentry:\n  %i_u = bitcast.i32.u32 %i\n  %x = extend.u32.i64 %i_u\n  ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n  %r = call @wide(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "narrow_stack").expect("lower narrow arguments");

    let emit = |os| -> Result<String, Box<dyn Error>> {
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, os, PointerWidth::Bits64)?;
        Ok(String::from_utf8(output)?)
    };

    let darwin = emit(TargetOperatingSystem::MacOS)?;
    assert!(darwin.contains("str w9, [sp, #0]"), "{darwin}");
    assert!(darwin.contains("str w10, [sp, #4]"), "{darwin}");
    let loaded: Vec<&str> = darwin
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("ldr w") && line.contains("[x29, #"))
        .collect();
    assert!(
        loaded.iter().any(|line| line.ends_with("[x29, #16]")),
        "{darwin}"
    );
    assert!(
        loaded.iter().any(|line| line.ends_with("[x29, #20]")),
        "{darwin}"
    );

    let linux = emit(TargetOperatingSystem::Linux)?;
    assert!(linux.contains("str x9, [sp, #0]"), "{linux}");
    assert!(linux.contains("str x10, [sp, #8]"), "{linux}");
    Ok(())
}

#[test]
fn darwin_sends_variadic_arguments_to_stack() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nglobal $fmt: [5 x i8] = \"%g\\n\"\n\nfn @main() -> i64 {\nentry:\n  %f = getelem.i8 $fmt, 0\n  %r = call @printf(%f, 7, 2.5)\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "darwin_variadic").expect("lower variadic call");

    let emit = |os| -> Result<String, Box<dyn Error>> {
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, os, PointerWidth::Bits64)?;
        Ok(String::from_utf8(output)?)
    };

    // The format string stays in x0; 7 and 2.5 take whole stack slots.
    let darwin = emit(TargetOperatingSystem::MacOS)?;
    let stored: Vec<&str> = darwin
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("str ") && line.contains("[sp, #"))
        .collect();
    assert!(
        stored.iter().any(|line| line.ends_with("[sp, #0]")),
        "{darwin}"
    );
    assert!(
        stored.iter().any(|line| line.ends_with("[sp, #8]")),
        "{darwin}"
    );
    assert!(!darwin.contains("fmov d0,"), "{darwin}");

    let linux = emit(TargetOperatingSystem::Linux)?;
    assert!(linux.contains("fmov d0,"), "{linux}");
    assert!(
        !linux
            .lines()
            .map(str::trim)
            .any(|line| line.starts_with("str ") && line.contains("[sp, #")),
        "{linux}"
    );
    Ok(())
}

#[test]
fn register_named_functions_are_rejected() -> Result<(), Box<dyn Error>> {
    use crate::mir::Block;
    for name in ["x0", "W30", "d0", "q0", "v0", "b1", "sp", "lr", "fp"] {
        assert!(is_aarch64_register_name(name), "{name}");
    }
    for name in ["x32", "x", "foo", "_x0", "xzr2", "main"] {
        assert!(!is_aarch64_register_name(name), "{name}");
    }
    let mut func =
        Function::new(Signature::new("x0").with_return(MirType::Scalar(ScalarType::I64)));
    let mut block = Block::new("entry");
    block.push(MirInst::Ret {
        value: Some(Operand::Immediate(Immediate::I64(7))),
    });
    func.add_block(block);
    let mut module = MirModule::new("registers");
    module.functions.insert("x0".to_string(), func);
    let mut out = Vec::new();
    let error = generate_mir_aarch64(
        &module,
        &mut out,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )
    .err()
    .ok_or("expected a rejection")?;
    assert!(
        error
            .to_string()
            .contains("@x0 is an AArch64 register name"),
        "{error}"
    );
    Ok(())
}

#[test]
fn mach_o_io_intrinsics_report_a_failed_syscall_as_minus_two() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(
        "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %w = write %p, 4\n  %r = read %p, 4\n  %b = writebyte 65\n  ret.i64 %w\n}\n",
    )?;
    let mir = from_ir_x86_64(&module, "io").expect("lower io");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::MacOS,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert_eq!(
        assembly
            .matches("svc #0\n    mov x9, #-2\n    csel x0, x9, x0, cs\n")
            .count(),
        3,
        "{assembly}"
    );
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains("csel x0, x9, x0, cs"), "{assembly}");
    Ok(())
}

fn darwin_add_immediates(assembly: &str) -> Vec<&str> {
    assembly
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("add ") && line.contains(", #"))
        .collect()
}

#[test]
fn darwin_variadic_callee_reads_its_arguments_from_the_stack() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(
        "fn @sum(i64 %n, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  %y = va_arg.i32 %ap\n  va_end %ap\n  %w = extend.i32.i64 %y\n  %s = add.i64 %x, %w\n  ret.i64 %s\n}\n",
    )?;
    let mir = from_ir_x86_64(&module, "darwin_va").expect("lower variadic callee");

    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::MacOS,
        PointerWidth::Bits64,
    )?;
    let darwin = String::from_utf8(output)?;
    assert!(
        darwin_add_immediates(&darwin)
            .iter()
            .any(|line| line.ends_with("x29, #16")),
        "{darwin}"
    );
    assert!(
        darwin_add_immediates(&darwin)
            .iter()
            .any(|line| line.ends_with(", #8")),
        "{darwin}"
    );
    assert!(darwin.contains("    sxtw "), "{darwin}");

    let mut output = Vec::new();
    let error = generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Windows,
        PointerWidth::Bits64,
    )
    .err()
    .ok_or("expected a rejection on Windows")?;
    assert!(
        error
            .to_string()
            .contains("is not yet supported by the AArch64 backend on Windows"),
        "{error}"
    );
    Ok(())
}

fn linux_va_assembly(source: &str, name: &str) -> Result<String, Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(source)?;
    let mir = from_ir_x86_64(&module, name).expect("lower variadic callee");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )?;
    Ok(String::from_utf8(output)?)
}

#[test]
fn linux_variadic_prologue_spills_all_argument_registers() -> Result<(), Box<dyn Error>> {
    let assembly = linux_va_assembly(
        "fn @sum(i64 %n, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  va_end %ap\n  ret.i64 %x\n}\n",
        "linux_va_spill",
    )?;
    let stores: Vec<&str> = assembly
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("str x") || line.starts_with("str q"))
        .collect();
    for index in 0..8 {
        assert!(
            stores.contains(&format!("str x{index}, [x16, #{}]", index * 8).as_str()),
            "{assembly}"
        );
        assert!(
            stores.contains(&format!("str q{index}, [x16, #{}]", 64 + index * 16).as_str()),
            "{assembly}"
        );
    }
    assert!(
        assembly
            .lines()
            .any(|line| line.trim().starts_with("sub x16, x29, #")),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn linux_variadic_va_arg_walks_general_then_vector_then_stack() -> Result<(), Box<dyn Error>> {
    let assembly = linux_va_assembly(
        "fn @mix(i64 %n, ...) -> f64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  %y = va_arg.f64 %ap\n  va_end %ap\n  ret.f64 %y\n}\n",
        "linux_va_walk",
    )?;
    let trimmed: Vec<&str> = assembly.lines().map(str::trim).collect();
    assert_eq!(
        trimmed
            .iter()
            .filter(|line| line.starts_with("ldrsw "))
            .count(),
        2,
        "{assembly}"
    );
    assert!(
        trimmed
            .iter()
            .any(|line| line.starts_with("b.ge .L_mix.va_arg")),
        "{assembly}"
    );
    assert!(
        trimmed
            .iter()
            .any(|line| line.ends_with(", #8]") && line.starts_with("ldr ")),
        "{assembly}"
    );
    assert!(
        trimmed
            .iter()
            .any(|line| line.ends_with(", #16]") && line.starts_with("ldr ")),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn linux_variadic_va_copy_moves_the_whole_va_list() -> Result<(), Box<dyn Error>> {
    let assembly = linux_va_assembly(
        "fn @copy(i64 %n, ...) -> i64 {\nentry:\n  %a = stackalloc.valist\n  %b = stackalloc.valist\n  va_start %a\n  va_copy %b, %a\n  %x = va_arg.i64 %b\n  va_end %a\n  va_end %b\n  ret.i64 %x\n}\n",
        "linux_va_copy",
    )?;
    assert!(
        assembly
            .lines()
            .filter(|line| line.trim().starts_with("ldr "))
            .count()
            >= 4,
        "{assembly}"
    );
    Ok(())
}

#[test]
fn darwin_variadic_area_starts_after_named_stack_arguments() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(
        "fn @f(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i32 %i, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  va_end %ap\n  ret.i64 %x\n}\n",
    )?;
    let mir = from_ir_x86_64(&module, "darwin_va_named").expect("lower variadic callee");
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::MacOS,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(
        darwin_add_immediates(&assembly)
            .iter()
            .any(|line| line.ends_with("x29, #24")),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn checked_arithmetic_reads_the_flags_or_the_widened_result() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let cases: [(&str, &str, &[&str]); 7] = [
        ("add.ovf.i64", "i64", &["cmn ", ", vs"]),
        ("add.ovf.u64", "i64", &["cmn ", ", cs"]),
        ("sub.ovf.u64", "i64", &["cmp ", ", cc"]),
        ("mul.ovf.i64", "i64", &["smulh ", "asr #63"]),
        ("mul.ovf.u64", "i64", &["umulh "]),
        ("mul.ovf.i32", "i32", &["sbfx ", "mul "]),
        ("add.ovf.u8", "i8", &["ubfx ", "cset "]),
    ];
    for (op, ty, expected) in cases {
        let source = format!(
            "fn @main({ty} %a, {ty} %b) -> i64 {{\nentry:\n  %f = {op} %a, %b\n  %e = extend.bool.i64 %f\n  ret.i64 %e\n}}\n"
        );
        let module = parse_module(&source)?;
        let mir = from_ir_x86_64(&module, "ovf").expect("lower checked arithmetic");
        let mut output = Vec::new();
        generate_mir_aarch64(
            &mir,
            &mut output,
            TargetOperatingSystem::Linux,
            PointerWidth::Bits64,
        )?;
        let assembly = String::from_utf8(output)?;
        for line in expected {
            assert!(assembly.contains(line), "{op} lacks `{line}`:\n{assembly}");
        }
    }
    Ok(())
}

#[test]
fn macho_thread_locals_use_tlv_descriptors_and_thread_sections() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "@thread_local global $t: i64 = 0\n@thread_local global $u: i64 = 5\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 $t\n  %b = load.i64 $u\n  %c = add.i64 %a, %b\n  ret.i64 %c\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "tls").expect("lower thread-local module");
    let mut out = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut out,
        TargetOperatingSystem::MacOS,
        PointerWidth::Bits64,
    )?;
    let asm = String::from_utf8(out)?;
    for symbol in ["_t", "_u"] {
        let access = format!(
            "    adrp x0, {symbol}.tlv@TLVPPAGE\n    ldr x0, [x0, {symbol}.tlv@TLVPPAGEOFF]\n    ldr x8, [x0]\n    blr x8\n"
        );
        assert!(asm.contains(&access), "{asm}");
    }
    assert!(
        asm.contains(
            ".section __DATA,__thread_vars,thread_local_variables\n.p2align 3\n_t.tlv:\n    .quad __tlv_bootstrap\n    .quad 0\n    .quad _t\n.p2align 3\n_u.tlv:\n    .quad __tlv_bootstrap\n    .quad 0\n    .quad _u\n"
        ),
        "{asm}"
    );
    assert!(
        asm.contains(
            ".section __DATA,__thread_bss,thread_local_zerofill\n.p2align 3\n_t:\n    .zero 8\n"
        ),
        "{asm}"
    );
    assert!(
        asm.contains(".section __DATA,__thread_data,thread_local_regular\n.p2align 3\n_u:\n    .byte 0x05, 0x00"),
        "{asm}"
    );
    assert!(!asm.contains(".tdata") && !asm.contains(".tbss"), "{asm}");
    Ok(())
}

fn aarch64_thread_local_asm(annotation: &str, os: TargetOperatingSystem) -> String {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let source = format!(
        "{annotation} global $t: i64 = 0\nfn @main() -> i64 {{\nentry:\n    store.i64 $t, 7\n    %v = load.i64 $t\n    ret.i64 %v\n}}\n"
    );
    let module = parse_module(&source).unwrap();
    let mir = from_ir_x86_64(&module, "tls").unwrap();
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, os, PointerWidth::Bits64).unwrap();
    String::from_utf8(output).unwrap()
}

#[test]
fn elf_local_exec_thread_local_reads_tp_and_adds_tprel_pair() {
    let asm = aarch64_thread_local_asm("@thread_local", TargetOperatingSystem::Linux);
    let mrs = asm
        .lines()
        .find(|line| line.starts_with("    mrs ") && line.ends_with(", tpidr_el0"))
        .expect(&asm);
    let reg = &mrs["    mrs ".len()..mrs.len() - ", tpidr_el0".len()];
    let sequence = format!(
        "{mrs}\n    add {reg}, {reg}, :tprel_hi12:t, lsl #12\n    add {reg}, {reg}, :tprel_lo12_nc:t\n"
    );
    assert!(asm.contains(&sequence), "{asm}");
    assert!(
        !asm.contains("gottprel") && !asm.contains("TLVPPAGE"),
        "{asm}"
    );
    assert!(asm.contains(".section .tbss,\"awT\",@nobits"), "{asm}");
}

#[test]
fn elf_initial_exec_thread_local_loads_gottprel_and_adds_tp() {
    let asm = aarch64_thread_local_asm("@thread_local(initial_exec)", TargetOperatingSystem::Linux);
    let adrp = asm
        .lines()
        .find(|line| line.starts_with("    adrp ") && line.ends_with(", :gottprel:t"))
        .expect(&asm);
    let reg = &adrp["    adrp ".len()..adrp.len() - ", :gottprel:t".len()];
    let ldr = format!("    ldr {reg}, [{reg}, :gottprel_lo12:t]");
    let mrs_line = asm
        .lines()
        .skip_while(|line| *line != ldr)
        .nth(1)
        .expect(&asm);
    let tp = mrs_line
        .strip_prefix("    mrs ")
        .and_then(|rest| rest.strip_suffix(", tpidr_el0"))
        .expect(&asm);
    let sequence = format!("{adrp}\n{ldr}\n{mrs_line}\n    add {reg}, {reg}, {tp}\n");
    assert!(asm.contains(&sequence), "{asm}");
    assert!(!asm.contains(":tprel_"), "{asm}");
    assert!(asm.contains(".section .tbss,\"awT\",@nobits"), "{asm}");
}

#[test]
fn elf_initialized_thread_local_goes_to_tdata() {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let source = "@thread_local global $t: i64 = 9\nfn @main() -> i64 {\nentry:\n    %v = load.i64 $t\n    ret.i64 %v\n}\n";
    let module = parse_module(source).unwrap();
    let mir = from_ir_x86_64(&module, "tls").unwrap();
    let mut output = Vec::new();
    generate_mir_aarch64(
        &mir,
        &mut output,
        TargetOperatingSystem::Linux,
        PointerWidth::Bits64,
    )
    .unwrap();
    let asm = String::from_utf8(output).unwrap();
    assert!(asm.contains(".section .tdata,\"awT\",@progbits"), "{asm}");
}

#[test]
fn thread_locals_stay_refused_on_coff() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let ir = "@thread_local global $t: i64 = 0\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 $t\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir_x86_64(&module, "tls").expect("lower thread-local module");
    let mut out = Vec::new();
    let error = generate_mir_aarch64(
        &mir,
        &mut out,
        TargetOperatingSystem::Windows,
        PointerWidth::Bits64,
    )
    .expect_err("thread-local global must be refused")
    .to_string();
    assert!(error.contains("thread-local storage"), "{error}");
    assert!(out.is_empty(), "wrote code before refusing");
    Ok(())
}

/// Register-hungry leaf whose live values outnumber the caller-saved pool.
#[cfg(test)]
const DENSE_LEAF_IR: &str = "fn @dense(i64 %s) -> i64 {\nentry:\n  %a1 = mul.i64 %s, 3\n  %a2 = mul.i64 %s, 5\n  %a3 = mul.i64 %s, 7\n  %a4 = mul.i64 %s, 11\n  %a5 = mul.i64 %s, 13\n  %a6 = mul.i64 %s, 17\n  %a7 = mul.i64 %s, 19\n  %a8 = mul.i64 %s, 23\n  %b1 = xor.i64 %a1, %a8\n  %b2 = xor.i64 %a2, %a7\n  %b3 = xor.i64 %a3, %a6\n  %b4 = xor.i64 %a4, %a5\n  %c1 = add.i64 %b1, %b2\n  %c2 = add.i64 %b3, %b4\n  %c3 = add.i64 %c1, %c2\n  %c4 = add.i64 %c3, %a1\n  %c5 = add.i64 %c4, %a8\n  ret.i64 %c5\n}\n";

#[test]
fn callee_saved_registers_are_saved_restored_and_described() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(DENSE_LEAF_IR)?;
    let mir = from_ir_x86_64(&module, "dense_leaf").expect("lower dense leaf");
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_aarch64_with_units_and_settings(
            &mir,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &settings,
            PointerWidth::Bits64,
        )?;
        let assembly = String::from_utf8(output)?;
        let (prologue, rest) = assembly
            .split_once(".L_dense.entry:")
            .ok_or("missing entry")?;
        let epilogue = rest.split_once(".Lret_dense:").ok_or("missing epilogue")?.1;

        let used: Vec<usize> = (21..=28)
            .filter(|index| rest.contains(&format!("x{index},")))
            .collect();
        assert!(
            !used.is_empty(),
            "{regalloc:?} spilled everything: {assembly}"
        );
        for index in used {
            let saved = prologue.lines().any(|line| {
                let line = line.trim();
                (line.starts_with("stp") || line.starts_with("stur") || line.starts_with("str"))
                    && line.contains(&format!("x{index},"))
            });
            assert!(saved, "x{index} is not saved: {assembly}");
            let restored = epilogue.lines().any(|line| {
                let line = line.trim();
                (line.starts_with("ldp") || line.starts_with("ldur") || line.starts_with("ldr"))
                    && line.contains(&format!("x{index},"))
            });
            assert!(restored, "x{index} is not restored: {assembly}");
            assert!(
                prologue.contains(&format!(".cfi_offset {index}, ")),
                "x{index} has no unwind rule: {assembly}"
            );
            assert!(
                epilogue.contains(&format!(".cfi_restore {index}")),
                "x{index} is never unwound back: {assembly}"
            );
        }
        for reserved in [16, 17, 18, 19, 20] {
            assert!(
                !prologue.contains(&format!(".cfi_offset {reserved}, ")),
                "x{reserved} must not be allocated: {assembly}"
            );
        }
    }
    Ok(())
}

#[test]
fn darwin_saves_callee_saved_registers_without_cfi() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
    let module = parse_module(DENSE_LEAF_IR)?;
    let mir = from_ir_x86_64(&module, "dense_leaf").expect("lower dense leaf");
    let settings = MirCodegenSettings {
        regalloc: RegallocStrategy::LinearScanGlobal,
        ..Default::default()
    };
    let mut output = Vec::new();
    generate_mir_aarch64_with_units_and_settings(
        &mir,
        &mut output,
        TargetOperatingSystem::MacOS,
        1,
        &settings,
        PointerWidth::Bits64,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains(".cfi_"), "{assembly}");
    let (prologue, rest) = assembly
        .split_once(".L_dense.entry:")
        .ok_or("missing entry")?;
    let epilogue = rest.split_once(".Lret_dense:").ok_or("missing epilogue")?.1;
    for index in 21..=28 {
        if rest.contains(&format!("x{index},")) {
            assert!(prologue.contains(&format!("x{index},")), "{assembly}");
            assert!(epilogue.contains(&format!("x{index},")), "{assembly}");
        }
    }
    Ok(())
}

#[test]
fn tail_calls_stay_off_callee_saved_registers() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    let x = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("f")
            .with_params(vec![Parameter::new(
                x.clone(),
                MirType::Scalar(ScalarType::I64),
            )])
            .with_return(MirType::Scalar(ScalarType::I64)),
    );
    let mut block = Block::new("entry");
    let mut live = Vec::new();
    for id in 1..=12 {
        let dst = Register::Virtual(VirtualReg::gpr(id));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: Operand::Register(x.clone()),
            rhs: Operand::Immediate(Immediate::I64(i64::from(id) * 2 + 1)),
        });
        live.push(dst);
    }
    let mut acc = Operand::Register(live[0].clone());
    for (step, reg) in live.iter().enumerate().skip(1) {
        let dst = Register::Virtual(VirtualReg::gpr(100 + step as u32));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Xor,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: acc,
            rhs: Operand::Register(reg.clone()),
        });
        acc = Operand::Register(dst);
    }
    block.push(MirInst::TailCall {
        name: "g".into(),
        args: vec![acc],
    });
    func.add_block(block);
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc,
            ..Default::default()
        };
        let asm = String::from_utf8(
            compile_single_function_aarch64(
                "f",
                &func,
                TargetOperatingSystem::Linux,
                &settings,
                &AggregateSignatures::empty(PointerWidth::Bits64),
            )
            .map_err(|error| error.to_string())?,
        )?;
        for index in 19..=28 {
            assert!(
                !asm.contains(&format!("x{index},")) && !asm.contains(&format!("x{index}]")),
                "x{index} survives no tail call: {asm}"
            );
        }
    }
    Ok(())
}

/// Single block: twelve values defined, a call, then every value used again.
#[cfg(test)]
fn function_with_values_live_across_a_call() -> Function {
    use crate::mir::{Block, Parameter};
    let x = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("across")
            .with_params(vec![Parameter::new(
                x.clone(),
                MirType::Scalar(ScalarType::I64),
            )])
            .with_return(MirType::Scalar(ScalarType::I64)),
    );
    let mut block = Block::new("entry");
    let mut live = Vec::new();
    for id in 1..=12 {
        let dst = Register::Virtual(VirtualReg::gpr(id));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: Operand::Register(x.clone()),
            rhs: Operand::Immediate(Immediate::I64(i64::from(id) * 2 + 1)),
        });
        live.push(dst);
    }
    block.push(MirInst::Call {
        name: "g".into(),
        args: vec![Operand::Immediate(Immediate::I64(1))],
        ret: Some(Register::Virtual(VirtualReg::gpr(50))),
    });
    let mut acc = Operand::Register(Register::Virtual(VirtualReg::gpr(50)));
    for (step, reg) in live.iter().enumerate() {
        let dst = Register::Virtual(VirtualReg::gpr(100 + step as u32));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Xor,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: acc,
            rhs: Operand::Register(reg.clone()),
        });
        acc = Operand::Register(dst);
    }
    block.push(MirInst::Ret { value: Some(acc) });
    func.add_block(block);
    func
}

#[cfg(test)]
fn compile_across(regalloc: RegallocStrategy) -> Result<String, Box<dyn Error>> {
    let settings = MirCodegenSettings {
        regalloc,
        ..Default::default()
    };
    Ok(String::from_utf8(
        compile_single_function_aarch64(
            "across",
            &function_with_values_live_across_a_call(),
            TargetOperatingSystem::Linux,
            &settings,
            &AggregateSignatures::empty(PointerWidth::Bits64),
        )
        .map_err(|error| error.to_string())?,
    )?)
}

#[test]
fn a_calling_function_still_draws_on_callee_saved_registers() -> Result<(), Box<dyn Error>> {
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let asm = compile_across(regalloc)?;
        assert!(asm.contains("    bl g"), "{asm}");
        let (prologue, rest) = asm.split_once(".L_across.entry:").ok_or("missing entry")?;
        let epilogue = rest
            .split_once(".Lret_across:")
            .ok_or("missing epilogue")?
            .1;
        let used: Vec<usize> = (21..=28)
            .filter(|index| rest.contains(&format!("x{index},")))
            .collect();
        assert!(!used.is_empty(), "{regalloc:?} planned nothing: {asm}");
        for index in used {
            assert!(prologue.contains(&format!("x{index},")), "{asm}");
            assert!(epilogue.contains(&format!("x{index},")), "{asm}");
        }
    }
    Ok(())
}

#[test]
fn no_caller_saved_register_is_read_back_after_a_call() -> Result<(), Box<dyn Error>> {
    for regalloc in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let asm = compile_across(regalloc)?;
        if let Some(reason) = caller_saved_read_back_after_a_call(&asm) {
            panic!("{regalloc:?}: {reason}: {asm}");
        }
    }
    Ok(())
}
