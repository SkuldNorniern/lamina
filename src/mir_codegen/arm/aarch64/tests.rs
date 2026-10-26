use super::*;
use crate::mir::{FunctionBuilder, Immediate, ScalarType};
use std::error::Error;

#[test]
fn global_parameters_are_initialized_before_entry() -> Result<(), Box<dyn Error>> {
    use crate::mir::{Block, Parameter};
    let reg = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(
        Signature::new("identity")
            .with_params(vec![Parameter::new(reg.clone(), MirType::i64())])
            .with_return(MirType::i64()),
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
                &AggregateSignatures::default(),
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
        (MirType::i8(), "sxtb x0, w0"),
        (MirType::i16(), "sxth x0, w0"),
        (MirType::i32(), "sxtw x0, w0"),
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
                &AggregateSignatures::default(),
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
            &AggregateSignatures::default(),
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
    generate_mir_aarch64(&module, &mut output, TargetOperatingSystem::Linux)
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
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir =
        "fn @main() -> i64 {\nentry:\n  %v = add.i64 1, 2\n  call @print(%v)\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "print_test").expect("lower print module");
    for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::IOS] {
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, os)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("bl _printf\n"), "{os:?}: {assembly}");
        assert!(assembly.contains("_main:\n"), "{os:?}: {assembly}");
        assert!(!assembly.contains("bl printf\n"), "{os:?}: {assembly}");
    }
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("bl printf\n"), "{assembly}");
    Ok(())
}

#[test]
fn global_allocation_keeps_values_out_of_caller_saved_pool_across_calls()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @id(i64 %x) -> i64 {\nentry:\n  %r = mul.i64 %x, 1\n  ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n  %a = add.i64 7, 0\n  %b = mul.i64 %a, 3\n  %c = call @id(5)\n  %d = add.i64 %b, %c\n  ret.i64 %d\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "call_crossing").expect("lower call module");
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
        )?;
        let assembly = String::from_utf8(output)?;
        let (_, main) = assembly.split_once("\nmain:\n").ok_or("missing main")?;
        let main = main.split_once(".Lret_main:").ok_or("missing epilogue")?.0;
        assert!(main.contains("bl id"), "{assembly}");
        for reg in ["x13", "x14", "x15"] {
            assert!(!main.contains(reg), "{regalloc:?} uses {reg}: {assembly}");
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
            .with_params(vec![Parameter::new(x.clone(), MirType::i64())])
            .with_return(MirType::i64()),
    );
    let mut block = Block::new("entry");
    let mut prev = Operand::Register(x);
    for id in 1..=101 {
        let dst = Register::Virtual(VirtualReg::gpr(id));
        block.push(MirInst::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::i64(),
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
            &AggregateSignatures::default(),
        )
        .map_err(|error| error.to_string())?,
    )?;
    let reserved = asm
        .lines()
        .find_map(|line| line.strip_prefix("    sub sp, sp, #"))
        .ok_or("missing frame reservation")?;
    assert!(
        asm.contains(&format!(
            "    add sp, sp, #{reserved}\n    ldp x29, x30, [sp], #16\n    b g\n"
        )),
        "{asm}"
    );
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
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %x = add.i64 100000, 0\n  switch.i64 %x, default, [100000, hit], [-5, neg]\nhit:\n  ret.i64 1\nneg:\n  ret.i64 2\ndefault:\n  ret.i64 3\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "switch_range").expect("lower switch module");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains("mov x9, #100000"), "{assembly}");
    assert!(assembly.contains("movz x9, #34464, lsl #0"), "{assembly}");
    Ok(())
}

#[test]
fn integer_float_conversions_select_width_and_saturation_forms() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %a = sitofp.i8.f32 -3\n  %b = fptosi.f64.i8 2.5\n  %c = fptoui.f64.u64 1.0\n  %d = uitofp.u64.f64 7\n  ret.i64 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "conversions").expect("lower conversion module");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
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
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "global @counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 @counter\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "globals").expect("lower global module");
    let mut linux = Vec::new();
    generate_mir_aarch64(&mir, &mut linux, TargetOperatingSystem::Linux)?;
    let linux = String::from_utf8(linux)?;
    assert!(linux.contains("adrp x9, counter\n"), "{linux}");
    assert!(linux.contains("add x9, x9, :lo12:counter\n"), "{linux}");
    assert!(
        linux.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
        "{linux}"
    );
    let mut macho = Vec::new();
    generate_mir_aarch64(&mir, &mut macho, TargetOperatingSystem::MacOS)?;
    let macho = String::from_utf8(macho)?;
    assert!(macho.contains("adrp x9, _counter@PAGE\n"), "{macho}");
    assert!(macho.contains("add x9, x9, _counter@PAGEOFF\n"), "{macho}");
    Ok(())
}

#[test]
fn frames_beyond_twelve_bit_immediates_are_addressed_through_a_register()
-> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %arr = alloc.stack [5000 x i8]\n  %p = getelem.ptr %arr, 4999, i8\n  store.i8 %p, 42\n  %v = load.i8 %p\n  %w = zext.i8.i64 %v\n  ret.i64 %w\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "big_frame").expect("lower big frame module");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
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
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @sum(struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  %ap = getfieldptr %p, 0\n  %a = load.i64 %ap\n  ret.i64 %a\n}\nfn @wide(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  ret.i64 %a\n}\nfn @main() -> i64 {\nentry:\n  %q = alloc.stack struct { a: i64, b: i64 }\n  %r = call @sum(%q)\n  %s = call @wide(1, 2, 3, 4, 5, 6, 7, %q)\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "aggregates").expect("lower aggregates");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
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
    assert!(wide.contains("ldr x11, [x29, #16]"), "{assembly}");
    assert!(wide.contains("ldr x11, [x29, #24]"), "{assembly}");
    Ok(())
}

#[test]
fn floats_cross_the_boundary_in_v_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @add(f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  ret.f64 %s\n}\nfn @addf(f32 %a, i64 %n) -> f32 {\nentry:\n  %s = add.f32 %a, 1.0\n  ret.f32 %s\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(1.5, 2.5)\n  %i = fptosi.f64.i64 %r\n  ret.i64 %i\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "floats").expect("lower floats");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
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
fn integer_aggregate_returns_come_back_in_x0_and_x1() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n  %p = alloc.stack struct { a: i64, b: i64 }\n  %pa = getfieldptr %p, 0\n  store.i64 %pa, 7\n  ret.ptr %p\n}\nfn @main() -> i64 {\nentry:\n  %r = call @make()\n  %ra = getfieldptr %r, 0\n  %a = load.i64 %ra\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "returns").expect("lower aggregate return");
    let mut output = Vec::new();
    generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
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
