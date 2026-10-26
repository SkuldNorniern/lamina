use super::*;
use crate::mir::FunctionBuilder;
use crate::mir_codegen::test_support::cross_assembler_ready;
use std::error::Error;

/// Replays the `li`, `slli`, `addi` and `add` lines that build the `t0` and `a0`
/// values. Returns what the prologue does to `sp`, and each address that is built
/// from `fp` or offset from an address already in `a0`.
fn riscv_frame_arithmetic(asm: &str) -> (i64, Vec<i64>) {
    let (mut t0, mut a0, mut sp_delta, mut addresses) = (0i64, 0i64, 0i64, Vec::new());
    let mut in_prologue = true;
    for line in asm.lines().map(str::trim) {
        if line.starts_with(".L_reserved.") {
            in_prologue = false;
        }
        let (op, args) = line.split_once(' ').unwrap_or((line, ""));
        let parts: Vec<&str> = args.split(", ").collect();
        match (op, parts.as_slice()) {
            ("li", [r, v]) => {
                *(if *r == "t0" { &mut t0 } else { &mut a0 }) = v.parse().unwrap_or(0)
            }
            ("slli", [r, _, k]) => {
                let v = if *r == "t0" { &mut t0 } else { &mut a0 };
                *v = v.wrapping_shl(k.parse().unwrap_or(0));
            }
            ("addi", ["sp", "sp", v]) if in_prologue => sp_delta += v.parse::<i64>().unwrap_or(0),
            ("addi", [r, _, v]) => {
                let v_ref = if *r == "t0" { &mut t0 } else { &mut a0 };
                *v_ref = v_ref.wrapping_add(v.parse().unwrap_or(0));
            }
            ("add", ["sp", "sp", "t0"]) if in_prologue => sp_delta += t0,
            ("add", ["a0", "fp", "a0"]) => addresses.push(a0),
            ("add", ["a0", "a0", "t0"]) => {
                a0 = a0.wrapping_add(t0);
                addresses.push(a0);
            }
            _ => {}
        }
    }
    (sp_delta, addresses)
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
    let mut output = Vec::new();
    generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    let (sp_delta, addresses) = riscv_frame_arithmetic(&assembly);
    assert_eq!(sp_delta, -(16 + 40000 + 16), "{assembly}");
    assert_eq!(addresses, [-(16 + 40000 + 8)], "{assembly}");
    Ok(())
}

#[test]
fn lea_offset_beyond_the_immediate_range_reaches_the_element() -> Result<(), Box<dyn Error>> {
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
    generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(!assembly.contains("addi a0, a0, 39999"), "{assembly}");
    let (_, addresses) = riscv_frame_arithmetic(&assembly);
    assert_eq!(
        addresses,
        [-(16 + 40000 + 16), -(16 + 40000 + 16) + 39999],
        "{assembly}"
    );
    Ok(())
}

#[test]
fn rv32_scalar_output_assembles() -> Result<(), Box<dyn Error>> {
    use std::{
        env::temp_dir,
        fs,
        process::{self, Command},
    };
    let directory = temp_dir().join(format!("lamina-rv32-scalar-{}", process::id()));
    fs::create_dir_all(&directory)?;
    let reg = |id| Register::Virtual(VirtualReg::gpr(id));
    let operand = |id| Operand::Register(reg(id));
    let addr = AddressMode::BaseOffset {
        base: reg(0),
        offset: 0,
        address_space: Default::default(),
    };
    let mut module = MirModule::new("rv32_scalar");
    for ty in [MirType::f32(), MirType::f64()] {
        let immediate = |v: f64| {
            Operand::Immediate(if ty == MirType::f32() {
                Immediate::F32(v as f32)
            } else {
                Immediate::F64(v)
            })
        };
        let mut function = FunctionBuilder::new(format!("float_{}", ty.size_bytes()))
            .param(reg(0), MirType::ptr())
            .returns(MirType::i32())
            .block("entry")
            .build();
        let instructions = &mut function.blocks[0].instructions;
        instructions.push(MirInst::Copy {
            ty,
            dst: reg(1),
            src: immediate(-1.23456789),
        });
        instructions.push(MirInst::Copy {
            ty,
            dst: reg(2),
            src: operand(1),
        });
        for (id, op) in [
            FloatBinOp::FAdd,
            FloatBinOp::FSub,
            FloatBinOp::FMul,
            FloatBinOp::FDiv,
        ]
        .into_iter()
        .enumerate()
        {
            instructions.push(MirInst::FloatBinary {
                op,
                ty,
                dst: reg(id as u32 + 3),
                lhs: operand(2),
                rhs: immediate(3.25),
            });
        }
        for (id, op) in [FloatUnOp::FNeg, FloatUnOp::FSqrt].into_iter().enumerate() {
            instructions.push(MirInst::FloatUnary {
                op,
                ty,
                dst: reg(id as u32 + 7),
                src: operand(3),
            });
        }
        instructions.push(MirInst::Store {
            ty,
            src: operand(7),
            addr: addr.clone(),
            attrs: Default::default(),
        });
        instructions.push(MirInst::Load {
            ty,
            dst: reg(9),
            addr: addr.clone(),
            attrs: Default::default(),
        });
        for (id, op) in [
            FloatCmpOp::Eq,
            FloatCmpOp::Ne,
            FloatCmpOp::Lt,
            FloatCmpOp::Le,
            FloatCmpOp::Gt,
            FloatCmpOp::Ge,
        ]
        .into_iter()
        .enumerate()
        {
            instructions.push(MirInst::FloatCmp {
                op,
                ty,
                dst: reg(id as u32 + 10),
                lhs: operand(9),
                rhs: immediate(-2.5),
            });
        }
        instructions.push(MirInst::Ret {
            value: Some(Operand::Immediate(Immediate::I32(0))),
        });
        module.add_function(function);
    }
    let mut function = FunctionBuilder::new("integers")
        .param(reg(0), MirType::ptr())
        .param(reg(1), MirType::i32())
        .returns(MirType::i32())
        .block("entry")
        .build();
    let instructions = &mut function.blocks[0].instructions;
    instructions.push(MirInst::Load {
        ty: MirType::ptr(),
        dst: reg(2),
        addr: addr.clone(),
        attrs: Default::default(),
    });
    instructions.push(MirInst::Copy {
        ty: MirType::ptr(),
        dst: reg(3),
        src: operand(2),
    });
    instructions.push(MirInst::Store {
        ty: MirType::ptr(),
        src: operand(3),
        addr: addr.clone(),
        attrs: Default::default(),
    });
    let mut id = 4;
    for ty in [MirType::i8(), MirType::i16(), MirType::i32()] {
        let input = id;
        instructions.push(MirInst::IntConvert {
            from: MirType::i32(),
            to: ty,
            signed: false,
            dst: reg(input),
            src: operand(1),
        });
        id += 1;
        for op in [
            IntBinOp::Add,
            IntBinOp::Sub,
            IntBinOp::Mul,
            IntBinOp::SDiv,
            IntBinOp::UDiv,
            IntBinOp::SRem,
            IntBinOp::URem,
            IntBinOp::And,
            IntBinOp::Or,
            IntBinOp::Xor,
            IntBinOp::Shl,
            IntBinOp::LShr,
            IntBinOp::AShr,
        ] {
            instructions.push(MirInst::IntBinary {
                op,
                ty,
                dst: reg(id),
                lhs: operand(input),
                rhs: Operand::Immediate(Immediate::I32(2)),
            });
            id += 1;
        }
    }
    for ty in [MirType::i8(), MirType::i16()] {
        instructions.push(MirInst::IntConvert {
            from: MirType::i32(),
            to: ty,
            signed: false,
            dst: reg(id),
            src: operand(1),
        });
        let truncated = id;
        for signed in [false, true] {
            instructions.push(MirInst::IntConvert {
                from: ty,
                to: MirType::i32(),
                signed,
                dst: reg(id + 1),
                src: operand(truncated),
            });
            id += 2;
        }
    }
    instructions.push(MirInst::Ret {
        value: Some(Operand::Immediate(Immediate::I32(0))),
    });
    for block in &mut function.blocks {
        for instruction in &mut block.instructions {
            instruction.normalize_integer_immediates(function.sig.ret_ty);
        }
    }
    module.add_function(function);
    if !cross_assembler_ready(
        &directory,
        &["--target=riscv32-unknown-elf", "-march=rv32imafd"],
    ) {
        fs::remove_dir_all(&directory)?;
        return Ok(());
    }
    for allocator in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let mut output = Vec::new();
        generate_mir_riscv_with_units_and_settings(
            &module,
            &mut output,
            RiscVTarget::general(Xlen::Rv32),
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings {
                regalloc: allocator,
                ..Default::default()
            },
        )?;
        let source = directory.join(format!("{allocator:?}.s"));
        let object = source.with_extension("o");
        fs::write(&source, output)?;
        let output = Command::new("clang")
            .args(["--target=riscv32-unknown-elf", "-march=rv32imafd", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .output()?;
        assert!(
            output.status.success(),
            "{allocator:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new("llvm-objdump-18")
            .arg("-d")
            .arg(&object)
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn narrow_loads_sign_extend_on_both_register_widths() -> Result<(), Box<dyn Error>> {
    for xlen in [Xlen::Rv32, Xlen::Rv64] {
        for (ty, mnemonic) in [
            (MirType::i8(), "lb"),
            (MirType::i16(), "lh"),
            (MirType::i32(), "lw"),
        ] {
            let base = Register::Virtual(VirtualReg::gpr(0));
            let dst = Register::Virtual(VirtualReg::gpr(1));
            let mut module = MirModule::new("loads");
            module.add_function(
                FunctionBuilder::new("load_bits")
                    .param(base.clone(), MirType::ptr())
                    .returns(ty)
                    .block("entry")
                    .instr(MirInst::Load {
                        ty,
                        dst: dst.clone(),
                        addr: AddressMode::BaseOffset {
                            base,
                            offset: 0,
                            address_space: Default::default(),
                        },
                        attrs: Default::default(),
                    })
                    .instr(MirInst::Ret {
                        value: Some(Operand::Register(dst)),
                    })
                    .build(),
            );
            let mut output = Vec::new();
            generate_mir_riscv_with_units_and_settings(
                &module,
                &mut output,
                RiscVTarget::general(xlen),
                TargetOperatingSystem::Linux,
                1,
                &MirCodegenSettings::default(),
            )?;
            let assembly = String::from_utf8(output)?;
            assert!(
                assembly.contains(&format!("    {mnemonic} a0, 0(t0)")),
                "{assembly}"
            );
            assert!(!assembly.contains("    lwu ") || xlen == Xlen::Rv64);
        }
    }
    Ok(())
}

#[test]
fn narrow_unsigned_compares_zero_extend_both_operands() -> Result<(), Box<dyn Error>> {
    let ty = MirType::Scalar(ScalarType::I8);
    let a = Register::Virtual(VirtualReg::gpr(0));
    let b = Register::Virtual(VirtualReg::gpr(1));
    let dst = Register::Virtual(VirtualReg::gpr(2));
    let mut module = MirModule::new("unsigned_cmp");
    module.add_function(
        FunctionBuilder::new("below")
            .param(a.clone(), ty)
            .param(b.clone(), ty)
            .returns(MirType::Scalar(ScalarType::I1))
            .block("entry")
            .instr(MirInst::IntCmp {
                op: IntCmpOp::ULt,
                dst: dst.clone(),
                lhs: Operand::Register(a),
                rhs: Operand::Register(b),
                ty,
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build(),
    );
    let mut output = Vec::new();
    generate_mir_riscv_with_units_and_settings(
        &module,
        &mut output,
        RiscVTarget::general(Xlen::Rv64),
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    srli a0, a0, 56\n"), "{assembly}");
    assert!(assembly.contains("    srli a1, a1, 56\n"), "{assembly}");
    assert!(assembly.contains("    sltu "), "{assembly}");
    Ok(())
}

#[test]
fn rv32_uses_word_memory_ops_not_doubleword() {
    let ty = MirType::Scalar(ScalarType::I32);
    let a = Register::Virtual(VirtualReg::gpr(0));
    let b = Register::Virtual(VirtualReg::gpr(1));
    let build = || {
        let mut module = MirModule::new("width_test");
        module.add_function(
            FunctionBuilder::new("addup")
                .param(a.clone(), ty)
                .returns(ty)
                .block("entry")
                .instr(MirInst::IntBinary {
                    op: IntBinOp::Add,
                    ty,
                    dst: b.clone(),
                    lhs: Operand::Register(a.clone()),
                    rhs: Operand::Immediate(Immediate::I32(1)),
                })
                .instr(MirInst::Ret {
                    value: Some(Operand::Register(b.clone())),
                })
                .build(),
        );
        module
    };

    let emit = |target| {
        let mut out = Vec::new();
        generate_mir_riscv_with_units_and_settings(
            &build(),
            &mut out,
            target,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )
        .expect("codegen");
        String::from_utf8(out).expect("utf8")
    };

    let rv32 = emit(RiscVTarget::general(Xlen::Rv32));
    assert!(!rv32.contains("    sd "), "rv32 emitted sd:\n{rv32}");
    assert!(!rv32.contains("    ld "), "rv32 emitted ld:\n{rv32}");
    assert!(rv32.contains("    sw "), "rv32 emitted no sw:\n{rv32}");

    let rv64 = emit(RiscVTarget::general(Xlen::Rv64));
    assert!(rv64.contains("    sd "), "rv64 lost sd:\n{rv64}");
}

#[test]
fn rv32_rejects_i64_instead_of_truncating_it() {
    let ty = MirType::Scalar(ScalarType::I64);
    let dst = Register::Virtual(VirtualReg::gpr(0));
    let function = FunctionBuilder::new("wide")
        .returns(ty)
        .block("entry")
        .instr(MirInst::Copy {
            ty,
            dst: dst.clone(),
            src: Operand::Immediate(Immediate::I64(1_i64 << 40)),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(dst)),
        })
        .build();
    let mut module = MirModule::new("wide_test");
    module.add_function(function);
    let mut output = Vec::new();
    let result = generate_mir_riscv_with_units_and_settings(
        &module,
        &mut output,
        RiscVTarget::general(Xlen::Rv32),
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    );
    assert!(result.is_err(), "rv32 i64 must be rejected");
    let Err(error) = result else {
        return;
    };
    let message = error.to_string();
    assert!(
        message.contains("i64"),
        "message did not name type: {message}"
    );
    assert!(
        message.contains("rv32imafd"),
        "message did not name target: {message}"
    );
}

#[test]
fn caller_saved_value_live_across_call_is_spilled_and_reloaded() -> Result<(), Box<dyn Error>> {
    let ty = MirType::Scalar(ScalarType::I64);
    let live = Register::Virtual(VirtualReg::gpr(0));
    let result = Register::Virtual(VirtualReg::gpr(1));
    let caller = FunctionBuilder::new("caller")
        .returns(ty)
        .block("entry")
        .instr(MirInst::Copy {
            ty,
            dst: live.clone(),
            src: Operand::Immediate(Immediate::I64(41)),
        })
        .instr(MirInst::Call {
            name: "callee".to_string(),
            args: Vec::new(),
            ret: None,
        })
        .instr(MirInst::IntBinary {
            op: IntBinOp::Add,
            ty,
            dst: result.clone(),
            lhs: Operand::Register(live),
            rhs: Operand::Immediate(Immediate::I64(1)),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(result)),
        })
        .build();
    let callee = FunctionBuilder::new("callee")
        .block("entry")
        .instr(MirInst::Ret { value: None })
        .build();
    let mut module = MirModule::new("call_preservation");
    module.add_function(caller);
    module.add_function(callee);
    let settings = MirCodegenSettings {
        regalloc: RegallocStrategy::GraphColorGlobal,
        ..MirCodegenSettings::default()
    };
    let mut output = Vec::new();
    generate_mir_riscv_with_units_and_settings(
        &module,
        &mut output,
        RiscVTarget::general(Xlen::Rv64),
        TargetOperatingSystem::Linux,
        1,
        &settings,
    )?;
    let assembly = String::from_utf8(output)?;
    let caller_body = assembly.split("caller:").nth(1);
    assert!(caller_body.is_some(), "missing caller body in:\n{assembly}");
    let Some(caller_body) = caller_body else {
        return Ok(());
    };
    let call_position = caller_body.find("call callee");
    assert!(
        call_position.is_some(),
        "missing call instruction in:\n{caller_body}"
    );
    let Some(call_position) = call_position else {
        return Ok(());
    };
    let before_call = &caller_body[..call_position];
    let after_call = &caller_body[call_position..];
    assert!(
        before_call.contains("sd t1,"),
        "missing spill:\n{caller_body}"
    );
    assert!(
        after_call.contains("ld t1,"),
        "missing reload:\n{caller_body}"
    );
    Ok(())
}

#[test]
fn globally_allocated_callee_saved_register_is_saved_by_frame() -> Result<(), Box<dyn Error>> {
    let ty = MirType::Scalar(ScalarType::I64);
    let value = Register::Virtual(VirtualReg::gpr(0));
    let function = FunctionBuilder::new("uses_saved")
        .returns(ty)
        .block("entry")
        .instr(MirInst::Copy {
            ty,
            dst: value.clone(),
            src: Operand::Immediate(Immediate::I64(7)),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(value)),
        })
        .build();
    let mut module = MirModule::new("saved_register");
    module.add_function(function);
    let settings = MirCodegenSettings {
        regalloc: RegallocStrategy::LinearScanGlobal,
        ..MirCodegenSettings::default()
    };
    let mut output = Vec::new();
    generate_mir_riscv_with_units_and_settings(
        &module,
        &mut output,
        RiscVTarget::general(Xlen::Rv64),
        TargetOperatingSystem::Linux,
        1,
        &settings,
    )?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("sd s11,"), "missing save:\n{assembly}");
    assert!(assembly.contains("ld s11,"), "missing restore:\n{assembly}");
    Ok(())
}

#[test]
fn base_isa_rejects_multiply_instead_of_emitting_it() {
    let ty = MirType::Scalar(ScalarType::I64);
    let dst = Register::Virtual(VirtualReg::gpr(0));
    let function = FunctionBuilder::new("times")
        .returns(ty)
        .block("entry")
        .instr(MirInst::IntBinary {
            op: IntBinOp::Mul,
            ty,
            dst: dst.clone(),
            lhs: Operand::Immediate(Immediate::I64(6)),
            rhs: Operand::Immediate(Immediate::I64(7)),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(dst)),
        })
        .build();
    let mut module = MirModule::new("ext_test");
    module.add_function(function);

    let mut out = Vec::new();
    let err = generate_mir_riscv_with_units_and_settings(
        &module,
        &mut out,
        RiscVTarget::base(Xlen::Rv64),
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )
    .expect_err("rv64i has no mul");
    let msg = err.to_string();
    assert!(msg.contains("M extension"), "unhelpful message: {msg}");
    assert!(msg.contains("rv64i"), "message should name the ISA: {msg}");

    let mut out = Vec::new();
    generate_mir_riscv_with_units_and_settings(
        &module,
        &mut out,
        RiscVTarget::general(Xlen::Rv64),
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )
    .expect("rv64g has mul");
    assert!(String::from_utf8_lossy(&out).contains("mul "));
}

#[test]
fn module_globals_are_addressed_through_symbol_loads() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "global @counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 @counter\n  ret.i64 %a\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "globals").expect("lower global module");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    la a0, counter\n"), "{assembly}");
    assert!(
        assembly.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
        "{assembly}"
    );
    Ok(())
}

#[test]
fn floats_cross_the_boundary_in_fa_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @add(f64 %a, i64 %n, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  %w = sitofp.i64.f64 %n\n  %t = add.f64 %s, %w\n  ret.f64 %t\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(1.5, 7, 2.5)\n  %i = fptosi.f64.i64 %r\n  ret.i64 %i\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "floats").expect("lower floats");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // The banks run out separately, so the integer still takes a0 while the two
    // floats take fa0 and fa1.
    let callee = assembly.split("\nadd:\n").nth(1).ok_or("missing add")?;
    let prologue = callee
        .split(".L_add.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(prologue.contains("    fsd fa0, "), "{assembly}");
    assert!(prologue.contains("    fsd fa1, "), "{assembly}");
    assert!(prologue.contains("    sd a0, "), "{assembly}");

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("call add").next().ok_or("missing call")?;
    assert!(call.contains("    fmv.d.x fa0, "), "{assembly}");
    assert!(call.contains("    fmv.d.x fa1, "), "{assembly}");
    Ok(())
}

#[test]
fn floats_past_the_float_bank_take_integer_registers() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let params: Vec<String> = (0..10).map(|i| format!("f64 %p{i}")).collect();
    let sums: Vec<String> = (0..10)
        .map(|i| format!("  %w{i} = fptosi.f64.i64 %p{i}\n  %a{i} = add.i64 %acc, %w{i}\n  %acc = add.i64 %a{i}, 0"))
        .collect();
    let args: Vec<String> = (0..10).map(|i| format!("{i}.5")).collect();
    let ir = format!(
        "fn @many({}) -> i64 {{\nentry:\n  %acc = add.i64 0, 0\n{}\n  ret.i64 %acc\n}}\nfn @main() -> i64 {{\nentry:\n  %r = call @many({})\n  ret.i64 %r\n}}\n",
        params.join(", "),
        sums.join("\n"),
        args.join(", ")
    );
    let module = parse_module(&ir)?;
    let mir = from_ir(&module, "overflow").expect("lower overflowing floats");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // Eight floats fill fa0 to fa7; the ninth and tenth carry their bits in a0 and
    // a1 rather than going to the stack.
    let callee = assembly.split("\nmany:\n").nth(1).ok_or("missing many")?;
    let prologue = callee
        .split(".L_many.entry:")
        .next()
        .ok_or("missing entry")?;
    assert!(prologue.contains("    fsd fa7, "), "{assembly}");
    assert!(prologue.contains("    sd a0, "), "{assembly}");
    assert!(prologue.contains("    sd a1, "), "{assembly}");

    let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
    let call = caller.split("call many").next().ok_or("missing call")?;
    assert!(call.contains("    li a0, "), "{assembly}");
    assert!(call.contains("    li a1, "), "{assembly}");
    Ok(())
}

#[test]
fn conversions_saturate_clamp_and_zero_nan() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %f = add.f64 1.5, 0.0\n  %wide = fptosi.f64.i64 %f\n  %narrow = fptosi.f64.i8 %f\n  %unsigned = fptoui.f64.u32 %f\n  %back = sitofp.i64.f64 %wide\n  %from_u = uitofp.u32.f64 %unsigned\n  ret.i64 %wide\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "conversions").expect("lower conversions");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;

    // Every float to integer conversion masks NaN to zero after the saturating
    // convert, and a narrow destination is clamped to its own range.
    assert!(
        assembly.contains("    fcvt.l.d a0, fa0, rtz\n"),
        "{assembly}"
    );
    assert!(
        assembly.contains("    fcvt.lu.d a0, fa0, rtz\n"),
        "{assembly}"
    );
    assert!(assembly.contains("    feq.d t0, fa0, fa0\n"), "{assembly}");
    assert!(assembly.contains("    li t1, 127\n"), "{assembly}");
    assert!(assembly.contains("    li t1, -128\n"), "{assembly}");
    assert!(assembly.contains("    sltu t2, t1, a0\n"), "{assembly}");
    assert!(assembly.contains("    fcvt.d.l fa0, a0\n"), "{assembly}");
    assert!(assembly.contains("    fcvt.d.lu fa0, a0\n"), "{assembly}");
    Ok(())
}

#[test]
fn byte_io_goes_through_putchar_and_getchar() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i64 {\nentry:\n  %w = writebyte 72\n  %r = readbyte\n  ret.i64 %r\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "byte_io").expect("lower byte io");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    call putchar\n"), "{assembly}");
    assert!(assembly.contains("    call getchar\n"), "{assembly}");
    assert!(!assembly.contains("call writebyte"), "{assembly}");
    assert!(!assembly.contains("call readbyte"), "{assembly}");
    Ok(())
}

#[test]
fn writeptr_sends_the_pointee_bytes_through_write() -> Result<(), Box<dyn Error>> {
    use crate::{mir::codegen::from_ir, parser::parse_module};
    let ir = "fn @main() -> i32 {\nentry:\n  %p = alloc.stack i32\n  store.i32 %p, 72\n  %n = writeptr %p\n  ret.i32 0\n}\n";
    let module = parse_module(ir)?;
    let mir = from_ir(&module, "writeptr").expect("lower writeptr");
    let mut output = Vec::new();
    generate_mir_riscv(&mir, &mut output, TargetOperatingSystem::Linux)?;
    let assembly = String::from_utf8(output)?;
    assert!(assembly.contains("    li a2, 4\n"), "{assembly}");
    assert!(assembly.contains("    li a0, 1\n"), "{assembly}");
    assert!(assembly.contains("    call write\n"), "{assembly}");
    Ok(())
}

#[test]
fn block_labels_are_unique_across_functions() {
    // Both functions have a block called "entry".
    let ty = MirType::Scalar(ScalarType::I64);
    let mut module = MirModule::new("label_test");
    for name in ["first", "second"] {
        let dst = Register::Virtual(VirtualReg::gpr(0));
        module.add_function(
            FunctionBuilder::new(name)
                .returns(ty)
                .block("entry")
                .instr(MirInst::Copy {
                    ty,
                    dst: dst.clone(),
                    src: Operand::Immediate(Immediate::I64(1)),
                })
                .instr(MirInst::Ret {
                    value: Some(Operand::Register(dst)),
                })
                .build(),
        );
    }
    let mut output = Vec::new();
    generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
        .expect("RISC-V codegen should succeed");
    let asm = String::from_utf8(output).expect("assembly should be UTF-8");

    let mut labels: Vec<&str> = asm
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with(".L_") && l.ends_with(':'))
        .collect();
    let before = labels.len();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(before, labels.len(), "duplicate labels in:\n{asm}");
}

#[test]
fn incoming_arguments_are_stored_into_their_slots() {
    let ty = MirType::Scalar(ScalarType::I64);
    let p0 = Register::Virtual(VirtualReg::gpr(0));
    let p1 = Register::Virtual(VirtualReg::gpr(1));
    let sum = Register::Virtual(VirtualReg::gpr(2));
    let function = FunctionBuilder::new("add_two")
        .param(p0.clone(), ty)
        .param(p1.clone(), ty)
        .returns(ty)
        .block("entry")
        .instr(MirInst::IntBinary {
            op: IntBinOp::Add,
            ty,
            dst: sum.clone(),
            lhs: Operand::Register(p0),
            rhs: Operand::Register(p1),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(sum)),
        })
        .build();
    let mut module = MirModule::new("param_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
        .expect("RISC-V codegen should succeed");
    let asm = String::from_utf8(output).expect("assembly should be UTF-8");

    let body = asm.split("add_two:").nth(1).expect("function label");
    let prologue_end = body.find(".L_").unwrap_or(body.len());
    let prologue = &body[..prologue_end];
    assert!(
        prologue.contains("sd a0,") && prologue.contains("sd a1,"),
        "arguments were never stored, prologue was:\n{prologue}"
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
    generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
        .expect("RISC-V codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(assembly.contains("ld a0"), "expected bit load: {assembly}");
    assert!(assembly.contains("sd a0"), "expected bit store: {assembly}");
    assert!(
        !assembly.contains("fadd"),
        "unexpected float add: {assembly}"
    );
}
