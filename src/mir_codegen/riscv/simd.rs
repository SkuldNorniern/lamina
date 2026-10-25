use super::*;
use crate::mir::{Instruction, VectorLane, VectorOp};
use crate::mir_codegen::simd::{index, lane, scalar_bits, unsupported, width};

fn slot(reg: &Register, slots: &HashMap<VirtualReg, i32>) -> Result<i32, LaminaError> {
    if let Register::Virtual(v) = reg
        && let Some(off) = slots.get(v)
    {
        return Ok(*off);
    }
    Err(unsupported("vector requires virtual register stack slot"))
}

fn operand_slot(operand: &Operand, slots: &HashMap<VirtualReg, i32>) -> Result<i32, LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    slot(reg, slots)
}

fn address(w: &mut impl Write, offset: i32) -> Result<(), LaminaError> {
    RiscVFrame::fp_address(w, "a1", offset)?;
    Ok(())
}

fn load_vector(
    w: &mut impl Write,
    operand: &Operand,
    v: &str,
    slots: &HashMap<VirtualReg, i32>,
    bits: usize,
) -> Result<(), LaminaError> {
    address(w, operand_slot(operand, slots)?)?;
    writeln!(w, "    vle{bits}.v {v}, (a1)")?;
    Ok(())
}

pub(super) fn emit(
    inst: &Instruction,
    w: &mut impl Write,
    ra: &RiscVRegAlloc,
    slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
) -> Result<(), LaminaError> {
    let ty = match inst {
        Instruction::Load { ty, .. }
        | Instruction::Store { ty, .. }
        | Instruction::VectorOp { ty, .. } => *ty,
        _ => unreachable!(),
    };
    let lane = lane(ty)?;
    target.require_vector().map_err(unsupported)?;
    if target.xlen != Xlen::Rv64 {
        return Err(unsupported("RVV V128 lowering requires RV64"));
    }
    let bits = width(lane);
    let lanes = 128 / bits;
    writeln!(w, "    vsetivli zero, {lanes}, e{bits}, m1, tu, mu")?;
    match inst {
        Instruction::Load { dst, addr, .. }
        | Instruction::Store {
            src: Operand::Register(dst),
            addr,
            ..
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported("RVV vector memory requires base and offset"));
            };
            load_operand_to_register(&Operand::reg(base.clone()), w, ra, slots, "t0")?;
            writeln!(w, "    li a1, {offset}\n    add t0, t0, a1")?;
            if matches!(inst, Instruction::Load { .. }) {
                writeln!(w, "    vle{bits}.v v0, (t0)")?;
                address(w, slot(dst, slots)?)?;
                writeln!(w, "    vse{bits}.v v0, (a1)")?;
            } else {
                address(w, slot(dst, slots)?)?;
                writeln!(w, "    vle{bits}.v v0, (a1)\n    vse{bits}.v v0, (t0)")?;
            }
        }
        Instruction::VectorOp {
            op, dst, operands, ..
        } => {
            use VectorOp::*;
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            if float && matches!(op, VShl | VLShr | VAShr) {
                return Err(unsupported(format!(
                    "RISC-V unsupported {op:?} for {lane:?}"
                )));
            }
            if float && matches!(op, VAdd | VSub | VMul) {
                target.require_float(bits == 32).map_err(unsupported)?;
            }
            if matches!(op, VExtractLane | VInsertLane) {
                let i = index(*op, lane, operands)?;
                load_vector(w, &operands[0], "v0", slots, bits)?;
                if *op == VExtractLane {
                    writeln!(w, "    vslidedown.vi v1, v0, {i}\n    vmv.x.s t0, v1")?;
                    if bits < 64 {
                        writeln!(
                            w,
                            "    slli t0, t0, {}\n    srli t0, t0, {}",
                            64 - bits,
                            64 - bits
                        )?;
                    }
                    let Register::Virtual(v) = dst else {
                        return Err(unsupported("RVV extract requires virtual destination"));
                    };
                    store_register_to_register("t0", v, w, ra, slots)?;
                    if float && ra.get_mapping(v).is_some() {
                        address(w, slot(dst, slots)?)?;
                        writeln!(w, "    sd t0, 0(a1)")?;
                    }
                    return Ok(());
                }
                load_operand_to_register(&scalar_bits(&operands[1]), w, ra, slots, "t0")?;
                writeln!(
                    w,
                    "    vmv.v.x v1, t0\n    vsetivli zero, {}, e{bits}, m1, tu, mu\n    vslideup.vi v0, v1, {i}\n    vsetivli zero, {lanes}, e{bits}, m1, tu, mu",
                    i + 1
                )?;
            } else if *op == VSplat {
                if operands.len() != 1 {
                    return Err(unsupported("VSplat requires one operand"));
                }
                load_operand_to_register(&scalar_bits(&operands[0]), w, ra, slots, "t0")?;
                writeln!(w, "    vmv.v.x v0, t0")?;
            } else {
                if operands.len() != 2 {
                    return Err(unsupported(format!("{op:?} requires two operands")));
                }
                load_vector(w, &operands[0], "v0", slots, bits)?;
                if matches!(op, VShl | VLShr | VAShr) {
                    address(w, operand_slot(&operands[1], slots)?)?;
                    writeln!(
                        w,
                        "    ld t0, 0(a1)\n    li a1, {bits}\n    bltu t0, a1, 1f"
                    )?;
                    if *op == VAShr {
                        writeln!(w, "    li t0, {}", bits - 1)?;
                    } else {
                        writeln!(w, "    vmv.v.i v0, 0\n    j 2f")?;
                    }
                    let mnemonic = match op {
                        VShl => "vsll.vx",
                        VLShr => "vsrl.vx",
                        _ => "vsra.vx",
                    };
                    writeln!(w, "1:\n    {mnemonic} v0, v0, t0\n2:")?;
                } else {
                    load_vector(w, &operands[1], "v1", slots, bits)?;
                    let mnemonic = match (op, float) {
                        (VAdd, false) => "vadd.vv",
                        (VSub, false) => "vsub.vv",
                        (VMul, false) => "vmul.vv",
                        (VAdd, true) => "vfadd.vv",
                        (VSub, true) => "vfsub.vv",
                        (VMul, true) => "vfmul.vv",
                        (VAnd, _) => "vand.vv",
                        (VOr, _) => "vor.vv",
                        (VXor, _) => "vxor.vv",
                        _ => {
                            return Err(unsupported(format!(
                                "RISC-V unsupported {op:?} for {lane:?}"
                            )));
                        }
                    };
                    writeln!(w, "    {mnemonic} v0, v0, v1")?;
                }
            }
            address(w, slot(dst, slots)?)?;
            writeln!(w, "    vse{bits}.v v0, (a1)")?;
        }
        _ => return Err(unsupported("RVV vector store requires register")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::mir::{VectorLane, VectorOp};
    use crate::mir_codegen::simd_support::*;

    #[test]
    fn vector_slot_address_beyond_the_immediate_range_avoids_li() {
        let mut out = Vec::new();
        address(&mut out, -40016).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(
            text,
            "    li a1, -10\n    slli a1, a1, 12\n    addi a1, a1, 944\n    add a1, fp, a1\n"
        );
    }

    #[test]
    fn rvv_assembles_every_supported_pair() {
        native(Backend::Rvv, |op, lane, index| {
            use VectorOp::*;
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            match op {
                VAdd | VSub | VMul => format!(
                    "v{}{}.vv v0, v0, v1",
                    if float { "f" } else { "" },
                    match op {
                        VAdd => "add",
                        VSub => "sub",
                        _ => "mul",
                    }
                ),
                VAnd | VOr | VXor => format!(
                    "v{}.vv v0, v0, v1",
                    match op {
                        VAnd => "and",
                        VOr => "or",
                        _ => "xor",
                    }
                ),
                VShl | VLShr | VAShr => format!(
                    "v{}.vx v0, v0, t0",
                    match op {
                        VShl => "sll",
                        VLShr => "srl",
                        _ => "sra",
                    }
                ),
                VSplat => "vmv.v.x v0, t0".into(),
                VExtractLane => format!("vslidedown.vi v1, v0, {index}"),
                VInsertLane => format!("vslideup.vi v0, v1, {index}"),
            }
        });
    }

    #[test]
    fn rvv_rejects_unsupported_pairs_and_missing_extension() {
        negative(Backend::Rvv);
    }

    #[test]
    fn rvv_checks_float_extensions_and_register_width() {
        use crate::{
            mir::{Instruction, MirType, Module, VectorType},
            mir_codegen::{MirCodegenSettings, riscv::generate_mir_riscv_with_units_and_settings},
        };
        use lamina_codegen::riscv::{RiscVTarget, Xlen};
        use lamina_platform::TargetOperatingSystem;
        for (lane, extension) in [(VectorLane::F32, "F"), (VectorLane::F64, "D")] {
            let mut module = Module::new("float");
            module.add_function(function(
                "bad",
                MirType::Vector(VectorType::V128(lane)),
                VectorOp::VAdd,
                0,
            ));
            let mut output = Vec::new();
            let error = generate_mir_riscv_with_units_and_settings(
                &module,
                &mut output,
                RiscVTarget {
                    v: true,
                    ..RiscVTarget::base(Xlen::Rv64)
                },
                TargetOperatingSystem::Linux,
                1,
                &MirCodegenSettings::default(),
            )
            .expect_err("requires float extension");
            assert!(
                error
                    .to_string()
                    .contains(&format!("{extension} extension")),
                "{error}"
            );
            assert!(output.is_empty());
        }
        for lane in LANES {
            let mut module = Module::new("rv32");
            let mut f = function(
                "bad",
                MirType::Vector(VectorType::V128(lane)),
                VectorOp::VAdd,
                0,
            );
            f.blocks[0]
                .instructions
                .retain(|i| matches!(i, Instruction::VectorOp { .. } | Instruction::Ret { .. }));
            module.add_function(f);
            let mut output = Vec::new();
            let error = generate_mir_riscv_with_units_and_settings(
                &module,
                &mut output,
                RiscVTarget {
                    v: true,
                    ..RiscVTarget::general(Xlen::Rv32)
                },
                TargetOperatingSystem::Linux,
                1,
                &MirCodegenSettings::default(),
            )
            .expect_err("requires RV64");
            assert!(error.to_string().contains("RV64"), "{error}");
            assert!(output.is_empty());
        }
    }
}
