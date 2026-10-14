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
    writeln!(w, "    li a1, {offset}\n    add a1, fp, a1")?;
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
