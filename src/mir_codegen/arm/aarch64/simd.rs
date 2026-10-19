use super::*;
use crate::mir::{Instruction, VectorLane, VectorOp};
use crate::mir_codegen::simd::{index, lane, scalar_bits, unsupported, width};

fn vector_slot(frame: &FrameMap, reg: &Register) -> Result<i32, LaminaError> {
    frame
        .slot_of(reg)
        .ok_or_else(|| unsupported("vector requires virtual register stack slot"))
}

fn load_vector(
    w: &mut impl Write,
    operand: &Operand,
    v: &str,
    frame: &FrameMap,
    scratch: &str,
) -> Result<(), LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    emit_frame_slot_address(w, scratch, vector_slot(frame, reg)?)?;
    writeln!(w, "    ldr {v}, [{scratch}]")?;
    Ok(())
}

pub(super) fn emit(
    inst: &Instruction,
    w: &mut impl Write,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    let a = ra
        .alloc_scratch()
        .ok_or_else(|| unsupported("NEON requires address scratch register"))?;
    let t = ra
        .alloc_scratch()
        .ok_or_else(|| unsupported("NEON requires scalar scratch register"))?;
    let result = emit_inner(inst, w, frame, ra, a, t);
    ra.free_scratch(t);
    ra.free_scratch(a);
    result
}

fn emit_inner(
    inst: &Instruction,
    w: &mut impl Write,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    a: &str,
    t: &str,
) -> Result<(), LaminaError> {
    match inst {
        Instruction::Load { ty, dst, addr, .. } => {
            lane(*ty)?;
            materialize_address(w, addr, a, frame, ra)?;
            writeln!(w, "    ldr q0, [{a}]")?;
            emit_frame_slot_address(w, a, vector_slot(frame, dst)?)?;
            writeln!(w, "    str q0, [{a}]")?;
        }
        Instruction::Store { ty, src, addr, .. } => {
            lane(*ty)?;
            load_vector(w, src, "q0", frame, a)?;
            materialize_address(w, addr, a, frame, ra)?;
            writeln!(w, "    str q0, [{a}]")?;
        }
        Instruction::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            use VectorOp::*;
            let lane = lane(*ty)?;
            let bits = width(lane);
            let (arr, element, scalar) = match bits {
                8 => ("16b", "b", w_alias(t)),
                16 => ("8h", "h", w_alias(t)),
                32 => ("4s", "s", w_alias(t)),
                _ => ("2d", "d", t.to_string()),
            };
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            if (*op == VMul && lane == VectorLane::I64)
                || (float && matches!(op, VShl | VLShr | VAShr))
            {
                return Err(unsupported(format!(
                    "AArch64 unsupported {op:?} for {lane:?}"
                )));
            }
            if matches!(op, VExtractLane | VInsertLane) {
                let i = index(*op, lane, operands)?;
                load_vector(w, &operands[0], "q0", frame, a)?;
                if *op == VExtractLane {
                    writeln!(w, "    umov {scalar}, v0.{element}[{i}]")?;
                    store_result(w, dst, t, frame, ra)?;
                    return Ok(());
                }
                emit_materialize_operand(w, &scalar_bits(&operands[1]), t, frame, ra)?;
                writeln!(w, "    ins v0.{element}[{i}], {scalar}")?;
            } else if *op == VSplat {
                if operands.len() != 1 {
                    return Err(unsupported("VSplat requires one operand"));
                }
                emit_materialize_operand(w, &scalar_bits(&operands[0]), t, frame, ra)?;
                writeln!(w, "    dup v0.{arr}, {scalar}")?;
            } else {
                if operands.len() != 2 {
                    return Err(unsupported(format!("{op:?} requires two operands")));
                }
                load_vector(w, &operands[0], "q0", frame, a)?;
                load_vector(w, &operands[1], "q1", frame, a)?;
                match op {
                    VShl | VLShr | VAShr => {
                        writeln!(w, "    umov {t}, v1.d[0]")?;
                        writeln!(w, "    mov {a}, #{bits}")?;
                        writeln!(w, "    cmp {t}, {a}")?;
                        writeln!(w, "    csel {t}, {t}, {a}, lo")?;
                        if *op != VShl {
                            writeln!(w, "    neg {t}, {t}")?;
                        }
                        writeln!(w, "    dup v1.{arr}, {scalar}")?;
                        let mnemonic = if *op == VAShr { "sshl" } else { "ushl" };
                        writeln!(w, "    {mnemonic} v0.{arr}, v0.{arr}, v1.{arr}")?;
                    }
                    VAnd | VOr | VXor => {
                        let mnemonic = match op {
                            VAnd => "and",
                            VOr => "orr",
                            _ => "eor",
                        };
                        writeln!(w, "    {mnemonic} v0.16b, v0.16b, v1.16b")?;
                    }
                    _ => {
                        let mnemonic = match (op, float) {
                            (VAdd, false) => "add",
                            (VSub, false) => "sub",
                            (VMul, false) => "mul",
                            (VAdd, true) => "fadd",
                            (VSub, true) => "fsub",
                            (VMul, true) => "fmul",
                            _ => {
                                return Err(unsupported(format!(
                                    "AArch64 unsupported {op:?} for {lane:?}"
                                )));
                            }
                        };
                        writeln!(w, "    {mnemonic} v0.{arr}, v0.{arr}, v1.{arr}")?;
                    }
                }
            }
            emit_frame_slot_address(w, a, vector_slot(frame, dst)?)?;
            writeln!(w, "    str q0, [{a}]")?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use crate::mir::{VectorLane, VectorOp};
    use crate::mir_codegen::simd_support::*;

    #[test]
    fn neon_assembles_every_supported_pair() {
        native(Backend::Neon, |op, lane, index| {
            use VectorOp::*;
            let bits = scalar(lane).size_bytes() * 8;
            let arr = match bits {
                8 => "16b",
                16 => "8h",
                32 => "4s",
                _ => "2d",
            };
            let elem = match bits {
                8 => "b",
                16 => "h",
                32 => "s",
                _ => "d",
            };
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            match op {
                VAdd | VSub | VMul => format!(
                    "{}{} v0.{arr}, v0.{arr}, v1.{arr}",
                    if float { "f" } else { "" },
                    match op {
                        VAdd => "add",
                        VSub => "sub",
                        _ => "mul",
                    }
                ),
                VAnd | VOr | VXor => format!(
                    "{} v0.16b, v0.16b, v1.16b",
                    match op {
                        VAnd => "and",
                        VOr => "orr",
                        _ => "eor",
                    }
                ),
                VShl | VLShr | VAShr => format!(
                    "{} v0.{arr}, v0.{arr}, v1.{arr}",
                    if op == VAShr { "sshl" } else { "ushl" }
                ),
                VSplat => format!("dup v0.{arr},"),
                VExtractLane => format!("v0.{elem}[{index}]"),
                VInsertLane => format!("ins v0.{elem}[{index}],"),
            }
        });
    }

    #[test]
    fn neon_rejects_unsupported_pairs() {
        negative(Backend::Neon);
    }
}
