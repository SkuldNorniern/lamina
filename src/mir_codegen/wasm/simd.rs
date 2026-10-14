use super::*;
use crate::mir::{Instruction, VectorLane, VectorOp};
use crate::mir_codegen::simd::{index, lane, unsupported, width};

fn address(
    w: &mut impl Write,
    addr: &AddressMode,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            load_register_wasm(base, w, locals)?;
            writeln!(w, "    i64.const {offset}\n    i64.add")?;
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => {
            load_register_wasm(base, w, locals)?;
            load_register_wasm(index, w, locals)?;
            writeln!(
                w,
                "    i64.const {scale}\n    i64.mul\n    i64.add\n    i64.const {offset}\n    i64.add"
            )?;
        }
    }
    writeln!(w, "    i32.wrap_i64")?;
    Ok(())
}

fn scalar(
    w: &mut impl Write,
    operand: &Operand,
    lane: VectorLane,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    load_operand_wasm(operand, w, locals)?;
    if width(lane) < 64 {
        writeln!(w, "    i32.wrap_i64")?;
    }
    match lane {
        VectorLane::F32 => writeln!(w, "    f32.reinterpret_i32")?,
        VectorLane::F64 => writeln!(w, "    f64.reinterpret_i64")?,
        _ => (),
    }
    Ok(())
}

pub(super) fn emit(
    inst: &Instruction,
    w: &mut impl Write,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match inst {
        Instruction::Load { ty, dst, addr, .. } => {
            lane(*ty)?;
            address(w, addr, locals)?;
            writeln!(w, "    v128.load align=1")?;
            store_to_register_wasm(dst, w, locals)?;
        }
        Instruction::Store { ty, src, addr, .. } => {
            lane(*ty)?;
            address(w, addr, locals)?;
            load_operand_wasm(src, w, locals)?;
            writeln!(w, "    v128.store align=1")?;
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
            let prefix = match lane {
                VectorLane::I8 => "i8x16",
                VectorLane::I16 => "i16x8",
                VectorLane::I32 => "i32x4",
                VectorLane::I64 => "i64x2",
                VectorLane::F32 => "f32x4",
                VectorLane::F64 => "f64x2",
            };
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            if (*op == VMul && lane == VectorLane::I8)
                || (float && matches!(op, VShl | VLShr | VAShr))
            {
                return Err(unsupported(format!("WASM unsupported {op:?} for {lane:?}")));
            }
            if matches!(op, VExtractLane | VInsertLane) {
                let i = index(*op, lane, operands)?;
                load_operand_wasm(&operands[0], w, locals)?;
                if *op == VExtractLane {
                    let suffix = if bits < 32 { "_u" } else { "" };
                    writeln!(w, "    {prefix}.extract_lane{suffix} {i}")?;
                    match lane {
                        VectorLane::F32 => {
                            writeln!(w, "    i32.reinterpret_f32\n    i64.extend_i32_u")?
                        }
                        VectorLane::F64 => writeln!(w, "    i64.reinterpret_f64")?,
                        _ if bits < 64 => writeln!(w, "    i64.extend_i32_u")?,
                        _ => (),
                    }
                } else {
                    scalar(w, &operands[1], lane, locals)?;
                    writeln!(w, "    {prefix}.replace_lane {i}")?;
                }
            } else if *op == VSplat {
                if operands.len() != 1 {
                    return Err(unsupported("VSplat requires one operand"));
                }
                scalar(w, &operands[0], lane, locals)?;
                writeln!(w, "    {prefix}.splat")?;
            } else {
                if operands.len() != 2 {
                    return Err(unsupported(format!("{op:?} requires two operands")));
                }
                if matches!(op, VShl | VLShr | VAShr) {
                    load_operand_wasm(&operands[1], w, locals)?;
                    writeln!(
                        w,
                        "    i64x2.extract_lane 0\n    i64.const {bits}\n    i64.ge_u\n    if (result v128)"
                    )?;
                    if *op == VAShr {
                        load_operand_wasm(&operands[0], w, locals)?;
                        writeln!(w, "    i32.const {}\n    {prefix}.shr_s", bits - 1)?;
                    } else {
                        writeln!(w, "    v128.const i32x4 0 0 0 0")?;
                    }
                    writeln!(w, "    else")?;
                    load_operand_wasm(&operands[0], w, locals)?;
                    load_operand_wasm(&operands[1], w, locals)?;
                    writeln!(w, "    i64x2.extract_lane 0\n    i32.wrap_i64")?;
                    let mnemonic = match op {
                        VShl => "shl",
                        VLShr => "shr_u",
                        _ => "shr_s",
                    };
                    writeln!(w, "    {prefix}.{mnemonic}\n    end")?;
                } else {
                    load_operand_wasm(&operands[0], w, locals)?;
                    load_operand_wasm(&operands[1], w, locals)?;
                    let mnemonic = match op {
                        VAdd => format!("{prefix}.add"),
                        VSub => format!("{prefix}.sub"),
                        VMul => format!("{prefix}.mul"),
                        VAnd => "v128.and".into(),
                        VOr => "v128.or".into(),
                        VXor => "v128.xor".into(),
                        _ => {
                            return Err(unsupported(format!(
                                "WASM unsupported {op:?} for {lane:?}"
                            )));
                        }
                    };
                    writeln!(w, "    {mnemonic}")?;
                }
            }
            store_to_register_wasm(dst, w, locals)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
