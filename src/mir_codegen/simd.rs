use crate::mir::{Function, Instruction, MirType, Operand, Register, VectorOp};

pub(super) fn vector_slots(func: &Function) -> Vec<(VirtualReg, usize)> {
    let mut vectors = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        let (ty, dst, operands) = match inst {
            Instruction::Load { ty, dst, .. } => (*ty, Some(dst), Vec::new()),
            Instruction::Store { ty, src, .. } => (*ty, None, vec![src]),
            Instruction::VectorOp {
                ty,
                dst,
                operands,
                op,
            } => (
                *ty,
                (*op != VectorOp::VExtractLane).then_some(dst),
                if *op == VectorOp::VSplat {
                    Vec::new()
                } else {
                    operands
                        .iter()
                        .take(
                            if *op == VectorOp::VInsertLane || *op == VectorOp::VExtractLane {
                                1
                            } else {
                                2
                            },
                        )
                        .collect()
                },
            ),
            _ => continue,
        };
        if !ty.is_vector() {
            continue;
        }
        for reg in dst
            .into_iter()
            .chain(operands.into_iter().filter_map(|o| match o {
                Operand::Register(r) => Some(r),
                _ => None,
            }))
        {
            if let Register::Virtual(v) = reg {
                vectors.push((*v, ty.size_bytes() / 8 - 1));
            }
        }
    }
    vectors
}

pub(super) fn reserve_vector_slots(func: &Function) -> Option<Function> {
    let vectors = vector_slots(func);
    if vectors.is_empty() {
        return None;
    }
    let mut reserved = func.clone();
    for (v, extra) in vectors {
        let entry = reserved.stack_reservations.entry(v).or_default();
        *entry = (*entry).max(extra);
    }
    Some(reserved)
}

use crate::{
    error::LaminaError,
    mir::{Immediate, VectorLane, VectorType, VirtualReg},
    mir_codegen::CodegenError,
};

pub(super) fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

pub(super) fn lane(ty: MirType) -> Result<VectorLane, LaminaError> {
    match ty {
        MirType::Vector(VectorType::V128(lane)) => Ok(lane),
        MirType::Vector(VectorType::V256(_)) => Err(unsupported("V256 is unsupported")),
        _ => Err(unsupported(format!("expected V128, got {ty:?}"))),
    }
}

pub(super) fn width(lane: VectorLane) -> usize {
    match lane {
        VectorLane::I8 => 8,
        VectorLane::I16 => 16,
        VectorLane::I32 | VectorLane::F32 => 32,
        VectorLane::I64 | VectorLane::F64 => 64,
    }
}

pub(super) fn index(
    op: VectorOp,
    lane: VectorLane,
    operands: &[Operand],
) -> Result<usize, LaminaError> {
    let count = if op == VectorOp::VExtractLane { 2 } else { 3 };
    if operands.len() != count {
        return Err(unsupported(format!("{op:?} requires {count} operands")));
    }
    let index = match &operands[count - 1] {
        Operand::Immediate(Immediate::I8(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I16(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I32(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I64(v)) => *v,
        _ => {
            return Err(unsupported(format!(
                "{op:?} {lane:?} lane index must be integer immediate"
            )));
        }
    };
    if !(0..128 / width(lane) as i64).contains(&index) {
        return Err(unsupported(format!(
            "{op:?} {lane:?} lane index {index} out of range"
        )));
    }
    Ok(index as usize)
}

pub(super) fn scalar_bits(operand: &Operand) -> Operand {
    match operand {
        Operand::Immediate(Immediate::F32(v)) => Operand::imm_i64(i64::from(v.to_bits())),
        Operand::Immediate(Immediate::F64(v)) => Operand::imm_i64(v.to_bits() as i64),
        _ => operand.clone(),
    }
}
