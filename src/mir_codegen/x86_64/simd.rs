use crate::mir::{Function, Instruction, MirType, Operand, Register, VectorOp};

pub(super) fn reserve_vector_slots(func: &Function) -> Option<Function> {
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

use super::{RaxTrackingWriter, util::load_operand_to_rax};
use crate::{
    error::LaminaError,
    mir::{AddressMode, VectorLane, VectorType, VirtualReg},
    mir_codegen::CodegenError,
};
use lamina_codegen::x86_64::X64RegAlloc;
use std::{collections::HashMap, io::Write};

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

fn lane(ty: MirType) -> Result<VectorLane, LaminaError> {
    match ty {
        MirType::Vector(VectorType::V128(lane)) => Ok(lane),
        MirType::Vector(VectorType::V256(_)) => Err(unsupported("x86_64 V256 requires AVX")),
        _ => Err(unsupported(format!("expected V128, got {ty:?}"))),
    }
}

fn slot(reg: &Register, slots: &HashMap<VirtualReg, i32>) -> Result<i32, LaminaError> {
    if let Register::Virtual(v) = reg
        && let Some(offset) = slots.get(v)
    {
        return Ok(*offset);
    }
    Err(unsupported(
        "vector value requires virtual register stack slot",
    ))
}

fn load_vector<W: Write>(
    operand: &Operand,
    xmm: &str,
    writer: &mut RaxTrackingWriter<W>,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    let offset = slot(reg, slots)?;
    writeln!(writer, "    movdqu {offset}(%rbp), %{xmm}")?;
    Ok(())
}

pub(super) fn memory<W: Write>(
    inst: &Instruction,
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let (ty, addr) = match inst {
        Instruction::Load { ty, addr, .. } | Instruction::Store { ty, addr, .. } => (*ty, addr),
        _ => unreachable!(),
    };
    lane(ty)?;
    let AddressMode::BaseOffset { base, offset, .. } = addr else {
        return Err(unsupported(format!("vector memory address {addr:?}")));
    };
    load_operand_to_rax(&Operand::Register(base.clone()), writer, ra, slots)?;
    match inst {
        Instruction::Load { dst, .. } => {
            let dst = slot(dst, slots)?;
            writeln!(writer, "    movdqu {offset}(%rax), %xmm0")?;
            writeln!(writer, "    movdqu %xmm0, {dst}(%rbp)")?;
        }
        Instruction::Store { src, .. } => {
            load_vector(src, "xmm0", writer, slots)?;
            writeln!(writer, "    movdqu %xmm0, {offset}(%rax)")?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
