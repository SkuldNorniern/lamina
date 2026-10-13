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
