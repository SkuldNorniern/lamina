//! Edge moves for block arguments: sequencing parallel copies and placing them on
//! CFG edges. Used by SSA destruction.

use crate::{
    block::Block,
    function::Function,
    instruction::{Instruction, Operand},
    loc::Located,
    register::{Register, VirtualRegAllocator},
    types::MirType,
};

/// Places the moves of each edge into the function. Moves go before a jump, into a fresh
/// trampoline block for a branch or switch arm, and otherwise at the end of the predecessor.
/// Edges are visited in label order so trampoline labels are deterministic.
pub fn insert_edge_moves(
    mir_func: &mut Function,
    mut edges: Vec<((String, String), Vec<Instruction>)>,
    vreg_alloc: &mut VirtualRegAllocator,
) {
    edges.sort_by(|left, right| left.0.cmp(&right.0));
    let mut trampoline_counter: u64 = 0;
    for ((pred, succ), moves) in edges {
        let moves = sequence_parallel_copies(moves, vreg_alloc);
        // Snapshot the current terminator without holding a mutable borrow
        let term = mir_func
            .get_block(&pred)
            .and_then(|bb| bb.terminator().cloned());

        let mut handled = false;
        if let Some(Instruction::Jmp { target }) = term.as_ref()
            && target == &succ
            && let Some(pred_bb) = mir_func.get_block_mut(&pred)
            && !pred_bb.instructions.is_empty()
        {
            let insert_pos = pred_bb.instructions.len().saturating_sub(1);
            pred_bb.instructions.splice(
                insert_pos..insert_pos,
                moves.iter().cloned().map(Located::from),
            );
            handled = true;
        }
        if !handled
            && let Some(Instruction::Br {
                cond: _,
                true_target,
                false_target,
            }) = term.as_ref()
            && (true_target == &succ || false_target == &succ)
        {
            {
                trampoline_counter = trampoline_counter.saturating_add(1);
                let tramp_label = format!(
                    "{}_to_{}_phi{}",
                    pred.replace('%', "_"),
                    succ.replace('%', "_"),
                    trampoline_counter
                );
                // Create trampoline and add it to the function
                let mut tramp = Block::new(tramp_label.clone());
                for m in &moves {
                    tramp.push(m.clone());
                }
                tramp.push(Instruction::Jmp {
                    target: succ.clone(),
                });
                mir_func.add_block(tramp);
                if let Some(pred_mut) = mir_func.get_block_mut(&pred)
                    && let Some(last) = pred_mut.instructions.last_mut().map(|i| &mut i.instr)
                    && let Instruction::Br {
                        cond: _,
                        true_target: t,
                        false_target: f,
                    } = last
                {
                    if *t == succ {
                        *t = tramp_label.clone();
                    }
                    if *f == succ {
                        *f = tramp_label.clone();
                    }
                    handled = true;
                }
            }
        }
        // A switch reaches several blocks from one terminator, and the scrutinee is
        // read by the terminator itself, so the moves cannot sit in front of it.
        if !handled
            && let Some(Instruction::Switch { cases, default, .. }) = term.as_ref()
            && (default == &succ || cases.iter().any(|(_, target)| target == &succ))
        {
            trampoline_counter = trampoline_counter.saturating_add(1);
            let tramp_label = format!(
                "{}_to_{}_phi{}",
                pred.replace('%', "_"),
                succ.replace('%', "_"),
                trampoline_counter
            );
            let mut tramp = Block::new(tramp_label.clone());
            for m in &moves {
                tramp.push(m.clone());
            }
            tramp.push(Instruction::Jmp {
                target: succ.clone(),
            });
            mir_func.add_block(tramp);
            if let Some(pred_mut) = mir_func.get_block_mut(&pred)
                && let Some(Instruction::Switch { cases, default, .. }) =
                    pred_mut.instructions.last_mut().map(|i| &mut i.instr)
            {
                if default == &succ {
                    *default = tramp_label.clone();
                }
                for (_, target) in cases.iter_mut().filter(|(_, t)| t == &succ) {
                    *target = tramp_label.clone();
                }
                handled = true;
            }
        }
        if !handled && let Some(pred_bb) = mir_func.get_block_mut(&pred) {
            if !pred_bb.instructions.is_empty() {
                let insert_pos = pred_bb.instructions.len().saturating_sub(1);
                pred_bb.instructions.splice(
                    insert_pos..insert_pos,
                    moves.iter().cloned().map(Located::from),
                );
            } else {
                for m in &moves {
                    pred_bb.push(m.clone());
                }
            }
        }
    }
}

/// Orders the copies on one edge so none overwrites a value another still reads. The
/// copies on an edge read all of their operands at once, and emitting a swap in source order would make
/// both destinations hold the same value. A cycle is broken by saving one destination
/// into a fresh register first.
pub fn sequence_parallel_copies(
    moves: Vec<Instruction>,
    vreg_alloc: &mut VirtualRegAllocator,
) -> Vec<Instruction> {
    let mut pending: Vec<(MirType, Register, Operand)> = Vec::new();
    let mut others = Vec::new();
    for instruction in moves {
        match instruction {
            Instruction::Copy { ty, dst, src } => pending.push((ty, dst, src)),
            other => others.push(other),
        }
    }

    let source = |operand: &Operand| match operand {
        Operand::Register(reg) => Some(reg.clone()),
        Operand::Immediate(_) => None,
    };
    let mut out = Vec::with_capacity(pending.len() + others.len());
    while !pending.is_empty() {
        let free = pending.iter().position(|(_, dst, _)| {
            !pending
                .iter()
                .any(|(_, other, src)| other != dst && source(src).as_ref() == Some(dst))
        });
        if let Some(index) = free {
            let (ty, dst, src) = pending.remove(index);
            out.push(Instruction::Copy { ty, dst, src });
            continue;
        }

        let (ty, dst, src) = pending.remove(0);
        let saved = Register::Virtual(if ty.is_float() {
            vreg_alloc.allocate_fpr()
        } else {
            vreg_alloc.allocate_gpr()
        });
        out.push(Instruction::Copy {
            ty,
            dst: saved.clone(),
            src: Operand::Register(dst.clone()),
        });
        for (_, _, pending_src) in pending.iter_mut() {
            if source(pending_src).as_ref() == Some(&dst) {
                *pending_src = Operand::Register(saved.clone());
            }
        }
        out.push(Instruction::Copy { ty, dst, src });
    }
    out.extend(others);
    out
}
