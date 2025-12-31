//! Control flow graph (CFG) simplification transforms.
//!
//! This module provides transforms that simplify the control flow graph:
//!
//! - **CfgSimplify**: Simplifies trivial branches and selects
//! - **JumpThreading**: Bypasses trivial jump-only blocks

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction, Operand};
use std::collections::HashMap;

/// Simple CFG simplifications:
/// - br with identical true/false targets -> jmp
/// - select with identical true/false values -> move
#[derive(Default)]
pub struct CfgSimplify;

impl Transform for CfgSimplify {
    fn name(&self) -> &'static str {
        "cfg_simplify"
    }

    fn description(&self) -> &'static str {
        "Simplifies trivial branches and selects"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        // Safe local rewrites; treat as stable for -O1
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CfgSimplify {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        // First pass: local simplifications within blocks
        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                match instr {
                    // br cond, A, A -> jmp A
                    Instruction::Br {
                        cond: _,
                        true_target,
                        false_target,
                    } if true_target == false_target => {
                        let target = true_target.clone();
                        *instr = Instruction::Jmp { target };
                        changed = true;
                    }
                    // select cond, x, x -> x
                    Instruction::Select {
                        dst,
                        ty,
                        cond: _,
                        true_val,
                        false_val,
                    } if true_val == false_val => {
                        // Replace with a move via add of immediate 0
                        let replacement = Instruction::IntBinary {
                            op: crate::mir::IntBinOp::Add,
                            ty: *ty,
                            dst: dst.clone(),
                            lhs: true_val.clone(),
                            rhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(0)),
                        };
                        *instr = replacement;
                        changed = true;
                    }
                    _ => {}
                }
            }
        }

        // Second pass: trivial block merge
        let mut preds: HashMap<String, Vec<String>> = HashMap::new();
        for block in &func.blocks {
            if let Some(term) = block.instructions.last()
                && term.is_terminator()
            {
                match term {
                    Instruction::Jmp { target } => {
                        preds
                            .entry(target.clone())
                            .or_default()
                            .push(block.label.clone());
                    }
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        preds
                            .entry(true_target.clone())
                            .or_default()
                            .push(block.label.clone());
                        preds
                            .entry(false_target.clone())
                            .or_default()
                            .push(block.label.clone());
                    }
                    _ => {} // Ignore switch, ret, etc. for simplicity
                }
            }
        }

        let mut merges = Vec::new();
        for block in &func.blocks {
            if block.instructions.len() == 1
                && let Some(Instruction::Jmp { target }) = block.instructions.last()
                && let Some(preds_list) = preds.get(&block.label)
                && preds_list.len() == 1
            {
                let pred_label = preds_list[0].clone();
                merges.push((pred_label, target.clone(), block.label.clone()));
            }
        }

        // Now perform the merges
        let mut to_remove = Vec::new();
        for (pred_label, new_target, trivial_label) in merges {
            if let Some(pred_block) = func.blocks.iter_mut().find(|b| b.label == pred_label)
                && let Some(pred_term) = pred_block.instructions.last_mut()
                && let Instruction::Jmp { target } = pred_term
                && *target == trivial_label
            {
                *target = new_target;
                changed = true;
                to_remove.push(trivial_label);
            }
        }

        // Remove merged blocks
        func.blocks.retain(|b| !to_remove.contains(&b.label));
        // Sort blocks by label for consistent ordering
        func.blocks.sort_by_key(|b| b.label.clone());

        Ok(changed)
    }
}

/// CFG Jump Threading / Target Bypass
/// Rewrites branches/jumps that target trivial jump-only blocks to jump directly
/// to the final destination. This reduces unnecessary hops and helps other passes.
#[derive(Default)]
pub struct JumpThreading;

impl Transform for JumpThreading {
    fn name(&self) -> &'static str {
        "jump_threading"
    }

    fn description(&self) -> &'static str {
        "Bypass trivial jump-only blocks in branch/jump targets"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        // Safe as it preserves semantics of jumps/branches; treat as stable
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl JumpThreading {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Build a mapping from block label to its simple jump target if it's a trivial block
        let mut simple_jumps: HashMap<String, String> = HashMap::new();

        for block in &func.blocks {
            if block.instructions.len() == 1
                && let Instruction::Jmp { target } = &block.instructions[0]
            {
                simple_jumps.insert(block.label.clone(), target.clone());
            }
        }

        // Resolve to ultimate targets (follow chains)
        // Safety: limit chain length to prevent excessive resolution
        fn resolve_target(map: &HashMap<String, String>, mut tgt: String) -> String {
            let mut seen = std::collections::HashSet::new();
            const MAX_CHAIN_LENGTH: usize = 100; // Safety limit
            let mut iterations = 0;

            while let Some(next) = map.get(&tgt) {
                if iterations >= MAX_CHAIN_LENGTH {
                    // Chain too long, return current target to avoid infinite loops
                    break;
                }
                if !seen.insert(tgt.clone()) {
                    // Cycle detected, return current target
                    break;
                }
                tgt = next.clone();
                iterations += 1;
            }
            tgt
        }

        for (k, v) in simple_jumps.clone() {
            let resolved = resolve_target(&simple_jumps, v.clone());
            if resolved != v {
                // compress path
                simple_jumps.insert(k, resolved);
            }
        }

        let mut changed = false;

        // Rewrite all Jmp/Br targets through the mapping
        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                match instr {
                    Instruction::Jmp { target } => {
                        if let Some(new_tgt) = simple_jumps.get(target)
                            && new_tgt != target
                        {
                            *target = new_tgt.clone();
                            changed = true;
                        }
                    }
                    Instruction::Br {
                        cond: _,
                        true_target,
                        false_target,
                    } => {
                        if let Some(new_tgt) = simple_jumps.get(true_target)
                            && new_tgt != true_target
                        {
                            *true_target = new_tgt.clone();
                            changed = true;
                        }
                        if let Some(new_tgt) = simple_jumps.get(false_target)
                            && new_tgt != false_target
                        {
                            *false_target = new_tgt.clone();
                            changed = true;
                        }
                    }
                    Instruction::Switch {
                        value: _, cases, ..
                    } => {
                        let mut local_change = false;
                        for (_val, tgt) in cases.iter_mut() {
                            if let Some(new_tgt) = simple_jumps.get(tgt)
                                && new_tgt != tgt
                            {
                                *tgt = new_tgt.clone();
                                local_change = true;
                            }
                        }
                        if local_change {
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(changed)
    }
}
