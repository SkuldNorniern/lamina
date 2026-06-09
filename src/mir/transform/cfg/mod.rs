//! Control flow graph (CFG) simplification transforms.
//!
//! Transforms that simplify the control flow graph:
//!
//! - **CfgSimplify**: Simplifies trivial branches and selects
//! - **JumpThreading**: Bypasses trivial jump-only blocks

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction, Operand};
use std::collections::{HashMap, HashSet};

/// Identify loop headers via back-edge detection (target block index ≤ source block index).
pub(crate) fn compute_back_edge_headers(func: &Function) -> HashSet<String> {
    let mut label_index: HashMap<&str, usize> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        label_index.insert(&b.label, i);
    }
    let mut headers: HashSet<String> = HashSet::new();
    for (i, b) in func.blocks.iter().enumerate() {
        for succ in b.successors() {
            if let Some(&tidx) = label_index.get(succ.as_str())
                && tidx <= i
            {
                headers.insert(succ);
            }
        }
    }
    headers
}

/// Compute dominator sets via iterative dataflow.
///
/// Returns a map from block label → set of block labels that dominate it (including itself).
pub(crate) fn calculate_dominators(func: &Function) -> HashMap<String, HashSet<String>> {
    let all_blocks: HashSet<String> = func.blocks.iter().map(|b| b.label.clone()).collect();

    let mut pred_map: HashMap<String, Vec<String>> = HashMap::new();
    for label in &all_blocks {
        pred_map.insert(label.clone(), Vec::new());
    }
    for block in &func.blocks {
        for succ in block.successors() {
            pred_map.entry(succ).or_default().push(block.label.clone());
        }
    }

    let mut dominators: HashMap<String, HashSet<String>> = HashMap::new();
    for block in &func.blocks {
        if block.label == func.entry {
            let mut set = HashSet::new();
            set.insert(block.label.clone());
            dominators.insert(block.label.clone(), set);
        } else {
            dominators.insert(block.label.clone(), all_blocks.clone());
        }
    }

    let mut changed = true;
    const MAX_ITERATIONS: usize = 1000;
    let mut iterations = 0;
    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;
        for block in &func.blocks {
            if block.label == func.entry {
                continue;
            }
            let preds: &[String] = pred_map.get(&block.label).map_or(&[], Vec::as_slice);
            if preds.is_empty() {
                continue;
            }
            let mut new_doms: HashSet<String> =
                dominators.get(&preds[0]).cloned().unwrap_or_default();
            for pred in &preds[1..] {
                if let Some(pred_doms) = dominators.get(pred) {
                    new_doms = new_doms.intersection(pred_doms).cloned().collect();
                }
            }
            new_doms.insert(block.label.clone());
            if dominators.get(&block.label) != Some(&new_doms) {
                dominators.insert(block.label.clone(), new_doms);
                changed = true;
            }
        }
    }
    dominators
}

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

    fn apply(&self, func: &mut Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CfgSimplify {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

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
                    // select cond, x, x -> add x, 0
                    Instruction::Select {
                        dst,
                        ty,
                        cond: _,
                        true_val,
                        false_val,
                    } if true_val == false_val => {
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
                    Instruction::Switch { cases, default, .. } => {
                        preds
                            .entry(default.clone())
                            .or_default()
                            .push(block.label.clone());
                        for (_, case_target) in cases {
                            preds
                                .entry(case_target.clone())
                                .or_default()
                                .push(block.label.clone());
                        }
                    }
                    _ => {}
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

        func.blocks.retain(|b| !to_remove.contains(&b.label));
        func.blocks.sort_by_key(|b| b.label.clone());

        Ok(changed)
    }
}

/// CFG Jump Threading / Target Bypass
/// Rewrites branches/jumps that target trivial jump-only blocks to jump directly
/// to the final destination. This reduces unnecessary hops and makes other passes more effective.
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

    fn apply(&self, func: &mut Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl JumpThreading {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut simple_jumps: HashMap<String, String> = HashMap::new();

        for block in &func.blocks {
            if block.instructions.len() == 1
                && let Instruction::Jmp { target } = &block.instructions[0]
            {
                simple_jumps.insert(block.label.clone(), target.clone());
            }
        }

        fn resolve_target(map: &HashMap<String, String>, mut tgt: String) -> String {
            let mut seen = HashSet::new();
            const MAX_CHAIN: usize = 100;
            let mut i = 0;
            while let Some(next) = map.get(&tgt) {
                if i >= MAX_CHAIN || !seen.insert(tgt.clone()) {
                    break;
                }
                tgt = next.clone();
                i += 1;
            }
            tgt
        }

        // Map each trivial block to its non-trivial terminal, skipping cycles.
        let mut resolved_targets: HashMap<String, String> = HashMap::new();
        for k in simple_jumps.keys() {
            let resolved = resolve_target(&simple_jumps, k.clone());
            if resolved != *k && !simple_jumps.contains_key(&resolved) {
                resolved_targets.insert(k.clone(), resolved);
            }
        }

        let mut changed = false;
        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                match instr {
                    Instruction::Jmp { target } => {
                        if let Some(new_tgt) = resolved_targets.get(target)
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
                        if let Some(new_tgt) = resolved_targets.get(true_target)
                            && new_tgt != true_target
                        {
                            *true_target = new_tgt.clone();
                            changed = true;
                        }
                        if let Some(new_tgt) = resolved_targets.get(false_target)
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
                            if let Some(new_tgt) = resolved_targets.get(tgt)
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
