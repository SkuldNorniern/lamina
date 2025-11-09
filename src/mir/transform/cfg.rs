use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction};
use std::collections::HashMap;

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
        TransformLevel::Experimental
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
            if block.instructions.len() == 1 {
                if let Instruction::Jmp { target } = &block.instructions[0] {
                    simple_jumps.insert(block.label.clone(), target.clone());
                }
            }
        }

        // Resolve to ultimate targets (follow chains)
        fn resolve_target(map: &HashMap<String, String>, mut tgt: String) -> String {
            let mut seen = std::collections::HashSet::new();
            while let Some(next) = map.get(&tgt) {
                if !seen.insert(tgt.clone()) {
                    break;
                }
                tgt = next.clone();
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
                        if let Some(new_tgt) = simple_jumps.get(target) {
                            if new_tgt != target {
                                *target = new_tgt.clone();
                                changed = true;
                            }
                        }
                    }
                    Instruction::Br {
                        cond: _,
                        true_target,
                        false_target,
                    } => {
                        if let Some(new_tgt) = simple_jumps.get(true_target) {
                            if new_tgt != true_target {
                                *true_target = new_tgt.clone();
                                changed = true;
                            }
                        }
                        if let Some(new_tgt) = simple_jumps.get(false_target) {
                            if new_tgt != false_target {
                                *false_target = new_tgt.clone();
                                changed = true;
                            }
                        }
                    }
                    Instruction::Switch { value: _, cases, .. } => {
                        let mut local_change = false;
                        for (_val, tgt) in cases.iter_mut() {
                            if let Some(new_tgt) = simple_jumps.get(tgt) {
                                if new_tgt != tgt {
                                    *tgt = new_tgt.clone();
                                    local_change = true;
                                }
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


