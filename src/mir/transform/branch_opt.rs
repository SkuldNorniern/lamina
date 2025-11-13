use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction};

/// Branch Optimization Transform
/// Performs simple branch optimizations like eliminating unreachable branches
/// and simplifying conditional branches with constant conditions
#[derive(Default)]
pub struct BranchOptimization;

impl Transform for BranchOptimization {
    fn name(&self) -> &'static str {
        "branch_optimization"
    }

    fn description(&self) -> &'static str {
        "Optimizes branch instructions and eliminates unreachable code"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl BranchOptimization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        // For now, just remove obviously unreachable blocks
        // More sophisticated branch optimizations could be added later
        let reachable_blocks = self.compute_reachable_blocks(func);
        let original_count = func.blocks.len();

        func.blocks
            .retain(|block| reachable_blocks.contains(&block.label));

        if func.blocks.len() != original_count {
            changed = true;
        }

        Ok(changed)
    }

    /// Compute which blocks are reachable from the entry block
    fn compute_reachable_blocks(&self, func: &Function) -> std::collections::HashSet<String> {
        use std::collections::{HashSet, VecDeque};

        let mut reachable = HashSet::new();
        let mut worklist = VecDeque::new();

        // Start with the entry block
        if let Some(entry) = func.blocks.first() {
            reachable.insert(entry.label.clone());
            worklist.push_back(entry.label.clone());
        }

        // BFS to find all reachable blocks
        while let Some(current_label) = worklist.pop_front() {
            if let Some(block) = func.get_block(&current_label)
                && let Some(last_instr) = block.instructions.last() {
                    match last_instr {
                        Instruction::Jmp { target } => {
                            if reachable.insert(target.clone()) {
                                worklist.push_back(target.clone());
                            }
                        }
                        Instruction::Br {
                            true_target,
                            false_target,
                            ..
                        } => {
                            if reachable.insert(true_target.clone()) {
                                worklist.push_back(true_target.clone());
                            }
                            if reachable.insert(false_target.clone()) {
                                worklist.push_back(false_target.clone());
                            }
                        }
                        Instruction::Ret { .. } => {
                            // Terminal instruction, no successors
                        }
                        _ => {
                            // For other instructions, assume fallthrough to next block
                            // (though MIR doesn't have implicit fallthrough)
                        }
                    }
                }
        }

        reachable
    }
}

