//! Branch optimization transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction};

/// Branch optimization that eliminates unreachable branches.
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
        // Safety check: prevent transforms on extremely large functions
        const MAX_BLOCKS: usize = 1_000;
        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for branch optimization ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        // Safety: Skip if function has no blocks or only one block
        if func.blocks.len() <= 1 {
            return Ok(false);
        }

        let mut changed = false;

        // For now, just remove obviously unreachable blocks
        // More sophisticated branch optimizations could be added later
        let reachable_blocks = self.compute_reachable_blocks(func);
        
        // Ensure entry block is always in reachable set
        let entry_label = func.blocks.first().map(|b| b.label.clone());
        let mut reachable_blocks = reachable_blocks;
        if let Some(ref entry) = entry_label {
            reachable_blocks.insert(entry.clone());
        }
        
        let original_count = func.blocks.len();
        
        func.blocks
            .retain(|block| {
                // Always keep entry block
                if Some(&block.label) == entry_label.as_ref() {
                    return true;
                }
                reachable_blocks.contains(&block.label)
            });

        // Safety: Ensure we never remove all blocks
        if func.blocks.is_empty() {
            return Err("Branch optimization would remove all blocks - aborting for safety".to_string());
        }

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
        const MAX_ITERATIONS: usize = 10_000;

        // Start with the entry block
        if let Some(entry) = func.blocks.first() {
            reachable.insert(entry.label.clone());
            worklist.push_back(entry.label.clone());
        }

        // BFS to find all reachable blocks with iteration limit
        let mut iterations = 0;
        while let Some(current_label) = worklist.pop_front() {
            if iterations >= MAX_ITERATIONS {
                // Safety: prevent infinite loops in malformed CFGs
                break;
            }
            iterations += 1;

            if let Some(block) = func.get_block(&current_label)
                && let Some(last_instr) = block.instructions.last()
            {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, Instruction, MirType, Operand, ScalarType, VirtualReg};

    #[test]
    fn test_remove_unreachable_blocks() {
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "block1".to_string(),
                false_target: "block2".to_string(),
            })
            .block("block1")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(1))),
            })
            .block("block2")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(2))),
            })
            .block("unreachable")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(3))),
            })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        let block_labels: Vec<String> = func.blocks.iter().map(|b| b.label.clone()).collect();
        assert!(block_labels.contains(&"entry".to_string()));
        assert!(block_labels.contains(&"block1".to_string()));
        assert!(block_labels.contains(&"block2".to_string()));
        assert!(!block_labels.contains(&"unreachable".to_string()));
    }

    #[test]
    fn test_preserve_entry_block() {
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(0))),
            })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed); // No unreachable blocks

        let block_labels: Vec<String> = func.blocks.iter().map(|b| b.label.clone()).collect();
        assert!(block_labels.contains(&"entry".to_string()));
    }

    #[test]
    fn test_large_function_rejected() {
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(0))),
            })
            .build();

        // Add many blocks to exceed limit
        for i in 0..2000 {
            let label = format!("block{}", i);
            let mut block = crate::mir::Block::new(&label);
            block.push(Instruction::Ret {
                value: Some(Operand::Immediate(crate::mir::Immediate::I64(0))),
            });
            func.add_block(block);
        }

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_err());
    }
}
