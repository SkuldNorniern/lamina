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
        // More branch optimizations could be added later
        let reachable_blocks = self.compute_reachable_blocks(func);

        // Ensure entry block is always in reachable set
        let entry_label = func.blocks.first().map(|b| b.label.clone());
        let mut reachable_blocks = reachable_blocks;
        if let Some(ref entry) = entry_label {
            reachable_blocks.insert(entry.clone());
        }

        let original_count = func.blocks.len();

        func.blocks.retain(|block| {
            // Always keep entry block
            if Some(&block.label) == entry_label.as_ref() {
                return true;
            }
            reachable_blocks.contains(&block.label)
        });

        // Safety: Ensure we never remove all blocks
        if func.blocks.is_empty() {
            return Err(
                "Branch optimization would remove all blocks - aborting for safety".to_string(),
            );
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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

    #[test]
    fn test_branch_opt_empty_function_single_block() {
        // Single block with just ret - should not crash or change
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_branch_opt_all_reachable() {
        // All blocks are reachable - nothing should be removed
        let mut func = FunctionBuilder::new("all_reachable")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "middle".to_string(),
            })
            .block("middle")
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed);
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_branch_opt_multiple_unreachable() {
        // Multiple unreachable blocks should all be removed
        let mut func = FunctionBuilder::new("multi_unreachable")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .block("dead1")
            .instr(Instruction::Ret { value: None })
            .block("dead2")
            .instr(Instruction::Ret { value: None })
            .block("dead3")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].label, "entry");
    }

    #[test]
    fn test_branch_opt_loop_reachable() {
        // Blocks reachable via back-edge should NOT be removed
        let mut func = FunctionBuilder::new("loop")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed);
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_branch_opt_diamond_cfg() {
        // Diamond CFG: entry -> (left | right) -> merge
        let mut func = FunctionBuilder::new("diamond")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "left".to_string(),
                false_target: "right".to_string(),
            })
            .block("left")
            .instr(Instruction::Jmp {
                target: "merge".to_string(),
            })
            .block("right")
            .instr(Instruction::Jmp {
                target: "merge".to_string(),
            })
            .block("merge")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed); // All blocks reachable
        assert_eq!(func.blocks.len(), 4);
    }

    #[test]
    fn test_branch_opt_preserves_entry() {
        // Entry block with no predecessors must always be kept
        let mut func = FunctionBuilder::new("entry_only")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        // Add unreachable block that could confuse algorithm
        let mut dead = crate::mir::Block::new("dead");
        dead.push(Instruction::Jmp {
            target: "entry".to_string(), // Points TO entry
        });
        func.add_block(dead);

        let pass = BranchOptimization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed); // Dead block removed
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].label, "entry");
    }

    #[test]
    fn test_branch_opt_stress_no_infinite_loop() {
        // Create many blocks in chain - should not cause infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Create chain of 500 blocks (under limit)
        for i in 0..500 {
            let label = format!("block{}", i);
            let next = if i < 499 {
                format!("block{}", i + 1)
            } else {
                "exit".to_string()
            };

            let mut block = crate::mir::Block::new(&label);
            block.push(Instruction::Jmp { target: next });
            func.add_block(block);
        }

        let mut exit = crate::mir::Block::new("exit");
        exit.push(Instruction::Ret { value: None });
        func.add_block(exit);

        // Entry jumps to first block
        func.blocks[0].instructions.push(Instruction::Jmp {
            target: "block0".to_string(),
        });

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        // All blocks should remain (all reachable)
        assert!(!result.unwrap());
    }

    #[test]
    fn test_branch_opt_nested_loops_three_levels() {
        // Triple-nested loop with block structure
        // i_loop -> j_loop -> k_loop with back-edges
        let mut func = FunctionBuilder::new("nested_triple")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "i_loop".to_string(),
            })
            .block("i_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "done".to_string(),
                false_target: "j_loop".to_string(),
            })
            .block("j_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "next_i".to_string(),
                false_target: "k_loop".to_string(),
            })
            .block("k_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "next_j".to_string(),
                false_target: "process_k".to_string(),
            })
            .block("process_k")
            .instr(Instruction::Jmp {
                target: "k_loop".to_string(),
            })
            .block("next_j")
            .instr(Instruction::Jmp {
                target: "j_loop".to_string(),
            })
            .block("next_i")
            .instr(Instruction::Jmp {
                target: "i_loop".to_string(),
            })
            .block("done")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        // All blocks are reachable - nothing should be removed
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 8);
    }

    #[test]
    fn test_branch_opt_diamond_with_variable_bounds_check() {
        // Diamond CFG for bounds checking
        // if i_end > n_rows: use_n_rows else: keep_i_end -> continue
        let mut func = FunctionBuilder::new("diamond_bounds")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "use_n_rows".to_string(),
                false_target: "keep_i_end".to_string(),
            })
            .block("use_n_rows")
            .instr(Instruction::Jmp {
                target: "continue_block".to_string(),
            })
            .block("keep_i_end")
            .instr(Instruction::Jmp {
                target: "continue_block".to_string(),
            })
            .block("continue_block")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        // All blocks reachable
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 4);
    }

    #[test]
    fn test_branch_opt_unroll_dispatch_pattern() {
        // Check unroll 16 -> check unroll 8 -> single loop
        // This is a chain of conditional checks leading to different loop bodies
        let mut func = FunctionBuilder::new("unroll_dispatch")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "process_k".to_string(),
            })
            .block("process_k")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(), // can_unroll_16
                true_target: "j_unrolled_16".to_string(),
                false_target: "check_unroll_8".to_string(),
            })
            .block("check_unroll_8")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(), // can_unroll_8
                true_target: "j_unrolled_8".to_string(),
                false_target: "j_single".to_string(),
            })
            .block("j_unrolled_16")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "next_k".to_string(),
                false_target: "process_j_16".to_string(),
            })
            .block("process_j_16")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(3).into(),
                true_target: "unroll_16_body".to_string(),
                false_target: "check_unroll_8_remainder".to_string(),
            })
            .block("unroll_16_body")
            .instr(Instruction::Jmp {
                target: "j_unrolled_16".to_string(),
            })
            .block("check_unroll_8_remainder")
            .instr(Instruction::Jmp {
                target: "check_unroll_8".to_string(),
            })
            .block("j_unrolled_8")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(4).into(),
                true_target: "next_k".to_string(),
                false_target: "process_j_8".to_string(),
            })
            .block("process_j_8")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(5).into(),
                true_target: "unroll_8_body".to_string(),
                false_target: "j_single".to_string(),
            })
            .block("unroll_8_body")
            .instr(Instruction::Jmp {
                target: "j_unrolled_8".to_string(),
            })
            .block("j_single")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(6).into(),
                true_target: "next_k".to_string(),
                false_target: "process_j_single".to_string(),
            })
            .block("process_j_single")
            .instr(Instruction::Jmp {
                target: "j_single".to_string(),
            })
            .block("next_k")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        // All blocks are reachable through different paths
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 13);
    }

    #[test]
    fn test_branch_opt_nested_block_loops_seven_levels() {
        // 7 levels of nesting (i_block, j_block, k_block, i, j, k, unroll)
        // This is a stress test of deeply nested loops
        let mut func = FunctionBuilder::new("seven_levels")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "i_block_loop".to_string(),
            })
            .block("i_block_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "done".to_string(),
                false_target: "j_block_loop".to_string(),
            })
            .block("j_block_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "next_i_block".to_string(),
                false_target: "k_block_loop".to_string(),
            })
            .block("k_block_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "next_j_block".to_string(),
                false_target: "i_loop".to_string(),
            })
            .block("i_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(3).into(),
                true_target: "next_k_block".to_string(),
                false_target: "k_loop".to_string(),
            })
            .block("k_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(4).into(),
                true_target: "next_i".to_string(),
                false_target: "j_loop".to_string(),
            })
            .block("j_loop")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(5).into(),
                true_target: "next_k".to_string(),
                false_target: "process_elem".to_string(),
            })
            .block("process_elem")
            .instr(Instruction::Jmp {
                target: "j_loop".to_string(),
            })
            .block("next_k")
            .instr(Instruction::Jmp {
                target: "k_loop".to_string(),
            })
            .block("next_i")
            .instr(Instruction::Jmp {
                target: "i_loop".to_string(),
            })
            .block("next_k_block")
            .instr(Instruction::Jmp {
                target: "k_block_loop".to_string(),
            })
            .block("next_j_block")
            .instr(Instruction::Jmp {
                target: "j_block_loop".to_string(),
            })
            .block("next_i_block")
            .instr(Instruction::Jmp {
                target: "i_block_loop".to_string(),
            })
            .block("done")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        // All blocks are reachable
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 14);
    }

    #[test]
    fn test_branch_opt_multiple_backedges_to_same_block() {
        // Pattern: Multiple different blocks jump back to the same loop header
        // This is common in unrolled loops with remainder handling
        let mut func = FunctionBuilder::new("multi_backedge")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop_header".to_string(),
            })
            .block("loop_header")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "exit".to_string(),
                false_target: "dispatch".to_string(),
            })
            .block("dispatch")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "path_a".to_string(),
                false_target: "path_b".to_string(),
            })
            .block("path_a")
            .instr(Instruction::Jmp {
                target: "loop_header".to_string(), // Back-edge 1
            })
            .block("path_b")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "path_c".to_string(),
                false_target: "path_d".to_string(),
            })
            .block("path_c")
            .instr(Instruction::Jmp {
                target: "loop_header".to_string(), // Back-edge 2
            })
            .block("path_d")
            .instr(Instruction::Jmp {
                target: "loop_header".to_string(), // Back-edge 3
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = BranchOptimization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
        assert_eq!(func.blocks.len(), 8);
    }
}
