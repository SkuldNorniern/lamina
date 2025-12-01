//! Instruction scheduling transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Register};

/// Instruction scheduling that reorders instructions for better ILP.
#[derive(Default)]
pub struct InstructionScheduling;

impl Transform for InstructionScheduling {
    fn name(&self) -> &'static str {
        "instruction_scheduling"
    }

    fn description(&self) -> &'static str {
        "Reorders instructions for better instruction-level parallelism"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl InstructionScheduling {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit function size
        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 1_000;
        
        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for instruction scheduling ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        for block in &func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                return Err(format!(
                    "Block '{}' too large for instruction scheduling ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_INSTRUCTIONS_PER_BLOCK
                ));
            }
        }

        let mut changed = false;

        for block in &mut func.blocks {
            if self.schedule_block_instructions(block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Schedule instructions within a basic block for better ILP
    fn schedule_block_instructions(&self, block: &mut Block) -> bool {
        if block.instructions.len() < 3 {
            return false; // Not enough instructions to reorder
        }

        let mut changed = false;

        // Look for multiply-accumulate patterns that can be reordered
        if self.schedule_multiply_accumulate_patterns(block) {
            changed = true;
        }

        // Look for load-use chains that can be optimized
        if self.schedule_load_use_chains(block) {
            changed = true;
        }

        changed
    }

    /// Schedule multiply-accumulate patterns for better ILP
    /// This is particularly important for matrix operations
    fn schedule_multiply_accumulate_patterns(&self, block: &mut Block) -> bool {
        let changed = false;

        // Find sequences of multiply and add operations
        let mut i = 0;
        while i < block.instructions.len().saturating_sub(2) {
            // Look for pattern: mul -> add (accumulate)
            if let (
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Mul,
                    dst: mul_dst,
                    ..
                },
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Add,
                    dst: add_dst,
                    lhs: add_lhs,
                    rhs: add_rhs,
                    ..
                },
            ) = (&block.instructions[i], &block.instructions[i + 1])
            {
                // Check if the add uses the result of the mul
                if let crate::mir::Operand::Register(rhs_reg) = add_rhs
                    && self.is_same_register(mul_dst, rhs_reg)
                {
                    // Check if this is an accumulation: dst += (lhs * rhs)
                    if let crate::mir::Operand::Register(lhs_reg) = add_lhs
                        && self.is_same_register(add_dst, lhs_reg)
                    {
                        // Found multiply-accumulate pattern
                        // In a real scheduler, we might:
                        // 1. Move independent instructions between mul and add
                        // 2. Schedule loads early to hide latency
                        // 3. Group similar operations together

                        // For now, this serves as pattern recognition
                    }
                }
            }
            i += 1;
        }

        changed
    }

    /// Schedule load-use chains to hide memory latency
    fn schedule_load_use_chains(&self, block: &mut Block) -> bool {
        let changed = false;

        // Find load instructions and try to schedule independent work between
        // the load and its first use

        let mut load_positions = Vec::new();

        // Find all loads and their positions
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Load { dst, .. } = inst {
                load_positions.push((idx, dst.clone()));
            }
        }

        // For each load, find its first use and see if we can schedule work between them
        for (load_idx, loaded_reg) in load_positions {
            // Find first use of this register after the load
            let first_use =
                self.find_first_use_after(&block.instructions, &loaded_reg, load_idx + 1);

            if let Some(use_idx) = first_use
                && use_idx > load_idx + 1
            {
                // There are instructions between load and use
                // Check if any can be moved or reordered for better scheduling
                // This is complex and would need sophisticated dependency analysis
            }
        }

        changed
    }

    /// Find the first use of a register after a given position
    fn find_first_use_after(
        &self,
        instructions: &[Instruction],
        reg: &Register,
        start_idx: usize,
    ) -> Option<usize> {
        for (idx, inst) in instructions.iter().enumerate().skip(start_idx) {
            if inst.use_regs().contains(&reg) {
                return Some(idx);
            }
        }
        None
    }

    /// Check if two registers refer to the same virtual register
    fn is_same_register(&self, reg1: &Register, reg2: &Register) -> bool {
        match (reg1, reg2) {
            (Register::Virtual(v1), Register::Virtual(v2)) => v1 == v2,
            _ => false,
        }
    }
}
