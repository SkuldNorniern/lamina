//! Dead code elimination transform for MIR.

use super::super::{Block, Function, Instruction, Register};
use super::{Transform, TransformCategory, TransformLevel};
use std::collections::{HashMap, HashSet};

/// Statistics about dead code elimination
#[derive(Debug, Default)]
pub struct DeadCodeStats {
    /// Number of instructions removed
    pub instructions_removed: usize,
    /// Number of registers that became dead
    pub registers_freed: usize,
}

/// Dead Code Elimination transform
#[derive(Default)]
pub struct DeadCodeElimination;

impl Transform for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead_code_elimination"
    }

    fn description(&self) -> &'static str {
        "Removes instructions that define registers which are never used"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::DeadCodeElimination
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
            .map(|stats| stats.instructions_removed > 0)
    }
}

impl DeadCodeElimination {
    /// Apply dead code elimination to a function
    pub fn apply_internal(&self, func: &mut Function) -> Result<DeadCodeStats, String> {
        let mut stats = DeadCodeStats::default();

        // 1. Compute liveness analysis (inter-block)
        let live_out_map = self.compute_liveness(func)?;

        // 2. Remove dead instructions using liveness info
        for block in &mut func.blocks {
            // Start with registers live at the end of the block
            let mut live_regs = live_out_map.get(&block.label).cloned().unwrap_or_default();

            let removed = self.remove_dead_instructions_in_block(block, &mut live_regs);
            stats.instructions_removed += removed;
        }

        Ok(stats)
    }

    /// Compute liveness analysis (live-out sets for each block)
    fn compute_liveness(
        &self,
        func: &Function,
    ) -> Result<HashMap<String, HashSet<Register>>, String> {
        let mut live_in: HashMap<String, HashSet<Register>> = HashMap::new();
        let mut live_out: HashMap<String, HashSet<Register>> = HashMap::new();

        // Initialize sets
        for block in &func.blocks {
            live_in.insert(block.label.clone(), HashSet::new());
            live_out.insert(block.label.clone(), HashSet::new());
        }

        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed {
            if iterations > MAX_ITERATIONS {
                return Err("Liveness analysis failed to converge".to_string());
            }
            iterations += 1;
            changed = false;

            // Process blocks in reverse order (heuristic for faster convergence)
            for block in func.blocks.iter().rev() {
                let label = &block.label;

                // 1. Calculate LiveOut = Union(LiveIn(successors))
                let mut current_live_out = HashSet::new();
                if let Some(terminator) = block.instructions.last() {
                    match terminator {
                        Instruction::Jmp { target } => {
                            if let Some(succ_live_in) = live_in.get(target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                        }
                        Instruction::Br {
                            true_target,
                            false_target,
                            ..
                        } => {
                            if let Some(succ_live_in) = live_in.get(true_target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                            if let Some(succ_live_in) = live_in.get(false_target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                        }
                        _ => {} // Return or others have no successors within function
                    }
                }

                if current_live_out != *live_out.get(label).unwrap() {
                    live_out.insert(label.clone(), current_live_out.clone());
                    changed = true;
                }

                // 2. Calculate LiveIn = Use U (LiveOut - Def)
                let mut current_live_in = current_live_out.clone();
                // Iterate backwards through instructions
                for instr in block.instructions.iter().rev() {
                    if let Some(def) = instr.def_reg() {
                        current_live_in.remove(def);
                    }
                    for use_reg in instr.use_regs() {
                        current_live_in.insert(use_reg.clone());
                    }
                }

                if current_live_in != *live_in.get(label).unwrap() {
                    live_in.insert(label.clone(), current_live_in);
                    changed = true;
                }
            }
        }

        Ok(live_out)
    }

    /// Remove dead instructions within a single basic block
    fn remove_dead_instructions_in_block(
        &self,
        block: &mut Block,
        live_regs: &mut HashSet<Register>,
    ) -> usize {
        let mut removed_count = 0;
        let mut instructions_to_keep = Vec::new();

        // Iterate backwards
        // We collect instructions to keep, then reverse them back
        for instr in block.instructions.iter().rev() {
            if self.is_dead_instruction(instr, live_regs) {
                removed_count += 1;
                // Don't add to keep list
            } else {
                instructions_to_keep.push(instr.clone());

                // Update liveness for the kept instruction
                if let Some(def_reg) = instr.def_reg() {
                    live_regs.remove(def_reg);
                }
                for use_reg in instr.use_regs() {
                    live_regs.insert(use_reg.clone());
                }
            }
        }

        if removed_count > 0 {
            instructions_to_keep.reverse();
            block.instructions = instructions_to_keep;
        }

        removed_count
    }

    /// Check if an instruction is dead and can be safely removed
    fn is_dead_instruction(&self, instr: &Instruction, live_regs: &HashSet<Register>) -> bool {
        // Never remove terminators
        if instr.is_terminator() {
            return false;
        }

        // Check if instruction defines a register
        if let Some(def_reg) = instr.def_reg() {
            // If the defined register is not live, instruction is dead
            if !live_regs.contains(def_reg) {
                // Additional check: ensure instruction has no side effects
                return self.has_no_side_effects(instr);
            }
        } else {
            // Instructions that don't define a register are usually for side effects (e.g. stores)
            // But some might be useless?
            // Assuming has_no_side_effects check covers it.
            return self.has_no_side_effects(instr);
        }

        false
    }

    /// Check if an instruction has no side effects and can be safely removed
    fn has_no_side_effects(&self, instr: &Instruction) -> bool {
        match instr {
            // Pure arithmetic operations - safe to remove
            Instruction::IntBinary { .. }
            | Instruction::FloatBinary { .. }
            | Instruction::FloatUnary { .. }
            | Instruction::IntCmp { .. }
            | Instruction::FloatCmp { .. }
            | Instruction::Select { .. }
            | Instruction::VectorOp { .. }
            | Instruction::Lea { .. } => true,

            // SIMD register-only ops are also pure (nightly only)
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { .. }
            | Instruction::SimdUnary { .. }
            | Instruction::SimdTernary { .. }
            | Instruction::SimdShuffle { .. }
            | Instruction::SimdExtract { .. }
            | Instruction::SimdInsert { .. } => true,

            // Instructions with side effects - never remove
            Instruction::Load { .. } // Reads memory (could fault)
            | Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::TailCall { .. }
            | Instruction::Ret { .. }
            | Instruction::Jmp { .. }
            | Instruction::Br { .. }
            | Instruction::Switch { .. }
            | Instruction::Unreachable
            | Instruction::SafePoint
            | Instruction::StackMap { .. }
            | Instruction::PatchPoint { .. }
            | Instruction::Comment { .. } => false,

            // SIMD memory and atomic operations have side effects (nightly only)
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad { .. }
            | Instruction::SimdStore { .. }
            | Instruction::AtomicLoad { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::AtomicBinary { .. }
            | Instruction::AtomicCompareExchange { .. }
            | Instruction::Fence { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_dead_code_elimination_basic() {
        // Create a function with dead instructions
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Dead instruction: v1 = v0 + 42 (v1 never used)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(42)),
            })
            // Live instruction: v2 = v0 + 10 (v2 used in ret)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination::default();

        let changed = dce.apply(&mut func).expect("DCE should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");
        // Original: 3 instructions. Dead removed -> 2 remaining.
        assert_eq!(entry.instructions.len(), 2);

        // Verify content
        // First instr should be the live add
        match &entry.instructions[0] {
            Instruction::IntBinary { dst, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
            }
            _ => panic!("Expected IntBinary"),
        }
    }
}
