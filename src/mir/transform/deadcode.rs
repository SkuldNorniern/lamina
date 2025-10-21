/// Dead Code Elimination Transform for LUMIR
///
/// This transform removes instructions that define registers which are never used,
/// improving code size and reducing unnecessary computations.
use super::super::{Block, Function, Instruction, Register};
use super::{Transform, TransformCategory, TransformLevel};
use std::collections::HashSet;

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
}

impl DeadCodeElimination {
    /// Apply dead code elimination to a function
    ///
    /// This method performs a backward liveness analysis to identify dead registers
    /// and removes instructions that only define dead registers.
    pub fn apply(&self, func: &mut Function) -> Result<DeadCodeStats, String> {
        let mut stats = DeadCodeStats::default();
        let mut changed = true;

        // Collect all registers that were defined before DCE
        let _initial_live_regs = self.compute_live_registers(func)?;
        let mut all_defined_regs = HashSet::new();
        for block in &func.blocks {
            for instr in &block.instructions {
                if let Some(def_reg) = instr.def_reg() {
                    all_defined_regs.insert(def_reg.clone());
                }
            }
        }

        // Iterate until no more dead code can be removed
        while changed {
            changed = false;

            // Perform liveness analysis to find live registers
            let live_regs = self.compute_live_registers(func)?;

            // Remove dead instructions from each block
            for block in &mut func.blocks {
                let removed = self.remove_dead_instructions(block, &live_regs);
                if removed > 0 {
                    changed = true;
                    stats.instructions_removed += removed;
                }
            }
        }

        // Count freed registers (registers that were defined but never used)
        let final_live_regs = self.compute_live_registers(func)?;
        stats.registers_freed = all_defined_regs.difference(&final_live_regs).count();

        Ok(stats)
    }

    /// Compute which registers are live at each point in the function
    ///
    /// Uses backward dataflow analysis to determine register liveness.
    /// A register is live if its value may be used before being redefined.
    fn compute_live_registers(&self, func: &Function) -> Result<HashSet<Register>, String> {
        let mut live_regs = HashSet::new();
        let mut changed = true;

        // Iterate until fixed point is reached
        while changed {
            changed = false;
            let old_live_count = live_regs.len();

            // Process each block in reverse order
            for block in func.blocks.iter().rev() {
                self.process_block_liveness(block, &mut live_regs);
            }

            // Check if we made progress
            if live_regs.len() != old_live_count {
                changed = true;
            }
        }

        Ok(live_regs)
    }

    /// Process liveness for a single block
    fn process_block_liveness(&self, block: &Block, live_regs: &mut HashSet<Register>) {
        // Process instructions in reverse order
        for instr in block.instructions.iter().rev() {
            // Registers defined by this instruction are no longer live
            if let Some(def_reg) = instr.def_reg() {
                live_regs.remove(def_reg);
            }

            // Registers used by this instruction become live
            for reg in instr.use_regs() {
                live_regs.insert(reg.clone());
            }
        }

        // Handle control flow - registers used in terminators are live
        if let Some(terminator) = block.terminator() {
            for reg in terminator.use_regs() {
                live_regs.insert(reg.clone());
            }
        }
    }

    /// Remove dead instructions from a block
    ///
    /// An instruction is considered dead if:
    /// 1. It defines a register that is not live
    /// 2. It has no side effects (no memory operations, calls, etc.)
    fn remove_dead_instructions(&self, block: &mut Block, live_regs: &HashSet<Register>) -> usize {
        let mut removed_count = 0;
        let mut new_instructions = Vec::new();

        for instr in block.instructions.drain(..) {
            let should_remove = self.is_dead_instruction(&instr, live_regs);

            if should_remove {
                removed_count += 1;
            } else {
                new_instructions.push(instr);
            }
        }

        block.instructions = new_instructions;
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

            // Instructions with side effects - never remove
            Instruction::Load { .. }
            | Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::Ret { .. }
            | Instruction::Jmp { .. }
            | Instruction::Br { .. }
            | Instruction::Switch { .. }
            | Instruction::Unreachable
            | Instruction::SafePoint
            | Instruction::StackMap { .. }
            | Instruction::PatchPoint { .. }
            | Instruction::Comment { .. } => false,
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
            // Dead instruction: v1 is never used
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(42)),
            })
            // Live instruction: v2 is used in return
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

        let stats = dce.apply(&mut func).expect("DCE should succeed");

        // Should have removed 1 dead instruction
        assert_eq!(stats.instructions_removed, 1);
        assert_eq!(stats.registers_freed, 1);

        // Check that the dead instruction was removed
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2); // Only live instruction + ret

        // Verify the remaining instruction is the live one
        match &entry.instructions[0] {
            Instruction::IntBinary { dst, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_dead_code_elimination_side_effects() {
        // Create a function with side-effect instructions that should not be removed
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::Ptr))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Store instruction - has side effects, should not be removed even if result unused
            .instr(Instruction::Store {
                ty: MirType::Scalar(ScalarType::I64),
                src: Operand::Immediate(Immediate::I64(42)),
                addr: crate::mir::AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                },
                attrs: crate::mir::MemoryAttrs {
                    align: 8,
                    volatile: false,
                },
            })
            // Dead arithmetic instruction
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination::default();
        let stats = dce.apply(&mut func).expect("DCE should succeed");

        // Should have removed 1 dead instruction (the arithmetic), but not the store
        assert_eq!(stats.instructions_removed, 1);

        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2); // Store + ret

        // Verify the store instruction is still there
        assert!(matches!(&entry.instructions[0], Instruction::Store { .. }));
    }

    #[test]
    fn test_dead_code_elimination_terminators() {
        // Create a function with dead instructions before terminators
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Dead instruction before terminator
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination::default();
        let stats = dce.apply(&mut func).expect("DCE should succeed");

        // Should have removed the dead instruction
        assert_eq!(stats.instructions_removed, 1);

        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 1); // Only ret

        // Verify the terminator is still there
        assert!(matches!(&entry.instructions[0], Instruction::Ret { .. }));
    }

    #[test]
    fn test_dead_code_elimination_no_changes() {
        // Create a function with no dead instructions
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(42)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination::default();
        let stats = dce.apply(&mut func).expect("DCE should succeed");

        // Should have removed no instructions
        assert_eq!(stats.instructions_removed, 0);
        assert_eq!(stats.registers_freed, 0);

        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2); // All instructions preserved
    }
}
