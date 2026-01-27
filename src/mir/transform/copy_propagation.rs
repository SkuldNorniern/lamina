//! Copy propagation transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Immediate, Instruction, Operand, Register};
use std::collections::HashMap;

/// Copy propagation transform that replaces variable uses with their source values.
#[derive(Default)]
pub struct CopyPropagation;

impl Transform for CopyPropagation {
    fn name(&self) -> &'static str {
        "copy_propagation"
    }

    fn description(&self) -> &'static str {
        "Replaces variable uses with their source values"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::CopyPropagation
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CopyPropagation {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit function size
        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 1_000;

        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for copy propagation ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        for block in &func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                return Err(format!(
                    "Block '{}' too large for copy propagation ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_INSTRUCTIONS_PER_BLOCK
                ));
            }
        }

        let mut changed = false;

        // Apply copy propagation within each basic block (safe: no control flow issues)
        for block in &mut func.blocks {
            if self.propagate_copies_in_block(block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn propagate_copies_in_block(&self, block: &mut Block) -> bool {
        let mut changed = false;
        let mut value_map = HashMap::new();

        let mut propagations_this_block = 0;
        let max_propagations_per_block = 50;

        let mut new_instructions = Vec::new();

        for instr in &block.instructions {
            let mut new_instr = instr.clone();

            if propagations_this_block < max_propagations_per_block
                && self.propagate_copies(&mut new_instr, &value_map)
            {
                changed = true;
                propagations_this_block += 1;
            }

            if let Some(def_reg) = new_instr.def_reg() {
                value_map.remove(def_reg);
            }
            if let Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst,
                lhs,
                rhs,
                ..
            } = &new_instr
                && let Operand::Immediate(Immediate::I64(0)) = rhs
                && let Operand::Register(src_reg) = lhs
            {
                value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
            }
            match &new_instr {
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Sub,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y - 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Mul,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y * 1 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(1)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                    // Check for x = 1 * y (copy pattern)
                    if let Operand::Immediate(Immediate::I64(1)) = lhs
                        && let Operand::Register(src_reg) = rhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Or,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y | 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                    // Check for x = 0 | y (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = lhs
                        && let Operand::Register(src_reg) = rhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::And,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y & -1 (copy pattern, since -1 is all bits set)
                    if let Operand::Immediate(Immediate::I64(-1)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Xor,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y ^ 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                _ => {}
            }

            new_instructions.push(new_instr);
        }

        block.instructions = new_instructions;
        changed
    }

    fn propagate_copies(
        &self,
        instr: &mut Instruction,
        value_map: &HashMap<Register, Operand>,
    ) -> bool {
        let mut changed = false;

        match instr {
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => {
                changed |= self.replace_operand(lhs, value_map);
                changed |= self.replace_operand(rhs, value_map);
            }
            Instruction::FloatUnary { src, .. } => {
                changed |= self.replace_operand(src, value_map);
            }
            Instruction::Select {
                cond: _,
                true_val,
                false_val,
                ..
            } => {
                changed |= self.replace_operand(true_val, value_map);
                changed |= self.replace_operand(false_val, value_map);
            }
            Instruction::Load { addr, .. } => {
                if let crate::mir::AddressMode::BaseOffset { base, offset: _ } = addr
                    && let Some(new_base) = value_map.get(base)
                    && let Operand::Register(new_reg) = new_base
                {
                    *base = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Store { src, addr, .. } => {
                changed |= self.replace_operand(src, value_map);
                if let crate::mir::AddressMode::BaseOffset { base, offset: _ } = addr
                    && let Some(new_base) = value_map.get(base)
                    && let Operand::Register(new_reg) = new_base
                {
                    *base = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    changed |= self.replace_operand(arg, value_map);
                }
            }
            Instruction::Br { cond, .. } => {
                if let Some(new_cond) = value_map.get(cond)
                    && let Operand::Register(new_reg) = new_cond
                {
                    *cond = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Switch { value, .. } => {
                if let Some(new_value) = value_map.get(value)
                    && let Operand::Register(new_reg) = new_value
                {
                    *value = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Ret { value: Some(val) } => {
                changed |= self.replace_operand(val, value_map);
            }
            _ => {}
        }

        changed
    }

    fn replace_operand(
        &self,
        operand: &mut Operand,
        value_map: &HashMap<Register, Operand>,
    ) -> bool {
        if let Operand::Register(reg) = operand
            && let Some(replacement) = value_map.get(reg)
        {
            *operand = replacement.clone();
            return true;
        }
        false
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_copy_propagation_basic() {
        // Test basic copy propagation within a block
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Copy: v1 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // Should be replaced: v2 = v1 + 0 -> v2 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let cp = CopyPropagation::default();
        let changed = cp
            .apply(&mut func)
            .expect("Copy propagation should succeed");

        assert!(changed);
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 3);

        // Check that the second instruction was modified to use v0 directly
        match &entry.instructions[1] {
            Instruction::IntBinary { dst, lhs, rhs, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_copy_propagation_register_redefinition() {
        // Test that copy mappings are invalidated when registers are redefined
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Copy: v1 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // Redefine v1: v1 = v3 + v4 (not a copy)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(3).into()),
                rhs: Operand::Register(VirtualReg::gpr(4).into()),
            })
            // Should NOT be replaced: v2 = v1 + 0 should stay as v2 = v1 + 0
            // (not become v2 = v0 + 0 since v1 was redefined)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let cp = CopyPropagation::default();
        let changed = cp
            .apply(&mut func)
            .expect("Copy propagation should succeed");

        // Should have made changes (first copy propagation), but not invalid ones
        assert!(changed);
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 4);

        // Check that the third instruction was NOT modified (still uses v1)
        match &entry.instructions[2] {
            Instruction::IntBinary { dst, lhs, rhs, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(1).into())); // Should still be v1
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_copy_propagation_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_copy_propagation_no_infinite_loop() {
        // Ensure copy propagation terminates on circular-looking patterns
        let mut func = FunctionBuilder::new("circular")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // Redefine v0
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        // Run multiple times to ensure no infinite loop
        for _ in 0..10 {
            let _ = cp.apply(&mut func);
        }
    }

    #[test]
    fn test_copy_propagation_mul_by_one() {
        // x * 1 is a copy pattern
        let mut func = FunctionBuilder::new("mul_one")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(5)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let changed = cp.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Second instruction should now use v0 directly
        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[1] {
            Instruction::IntBinary { lhs, .. } => {
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_copy_propagation_loop_counter_pattern() {
        // %i = add.i64 0, 0 (loop counter init)
        // followed by %i = add.i64 %i, 1 (loop increment)
        // Copy propagation should handle the redefinition correctly
        let mut func = FunctionBuilder::new("loop_counter")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %i = 0 + 0
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %i = %i + 1 (redefine same reg)
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should not crash on redefined registers
    }

    #[test]
    fn test_copy_propagation_block_size_pattern() {
        // %block_size = add.i64 8, 0 (constant via add)
        // This is a copy pattern that should be propagated
        let mut func = FunctionBuilder::new("block_size")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %block_size_i = 8 + 0
                lhs: Operand::Immediate(Immediate::I64(8)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(), // %i_end = %i_block + %block_size_i
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Register(VirtualReg::gpr(0).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
    }

    #[test]
    fn test_copy_propagation_chained_adds_zero() {
        // Multiple %x = add.i64 %y, 0 in sequence
        let mut func = FunctionBuilder::new("chained_zero")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(), // %a = %src + 0
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(), // %b = %a + 0
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(), // %c = %b + 0
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(), // use %c
                lhs: Operand::Register(VirtualReg::gpr(3).into()),
                rhs: Operand::Immediate(Immediate::I64(100)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let changed = cp.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Final add should use v0 directly (all copies should chain)
        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[3] {
            Instruction::IntBinary { lhs, .. } => {
                // After propagation, should trace back to v0
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_copy_propagation_same_reg_redefined_in_block() {
        // %a_elem = mul %i %k; %a_elem = add %a_elem 1
        // Same register redefined - must NOT propagate old value after redefinition
        let mut func = FunctionBuilder::new("redefine_same")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = 0 + 0 (creates copy mapping)
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = %i * %k (REDEFINE - invalidates mapping)
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = %x + 1 (should use the mul result)
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should not crash and should handle redefinitions correctly
    }

    #[test]
    fn test_copy_propagation_unroll_pattern() {
        // Unrolled loop with many similar instructions
        // %j_plus_1 = add %j 1; %j_plus_2 = add %j 2; etc.
        let mut func = FunctionBuilder::new("unroll_pattern")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %j = %j_block + 0
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .build();

        // Add 16 unrolled iterations
        let entry = func.blocks.get_mut(0).unwrap();
        for i in 1..=16 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(i).into(), // %j_plus_i = %j + i
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(i as i64)),
            });
        }

        // Use all the computed values
        let mut sum_reg = VirtualReg::gpr(1);
        for i in 2..=16 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(50 + i).into(),
                lhs: Operand::Register(sum_reg.into()),
                rhs: Operand::Register(VirtualReg::gpr(i).into()),
            });
            sum_reg = VirtualReg::gpr(50 + i);
        }

        entry.push(Instruction::Ret {
            value: Some(Operand::Register(sum_reg.into())),
        });

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        let changed = result.unwrap();
        assert!(changed); // Should propagate v10 -> v0
    }

    #[test]
    fn test_copy_propagation_accumulator_pattern() {
        // %total_sum = add %total_sum %product
        // Accumulator that gets updated repeatedly
        let mut func = FunctionBuilder::new("accumulator")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %total_sum = 0 + 0
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .build();

        let entry = func.blocks.get_mut(0).unwrap();
        // Simulate 8 accumulation steps
        for i in 1..=8 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %total_sum = %total_sum + %product_i
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(100 + i).into()),
            });
        }

        entry.push(Instruction::Ret {
            value: Some(Operand::Register(VirtualReg::gpr(0).into())),
        });

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should handle accumulator pattern without issues
    }

    #[test]
    fn test_copy_propagation_many_blocks_with_copies() {
        // Pattern: Multiple blocks each with copy patterns
        // Tests cross-block isolation (copies don't propagate across blocks)
        let mut func = FunctionBuilder::new("multi_block_copies")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "block1".to_string(),
            })
            .block("block1")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "block2".to_string(),
            })
            .block("block2")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(3).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Each block should handle its own copies independently
    }
}
