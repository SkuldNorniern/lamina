//! Constant folding transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Immediate, Instruction, Operand};

/// Constant folding that evaluates constant expressions at compile time.
#[derive(Default)]
pub struct ConstantFolding;

impl Transform for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn description(&self) -> &'static str {
        "Evaluates constant expressions at compile time"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ConstantFolding
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl ConstantFolding {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                if self.try_fold_constants(instr) {
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    fn try_fold_constants(&self, instr: &mut Instruction) -> bool {
        if let Instruction::IntBinary {
            op, dst, lhs, rhs, ..
        } = instr
            && let (Some(lhs_val), Some(rhs_val)) =
                (self.extract_constant(lhs), self.extract_constant(rhs))
        {
            let result = match op {
                crate::mir::IntBinOp::Add => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_add(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::Sub => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_sub(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::Mul => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_mul(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::UDiv if rhs_val != 0 => {
                    // Cast to u64 for proper unsigned division semantics
                    let lhs_u = lhs_val as u64;
                    let rhs_u = rhs_val as u64;
                    (lhs_u / rhs_u) as i64
                }
                crate::mir::IntBinOp::SDiv if rhs_val != 0 => {
                    // Check for overflow: i64::MIN / -1
                    if lhs_val == i64::MIN && rhs_val == -1 {
                        return false; // Skip folding to avoid overflow
                    }
                    // Keep signed division for SDiv
                    lhs_val / rhs_val
                }
                crate::mir::IntBinOp::URem if rhs_val != 0 => {
                    // Cast to u64 for proper unsigned remainder semantics
                    let lhs_u = lhs_val as u64;
                    let rhs_u = rhs_val as u64;
                    (lhs_u % rhs_u) as i64
                }
                crate::mir::IntBinOp::SRem if rhs_val != 0 => {
                    // Keep signed remainder for SRem
                    lhs_val % rhs_val
                }
                _ => return false,
            };

            // Replace the instruction with a load immediate
            *instr = Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst: dst.clone(),
                ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                lhs: Operand::Immediate(Immediate::I64(result)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            };
            return true;
        }
        false
    }

    fn extract_constant(&self, operand: &Operand) -> Option<i64> {
        match operand {
            Operand::Immediate(Immediate::I64(val)) => Some(*val),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{Function, FunctionBuilder, Immediate, IntBinOp, MirType, ScalarType, VirtualReg};

    #[test]
    fn test_signed_division_overflow_prevention() {
        // Test that i64::MIN / -1 is prevented (would overflow)
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = crate::mir::Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(-1)),
        });
        func.add_block(bb);

        let pass = ConstantFolding::default();
        let changed = pass.try_fold_constants(&mut func.blocks[0].instructions[0]);
        // Should NOT change due to overflow prevention
        assert!(!changed);
    }

    #[test]
    fn test_constant_folding_division_by_zero() {
        // Division by zero should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = crate::mir::Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(100)),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed);
    }

    #[test]
    fn test_constant_folding_overflow_sub() {
        // Subtraction overflow should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = crate::mir::Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Sub,
            ty: MirType::Scalar(ScalarType::I64),
            dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed); // Overflow, no fold
    }

    #[test]
    fn test_constant_folding_overflow_mul() {
        // Multiplication overflow should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = crate::mir::Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed); // Overflow, no fold
    }

    #[test]
    fn test_constant_folding_unsigned_division() {
        // Unsigned division: large positive / 2 should fold correctly
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = crate::mir::Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(-2)), // Interpreted as u64::MAX - 1
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(changed);

        // Check the result: (u64::MAX - 1) / 2 = 0x7FFFFFFFFFFFFFFE
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, .. } => {
                let expected = ((-2i64 as u64) / 2) as i64;
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(expected)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }
}
