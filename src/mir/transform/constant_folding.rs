//! Constant folding transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{FloatBinOp, Function, Immediate, Instruction, IntCmpOp, Operand};

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

    fn apply(&self, func: &mut Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl ConstantFolding {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                if self.try_fold_int(instr)
                    || self.try_fold_float(instr)
                    || self.try_fold_cmp(instr)
                {
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    fn try_fold_int(&self, instr: &mut Instruction) -> bool {
        if let Instruction::IntBinary {
            op, dst, lhs, rhs, ..
        } = instr
            && let (Some(lhs_val), Some(rhs_val)) = (self.extract_i64(lhs), self.extract_i64(rhs))
        {
            let result: i64 = match op {
                crate::mir::IntBinOp::Add => match lhs_val.checked_add(rhs_val) {
                    Some(r) => r,
                    None => return false,
                },
                crate::mir::IntBinOp::Sub => match lhs_val.checked_sub(rhs_val) {
                    Some(r) => r,
                    None => return false,
                },
                crate::mir::IntBinOp::Mul => match lhs_val.checked_mul(rhs_val) {
                    Some(r) => r,
                    None => return false,
                },
                crate::mir::IntBinOp::UDiv if rhs_val != 0 => {
                    ((lhs_val as u64) / (rhs_val as u64)) as i64
                }
                crate::mir::IntBinOp::SDiv if rhs_val != 0 => {
                    if lhs_val == i64::MIN && rhs_val == -1 {
                        return false;
                    }
                    lhs_val / rhs_val
                }
                crate::mir::IntBinOp::URem if rhs_val != 0 => {
                    ((lhs_val as u64) % (rhs_val as u64)) as i64
                }
                crate::mir::IntBinOp::SRem if rhs_val != 0 => lhs_val % rhs_val,
                crate::mir::IntBinOp::And => lhs_val & rhs_val,
                crate::mir::IntBinOp::Or => lhs_val | rhs_val,
                crate::mir::IntBinOp::Xor => lhs_val ^ rhs_val,
                crate::mir::IntBinOp::Shl => {
                    if !(0..64).contains(&rhs_val) {
                        return false;
                    }
                    lhs_val << rhs_val
                }
                crate::mir::IntBinOp::LShr => {
                    if !(0..64).contains(&rhs_val) {
                        return false;
                    }
                    ((lhs_val as u64) >> rhs_val) as i64
                }
                crate::mir::IntBinOp::AShr => {
                    if !(0..64).contains(&rhs_val) {
                        return false;
                    }
                    lhs_val >> rhs_val
                }
                _ => return false,
            };

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

    fn try_fold_float(&self, instr: &mut Instruction) -> bool {
        if let Instruction::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } = instr
            && let (Some(l), Some(r)) = (self.extract_f64(lhs), self.extract_f64(rhs))
        {
            let result = match op {
                FloatBinOp::FAdd => l + r,
                FloatBinOp::FSub => l - r,
                FloatBinOp::FMul => l * r,
                FloatBinOp::FDiv => {
                    if r == 0.0 {
                        return false;
                    }
                    l / r
                }
            };
            if result.is_nan() || result.is_infinite() {
                return false;
            }
            let saved_ty = *ty;
            let saved_dst = dst.clone();
            *instr = Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                dst: saved_dst,
                ty: saved_ty,
                lhs: Operand::Immediate(Immediate::F64(result)),
                rhs: Operand::Immediate(Immediate::F64(0.0)),
            };
            return true;
        }
        false
    }

    fn try_fold_cmp(&self, instr: &mut Instruction) -> bool {
        if let Instruction::IntCmp {
            op, dst, lhs, rhs, ..
        } = instr
            && let (Some(lhs_val), Some(rhs_val)) = (self.extract_i64(lhs), self.extract_i64(rhs))
        {
            let result: bool = match op {
                IntCmpOp::Eq => lhs_val == rhs_val,
                IntCmpOp::Ne => lhs_val != rhs_val,
                IntCmpOp::SLt => lhs_val < rhs_val,
                IntCmpOp::SLe => lhs_val <= rhs_val,
                IntCmpOp::SGt => lhs_val > rhs_val,
                IntCmpOp::SGe => lhs_val >= rhs_val,
                IntCmpOp::ULt => (lhs_val as u64) < (rhs_val as u64),
                IntCmpOp::ULe => (lhs_val as u64) <= (rhs_val as u64),
                IntCmpOp::UGt => (lhs_val as u64) > (rhs_val as u64),
                IntCmpOp::UGe => (lhs_val as u64) >= (rhs_val as u64),
            };
            let saved_dst = dst.clone();
            *instr = Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst: saved_dst,
                ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I1),
                lhs: Operand::Immediate(Immediate::I64(result as i64)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            };
            return true;
        }
        false
    }

    fn extract_i64(&self, operand: &Operand) -> Option<i64> {
        match operand {
            Operand::Immediate(Immediate::I64(val)) => Some(*val),
            Operand::Immediate(Immediate::I32(val)) => Some(*val as i64),
            Operand::Immediate(Immediate::I16(val)) => Some(*val as i64),
            Operand::Immediate(Immediate::I8(val)) => Some(*val as i64),
            _ => None,
        }
    }

    fn extract_f64(&self, operand: &Operand) -> Option<f64> {
        match operand {
            Operand::Immediate(Immediate::F64(val)) => Some(*val),
            Operand::Immediate(Immediate::F32(val)) => Some(*val as f64),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{Function, Immediate, IntBinOp, MirType, ScalarType, VirtualReg};

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

        let pass = ConstantFolding;
        let changed = pass.try_fold_int(&mut func.blocks[0].instructions[0]);
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

        let cf = ConstantFolding;
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

        let cf = ConstantFolding;
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

        let cf = ConstantFolding;
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

        let cf = ConstantFolding;
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

    #[test]
    fn test_constant_folding_bitwise() {
        let cf = ConstantFolding;

        let mk = |op: IntBinOp, l: i64, r: i64| {
            let mut func = Function::new(crate::mir::function::Signature::new("f"))
                .with_entry("entry".to_string());
            let mut bb = crate::mir::Block::new("entry");
            bb.push(Instruction::IntBinary {
                op,
                ty: MirType::Scalar(ScalarType::I64),
                dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::I64(l)),
                rhs: Operand::Immediate(Immediate::I64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::IntBinary {
                lhs: Operand::Immediate(Immediate::I64(v)),
                ..
            } => *v,
            _ => panic!("Expected folded IntBinary"),
        };

        let mut f = mk(IntBinOp::And, 0b1100, 0b1010);
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), 0b1000);

        let mut f = mk(IntBinOp::Or, 0b1100, 0b1010);
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), 0b1110);

        let mut f = mk(IntBinOp::Xor, 0b1100, 0b1010);
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), 0b0110);

        let mut f = mk(IntBinOp::Shl, 1, 3);
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), 8);

        let mut f = mk(IntBinOp::LShr, -1i64, 1); // logical: u64::MAX >> 1
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), (u64::MAX >> 1) as i64);

        let mut f = mk(IntBinOp::AShr, -8i64, 2); // arithmetic: -8 >> 2 = -2
        cf.apply(&mut f).unwrap();
        assert_eq!(extract(&f), -2);
    }

    #[test]
    fn test_constant_folding_float() {
        let cf = ConstantFolding;

        let mk = |op: FloatBinOp, l: f64, r: f64| {
            let mut func = Function::new(crate::mir::function::Signature::new("f"))
                .with_entry("entry".to_string());
            let mut bb = crate::mir::Block::new("entry");
            bb.push(Instruction::FloatBinary {
                op,
                ty: MirType::Scalar(ScalarType::F64),
                dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::F64(l)),
                rhs: Operand::Immediate(Immediate::F64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::FloatBinary {
                lhs: Operand::Immediate(Immediate::F64(v)),
                ..
            } => *v,
            _ => panic!("Expected folded FloatBinary"),
        };

        let mut f = mk(FloatBinOp::FAdd, 1.5, 2.5);
        assert!(cf.apply(&mut f).unwrap());
        assert_eq!(extract(&f), 4.0);

        let mut f = mk(FloatBinOp::FMul, 3.0, 4.0);
        assert!(cf.apply(&mut f).unwrap());
        assert_eq!(extract(&f), 12.0);

        let mut f = mk(FloatBinOp::FDiv, 10.0, 0.0); // divide by zero — no fold
        assert!(!cf.apply(&mut f).unwrap());
    }

    #[test]
    fn test_constant_folding_cmp() {
        let cf = ConstantFolding;

        let mk = |op: IntCmpOp, l: i64, r: i64| {
            let mut func = Function::new(crate::mir::function::Signature::new("f"))
                .with_entry("entry".to_string());
            let mut bb = crate::mir::Block::new("entry");
            bb.push(Instruction::IntCmp {
                op,
                ty: MirType::Scalar(ScalarType::I1),
                dst: crate::mir::Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::I64(l)),
                rhs: Operand::Immediate(Immediate::I64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::IntBinary {
                lhs: Operand::Immediate(Immediate::I64(v)),
                ..
            } => *v,
            _ => panic!("Expected folded to IntBinary"),
        };

        let mut f = mk(IntCmpOp::Eq, 5, 5);
        assert!(cf.apply(&mut f).unwrap());
        assert_eq!(extract(&f), 1);

        let mut f = mk(IntCmpOp::SLt, 3, 7);
        assert!(cf.apply(&mut f).unwrap());
        assert_eq!(extract(&f), 1);

        let mut f = mk(IntCmpOp::SGt, 3, 7);
        assert!(cf.apply(&mut f).unwrap());
        assert_eq!(extract(&f), 0);
    }
}
