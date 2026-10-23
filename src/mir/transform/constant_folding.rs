//! Constant folding transform for MIR.

use crate::mir::{
    FloatBinOp, Function, Immediate, Instruction, IntCmpOp, MirType, Operand, ScalarType,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};

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

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl ConstantFolding {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                if self.try_fold_int(instr)
                    || self.try_fold_float(instr)
                    || self.try_fold_cmp(instr)
                {
                    instr.normalize_integer_immediates(None);
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    pub(super) fn try_fold_int(&self, instr: &mut Instruction) -> bool {
        if let Instruction::IntBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } = instr
            && let (Some(lhs_val), Some(rhs_val)) = (self.extract_i64(lhs), self.extract_i64(rhs))
        {
            let Some(result) = op.evaluate(*ty, lhs_val, rhs_val) else {
                return false;
            };

            *instr = Instruction::Copy {
                dst: dst.clone(),
                ty: *ty,
                src: Operand::Immediate(Immediate::I64(result)),
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
            let immediate = match saved_ty {
                MirType::Scalar(ScalarType::F32) => Immediate::F32(result as f32),
                _ => Immediate::F64(result),
            };
            *instr = Instruction::Copy {
                dst: saved_dst,
                ty: saved_ty,
                src: Operand::Immediate(immediate),
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
            *instr = Instruction::Copy {
                dst: saved_dst,
                ty: MirType::Scalar(ScalarType::I1),
                src: Operand::Immediate(Immediate::I64(result as i64)),
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
    use crate::mir::{Block, IntBinOp, Register, VirtualReg, function::Signature};

    #[test]
    fn folds_wrapping_arithmetic_and_masked_shifts_preserving_traps() {
        for ty in [
            MirType::i8(),
            MirType::i16(),
            MirType::i32(),
            MirType::i64(),
        ] {
            let width = ty.size_bytes() * 8;
            let minimum = ty.signed_integer(1i64 << (width - 1));
            let maximum = minimum.wrapping_sub(1);
            for (op, lhs, rhs, expected) in [
                (IntBinOp::Add, maximum, 1, Some(minimum)),
                (IntBinOp::Add, maximum, 2, Some(minimum + 1)),
                (IntBinOp::Sub, minimum, 1, Some(ty.signed_integer(maximum))),
                (IntBinOp::Mul, minimum, 2, Some(0)),
                (IntBinOp::Mul, minimum, 3, Some(minimum)),
                (IntBinOp::Shl, 1, width as i64, Some(1)),
                (IntBinOp::Shl, 1, width as i64 * 1024 + 1, Some(2)),
                (IntBinOp::SDiv, minimum, -1, None),
                (IntBinOp::SRem, minimum, -1, None),
                (IntBinOp::SDiv, 1, 0, None),
                (IntBinOp::SRem, 1, 0, None),
                (IntBinOp::UDiv, 1, 0, None),
                (IntBinOp::URem, 1, 0, None),
            ] {
                let mut instruction = Instruction::IntBinary {
                    op,
                    ty,
                    dst: Register::Virtual(VirtualReg::gpr(0)),
                    lhs: Operand::Immediate(Immediate::I64(lhs)),
                    rhs: Operand::Immediate(Immediate::I64(rhs)),
                };
                assert_eq!(
                    ConstantFolding.try_fold_int(&mut instruction),
                    expected.is_some(),
                    "{ty} {op}"
                );
                if let Some(expected) = expected {
                    let Instruction::Copy { src, .. } = instruction else {
                        panic!("expected copy")
                    };
                    assert_eq!(src.integer_constant(), Some(expected), "{ty} {op}");
                }
            }
        }
    }

    #[test]
    fn test_signed_division_overflow_prevention() {
        // Test that i64::MIN / -1 is prevented (would overflow)
        let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
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
        let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
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
        let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Sub,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        func.add_block(bb);

        let cf = ConstantFolding;
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(changed);
    }

    #[test]
    fn test_constant_folding_overflow_mul() {
        let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding;
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(changed);
    }

    #[test]
    fn test_constant_folding_unsigned_division() {
        // Unsigned division: large positive / 2 should fold correctly
        let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(-2)), // Interpreted as u64::MAX - 1
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding;
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(changed);

        // Check the result: (u64::MAX - 1) / 2 = 0x7FFFFFFFFFFFFFFE
        match &func.blocks[0].instructions[0] {
            Instruction::Copy { src, .. } => {
                let expected = ((-2i64 as u64) / 2) as i64;
                assert_eq!(src, &Operand::Immediate(Immediate::I64(expected)));
            }
            _ => panic!("Expected Copy"),
        }
    }

    #[test]
    fn test_constant_folding_bitwise() {
        let cf = ConstantFolding;

        let mk = |op: IntBinOp, l: i64, r: i64| {
            let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
            let mut bb = Block::new("entry");
            bb.push(Instruction::IntBinary {
                op,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::I64(l)),
                rhs: Operand::Immediate(Immediate::I64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::Copy {
                src: Operand::Immediate(Immediate::I64(v)),
                ..
            } => *v,
            _ => panic!("Expected folded Copy"),
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
            let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
            let mut bb = Block::new("entry");
            bb.push(Instruction::FloatBinary {
                op,
                ty: MirType::Scalar(ScalarType::F64),
                dst: Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::F64(l)),
                rhs: Operand::Immediate(Immediate::F64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::Copy {
                src: Operand::Immediate(Immediate::F64(v)),
                ..
            } => *v,
            _ => panic!("Expected folded Copy"),
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
            let mut func = Function::new(Signature::new("f")).with_entry("entry".to_string());
            let mut bb = Block::new("entry");
            bb.push(Instruction::IntCmp {
                op,
                ty: MirType::Scalar(ScalarType::I1),
                dst: Register::Virtual(VirtualReg::gpr(0)),
                lhs: Operand::Immediate(Immediate::I64(l)),
                rhs: Operand::Immediate(Immediate::I64(r)),
            });
            func.add_block(bb);
            func
        };

        let extract = |func: &Function| match &func.blocks[0].instructions[0] {
            Instruction::Copy {
                src: Operand::Immediate(Immediate::I8(v)),
                ..
            } => *v,
            _ => panic!("Expected folded to Copy"),
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
