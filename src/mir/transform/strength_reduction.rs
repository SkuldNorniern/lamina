use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, IntBinOp, MirType, Operand, Register};
use crate::mir::instruction::Immediate;

/// Strength Reduction Transform
///
/// Replaces expensive operations with cheaper equivalents:
/// - Multiplication by powers of 2 → left shifts
/// - Division by powers of 2 → right shifts (unsigned)
/// - Modulo by powers of 2 → bitwise AND
/// - Multiplication by constants → optimized sequences
#[derive(Default)]
pub struct StrengthReduction;

impl Transform for StrengthReduction {
    fn name(&self) -> &'static str {
        "strength_reduction"
    }

    fn description(&self) -> &'static str {
        "Replace expensive operations with cheaper equivalents (shifts, AND, etc.)"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl StrengthReduction {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                if self.try_reduce_strength(instr) {
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    /// Try to apply strength reduction to an instruction
    fn try_reduce_strength(&self, instr: &mut Instruction) -> bool {
        match instr {
            Instruction::IntBinary {
                op,
                dst: _,
                ty,
                lhs,
                rhs,
            } => self.try_reduce_int_binary(op, lhs, rhs, ty),
            _ => false,
        }
    }

    /// Reduce integer binary operations
    fn try_reduce_int_binary(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        ty: &MirType,
    ) -> bool {
        let rhs_const = extract_constant(rhs);

        match *op {
            IntBinOp::Mul => self.reduce_multiplication(op, lhs, rhs, rhs_const, ty),
            IntBinOp::UDiv => self.reduce_unsigned_division(op, lhs, rhs, rhs_const, ty),
            IntBinOp::SDiv => self.reduce_signed_division(op, lhs, rhs, rhs_const, ty),
            IntBinOp::URem => self.reduce_unsigned_remainder(op, lhs, rhs, rhs_const, ty),
            IntBinOp::SRem => self.reduce_signed_remainder(op, lhs, rhs, rhs_const, ty),
            _ => false,
        }
    }

    /// Reduce multiplication operations
    fn reduce_multiplication(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        rhs_const: Option<i64>,
        ty: &MirType,
    ) -> bool {
        if let Some(power_of_2) = rhs_const.and_then(is_power_of_2) {
            // x * 2^k → x << k
            *op = IntBinOp::Shl;
            *rhs = Operand::Immediate(Immediate::I64(power_of_2));
            return true;
        }

        // Check for multiplication by other common constants used in algorithms
        if let Some(const_val) = rhs_const {
            match const_val {
                // Common small constants that appear in various algorithms
                3 => {
                    // x * 3 → (x << 1) + x, but this would require multiple instructions
                    // For now, leave as multiplication since it's not clearly faster
                    return false;
                }
                5 => {
                    // x * 5 → (x << 2) + x, but again requires multiple instructions
                    return false;
                }
                6 => {
                    // x * 6 → (x << 1) + (x << 2), requires multiple instructions
                    return false;
                }
                9 => {
                    // x * 9 → (x << 3) + x, requires multiple instructions
                    return false;
                }
                10 => {
                    // x * 10 → (x << 1) + (x << 3), requires multiple instructions
                    return false;
                }
                _ => {
                    // Check for more complex decompositions
                    if let Some((shift1, shift2)) = decompose_multiplication(const_val) {
                        // For now, keep simple case. More complex decompositions
                        // could be handled by generating multiple instructions,
                        // but that would require more sophisticated infrastructure.
                        return false;
                    }
                }
            }
        }

        false
    }

    /// Reduce unsigned division by powers of 2
    fn reduce_unsigned_division(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        rhs_const: Option<i64>,
        ty: &MirType,
    ) -> bool {
        if let Some(power_of_2) = rhs_const.and_then(is_power_of_2) {
            // x / 2^k → x >>> k (logical shift right)
            *op = IntBinOp::LShr;
            *rhs = Operand::Immediate(Immediate::I64(power_of_2));
            return true;
        }
        false
    }

    /// Reduce signed division by powers of 2
    fn reduce_signed_division(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        rhs_const: Option<i64>,
        ty: &MirType,
    ) -> bool {
        // Without range analysis we cannot prove non-negativity; do not transform.
        let _ = (op, lhs, rhs, rhs_const, ty);
        false
    }

    /// Reduce unsigned remainder by powers of 2
    fn reduce_unsigned_remainder(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        rhs_const: Option<i64>,
        ty: &MirType,
    ) -> bool {
        if let Some(power_of_2) = rhs_const.and_then(is_power_of_2) {
            // x % 2^k → x & (2^k - 1)
            *op = IntBinOp::And;
            *rhs = Operand::Immediate(Immediate::I64((1i64 << power_of_2) - 1));
            return true;
        }
        false
    }

    /// Reduce signed remainder by powers of 2
    fn reduce_signed_remainder(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        rhs_const: Option<i64>,
        ty: &MirType,
    ) -> bool {
        // Without range analysis, do not rewrite signed remainder to bitmasking.
        let _ = (op, lhs, rhs, rhs_const, ty);
        false
    }
}

/// Check if a number is a power of 2, return the exponent if so
fn is_power_of_2(n: i64) -> Option<i64> {
    if n > 0 && (n & (n - 1)) == 0 {
        Some(n.trailing_zeros() as i64)
    } else {
        None
    }
}

/// Decompose a multiplication constant into shift operations
/// Returns (shift1, shift2) for expressions like (x << shift1) + (x << shift2)
/// This is useful for constants like 3 = (1 << 0) + (1 << 1), 5 = (1 << 0) + (1 << 2), etc.
fn decompose_multiplication(n: i64) -> Option<(i64, i64)> {
    if n <= 1 {
        return None;
    }

    // Find two powers of 2 that sum to n
    for i in 0..64 {
        let pow_i = 1i64 << i;
        if pow_i >= n {
            break;
        }
        for j in (i + 1)..64 {
            let pow_j = 1i64 << j;
            if pow_i + pow_j == n {
                return Some((i, j));
            }
        }
    }

    None
}

/// Extract integer constant from operand
fn extract_constant(operand: &Operand) -> Option<i64> {
    match operand {
        Operand::Immediate(Immediate::I8(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I16(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I32(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I64(v)) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_power_of_2_detection() {
        assert_eq!(is_power_of_2(1), Some(0)); // 2^0 = 1
        assert_eq!(is_power_of_2(2), Some(1)); // 2^1 = 2
        assert_eq!(is_power_of_2(4), Some(2)); // 2^2 = 4
        assert_eq!(is_power_of_2(8), Some(3)); // 2^3 = 8
        assert_eq!(is_power_of_2(16), Some(4)); // 2^4 = 16

        assert_eq!(is_power_of_2(0), None); // 0 is not positive
        assert_eq!(is_power_of_2(3), None); // 3 is not a power of 2
        assert_eq!(is_power_of_2(6), None); // 6 is not a power of 2
        assert_eq!(is_power_of_2(-1), None); // Negative numbers
    }

    #[test]
    fn test_multiplication_by_power_of_2() {
        // Test x * 4 → x << 2
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(4)), // 2^2
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2);

        // Check that multiplication was converted to shift
        match &entry.instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::Shl);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(2))); // log2(4) = 2
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_unsigned_division_by_power_of_2() {
        // Test x / 8 → x >>> 3
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::UDiv,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(8)), // 2^3
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");

        // Check that unsigned division was converted to logical shift right
        match &entry.instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::LShr);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(3))); // log2(8) = 3
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_signed_division_by_power_of_2() {
        // Signed division by powers of 2 is disabled for safety (requires range analysis)
        // Test that x / 16 does NOT get transformed
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::SDiv,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(16)), // 2^4
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        // Signed division should NOT be transformed without range analysis
        assert!(!changed);

        let entry = func.get_block("entry").expect("entry block exists");

        // Check that signed division remains unchanged
        match &entry.instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::SDiv);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(16)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_unsigned_remainder_by_power_of_2() {
        // Test x % 32 → x & 31
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::URem,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(32)), // 2^5
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");

        // Check that unsigned remainder was converted to AND with mask
        match &entry.instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::And);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(31))); // 32 - 1 = 31
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_no_change_for_non_power_of_2() {
        // Test that non-powers of 2 are not changed
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(6)), // Not a power of 2
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        // Should not have changed
        assert!(!changed);

        let entry = func.get_block("entry").expect("entry block exists");

        // Check that multiplication is still multiplication
        match &entry.instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::Mul);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(6)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_no_change_for_non_constants() {
        // Test that operations with non-constant operands are not changed
        let mut func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()), // Variable, not constant
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let mut func = func;
        let pass = StrengthReduction::default();
        let changed = pass.apply(&mut func).expect("Strength reduction should succeed");

        // Should not have changed
        assert!(!changed);
    }
}
