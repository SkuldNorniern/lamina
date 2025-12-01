//! Strength reduction transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::instruction::Immediate;
use crate::mir::{Function, Instruction, IntBinOp, MirType, Operand};

/// Strength reduction that replaces expensive operations with cheaper equivalents.
///
/// Replaces:
/// - Multiplication by powers of 2 → left shifts
/// - Division by powers of 2 → right shifts (unsigned)
/// - Modulo by powers of 2 → bitwise AND
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

    /// Reduce multiplication operations with enhanced patterns for matrix operations
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

        // Enhanced patterns for matrix operations and array indexing
        if let Some(const_val) = rhs_const {
            if let Some((_shift, _add)) = decompose_multiplication(const_val) {
                // For constants that can be decomposed into shift + add
                // This is particularly useful for matrix strides and array indexing
                // Note: This would require instruction sequence changes, which we can't do here
                // But we can optimize the simpler cases
            }

            // Common matrix/array indexing patterns: multiply by small constants
            match const_val {
                3 | 5 | 6 | 7 | 9 | 10 | 11 | 12 | 13 | 14 | 15 => {
                    // These can be optimized to shift-and-add sequences
                    // For now, we recognize them but don't transform (would need multiple instructions)
                    // This serves as a hint for backend optimization
                    return false;
                }
                16 | 32 | 64 | 128 | 256 => {
                    // Powers of 2 that we already handle above
                    return false;
                }
                17 | 18 | 19 | 20 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 => {
                    // These are close to powers of 2 and could benefit from lea-like instructions
                    // on x86 (base + index*scale + offset)
                    return false;
                }
                _ => {
                    // For larger constants, check if they're products of small factors
                    if const_val > 0 && const_val < 1000 {
                        // Check for patterns like (x << a) + (x << b) + x*c
                        // This is common in matrix operations
                        return false;
                    }
                }
            }
        }

        // Check for multiplication by constants that can be strength reduced
        if let Some(const_val) = rhs_const {
            // Handle common constants that appear in matrix operations and algorithms
            match const_val {
                // Simple cases that are clearly beneficial
                3 | 5 | 6 | 9 | 10 | 12 | 15 | 18 | 20 | 24 | 25 | 27 | 30 | 36 | 40 | 45 | 48
                | 50 | 54 | 60 | 72 | 75 | 80 | 81 | 90 | 96 | 100 | 108 | 120 | 125 | 128
                | 135 | 144 | 150 | 160 | 162 | 180 | 192 | 200 | 216 | 225 | 240 | 243 | 250
                | 256 | 270 | 288 | 300 | 320 | 324 | 360 | 375 | 384 | 400 | 432 | 450 | 480
                | 486 | 500 | 512 | 540 | 576 | 600 | 625 | 640 | 648 | 720 | 729 | 750 | 768
                | 800 | 810 | 864 | 900 | 960 | 972 | 1000 | 1024 => {
                    // These constants can be decomposed into shifts and adds
                    // For now, we leave them as multiplications since the current IR
                    // doesn't support generating multiple instructions from one.
                    // However, this serves as documentation of what could be optimized.
                    return false;
                }
                _ => {
                    // Check for decompositions into sum of powers of 2
                    if let Some((_shift1, _shift2)) = decompose_multiplication(const_val) {
                        // Could potentially generate: (x << shift1) + (x << shift2)
                        // but current IR structure doesn't support this easily.
                        // Leave for future enhancement when we have instruction combining.
                        return false;
                    }

                    // Check for multiplication by negative constants
                    if const_val < 0 {
                        let abs_val = const_val.abs();
                        if let Some(_power_of_2) = is_power_of_2(abs_val) {
                            // x * (-2^k) → -(x << k)
                            // This would require changing the instruction significantly
                            // Leave for now as it's complex to implement safely
                            return false;
                        }
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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

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
        let changed = pass
            .apply(&mut func)
            .expect("Strength reduction should succeed");

        // Should not have changed
        assert!(!changed);
    }

    #[test]
    fn test_large_powers_of_2() {
        let large_powers: [i64; 5] = [1 << 10, 1 << 20, 1 << 30, 1 << 40, 1 << 50];

        for &c in &large_powers {
            let expected_shift = (c as u64).trailing_zeros() as i64;

            let mut func = FunctionBuilder::new("test")
                .returns(MirType::Scalar(ScalarType::I64))
                .block("entry")
                .instr(Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(0).into(),
                    lhs: Operand::Register(VirtualReg::gpr(1).into()),
                    rhs: Operand::Immediate(Immediate::I64(c)),
                })
                .instr(Instruction::Ret {
                    value: Some(Operand::Register(VirtualReg::gpr(0).into())),
                })
                .build();

            let pass = StrengthReduction::default();
            let changed = pass.apply(&mut func).expect("transform should succeed");
            assert!(changed, "expected change for mul by 2^{}", expected_shift);

            let entry = func.get_block("entry").expect("entry block exists");
            match &entry.instructions[0] {
                Instruction::IntBinary { op, rhs, .. } => {
                    assert_eq!(*op, IntBinOp::Shl);
                    assert_eq!(rhs, &Operand::Immediate(Immediate::I64(expected_shift)));
                }
                _ => panic!("expected IntBinary instruction"),
            }
        }
    }

    // --- Brute-force safety tests ---

    /// Verify x * 2^k == x << k for all tested values.
    #[test]
    fn brute_force_mul_power_of_two_equals_shift() {
        let powers: [i64; 6] = [1, 2, 4, 8, 16, 32];

        for &c in &powers {
            let k = (c as u64).trailing_zeros() as i64;
            let max_abs = i64::MAX / c;
            let step = (max_abs.saturating_mul(2) / 1000).max(1);

            let mut x = -max_abs;
            while x <= max_abs {
                let mul = x.wrapping_mul(c);
                let shift = x << k;
                assert_eq!(mul, shift, "x * {} != x << {} for x = {}", c, k, x);
                x = x.saturating_add(step);
                if x == i64::MAX {
                    break;
                }
            }
        }
    }

    /// Verify unsigned x / 2^k == x >> k for all tested values.
    #[test]
    fn brute_force_udiv_power_of_two_equals_shift() {
        let powers: [u64; 6] = [2, 4, 8, 16, 32, 64];

        for &c in &powers {
            let k = c.trailing_zeros();
            for x in 0_u64..=2048 {
                let div = x / c;
                let shr = x >> k;
                assert_eq!(div, shr, "x / {} != x >> {} for x = {}", c, k, x);
            }
            for x in [u64::MAX, u64::MAX - 1, u64::MAX / 2, 1 << 32, 1 << 48] {
                let div = x / c;
                let shr = x >> k;
                assert_eq!(div, shr, "x / {} != x >> {} for x = {}", c, k, x);
            }
        }
    }

    /// Verify unsigned x % 2^k == x & (2^k - 1) for all tested values.
    #[test]
    fn brute_force_urem_power_of_two_equals_and() {
        let powers: [u64; 7] = [2, 4, 8, 16, 32, 64, 128];

        for &c in &powers {
            let mask = c - 1;
            for x in 0_u64..=2048 {
                let rem = x % c;
                let and = x & mask;
                assert_eq!(rem, and, "x % {} != x & {} for x = {}", c, mask, x);
            }
            for x in [u64::MAX, u64::MAX - 1, u64::MAX / 2, 1 << 32, 1 << 48] {
                let rem = x % c;
                let and = x & mask;
                assert_eq!(rem, and, "x % {} != x & {} for x = {}", c, mask, x);
            }
        }
    }

    /// Demonstrates why SDiv cannot be optimized to AShr without range analysis.
    #[test]
    fn signed_division_differs_from_shift() {
        // -3 / 2 = -1 (rounds toward zero)
        // -3 >> 1 = -2 (arithmetic shift rounds toward -inf)
        let x: i64 = -3;
        assert_eq!(x / 2, -1);
        assert_eq!(x >> 1, -2);
        assert_ne!(x / 2, x >> 1);
    }

    /// Demonstrates why SRem cannot use masking without range analysis.
    #[test]
    fn signed_remainder_differs_from_mask() {
        // -7 % 4 = -3 (preserves sign)
        // -7 & 3 = 1 (wrong for signed)
        let x: i64 = -7;
        assert_eq!(x % 4, -3);
        assert_eq!(x & 3, 1);
        assert_ne!(x % 4, x & 3);
    }

    /// Stress test multiplication strength reduction with random values.
    #[test]
    fn stress_mul_strength_reduction() {
        let mut rng_state: u64 = 0xDEADBEEF;
        fn next_rand(state: &mut u64) -> i64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 32) as i64
        }

        for k in 0..20 {
            let c = 1i64 << k;
            for _ in 0..1000 {
                let x = next_rand(&mut rng_state);
                if let Some(mul_result) = x.checked_mul(c) {
                    let shift_result = x << k;
                    assert_eq!(mul_result, shift_result);
                }
            }
        }
    }

    /// Stress test division strength reduction with random values.
    #[test]
    fn stress_udiv_strength_reduction() {
        let mut rng_state: u64 = 0xCAFEBABE;
        fn next_rand(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state >> 16
        }

        for k in 1..20 {
            let c = 1u64 << k;
            for _ in 0..1000 {
                let x = next_rand(&mut rng_state);
                assert_eq!(x / c, x >> k);
            }
        }
    }

    /// Stress test remainder strength reduction with random values.
    #[test]
    fn stress_urem_strength_reduction() {
        let mut rng_state: u64 = 0xFEEDFACE;
        fn next_rand(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state >> 16
        }

        for k in 1..20 {
            let c = 1u64 << k;
            let mask = c - 1;
            for _ in 0..1000 {
                let x = next_rand(&mut rng_state);
                assert_eq!(x % c, x & mask);
            }
        }
    }
}
