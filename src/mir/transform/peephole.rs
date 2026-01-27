//! Peephole optimizations for MIR.

use crate::mir::instruction::{Immediate, Instruction, IntBinOp, IntCmpOp, Operand};
use crate::mir::{Block, Function};

use super::{Transform, TransformCategory, TransformLevel};

/// Peephole optimizations that do local rewrites.
///
/// Handles arithmetic identities, comparison optimizations, constant folding,
/// strength reduction, and address calculation optimizations.
#[derive(Default)]
pub struct Peephole;

impl Transform for Peephole {
    fn name(&self) -> &'static str {
        "peephole"
    }

    fn description(&self) -> &'static str {
        "Local rewrites for arithmetic, comparison, and algebraic optimizations"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        Ok(self.run_on_function(func))
    }
}

impl Peephole {
    /// Apply peephole rewrites to a function. Returns true if any change occurred.
    pub fn run_on_function(&self, func: &mut Function) -> bool {
        // Safety check: limit block size to prevent excessive processing
        const MAX_BLOCK_INSTRUCTIONS: usize = 10_000;
        for block in &func.blocks {
            if block.instructions.len() > MAX_BLOCK_INSTRUCTIONS {
                return false; // Skip optimization on oversized blocks
            }
        }

        let mut changed = false;
        let loop_headers = compute_back_edge_headers(func);
        for block in &mut func.blocks {
            let in_loop_block = loop_headers.contains(&block.label);
            if self.run_on_block(block, in_loop_block) {
                changed = true;
            }
        }
        changed
    }

    fn run_on_block(&self, block: &mut Block, in_loop_block: bool) -> bool {
        let mut changed = false;

        // First pass: optimize individual instructions
        for inst in &mut block.instructions {
            if self.try_optimize_instruction(inst, in_loop_block) {
                changed = true;
            }
        }

        changed
    }

    /// Try to optimize a single instruction through various peephole patterns
    fn try_optimize_instruction(&self, inst: &mut Instruction, _in_loop_block: bool) -> bool {
        match inst {
            Instruction::IntBinary { op, lhs, rhs, .. } => self.try_fold_int_binary(op, lhs, rhs),
            Instruction::IntCmp { .. } => self.try_fold_int_cmp(inst),
            Instruction::FloatUnary { op, src, .. } => self.try_fold_float_unary(op, src),
            Instruction::Select {
                cond,
                true_val,
                false_val,
                ..
            } => self.try_fold_select(cond, true_val, false_val),
            Instruction::Call { name, args, .. } => self.try_optimize_intrinsic_call(name, args),
            _ => false,
        }
    }

    /// Optimize integer binary operations
    fn try_fold_int_binary(&self, op: &mut IntBinOp, lhs: &mut Operand, rhs: &mut Operand) -> bool {
        let lhs_imm = extract_constant(lhs);
        let rhs_imm = extract_constant(rhs);

        match op {
            IntBinOp::Add => self.fold_add(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::Sub => self.fold_sub(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::Mul => self.fold_mul(op, lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::UDiv => self.fold_div(lhs, rhs, lhs_imm, rhs_imm, false),
            IntBinOp::SDiv => self.fold_div(lhs, rhs, lhs_imm, rhs_imm, true),
            IntBinOp::URem => {
                // Special optimizations: x % c -> x & (c-1) for powers of 2
                if let Some(c) = rhs_imm
                    && c > 0
                    && (c & (c - 1)) == 0
                {
                    let mask = c - 1;
                    *op = IntBinOp::And;
                    *rhs = Operand::Immediate(Immediate::I64(mask));
                    return true;
                }
                self.fold_rem(lhs, rhs, lhs_imm, rhs_imm, false)
            }
            IntBinOp::SRem => self.fold_rem(lhs, rhs, lhs_imm, rhs_imm, true),
            IntBinOp::And => self.fold_bitwise_and(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::Or => self.fold_bitwise_or(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::Xor => self.fold_bitwise_xor(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::Shl => self.fold_shift_left(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::LShr => self.fold_shift_right_logical(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::AShr => self.fold_shift_right_arithmetic(lhs, rhs, lhs_imm, rhs_imm),
        }
    }

    fn fold_add(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        // Canonicalize: prefer register on LHS
        if let (Operand::Immediate(_), Operand::Register(_)) = (&lhs, &rhs) {
            core::mem::swap(lhs, rhs);
            return true;
        }
        // x + 0 => x
        if is_zero(rhs_imm) {
            return false;
        }
        // Constant folding
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_add(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_sub(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        // x - 0 => x (no change needed, but don't perform constant folding)
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_sub(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_mul(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if let (Operand::Immediate(_), Operand::Register(_)) = (&lhs, &rhs) {
            core::mem::swap(lhs, rhs);
            return true;
        }
        if is_one(rhs_imm) {
            return false;
        }
        // x * 0 => 0
        if is_zero(lhs_imm) || is_zero(rhs_imm) {
            *op = IntBinOp::Add;
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }

        // Strength reduction
        if let Some(const_val) = rhs_imm
            && let Some((shift, add_val)) = decompose_multiplication(const_val)
        {
            // We can't easily expand into multiple instructions here (peephole works on single instr).
            // But if it's a pure power of 2, we can convert to shift.
            // decompose_multiplication handles "complex" decompositions which require adding instructions.
            // For simple power of 2:
            if add_val == 0 {
                // Just shift
                *op = IntBinOp::Shl;
                *rhs = Operand::Immediate(Immediate::I64(shift as i64));
                return true;
            }
        }

        // Power of 2 check directly
        if let Some(c) = rhs_imm
            && c > 0
            && (c & (c - 1)) == 0
        {
            let shift = c.trailing_zeros();
            *op = IntBinOp::Shl;
            *rhs = Operand::Immediate(Immediate::I64(shift as i64));
            return true;
        }

        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_mul(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_div(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
        signed: bool,
    ) -> bool {
        if is_one(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && c2 != 0
            && !(signed && c1 == i64::MIN && c2 == -1)
        {
            let result = if signed {
                c1 / c2
            } else {
                ((c1 as u64) / (c2 as u64)) as i64
            };
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_rem(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
        signed: bool,
    ) -> bool {
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && c2 != 0
        {
            let result = if signed {
                c1 % c2
            } else {
                ((c1 as u64) % (c2 as u64)) as i64
            };
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_bitwise_and(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if let (Operand::Immediate(_), Operand::Register(_)) = (&lhs, &rhs) {
            core::mem::swap(lhs, rhs);
            return true;
        }
        if is_all_ones(rhs_imm) {
            return false;
        }
        if is_zero(lhs_imm) || is_zero(rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(c1 & c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_bitwise_or(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if let (Operand::Immediate(_), Operand::Register(_)) = (&lhs, &rhs) {
            core::mem::swap(lhs, rhs);
            return true;
        }
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(c1 | c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_bitwise_xor(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if let (Operand::Immediate(_), Operand::Register(_)) = (&lhs, &rhs) {
            core::mem::swap(lhs, rhs);
            return true;
        }
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Operand::Register(r1), Operand::Register(r2)) = (&*lhs, &*rhs)
            && r1 == r2
        {
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(c1 ^ c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_shift_left(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && (0..64).contains(&c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(c1 << c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_shift_right_logical(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && (0..64).contains(&c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(((c1 as u64) >> c2) as i64));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn fold_shift_right_arithmetic(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        if is_zero(rhs_imm) {
            return false;
        }
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && (0..64).contains(&c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(c1 >> c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    fn try_fold_int_cmp(&self, inst: &mut Instruction) -> bool {
        if let Instruction::IntCmp {
            op,
            lhs,
            rhs,
            dst,
            ty,
        } = inst
        {
            let lhs_imm = extract_constant(lhs);
            let rhs_imm = extract_constant(rhs);

            if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm) {
                let result = match op {
                    IntCmpOp::Eq => c1 == c2,
                    IntCmpOp::Ne => c1 != c2,
                    IntCmpOp::SLt => c1 < c2,
                    IntCmpOp::SLe => c1 <= c2,
                    IntCmpOp::SGt => c1 > c2,
                    IntCmpOp::SGe => c1 >= c2,
                    IntCmpOp::ULt => (c1 as u64) < (c2 as u64),
                    IntCmpOp::ULe => (c1 as u64) <= (c2 as u64),
                    IntCmpOp::UGt => (c1 as u64) > (c2 as u64),
                    IntCmpOp::UGe => (c1 as u64) >= (c2 as u64),
                };

                // Replace with IntBinary Add 0 (Move)
                let result_val = if result { 1 } else { 0 };
                *inst = Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: *ty,
                    dst: dst.clone(),
                    lhs: Operand::Immediate(Immediate::I64(result_val)),
                    rhs: Operand::Immediate(Immediate::I64(0)),
                };
                return true;
            }
        }
        false
    }

    fn try_fold_float_unary(
        &self,
        op: &mut crate::mir::instruction::FloatUnOp,
        src: &mut Operand,
    ) -> bool {
        let src_imm = extract_float_constant(src);

        if let Some(c) = src_imm {
            let result = match op {
                crate::mir::instruction::FloatUnOp::FNeg => -c,
                crate::mir::instruction::FloatUnOp::FSqrt if c >= 0.0 => c.sqrt(),
                _ => return false,
            };
            *src = Operand::Immediate(Immediate::F64(result));
            return true;
        }

        false
    }

    fn try_fold_select(
        &self,
        _cond: &mut crate::mir::Register,
        true_val: &mut Operand,
        false_val: &mut Operand,
    ) -> bool {
        if true_val == false_val {
            // If both values are same, this is a redundant select.
            // But removing it requires changing the instruction to a Move.
            // The current peephole pass structure iterates mutably over instructions.
            // If we return true, we claim modification.
            // But we need to change Instruction::Select to Instruction::IntBinary (Add 0) or similar?
            // The `try_optimize_instruction` calls `try_fold_select`.
            // That function takes fields.
            // We should refactor `try_optimize_instruction` to handle opcode changes if we want to support this.
            // Currently, let's leave as is but just document it works for values.
            return false;
        }
        false
    }

    fn try_optimize_intrinsic_call(&self, name: &str, args: &mut [Operand]) -> bool {
        match name {
            "print" => {
                matches!(args.first(), Some(Operand::Immediate(Immediate::I64(0))))
            }
            _ => false,
        }
    }
}

fn extract_constant(operand: &Operand) -> Option<i64> {
    match operand {
        Operand::Immediate(Immediate::I8(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I16(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I32(v)) => Some(*v as i64),
        Operand::Immediate(Immediate::I64(v)) => Some(*v),
        _ => None,
    }
}

fn extract_float_constant(operand: &Operand) -> Option<f64> {
    match operand {
        Operand::Immediate(Immediate::F32(v)) => Some(*v as f64),
        Operand::Immediate(Immediate::F64(v)) => Some(*v),
        _ => None,
    }
}

fn is_zero(i: Option<i64>) -> bool {
    i == Some(0)
}

fn is_one(i: Option<i64>) -> bool {
    i == Some(1)
}

fn is_all_ones(i: Option<i64>) -> bool {
    i == Some(-1)
}

/// Decompose multiplication by constant into shift and add operations
fn decompose_multiplication(const_val: i64) -> Option<(u32, i64)> {
    let abs_val = const_val.abs();
    if abs_val > 0 && (abs_val & (abs_val - 1)) == 0 {
        return Some((abs_val.trailing_zeros(), 0));
    }
    // We can support simple shifts + add/sub.
    // This is primarily useful if we can replace the MUL instruction with a sequence,
    // which Peephole isn't well equipped for (1->N expansion).
    // So we return None for now unless it's a pure shift (handled above).
    None
}

/// Identify loop headers via simple back-edge detection using block order.
fn compute_back_edge_headers(func: &Function) -> std::collections::HashSet<String> {
    use std::collections::{HashMap, HashSet};
    let mut label_index: HashMap<&str, usize> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        label_index.insert(&b.label, i);
    }
    let mut headers: HashSet<String> = HashSet::new();
    for (i, b) in func.blocks.iter().enumerate() {
        if let Some(term) = b.instructions.last() {
            match term {
                Instruction::Jmp { target } => {
                    if let Some(&tidx) = label_index.get(target.as_str())
                        && tidx <= i
                    {
                        headers.insert(target.clone());
                    }
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if let Some(&tidx) = label_index.get(true_target.as_str())
                        && tidx <= i
                    {
                        headers.insert(true_target.clone());
                    }
                    if let Some(&fidx) = label_index.get(false_target.as_str())
                        && fidx <= i
                    {
                        headers.insert(false_target.clone());
                    }
                }
                _ => {}
            }
        }
    }
    headers
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::register::{Register, VirtualReg};
    use crate::mir::types::{MirType, ScalarType};

    #[test]
    fn fold_add_zero_right() {
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(!changed);
    }

    #[test]
    fn fold_mul_one_left() {
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Register(Register::Virtual(VirtualReg::gpr(2))),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);
    }

    #[test]
    fn fold_int_cmp_true() {
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntCmp {
            op: crate::mir::IntCmpOp::SLt,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        let bb = &func.blocks[0];
        match &bb.instructions[0] {
            Instruction::IntBinary { op, lhs, rhs, .. } => {
                assert_eq!(*op, crate::mir::IntBinOp::Add);
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(1))); // True result (1)
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary (move)"),
        }
    }

    #[test]
    fn test_peephole_empty_function() {
        let mut func = Function::new(crate::mir::function::Signature::new("empty"))
            .with_entry("entry".to_string());
        let bb = Block::new("entry");
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(!changed);
    }

    #[test]
    fn test_peephole_no_canonicalization_loop() {
        // Test that canonicalization doesn't cause infinite swap loop
        // imm + reg should become reg + imm and stay that way
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(5)),
            rhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
        });
        func.add_block(bb);

        let pass = Peephole::default();

        // Run multiple times to ensure no infinite loop
        for _ in 0..10 {
            pass.run_on_function(&mut func);
        }

        // LHS should be register now
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, rhs, .. } => {
                assert!(matches!(lhs, Operand::Register(_)));
                assert!(matches!(rhs, Operand::Immediate(_)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_division_by_zero_not_folded() {
        // Division by zero should not be constant-folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(10)),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        // Should NOT fold division by zero
        assert!(!changed);
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { op, rhs, .. } => {
                assert_eq!(*op, IntBinOp::SDiv);
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_i64_min_div_neg_one() {
        // i64::MIN / -1 would overflow, should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(-1)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        // Should NOT fold due to overflow
        assert!(!changed);
    }

    #[test]
    fn test_peephole_overflow_add_not_folded() {
        // Overflow in constant folding should be skipped
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        // Checked_add returns None on overflow, so no fold
        assert!(!changed);
    }

    #[test]
    fn test_peephole_mul_zero_result() {
        // x * 0 = 0 is a valid transformation
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        // Result should be 0 + 0 (constant zero)
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { op, lhs, rhs, .. } => {
                assert_eq!(*op, IntBinOp::Add);
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(0)));
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_xor_self_is_zero() {
        // x ^ x = 0
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        let reg = Register::Virtual(VirtualReg::gpr(1));
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Xor,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(reg.clone()),
            rhs: Operand::Register(reg),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, rhs, .. } => {
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(0)));
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_and_zero_is_zero() {
        // x & 0 = 0
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::And,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        assert!(changed);
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, rhs, .. } => {
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(0)));
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_shift_bounds() {
        // Shift by >= 64 should not be folded (undefined behavior)
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Shl,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(64)), // Out of bounds
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);

        // Should NOT fold shift by 64+
        assert!(!changed);
    }

    #[test]
    fn test_peephole_stress_many_instructions() {
        // Ensure peephole doesn't hang on large blocks
        let mut func = Function::new(crate::mir::function::Signature::new("stress"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");

        for i in 0..1000 {
            bb.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(i)),
                lhs: Operand::Immediate(Immediate::I64(i as i64)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            });
        }
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        // Constant folding should occur
        assert!(changed);
    }

    #[test]
    fn test_peephole_unsigned_cmp_folding() {
        // ULt: 0 < 1 is true for unsigned
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntCmp {
            op: crate::mir::IntCmpOp::ULt,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(0)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        // Should fold to 1 (true)
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, .. } => {
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(1)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_peephole_negative_unsigned_cmp() {
        // -1 as unsigned is MAX, so -1 > 0 (unsigned) is true
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntCmp {
            op: crate::mir::IntCmpOp::UGt,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(-1)), // u64::MAX
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, .. } => {
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(1))); // true
            }
            _ => panic!("Expected IntBinary"),
        }
    }
}
