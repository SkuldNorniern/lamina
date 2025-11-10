use crate::mir::instruction::{Immediate, Instruction, IntBinOp, IntCmpOp, Operand};
use crate::mir::types::{MirType, ScalarType};
use crate::mir::{Block, Function};

use super::{Transform, TransformCategory, TransformLevel};

/// Advanced peephole optimizations for MIR
///
/// This pass performs comprehensive local rewrites including:
/// - Arithmetic identities and simplifications
/// - Comparison optimizations
/// - Algebraic transformations
/// - Constant folding patterns
/// - Instruction strength reduction
#[derive(Default)]
pub struct Peephole;

impl Transform for Peephole {
    fn name(&self) -> &'static str {
        "peephole"
    }

    fn description(&self) -> &'static str {
        "Advanced local rewrites for arithmetic, comparison, and algebraic optimizations"
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

        for inst in &mut block.instructions {
            if self.try_optimize_instruction(inst, in_loop_block) {
                changed = true;
            }
        }

        changed
    }

    /// Try to optimize a single instruction through various peephole patterns
    fn try_optimize_instruction(&self, inst: &mut Instruction, in_loop_block: bool) -> bool {
        match inst {
            Instruction::IntBinary {
                op,
                dst: _,
                ty: _,
                lhs,
                rhs,
            } => self.try_fold_int_binary(op, lhs, rhs),
            // For IntCmp, allow full replacement with a constant-producing move
            Instruction::IntCmp { op, dst, ty: _cmp_ty, lhs, rhs } => {
                // Be conservative inside loop blocks to avoid changing loop progress/termination subtly
                if !in_loop_block {
                    if let Some(b) = Self::evaluate_int_cmp(op, lhs, rhs) {
                        // Replace comparison with a constant move: dst = (b ? 1 : 0)
                        let new_inst = Instruction::IntBinary {
                            op: IntBinOp::Add,
                            ty: MirType::Scalar(ScalarType::I64),
                            dst: dst.clone(),
                            lhs: Operand::Immediate(Immediate::I64(if b { 1 } else { 0 })),
                            rhs: Operand::Immediate(Immediate::I64(0)),
                        };
                        *inst = new_inst;
                        return true;
                    }
                }
                // If not fully reducible (or guarded), do not rewrite
                false
            }
            Instruction::FloatUnary {
                op,
                dst: _,
                ty: _,
                src,
            } => self.try_fold_float_unary(op, src),
            Instruction::Select {
                dst: _,
                ty: _,
                cond,
                true_val,
                false_val,
            } => self.try_fold_select(cond, true_val, false_val),
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
            IntBinOp::Mul => self.fold_mul(lhs, rhs, lhs_imm, rhs_imm),
            IntBinOp::UDiv => self.fold_div(lhs, rhs, lhs_imm, rhs_imm, false),
            IntBinOp::SDiv => self.fold_div(lhs, rhs, lhs_imm, rhs_imm, true),
            IntBinOp::URem => {
                // Special optimizations: x % c -> x & (c-1) for small powers of 2
                // Common in prime generation and divisibility testing
                if let Some(c) = rhs_imm {
                    if c == 2 {
                        // x % 2 -> x & 1 (extract LSB for even/odd)
                        *op = IntBinOp::And;
                        *rhs = Operand::Immediate(Immediate::I64(1));
                        return true;
                    } else if c == 4 {
                        // x % 4 -> x & 3 (mask bottom 2 bits)
                        *op = IntBinOp::And;
                        *rhs = Operand::Immediate(Immediate::I64(3));
                        return true;
                    } else if c == 8 {
                        // x % 8 -> x & 7 (mask bottom 3 bits)
                        *op = IntBinOp::And;
                        *rhs = Operand::Immediate(Immediate::I64(7));
                        return true;
                    } else if c == 16 {
                        // x % 16 -> x & 15 (mask bottom 4 bits)
                        *op = IntBinOp::And;
                        *rhs = Operand::Immediate(Immediate::I64(15));
                        return true;
                    }
                }
                self.fold_rem(lhs, rhs, lhs_imm, rhs_imm, false)
            }
            IntBinOp::SRem => {
                self.fold_rem(lhs, rhs, lhs_imm, rhs_imm, true)
            }
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
        // Canonicalize: prefer register on LHS, immediate on RHS
        match (&lhs, &rhs) {
            (Operand::Immediate(_), Operand::Register(_)) => {
                core::mem::swap(lhs, rhs);
                return true;
            }
            _ => {}
        }
        // x + 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // 0 + x => x (swap operands to canonical form)
        if is_zero(lhs_imm) {
            core::mem::swap(lhs, rhs);
            return true; // Operands were swapped
        }
        // Constant folding: c1 + c2 => (c1+c2) with overflow check
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_add(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Skip folding on overflow
        false
    }

    fn fold_sub(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        // x - 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // Constant folding: c1 - c2 => (c1-c2) with overflow check
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_sub(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Skip folding on overflow
        false
    }

    fn fold_mul(
        &self,
        lhs: &mut Operand,
        rhs: &mut Operand,
        lhs_imm: Option<i64>,
        rhs_imm: Option<i64>,
    ) -> bool {
        // Canonicalize: prefer register on LHS, immediate on RHS
        match (&lhs, &rhs) {
            (Operand::Immediate(_), Operand::Register(_)) => {
                core::mem::swap(lhs, rhs);
                return true;
            }
            _ => {}
        }
        // x * 1 => x (already in optimal form, no change needed)
        if is_one(rhs_imm) {
            return false; // No change made
        }
        // 1 * x => x (swap operands to canonical form)
        if is_one(lhs_imm) {
            core::mem::swap(lhs, rhs);
            return true; // Operands were swapped
        }
        // x * 0 => 0, 0 * x => 0
        if is_zero(lhs_imm) || is_zero(rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Constant folding: c1 * c2 => (c1*c2) with overflow check
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && let Some(result) = c1.checked_mul(c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(result));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Skip folding on overflow
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
        // x / 1 => x (already in optimal form, no change needed)
        if is_one(rhs_imm) {
            return false; // No change made
        }
        // Constant folding with safety check
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && c2 != 0
            && !(signed && c1 == i64::MIN && c2 == -1)
        // Overflow check for i64::MIN / -1
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
        // Constant folding with safety check
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
        // Canonicalize: prefer register on LHS, immediate on RHS
        match (&lhs, &rhs) {
            (Operand::Immediate(_), Operand::Register(_)) => {
                core::mem::swap(lhs, rhs);
                return true;
            }
            _ => {}
        }
        // x & -1 => x (already in optimal form, no change needed)
        if is_all_ones(rhs_imm) {
            return false; // No change made
        }
        // -1 & x => x (swap operands to canonical form)
        if is_all_ones(lhs_imm) {
            core::mem::swap(lhs, rhs);
            return true; // Operands were swapped
        }
        // x & 0 => 0, 0 & x => 0
        if is_zero(lhs_imm) || is_zero(rhs_imm) {
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Constant folding
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
        // Canonicalize: prefer register on LHS, immediate on RHS
        match (&lhs, &rhs) {
            (Operand::Immediate(_), Operand::Register(_)) => {
                core::mem::swap(lhs, rhs);
                return true;
            }
            _ => {}
        }
        // x | 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // 0 | x => x (swap operands to canonical form)
        if is_zero(lhs_imm) {
            core::mem::swap(lhs, rhs);
            return true; // Operands were swapped
        }
        // Constant folding
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
        // Canonicalize: prefer register on LHS, immediate on RHS
        match (&lhs, &rhs) {
            (Operand::Immediate(_), Operand::Register(_)) => {
                core::mem::swap(lhs, rhs);
                return true;
            }
            _ => {}
        }
        // x ^ 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // 0 ^ x => x (swap operands to canonical form)
        if is_zero(lhs_imm) {
            core::mem::swap(lhs, rhs);
            return true; // Operands were swapped
        }
        // x ^ x => 0 (if same register)
        if let (Operand::Register(r1), Operand::Register(r2)) = (&*lhs, &*rhs)
            && r1 == r2
        {
            *lhs = Operand::Immediate(Immediate::I64(0));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        // Constant folding
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
        // x << 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // Constant folding
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
        // x >>> 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // Constant folding
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
        // x >> 0 => x (already in optimal form, no change needed)
        if is_zero(rhs_imm) {
            return false; // No change made
        }
        // Constant folding
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm)
            && (0..64).contains(&c2)
        {
            *lhs = Operand::Immediate(Immediate::I64(c1 >> c2));
            *rhs = Operand::Immediate(Immediate::I64(0));
            return true;
        }
        false
    }

    /// Evaluate integer comparison into a boolean if possible (purely local).
    fn evaluate_int_cmp(op: &IntCmpOp, lhs: &Operand, rhs: &Operand) -> Option<bool> {
        let lhs_imm = extract_constant(lhs);
        let rhs_imm = extract_constant(rhs);
        // Constant-vs-constant
        if let (Some(c1), Some(c2)) = (lhs_imm, rhs_imm) {
            return Some(match op {
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
            });
        }
        // x ? x patterns
        if let (Operand::Register(r1), Operand::Register(r2)) = (lhs, rhs) {
            if r1 == r2 {
                return Some(match op {
                    IntCmpOp::Eq | IntCmpOp::SLe | IntCmpOp::ULe | IntCmpOp::SGe | IntCmpOp::UGe => true,
                    IntCmpOp::Ne | IntCmpOp::SLt | IntCmpOp::ULt | IntCmpOp::SGt | IntCmpOp::UGt => false,
                });
            }
        }
        None
    }

    /// Optimize float unary operations
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

    /// Optimize select operations
    fn try_fold_select(
        &self,
        _cond: &mut crate::mir::Register,
        true_val: &mut Operand,
        false_val: &mut Operand,
    ) -> bool {
        // For now, we can only optimize selects when both values are identical
        // More complex optimizations would require interprocedural analysis

        // If both values are the same, eliminate the select
        if true_val == false_val {
            // We can't change the condition type, so we can't actually optimize this here
            // This would be handled by a more sophisticated analysis
            return false;
        }

        false
    }
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

/// Extract float constant from operand
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
                    if let Some(&tidx) = label_index.get(target.as_str()) {
                        if tidx <= i {
                            headers.insert(target.clone());
                        }
                    }
                }
                Instruction::Br { true_target, false_target, .. } => {
                    if let Some(&tidx) = label_index.get(true_target.as_str()) {
                        if tidx <= i {
                            headers.insert(true_target.clone());
                        }
                    }
                    if let Some(&fidx) = label_index.get(false_target.as_str()) {
                        if fidx <= i {
                            headers.insert(false_target.clone());
                        }
                    }
                }
                _ => {}
            }
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::register::{Register, VirtualReg};
    use crate::mir::types::{MirType, ScalarType};

    #[test]
    fn fold_add_zero_right() {
        // x + 0 is already in optimal form (represents a move), so no change should be made
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
        assert!(!changed); // No change should be made
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
    fn fold_constant_addition() {
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(5)),
            rhs: Operand::Immediate(Immediate::I64(3)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        // Check that the result is folded to 8
        if let Some(block) = func.get_block("entry") {
            if let Some(Instruction::IntBinary { lhs, rhs, .. }) = block.instructions.first() {
                if let (
                    Operand::Immediate(Immediate::I64(val)),
                    Operand::Immediate(Immediate::I64(0)),
                ) = (lhs, rhs)
                {
                    assert_eq!(*val, 8);
                } else {
                    panic!("Expected constant 8 + 0");
                }
            } else {
                panic!("Expected IntBinary instruction");
            }
        }
    }

    #[test]
    fn fold_bitwise_and_all_ones() {
        // x & -1 is already in optimal form (no-op), so no change should be made
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::And,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            rhs: Operand::Immediate(Immediate::I64(-1)), // All bits set
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(!changed); // No change should be made
    }

    #[test]
    fn fold_shift_by_zero() {
        // x << 0 is already in optimal form (represents a move), so no change should be made
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Shl,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(!changed); // No change should be made
    }

    #[test]
    fn test_overflow_prevention_add() {
        // Test that overflow in addition skips folding (conservative behavior)
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(1)), // This would overflow
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        // Should NOT change due to overflow prevention
        assert!(!changed);
    }

    #[test]
    fn test_overflow_prevention_mul() {
        // Test that overflow in multiplication skips folding
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(2)), // This would overflow
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        // Should NOT change due to overflow prevention
        assert!(!changed);
    }

    #[test]
    fn test_unsigned_division() {
        // Test that UDiv uses proper unsigned semantics
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(-8)), // -8 as i64 = 18446744073709551608 as u64
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let pass = Peephole::default();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        // Check that the result is correct for unsigned division: -8 u64 / 2 = 9223372036854775804
        if let Some(block) = func.get_block("entry") {
            if let Some(Instruction::IntBinary { lhs, rhs, .. }) = block.instructions.first() {
                if let (
                    Operand::Immediate(Immediate::I64(result)),
                    Operand::Immediate(Immediate::I64(0)),
                ) = (lhs, rhs)
                {
                    // -8 as u64 = 18446744073709551608, divided by 2 = 9223372036854775804
                    assert_eq!(*result, 9223372036854775804i64);
                } else {
                    panic!("Expected constant result");
                }
            } else {
                panic!("Expected IntBinary instruction");
            }
        }
    }

    #[test]
    fn test_signed_division_overflow_prevention() {
        // Test that i64::MIN / -1 is prevented (would overflow)
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
        // Should NOT change due to overflow prevention
        assert!(!changed);
    }
}
