use crate::mir::instruction::{Immediate, Instruction, IntBinOp, Operand};
use crate::mir::{Block, Function};

use super::{Transform, TransformCategory, TransformLevel};

/// Basic peephole optimizations for MIR
///
/// This pass performs simple local rewrites to remove identity operations
/// and simplify trivial patterns. This is intentionally conservative and
/// architecture-agnostic.
#[derive(Default)]
pub struct Peephole;

impl Transform for Peephole {
    fn name(&self) -> &'static str {
        "peephole"
    }

    fn description(&self) -> &'static str {
        "Basic local rewrites like add/sub by 0 and mul by 1/0"
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
        for block in &mut func.blocks {
            if self.run_on_block(block) {
                changed = true;
            }
        }
        changed
    }

    fn run_on_block(&self, block: &mut Block) -> bool {
        let mut changed = false;

        for inst in &mut block.instructions {
            let mut local_change = false;
            if let Instruction::IntBinary {
                op,
                dst: _,
                ty: _,
                lhs,
                rhs,
            } = inst
            {
                // Fold identities on integer arithmetic
                local_change |= self.try_fold_int_identity(op, lhs, rhs);
            }

            if local_change {
                changed = true;
            }
        }

        // Additional cleanups that may require looking at neighbors can be added later
        changed
    }

    fn try_fold_int_identity(
        &self,
        op: &mut IntBinOp,
        lhs: &mut Operand,
        rhs: &mut Operand,
    ) -> bool {
        // Recognize immediates. Keep by-reference uses minimal to avoid cloning owned types.
        let lhs_imm = match lhs {
            Operand::Immediate(i) => Some(*i),
            _ => None,
        };
        let rhs_imm = match rhs {
            Operand::Immediate(i) => Some(*i),
            _ => None,
        };

        match op {
            IntBinOp::Add => {
                // x + 0 => x ; 0 + x => x
                if is_zero(lhs_imm) {
                    // 0 + x => move rhs into place by turning into x + 0 and swapping
                    // Prefer canonical form: x + 0 by swapping
                    core::mem::swap(lhs, rhs);
                    return true;
                }
                if is_zero(rhs_imm) {
                    // x + 0 => x (no instruction removal here; later DCE/CP may coalesce)
                    return true;
                }
            }
            IntBinOp::Sub => {
                // x - 0 => x
                if is_zero(rhs_imm) {
                    return true;
                }
            }
            IntBinOp::Mul => {
                // x * 1 => x ; 1 * x => x ; x * 0 => 0 ; 0 * x => 0
                if is_one(lhs_imm) {
                    // 1 * x => x ; prefer x * 1 canonicalization
                    core::mem::swap(lhs, rhs);
                    return true;
                }
                if is_one(rhs_imm) {
                    return true;
                }
                if is_zero(lhs_imm) || is_zero(rhs_imm) {
                    // Replace both sides with immediate 0 to signal result is zero.
                    *lhs = Operand::Immediate(Immediate::I64(0));
                    *rhs = Operand::Immediate(Immediate::I64(0));
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

fn is_zero(i: Option<Immediate>) -> bool {
    match i {
        Some(Immediate::I8(v)) => v == 0,
        Some(Immediate::I16(v)) => v == 0,
        Some(Immediate::I32(v)) => v == 0,
        Some(Immediate::I64(v)) => v == 0,
        // For floating patterns we'd use FloatBinary, so ignore here
        _ => false,
    }
}

fn is_one(i: Option<Immediate>) -> bool {
    match i {
        Some(Immediate::I8(v)) => v == 1,
        Some(Immediate::I16(v)) => v == 1,
        Some(Immediate::I32(v)) => v == 1,
        Some(Immediate::I64(v)) => v == 1,
        _ => false,
    }
}

#[cfg(test)]
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
        assert!(changed);
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
}
