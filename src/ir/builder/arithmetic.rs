//! Arithmetic and comparison operations for IR builder.
//!
//! This module provides methods for binary arithmetic operations, comparisons,
//! and unary operations (negation, logical not).

use super::IRBuilder;
use crate::ir::instruction::{BinaryOp, CmpOp, Instruction};
use crate::ir::types::{PrimitiveType, Value};

impl<'a> IRBuilder<'a> {
    /// Creates a binary operation instruction
    pub fn binary(
        &mut self,
        op: BinaryOp,
        result: &'a str,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }

    /// Creates a comparison operation instruction
    pub fn cmp(
        &mut self,
        op: CmpOp,
        result: &'a str,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }
}
