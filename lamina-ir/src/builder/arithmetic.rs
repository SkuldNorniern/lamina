//! Arithmetic and comparison operations for IR builder.
//!
//! This module provides methods for binary arithmetic operations, comparisons,
//! and unary operations (negation, bitwise complement, square root).

use crate::{
    builder::IRBuilder,
    instruction::{BinaryOp, CmpOp, Instruction, UnaryOp},
    types::{PrimitiveType, Value},
};

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

    /// Creates an arithmetic negation instruction (`neg.<ty>`).
    pub fn neg(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Unary {
            op: UnaryOp::Neg,
            result,
            ty,
            value,
        })
    }

    /// Creates a bitwise complement instruction (`not.<ty>`).
    ///
    /// # Panics
    /// Panics if `ty` is `f32` or `f64`.
    pub fn not(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        assert!(
            !matches!(ty, PrimitiveType::F32 | PrimitiveType::F64),
            "not takes an integer type, found {ty}"
        );
        self.inst(Instruction::Unary {
            op: UnaryOp::Not,
            result,
            ty,
            value,
        })
    }

    /// Creates a square root instruction (`sqrt.<ty>`).
    ///
    /// # Panics
    /// Panics if `ty` is not `f32` or `f64`.
    pub fn sqrt(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        assert!(
            matches!(ty, PrimitiveType::F32 | PrimitiveType::F64),
            "sqrt takes f32 or f64, found {ty}"
        );
        self.inst(Instruction::Unary {
            op: UnaryOp::Sqrt,
            result,
            ty,
            value,
        })
    }
}
