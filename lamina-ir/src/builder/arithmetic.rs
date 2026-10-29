//! Arithmetic and comparison operations for IR builder.
//!
//! This module provides methods for binary arithmetic operations, comparisons,
//! unary operations (negation, bitwise complement, square root), and the bit
//! intrinsics (population count, leading and trailing zeros, byte swap, rotation).

use crate::{
    builder::IRBuilder,
    instruction::{BinaryOp, CmpOp, Instruction, OverflowOp, UnaryOp},
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

    /// Creates an overflow-flag instruction (`<op>.ovf.<ty>`), whose result is a
    /// `bool` saying whether the same wrapping arithmetic left the range of `ty`.
    pub fn overflow_flag(
        &mut self,
        op: OverflowOp,
        result: &'a str,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::OverflowFlag {
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

    /// Creates a population count instruction (`popcount.<ty>`).
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn popcount(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.bit_unary(UnaryOp::Popcount, "popcount", result, ty, value)
    }

    /// Creates a count-leading-zeros instruction (`ctlz.<ty>`). `ctlz` of zero is the bit width.
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn ctlz(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.bit_unary(UnaryOp::Ctlz, "ctlz", result, ty, value)
    }

    /// Creates a count-trailing-zeros instruction (`cttz.<ty>`). `cttz` of zero is the bit width.
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn cttz(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.bit_unary(UnaryOp::Cttz, "cttz", result, ty, value)
    }

    /// Creates a byte-swap instruction (`bswap.<ty>`).
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn bswap(&mut self, result: &'a str, ty: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.bit_unary(UnaryOp::Bswap, "bswap", result, ty, value)
    }

    /// Creates a rotate-left instruction (`rotl.<ty>`). The count is taken modulo the bit width.
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn rotl(
        &mut self,
        result: &'a str,
        ty: PrimitiveType,
        value: Value<'a>,
        count: Value<'a>,
    ) -> &mut Self {
        self.bit_binary(BinaryOp::Rotl, "rotl", result, ty, value, count)
    }

    /// Creates a rotate-right instruction (`rotr.<ty>`). The count is taken modulo the bit width.
    ///
    /// # Panics
    /// Panics if `ty` is not an integer type.
    pub fn rotr(
        &mut self,
        result: &'a str,
        ty: PrimitiveType,
        value: Value<'a>,
        count: Value<'a>,
    ) -> &mut Self {
        self.bit_binary(BinaryOp::Rotr, "rotr", result, ty, value, count)
    }

    fn bit_unary(
        &mut self,
        op: UnaryOp,
        name: &str,
        result: &'a str,
        ty: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        assert!(is_integer(ty), "{name} takes an integer type, found {ty}");
        self.inst(Instruction::Unary {
            op,
            result,
            ty,
            value,
        })
    }

    fn bit_binary(
        &mut self,
        op: BinaryOp,
        name: &str,
        result: &'a str,
        ty: PrimitiveType,
        value: Value<'a>,
        count: Value<'a>,
    ) -> &mut Self {
        assert!(is_integer(ty), "{name} takes an integer type, found {ty}");
        self.inst(Instruction::Binary {
            op,
            result,
            ty,
            lhs: value,
            rhs: count,
        })
    }
}

fn is_integer(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
            | PrimitiveType::U128
    )
}
