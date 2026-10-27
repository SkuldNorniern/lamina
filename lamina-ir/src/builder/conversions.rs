//! Type conversion operations for IR builder.
//!
//! This module provides methods for type conversions including zero-extension,
//! sign-extension, truncation, bitcasting, and conditional selection.

use crate::{
    builder::IRBuilder,
    instruction::Instruction,
    types::{PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Creates a zero-extension instruction
    pub fn zext(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates an integer truncation instruction
    pub fn trunc(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Trunc {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a sign-extension instruction
    pub fn sext(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SignExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a bitcast instruction between equally-sized primitive types.
    pub fn bitcast(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Bitcast {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a select instruction (SSA conditional expression).
    pub fn select(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        cond: Value<'a>,
        true_val: Value<'a>,
        false_val: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        })
    }

    /// Creates an integer-to-float conversion (`inttofp.<int>.<float>`).
    ///
    /// The signedness is derived from `source_type`: `i8`, `i16`, `i32` and `i64`
    /// are signed, `u8`, `u16`, `u32` and `u64` are unsigned.
    ///
    /// # Panics
    /// Panics if `source_type` is not an integer type or `target_type` is not `f32` or `f64`.
    pub fn inttofp(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        let signed = integer_signedness(source_type).unwrap_or_else(|| {
            panic!("inttofp source must be an integer type, found {source_type}")
        });
        assert!(
            matches!(target_type, PrimitiveType::F32 | PrimitiveType::F64),
            "inttofp target must be f32 or f64, found {target_type}"
        );
        self.inst(Instruction::IntToFloat {
            result,
            signed,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a float-to-integer conversion that truncates toward zero and saturates
    /// (`fptoint.<float>.<int>`).
    ///
    /// The signedness is derived from `target_type`: `i8`, `i16`, `i32` and `i64`
    /// are signed, `u8`, `u16`, `u32` and `u64` are unsigned.
    ///
    /// # Panics
    /// Panics if `source_type` is not `f32` or `f64` or `target_type` is not an integer type.
    pub fn fptoint(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        assert!(
            matches!(source_type, PrimitiveType::F32 | PrimitiveType::F64),
            "fptoint source must be f32 or f64, found {source_type}"
        );
        let signed = integer_signedness(target_type).unwrap_or_else(|| {
            panic!("fptoint target must be an integer type, found {target_type}")
        });
        self.inst(Instruction::FloatToInt {
            result,
            signed,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a conversion between `f32` and `f64` (`fpcast.<float>.<float>`).
    ///
    /// # Panics
    /// Panics unless the pair is `f32` to `f64` or `f64` to `f32`.
    pub fn fpcast(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        assert!(
            matches!(
                (source_type, target_type),
                (PrimitiveType::F32, PrimitiveType::F64) | (PrimitiveType::F64, PrimitiveType::F32)
            ),
            "fpcast converts between f32 and f64, found {source_type} to {target_type}"
        );
        self.inst(Instruction::FloatConvert {
            result,
            source_type,
            target_type,
            value,
        })
    }
}

fn integer_signedness(ty: PrimitiveType) -> Option<bool> {
    match ty {
        PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
            Some(true)
        }
        PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
            Some(false)
        }
        _ => None,
    }
}
