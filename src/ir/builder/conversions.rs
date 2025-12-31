//! Type conversion operations for IR builder.
//!
//! This module provides methods for type conversions including zero-extension,
//! sign-extension, truncation, bitcasting, and conditional selection.

use super::IRBuilder;
use crate::ir::instruction::Instruction;
use crate::ir::types::{PrimitiveType, Type, Value};

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
}
