//! Pointer operations for IR builder.
//!
//! This module provides methods for pointer arithmetic, structure field access,
//! and pointer-integer conversions.

use super::IRBuilder;
use crate::ir::instruction::Instruction;
use crate::ir::types::{PrimitiveType, Value};

impl<'a> IRBuilder<'a> {
    /// Gets a pointer to an array element (pointer arithmetic)
    pub fn getelementptr(
        &mut self,
        result: &'a str,
        array_ptr: Value<'a>,
        index: Value<'a>,
        element_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        })
    }

    /// Convert pointer to integer for pointer arithmetic
    pub fn ptrtoint(
        &mut self,
        result: &'a str,
        ptr_value: Value<'a>,
        target_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::PtrToInt {
            result,
            ptr_value,
            target_type,
        })
    }

    /// Convert integer back to pointer
    pub fn inttoptr(
        &mut self,
        result: &'a str,
        int_value: Value<'a>,
        target_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::IntToPtr {
            result,
            int_value,
            target_type,
        })
    }

    /// Gets a pointer to a struct field (structure field access)
    pub fn struct_gep(
        &mut self,
        result: &'a str,
        struct_ptr: Value<'a>,
        field_index: usize,
    ) -> &mut Self {
        self.inst(Instruction::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
        })
    }
}

