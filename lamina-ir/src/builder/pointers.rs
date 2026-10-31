//! Pointer operations for IR builder.
//!
//! This module provides methods for pointer arithmetic, structure field access,
//! and pointer-integer conversions.

use crate::{
    builder::IRBuilder,
    instruction::Instruction,
    types::{PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Gets a pointer to an array element (pointer arithmetic)
    pub fn getelem(
        &mut self,
        result: &'a str,
        array_ptr: Value<'a>,
        index: Value<'a>,
        element_type: impl Into<Type<'a>>,
    ) -> &mut Self {
        self.inst(Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type: element_type.into(),
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

    /// Convert integer back to pointer. `target_type` is the integer type of
    /// `int_value`; the result is always `ptr`.
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

    /// Gets a pointer to a field of a struct laid out as `struct_type`.
    ///
    /// The byte offset is computed from the layout when the IR is lowered, using standard
    /// C-ABI alignment rules. A named layout is resolved against the module's type
    /// declarations.
    pub fn getfieldptr(
        &mut self,
        result: &'a str,
        struct_ptr: Value<'a>,
        struct_type: impl Into<Type<'a>>,
        field_index: usize,
    ) -> &mut Self {
        self.inst(Instruction::GetFieldPtr {
            result,
            struct_ptr,
            struct_type: struct_type.into(),
            field_index,
        })
    }

    /// Creates the address of a function as a pointer value (`funcaddr @name`).
    pub fn funcaddr(&mut self, result: &'a str, func_name: &'a str) -> &mut Self {
        self.inst(Instruction::FuncAddr { result, func_name })
    }
}
