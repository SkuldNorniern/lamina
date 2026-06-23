//! Pointer operations for IR builder.
//!
//! This module provides methods for pointer arithmetic, structure field access,
//! and pointer-integer conversions.

use crate::builder::IRBuilder;
use crate::instruction::Instruction;
use crate::types::{PrimitiveType, StructField, Value, struct_field_byte_offset};

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

    /// Gets a pointer to a struct field using a fixed 8-byte-per-field stride.
    ///
    /// For structs with non-uniform or non-8-byte fields, use [`Self::struct_gep_typed`]
    /// which computes the correct ABI offset from the field type list.
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
            field_byte_offset: None,
        })
    }

    /// Gets a pointer to a struct field with ABI-correct byte offset computation.
    ///
    /// The `fields` slice must describe the complete struct layout; `field_index` selects
    /// which field to address. The byte offset is computed from `fields` using standard
    /// C-ABI alignment rules (natural alignment, no packing) and embedded in the
    /// instruction so the MIR backend uses the correct offset instead of a fixed stride.
    pub fn struct_gep_typed(
        &mut self,
        result: &'a str,
        struct_ptr: Value<'a>,
        fields: &[StructField<'a>],
        field_index: usize,
    ) -> &mut Self {
        let field_byte_offset = struct_field_byte_offset(fields, field_index);
        self.inst(Instruction::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
            field_byte_offset,
        })
    }
}
