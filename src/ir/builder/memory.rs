//! Memory operations for IR builder.
//!
//! This module provides methods for memory allocation, access, and deallocation
//! operations in the IR builder API.

use super::IRBuilder;
use crate::ir::instruction::{AllocType, Instruction};
use crate::ir::types::{Type, Value};

impl<'a> IRBuilder<'a> {
    /// Allocates stack memory (automatic lifetime management)
    pub fn alloc_stack(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Stack,
            allocated_ty: ty,
        })
    }

    /// Allocates heap memory
    pub fn alloc_heap(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Heap,
            allocated_ty: ty,
        })
    }

    /// Stores a value to memory
    pub fn store(&mut self, ty: Type<'a>, ptr: Value<'a>, val: Value<'a>) -> &mut Self {
        self.inst(Instruction::Store {
            ty,
            ptr,
            value: val,
        })
    }

    /// Loads a value from memory
    pub fn load(&mut self, result: &'a str, ty: Type<'a>, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Load { result, ty, ptr })
    }

    /// Deallocates heap memory
    pub fn dealloc(&mut self, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Dealloc { ptr })
    }
}

