//! Memory operations for IR builder.
//!
//! This module provides methods for memory allocation, access, and deallocation
//! operations in the IR builder API.

use crate::builder::IRBuilder;
use crate::instruction::{AllocType, Instruction};
use crate::types::{Type, Value};

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

    /// Copies `size` bytes from `src` to `dst` (regions must not overlap).
    ///
    /// Lowered to a `memcpy` libc call in MIR/codegen.
    pub fn memcpy(&mut self, dst: Value<'a>, src: Value<'a>, size: Value<'a>) -> &mut Self {
        self.inst(Instruction::MemCpy { dst, src, size })
    }

    /// Copies `size` bytes from `src` to `dst`, allowing overlapping regions.
    ///
    /// Lowered to a `memmove` libc call in MIR/codegen.
    pub fn memmove(&mut self, dst: Value<'a>, src: Value<'a>, size: Value<'a>) -> &mut Self {
        self.inst(Instruction::MemMove { dst, src, size })
    }

    /// Sets `size` bytes at `dst` to the byte `value`.
    ///
    /// Lowered to a `memset` libc call in MIR/codegen.
    pub fn memset(&mut self, dst: Value<'a>, value: Value<'a>, size: Value<'a>) -> &mut Self {
        self.inst(Instruction::MemSet { dst, value, size })
    }
}
