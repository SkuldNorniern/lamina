//! Memory operations for IR builder.
//!
//! This module provides methods for memory allocation, access, and deallocation
//! operations in the IR builder API.

use crate::{
    builder::IRBuilder,
    instruction::{AllocType, Instruction},
    types::{PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Allocates stack memory (automatic lifetime management)
    pub fn stackalloc(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Stack,
            allocated_ty: ty,
        })
    }

    /// Starts a `valist` at `ap` over the variadic arguments of this function
    pub fn va_start(&mut self, ap: Value<'a>) -> &mut Self {
        self.inst(Instruction::VaStart { ap })
    }

    /// Reads the next variadic argument as `ty` through the `valist` at `ap`
    pub fn va_arg(&mut self, result: &'a str, ty: PrimitiveType, ap: Value<'a>) -> &mut Self {
        self.inst(Instruction::VaArg { result, ty, ap })
    }

    /// Ends the lifetime of the `valist` at `ap`
    pub fn va_end(&mut self, ap: Value<'a>) -> &mut Self {
        self.inst(Instruction::VaEnd { ap })
    }

    /// Copies the `valist` at `src` into the `valist` at `dst`
    pub fn va_copy(&mut self, dst: Value<'a>, src: Value<'a>) -> &mut Self {
        self.inst(Instruction::VaCopy { dst, src })
    }

    /// Allocates heap memory
    pub fn heapalloc(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
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
            volatile: false,
        })
    }

    /// Stores a value to memory as a volatile access
    pub fn volatile_store(&mut self, ty: Type<'a>, ptr: Value<'a>, val: Value<'a>) -> &mut Self {
        self.inst(Instruction::Store {
            ty,
            ptr,
            value: val,
            volatile: true,
        })
    }

    /// Loads a value from memory
    pub fn load(&mut self, result: &'a str, ty: Type<'a>, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Load {
            result,
            ty,
            ptr,
            volatile: false,
        })
    }

    /// Loads a value from memory as a volatile access
    pub fn volatile_load(&mut self, result: &'a str, ty: Type<'a>, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Load {
            result,
            ty,
            ptr,
            volatile: true,
        })
    }

    /// Deallocates heap memory
    pub fn heapfree(&mut self, ptr: Value<'a>) -> &mut Self {
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
