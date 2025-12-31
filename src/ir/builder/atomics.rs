//! Atomic operations for IR builder (nightly feature).
//!
//! This module provides methods for atomic memory operations including
//! atomic loads, stores, binary operations, and compare-and-swap.
//! These operations are only available when the `nightly` feature is enabled.

#[cfg(feature = "nightly")]
use super::IRBuilder;
#[cfg(feature = "nightly")]
use crate::ir::instruction::{AtomicBinOp, Instruction, MemoryOrdering};
#[cfg(feature = "nightly")]
use crate::ir::types::{Type, Value};

#[cfg(feature = "nightly")]
impl<'a> IRBuilder<'a> {
    /// Performs an atomic load operation with specified memory ordering.
    pub fn atomic_load(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicLoad {
            result: result.into(),
            ty,
            ptr,
            ordering,
        })
    }

    /// Performs an atomic store operation with specified memory ordering.
    pub fn atomic_store(
        &mut self,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        })
    }

    /// Performs an atomic binary operation (read-modify-write).
    pub fn atomic_binary(
        &mut self,
        op: AtomicBinOp,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicBinary {
            op,
            result: result.into(),
            ty,
            ptr,
            value,
            ordering,
        })
    }

    /// Performs an atomic compare-exchange operation.
    pub fn atomic_compare_exchange(
        &mut self,
        result: &'a str,
        success: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        expected: Value<'a>,
        desired: Value<'a>,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicCompareExchange {
            result: result.into(),
            success: success.into(),
            ty,
            ptr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        })
    }

    /// Inserts a memory fence/barrier with specified memory ordering.
    pub fn fence(&mut self, ordering: MemoryOrdering) -> &mut Self {
        self.inst(Instruction::Fence { ordering })
    }
}

