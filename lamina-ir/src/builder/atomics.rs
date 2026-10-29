//! Atomic operations for IR builder.
//!
//! This module provides methods for atomic memory operations including
//! atomic loads, stores, binary operations, and compare-and-swap.

use crate::builder::IRBuilder;
use crate::instruction::{AtomicBinOp, Instruction, MemoryOrdering};
use crate::types::{Type, Value};

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
            result,
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
    /// Creates an atomic read-modify-write (`%result = <op>.<ty> <ptr>, <value>, <ordering>`).
    ///
    /// # Panics
    /// Panics for `AtomicBinOp::CompareExchange`; use `atomic_compare_exchange`.
    pub fn atomic_binary(
        &mut self,
        op: AtomicBinOp,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        assert!(
            !matches!(op, AtomicBinOp::CompareExchange),
            "atomic_binary has no compare-exchange form; use atomic_compare_exchange"
        );
        self.inst(Instruction::AtomicBinary {
            op,
            result,
            ty,
            ptr,
            value,
            ordering,
        })
    }

    /// Performs an atomic compare-exchange and binds the old value to `result`.
    #[allow(clippy::too_many_arguments)]
    pub fn atomic_compare_exchange(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        expected: Value<'a>,
        desired: Value<'a>,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicCompareExchange {
            result,
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
