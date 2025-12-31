//! SIMD operations for IR builder (nightly feature).
//!
//! This module provides methods for SIMD (Single Instruction, Multiple Data)
//! operations including vector arithmetic, shuffling, and element extraction.
//! These operations are only available when the `nightly` feature is enabled.

#[cfg(feature = "nightly")]
use super::IRBuilder;
#[cfg(feature = "nightly")]
use crate::ir::instruction::{Instruction, SimdOp};
#[cfg(feature = "nightly")]
use crate::ir::types::{PrimitiveType, Type, Value};

#[cfg(feature = "nightly")]
impl<'a> IRBuilder<'a> {
    /// Performs a SIMD binary operation (element-wise).
    pub fn simd_binary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdBinary {
            op,
            result: result.into(),
            vector_type,
            lhs,
            rhs,
        })
    }

    /// Performs a SIMD unary operation.
    pub fn simd_unary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        operand: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdUnary {
            op,
            result: result.into(),
            vector_type,
            operand,
        })
    }

    /// Performs a SIMD ternary operation (e.g., fused multiply-add).
    pub fn simd_ternary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        acc: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdTernary {
            op,
            result: result.into(),
            vector_type,
            lhs,
            rhs,
            acc,
        })
    }

    /// Performs a SIMD shuffle operation.
    pub fn simd_shuffle(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        mask: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdShuffle {
            result: result.into(),
            vector_type,
            lhs,
            rhs,
            mask,
        })
    }

    /// Extracts a single element from a SIMD vector.
    pub fn simd_extract(
        &mut self,
        result: &'a str,
        scalar_type: PrimitiveType,
        vector: Value<'a>,
        lane_index: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdExtract {
            result: result.into(),
            scalar_type,
            vector,
            lane_index,
        })
    }

    /// Inserts a single element into a SIMD vector.
    pub fn simd_insert(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        vector: Value<'a>,
        scalar: Value<'a>,
        lane_index: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdInsert {
            result: result.into(),
            vector_type,
            vector,
            scalar,
            lane_index,
        })
    }

    /// Loads a SIMD vector from memory.
    pub fn simd_load(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        ptr: Value<'a>,
        alignment: Option<u32>,
    ) -> &mut Self {
        self.inst(Instruction::SimdLoad {
            result: result.into(),
            vector_type,
            ptr,
            alignment,
        })
    }

    /// Stores a SIMD vector to memory.
    pub fn simd_store(
        &mut self,
        vector_type: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        alignment: Option<u32>,
    ) -> &mut Self {
        self.inst(Instruction::SimdStore {
            vector_type,
            ptr,
            value,
            alignment,
        })
    }
}
