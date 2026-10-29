//! Vector operations for the IR builder.
//!
//! A vector is a type, not a second family of operations: element-wise arithmetic
//! reuses [`BinaryOp`] with a `<N x T>` suffix. Only the four operations that cannot
//! be written any other way get their own builder methods.

use crate::builder::IRBuilder;
use crate::instruction::{BinaryOp, Instruction};
use crate::types::{PrimitiveType, Value};

impl<'a> IRBuilder<'a> {
    /// Element-wise `op` over `<lanes x element_type>`.
    pub fn vector_binary(
        &mut self,
        op: BinaryOp,
        result: &'a str,
        element_type: PrimitiveType,
        lanes: u32,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::VectorBinary {
            op,
            result,
            element_type,
            lanes,
            lhs,
            rhs,
        })
    }

    /// Broadcasts `value` into every lane.
    pub fn splat(
        &mut self,
        result: &'a str,
        element_type: PrimitiveType,
        lanes: u32,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::VectorSplat {
            result,
            element_type,
            lanes,
            value,
        })
    }

    /// Reads lane `lane` out as a scalar.
    pub fn extract_lane(
        &mut self,
        result: &'a str,
        element_type: PrimitiveType,
        lanes: u32,
        vector: Value<'a>,
        lane: u32,
    ) -> &mut Self {
        self.inst(Instruction::VectorExtractLane {
            result,
            element_type,
            lanes,
            vector,
            lane,
        })
    }

    /// Writes `scalar` into lane `lane`.
    pub fn insert_lane(
        &mut self,
        result: &'a str,
        element_type: PrimitiveType,
        lanes: u32,
        vector: Value<'a>,
        scalar: Value<'a>,
        lane: u32,
    ) -> &mut Self {
        self.inst(Instruction::VectorInsertLane {
            result,
            element_type,
            lanes,
            vector,
            scalar,
            lane,
        })
    }

    /// Selects each result lane from `lhs ++ rhs` by a mask fixed at compile time.
    pub fn shuffle(
        &mut self,
        result: &'a str,
        element_type: PrimitiveType,
        lanes: u32,
        lhs: Value<'a>,
        rhs: Value<'a>,
        mask: Vec<u32>,
    ) -> &mut Self {
        self.inst(Instruction::VectorShuffle {
            result,
            element_type,
            lanes,
            lhs,
            rhs,
            mask,
        })
    }
}
