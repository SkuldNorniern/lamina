//! Tuple operations for IR builder.
//!
//! This module provides methods for creating tuples from multiple values
//! and extracting individual elements from tuples.

use super::IRBuilder;
use crate::ir::instruction::Instruction;
use crate::ir::types::Value;

impl<'a> IRBuilder<'a> {
    /// Creates a tuple from multiple values (composite data construction)
    pub fn tuple(&mut self, result: &'a str, elements: Vec<Value<'a>>) -> &mut Self {
        self.inst(Instruction::Tuple { result, elements })
    }

    /// Extracts a value from a tuple (tuple element access)
    pub fn extract_tuple(
        &mut self,
        result: &'a str,
        tuple_val: Value<'a>,
        index: usize,
    ) -> &mut Self {
        self.inst(Instruction::ExtractTuple {
            result,
            tuple_val,
            index,
        })
    }
}

