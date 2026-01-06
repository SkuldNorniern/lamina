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

    // --- Convenience methods for common SIMD operations ---

    /// Performs SIMD addition: `result = lhs + rhs` (element-wise).
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// # use lamina::ir::builder::{var};
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let mut builder = IRBuilder::new();
    /// let vec_type = Type::Vector {
    ///     element_type: PrimitiveType::I32,
    ///     lanes: 4,
    /// };
    /// builder.simd_add("sum", vec_type, var("a"), var("b"));
    /// # }
    /// ```
    pub fn simd_add(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Add, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD subtraction: `result = lhs - rhs` (element-wise).
    pub fn simd_sub(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Sub, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD multiplication: `result = lhs * rhs` (element-wise).
    pub fn simd_mul(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Mul, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD division: `result = lhs / rhs` (element-wise).
    pub fn simd_div(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Div, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD minimum: `result = min(lhs, rhs)` (element-wise).
    pub fn simd_min(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Min, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD maximum: `result = max(lhs, rhs)` (element-wise).
    pub fn simd_max(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.simd_binary(SimdOp::Max, result, vector_type, lhs, rhs)
    }

    /// Performs SIMD absolute value: `result = abs(value)`.
    pub fn simd_abs(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        operand: Value<'a>,
    ) -> &mut Self {
        self.simd_unary(SimdOp::Abs, result, vector_type, operand)
    }

    /// Performs SIMD negation: `result = -value`.
    pub fn simd_neg(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        operand: Value<'a>,
    ) -> &mut Self {
        self.simd_unary(SimdOp::Neg, result, vector_type, operand)
    }

    /// Performs SIMD square root: `result = sqrt(value)`.
    pub fn simd_sqrt(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        operand: Value<'a>,
    ) -> &mut Self {
        self.simd_unary(SimdOp::Sqrt, result, vector_type, operand)
    }

    /// Performs SIMD fused multiply-add: `result = (lhs * rhs) + acc`.
    ///
    /// FMA operations are typically faster and more accurate than separate
    /// multiply and add operations.
    pub fn simd_fma(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        acc: Value<'a>,
    ) -> &mut Self {
        self.simd_ternary(SimdOp::Fma, result, vector_type, lhs, rhs, acc)
    }

    /// Broadcasts a scalar value to all lanes of a vector (splat).
    ///
    /// This creates a vector where all lanes contain the same scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// # use lamina::ir::builder::i32;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let mut builder = IRBuilder::new();
    /// let vec_type = Type::Vector {
    ///     element_type: PrimitiveType::I32,
    ///     lanes: 4,
    /// };
    /// builder.simd_splat("vec", vec_type, i32(42));
    /// // Creates <4 x i32> with all lanes set to 42
    /// # }
    /// ```
    pub fn simd_splat(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        scalar: Value<'a>,
    ) -> &mut Self {
        // Splat is implemented as a special case of SimdUnary with Splat op
        // Note: The actual implementation may need to use a different instruction
        // For now, we'll use SimdUnary with Splat op
        self.inst(Instruction::SimdUnary {
            op: SimdOp::Splat,
            result: result.into(),
            vector_type,
            operand: scalar,
        })
    }

    /// Creates a vector type with the specified element type and number of lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina::ir::{IRBuilder, Type};
    /// # use lamina::ir::types::PrimitiveType;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let mut builder = IRBuilder::new();
    /// // Create a 4-lane vector of i32: <4 x i32>
    /// let vec_type = builder.vector_type(PrimitiveType::I32, 4);
    /// # }
    /// ```
    pub fn vector_type(&self, element_type: PrimitiveType, lanes: u32) -> Type<'a> {
        Type::Vector {
            element_type,
            lanes,
        }
    }

    /// Creates a 128-bit vector type (v128) with the specified element type.
    ///
    /// Automatically calculates the number of lanes based on the element size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina::ir::{IRBuilder, Type};
    /// # use lamina::ir::types::PrimitiveType;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let mut builder = IRBuilder::new();
    /// // Creates <4 x i32> (4 * 32 = 128 bits)
    /// let vec = builder.vector_128(PrimitiveType::I32);
    /// # }
    /// ```
    pub fn vector_128(&self, element_type: PrimitiveType) -> Type<'a> {
        let lane_size = match element_type {
            PrimitiveType::I8 | PrimitiveType::U8 => 8,
            PrimitiveType::I16 | PrimitiveType::U16 => 16,
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 32,
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 64,
            _ => 32, // Default fallback
        };
        Type::Vector {
            element_type,
            lanes: 128 / lane_size,
        }
    }

    /// Creates a 256-bit vector type (v256) with the specified element type.
    ///
    /// Automatically calculates the number of lanes based on the element size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina::ir::{IRBuilder, Type};
    /// # use lamina::ir::types::PrimitiveType;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let mut builder = IRBuilder::new();
    /// // Creates <8 x f32> (8 * 32 = 256 bits)
    /// let vec = builder.vector_256(PrimitiveType::F32);
    /// # }
    /// ```
    pub fn vector_256(&self, element_type: PrimitiveType) -> Type<'a> {
        let lane_size = match element_type {
            PrimitiveType::I8 | PrimitiveType::U8 => 8,
            PrimitiveType::I16 | PrimitiveType::U16 => 16,
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 32,
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 64,
            _ => 32, // Default fallback
        };
        Type::Vector {
            element_type,
            lanes: 256 / lane_size,
        }
    }
}
