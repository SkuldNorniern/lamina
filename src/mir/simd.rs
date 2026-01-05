//! SIMD types and operations for LUMIR.
//!
//! This module defines SIMD (Single Instruction, Multiple Data) vector types
//! and operations. SIMD operations enable parallel computation on multiple
//! data elements simultaneously, improving performance for data-parallel workloads.
//!
//! ## Vector Types
//!
//! - **v128**: 128-bit vectors (16 bytes)
//! - **v256**: 256-bit vectors (32 bytes)
//!
//! ## Lane Types
//!
//! Vector lanes can be:
//! - Integer lanes: i8, i16, i32, i64
//! - Floating-point lanes: f32, f64
//!
//! ## Operations
//!
//! SIMD operations include element-wise arithmetic, comparisons, shuffling,
//! and reduction operations.

#[cfg(feature = "nightly")]
use lamina_mir::types::{MirType, VectorLane, VectorType};

/// Calculate the number of lanes in a vector type.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::lane_count;
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert_eq!(lane_count(&v128_i32), 4); // 128 bits / 32 bits = 4 lanes
/// ```
#[cfg(feature = "nightly")]
pub fn lane_count(vector_type: &VectorType) -> usize {
    match vector_type {
        VectorType::V128(lane) => {
            128 / lane_size_bits(lane)
        }
        VectorType::V256(lane) => {
            256 / lane_size_bits(lane)
        }
    }
}

/// Get the size of a lane type in bits.
#[cfg(feature = "nightly")]
fn lane_size_bits(lane: &VectorLane) -> usize {
    match lane {
        VectorLane::I8 => 8,
        VectorLane::I16 => 16,
        VectorLane::I32 => 32,
        VectorLane::I64 => 64,
        VectorLane::F32 => 32,
        VectorLane::F64 => 64,
    }
}

/// Get the lane type from a vector type.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::get_lane_type;
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert_eq!(get_lane_type(&v128_i32), VectorLane::I32);
/// ```
#[cfg(feature = "nightly")]
pub fn get_lane_type(vector_type: &VectorType) -> VectorLane {
    match vector_type {
        VectorType::V128(lane) | VectorType::V256(lane) => *lane,
    }
}

/// Check if a vector type is a floating-point vector.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::is_float_vector;
/// let v128_f32 = VectorType::V128(VectorLane::F32);
/// assert!(is_float_vector(&v128_f32));
///
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert!(!is_float_vector(&v128_i32));
/// ```
#[cfg(feature = "nightly")]
pub fn is_float_vector(vector_type: &VectorType) -> bool {
    matches!(
        get_lane_type(vector_type),
        VectorLane::F32 | VectorLane::F64
    )
}

/// Check if a vector type is an integer vector.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::is_integer_vector;
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert!(is_integer_vector(&v128_i32));
///
/// let v128_f32 = VectorType::V128(VectorLane::F32);
/// assert!(!is_integer_vector(&v128_f32));
/// ```
#[cfg(feature = "nightly")]
pub fn is_integer_vector(vector_type: &VectorType) -> bool {
    matches!(
        get_lane_type(vector_type),
        VectorLane::I8 | VectorLane::I16 | VectorLane::I32 | VectorLane::I64
    )
}

/// Get the size of a vector type in bytes.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::vector_size_bytes;
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert_eq!(vector_size_bytes(&v128_i32), 16);
///
/// let v256_f64 = VectorType::V256(VectorLane::F64);
/// assert_eq!(vector_size_bytes(&v256_f64), 32);
/// ```
#[cfg(feature = "nightly")]
pub fn vector_size_bytes(vector_type: &VectorType) -> usize {
    match vector_type {
        VectorType::V128(_) => 16,
        VectorType::V256(_) => 32,
    }
}

/// Get the alignment requirement for a vector type in bytes.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{VectorType, VectorLane};
/// # use lamina::mir::simd::vector_alignment;
/// let v128_i32 = VectorType::V128(VectorLane::I32);
/// assert_eq!(vector_alignment(&v128_i32), 16);
///
/// let v256_f64 = VectorType::V256(VectorLane::F64);
/// assert_eq!(vector_alignment(&v256_f64), 32);
/// ```
#[cfg(feature = "nightly")]
pub fn vector_alignment(vector_type: &VectorType) -> usize {
    match vector_type {
        VectorType::V128(_) => 16,
        VectorType::V256(_) => 32,
    }
}

/// Extract the vector type from a MirType, returning None if it's not a vector.
///
/// # Examples
///
/// ```
/// # use lamina::mir::types::{MirType, VectorType, VectorLane, ScalarType};
/// # use lamina::mir::simd::extract_vector_type;
/// let vec_ty = MirType::Vector(VectorType::V128(VectorLane::I32));
/// assert_eq!(extract_vector_type(&vec_ty), Some(VectorType::V128(VectorLane::I32)));
///
/// let scalar_ty = MirType::Scalar(ScalarType::I32);
/// assert_eq!(extract_vector_type(&scalar_ty), None);
/// ```
#[cfg(feature = "nightly")]
pub fn extract_vector_type(ty: &MirType) -> Option<VectorType> {
    match ty {
        MirType::Vector(v) => Some(*v),
        MirType::Scalar(_) => None,
    }
}

#[cfg(test)]
#[cfg(feature = "nightly")]
mod tests {
    use super::*;
    use lamina_mir::types::{MirType, ScalarType};

    #[test]
    fn test_lane_count() {
        assert_eq!(lane_count(&VectorType::V128(VectorLane::I8)), 16);
        assert_eq!(lane_count(&VectorType::V128(VectorLane::I16)), 8);
        assert_eq!(lane_count(&VectorType::V128(VectorLane::I32)), 4);
        assert_eq!(lane_count(&VectorType::V128(VectorLane::I64)), 2);
        assert_eq!(lane_count(&VectorType::V128(VectorLane::F32)), 4);
        assert_eq!(lane_count(&VectorType::V128(VectorLane::F64)), 2);

        assert_eq!(lane_count(&VectorType::V256(VectorLane::I8)), 32);
        assert_eq!(lane_count(&VectorType::V256(VectorLane::I16)), 16);
        assert_eq!(lane_count(&VectorType::V256(VectorLane::I32)), 8);
        assert_eq!(lane_count(&VectorType::V256(VectorLane::I64)), 4);
        assert_eq!(lane_count(&VectorType::V256(VectorLane::F32)), 8);
        assert_eq!(lane_count(&VectorType::V256(VectorLane::F64)), 4);
    }

    #[test]
    fn test_get_lane_type() {
        assert_eq!(
            get_lane_type(&VectorType::V128(VectorLane::I32)),
            VectorLane::I32
        );
        assert_eq!(
            get_lane_type(&VectorType::V256(VectorLane::F64)),
            VectorLane::F64
        );
    }

    #[test]
    fn test_is_float_vector() {
        assert!(is_float_vector(&VectorType::V128(VectorLane::F32)));
        assert!(is_float_vector(&VectorType::V128(VectorLane::F64)));
        assert!(is_float_vector(&VectorType::V256(VectorLane::F32)));
        assert!(is_float_vector(&VectorType::V256(VectorLane::F64)));

        assert!(!is_float_vector(&VectorType::V128(VectorLane::I32)));
        assert!(!is_float_vector(&VectorType::V256(VectorLane::I64)));
    }

    #[test]
    fn test_is_integer_vector() {
        assert!(is_integer_vector(&VectorType::V128(VectorLane::I8)));
        assert!(is_integer_vector(&VectorType::V128(VectorLane::I16)));
        assert!(is_integer_vector(&VectorType::V128(VectorLane::I32)));
        assert!(is_integer_vector(&VectorType::V128(VectorLane::I64)));

        assert!(!is_integer_vector(&VectorType::V128(VectorLane::F32)));
        assert!(!is_integer_vector(&VectorType::V256(VectorLane::F64)));
    }

    #[test]
    fn test_vector_size_bytes() {
        assert_eq!(vector_size_bytes(&VectorType::V128(VectorLane::I32)), 16);
        assert_eq!(vector_size_bytes(&VectorType::V256(VectorLane::F64)), 32);
    }

    #[test]
    fn test_vector_alignment() {
        assert_eq!(vector_alignment(&VectorType::V128(VectorLane::I32)), 16);
        assert_eq!(vector_alignment(&VectorType::V256(VectorLane::F64)), 32);
    }

    #[test]
    fn test_extract_vector_type() {
        let vec_ty = MirType::Vector(VectorType::V128(VectorLane::I32));
        assert_eq!(
            extract_vector_type(&vec_ty),
            Some(VectorType::V128(VectorLane::I32))
        );

        let scalar_ty = MirType::Scalar(ScalarType::I32);
        assert_eq!(extract_vector_type(&scalar_ty), None);
    }
}
