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

