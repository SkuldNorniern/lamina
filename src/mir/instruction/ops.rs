//! Operator enums for MIR instructions.
//!
//! These types represent the specific operation being performed in each
//! MIR instruction variant (e.g. the `op` field of `IntBinary`).

use std::fmt;

/// Integer binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBinOp {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

impl fmt::Display for IntBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntBinOp::Add => "add",
            IntBinOp::Sub => "sub",
            IntBinOp::Mul => "mul",
            IntBinOp::UDiv => "udiv",
            IntBinOp::SDiv => "sdiv",
            IntBinOp::URem => "urem",
            IntBinOp::SRem => "srem",
            IntBinOp::And => "and",
            IntBinOp::Or => "or",
            IntBinOp::Xor => "xor",
            IntBinOp::Shl => "shl",
            IntBinOp::LShr => "lshr",
            IntBinOp::AShr => "ashr",
        };
        write!(f, "{}", s)
    }
}

/// Floating-point binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatBinOp {
    FAdd,
    FSub,
    FMul,
    FDiv,
}

impl fmt::Display for FloatBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatBinOp::FAdd => "fadd",
            FloatBinOp::FSub => "fsub",
            FloatBinOp::FMul => "fmul",
            FloatBinOp::FDiv => "fdiv",
        };
        write!(f, "{}", s)
    }
}

/// Floating-point unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatUnOp {
    FNeg,
    FSqrt,
}

impl fmt::Display for FloatUnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatUnOp::FNeg => "fneg",
            FloatUnOp::FSqrt => "fsqrt",
        };
        write!(f, "{}", s)
    }
}

/// Integer comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntCmpOp {
    Eq,
    Ne,
    ULt,
    ULe,
    UGt,
    UGe,
    SLt,
    SLe,
    SGt,
    SGe,
}

impl fmt::Display for IntCmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntCmpOp::Eq => "eq",
            IntCmpOp::Ne => "ne",
            IntCmpOp::ULt => "ult",
            IntCmpOp::ULe => "ule",
            IntCmpOp::UGt => "ugt",
            IntCmpOp::UGe => "uge",
            IntCmpOp::SLt => "slt",
            IntCmpOp::SLe => "sle",
            IntCmpOp::SGt => "sgt",
            IntCmpOp::SGe => "sge",
        };
        write!(f, "{}", s)
    }
}

/// Floating-point comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Ordered equal (legacy alias — prefer `Eq`)
    FEq,
    /// Ordered less-than (legacy alias — prefer `Lt`)
    FLt,
}

impl fmt::Display for FloatCmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatCmpOp::Eq | FloatCmpOp::FEq => "eq",
            FloatCmpOp::Ne => "ne",
            FloatCmpOp::Lt | FloatCmpOp::FLt => "lt",
            FloatCmpOp::Le => "le",
            FloatCmpOp::Gt => "gt",
            FloatCmpOp::Ge => "ge",
        };
        write!(f, "{}", s)
    }
}

/// Vector operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorOp {
    Add,
    VSub,
    VMul,
    VAnd,
    VOr,
    VXor,
    VShl,
    VLShr,
    VAShr,
    VSplat,
    VExtractLane,
    VInsertLane,
}

impl fmt::Display for VectorOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            VectorOp::Add => "vadd",
            VectorOp::VSub => "vsub",
            VectorOp::VMul => "vmul",
            VectorOp::VAnd => "vand",
            VectorOp::VOr => "vor",
            VectorOp::VXor => "vxor",
            VectorOp::VShl => "vshl",
            VectorOp::VLShr => "vlshr",
            VectorOp::VAShr => "vashr",
            VectorOp::VSplat => "vsplat",
            VectorOp::VExtractLane => "vextractlane",
            VectorOp::VInsertLane => "vinsertlane",
        };
        write!(f, "{}", s)
    }
}

/// SIMD (Single Instruction, Multiple Data) vector operations for MIR.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Abs,
    Neg,
    Sqrt,
    Fma,
    Shuffle,
    ExtractLane,
    InsertLane,
    Splat,
}

#[cfg(feature = "nightly")]
impl fmt::Display for SimdOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SimdOp::Add => "simd_add",
            SimdOp::Sub => "simd_sub",
            SimdOp::Mul => "simd_mul",
            SimdOp::Div => "simd_div",
            SimdOp::Min => "simd_min",
            SimdOp::Max => "simd_max",
            SimdOp::Abs => "simd_abs",
            SimdOp::Neg => "simd_neg",
            SimdOp::Sqrt => "simd_sqrt",
            SimdOp::Fma => "simd_fma",
            SimdOp::Shuffle => "simd_shuffle",
            SimdOp::ExtractLane => "simd_extract_lane",
            SimdOp::InsertLane => "simd_insert_lane",
            SimdOp::Splat => "simd_splat",
        };
        write!(f, "{}", s)
    }
}

/// Memory ordering constraints for atomic operations.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

#[cfg(feature = "nightly")]
impl fmt::Display for MemoryOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MemoryOrdering::Relaxed => "relaxed",
            MemoryOrdering::Acquire => "acquire",
            MemoryOrdering::Release => "release",
            MemoryOrdering::AcqRel => "acqrel",
            MemoryOrdering::SeqCst => "seqcst",
        };
        write!(f, "{}", s)
    }
}

/// Atomic binary operations for concurrent programming.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicBinOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Exchange,
    CompareExchange,
}

#[cfg(feature = "nightly")]
impl fmt::Display for AtomicBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AtomicBinOp::Add => "atomic_add",
            AtomicBinOp::Sub => "atomic_sub",
            AtomicBinOp::And => "atomic_and",
            AtomicBinOp::Or => "atomic_or",
            AtomicBinOp::Xor => "atomic_xor",
            AtomicBinOp::Exchange => "atomic_exchange",
            AtomicBinOp::CompareExchange => "atomic_cmpxchg",
        };
        write!(f, "{}", s)
    }
}
