//! Instruction representation for LUMIR.
//!
//! LUMIR instructions are low-level, machine-friendly operations that map
//! closely to actual assembly instructions. This design enables efficient
//! code generation and optimization.
use crate::types::AddressSpace;
use crate::{
    register::{Register, VirtualReg},
    types::{MirType, ScalarType},
};
use std::fmt;

use lamina_platform::PointerWidth;

/// Result type of the I/O intrinsics: a byte count, -1 at end of input, or -2 on error.
pub fn io_intrinsic_result_type(name: &str, pointer: PointerWidth) -> Option<MirType> {
    matches!(
        name,
        "lamina.write" | "lamina.read" | "lamina.writebyte" | "lamina.readbyte"
    )
    .then(|| match pointer {
        PointerWidth::Bits32 => MirType::Scalar(ScalarType::I32),
        PointerWidth::Bits64 => MirType::Scalar(ScalarType::I64),
    })
}

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

/// Integer bit operations on a single operand. The result has the operand's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntUnOp {
    /// Number of set bits within the operand's width.
    Popcount,
    /// Leading zero bits within the operand's width; zero yields the width.
    Ctlz,
    /// Trailing zero bits within the operand's width; zero yields the width.
    Cttz,
    /// Byte-reversed operand.
    Bswap,
}

impl IntUnOp {
    /// Evaluate on a concrete operand, returning the canonical value of the operand's type.
    pub fn evaluate(self, ty: MirType, value: i64, pointer: PointerWidth) -> i64 {
        let bits = ty.size_bytes(pointer) as u32 * 8;
        let unsigned = ty.unsigned_integer(value, pointer);
        let result = match self {
            Self::Popcount => u64::from(unsigned.count_ones()),
            Self::Ctlz => u64::from(unsigned.leading_zeros() - (64 - bits)),
            Self::Cttz => u64::from(unsigned.trailing_zeros().min(bits)),
            Self::Bswap => unsigned.swap_bytes() >> (64 - bits),
        };
        ty.signed_integer(result as i64, pointer)
    }
}

impl fmt::Display for IntUnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntUnOp::Popcount => "popcount",
            IntUnOp::Ctlz => "ctlz",
            IntUnOp::Cttz => "cttz",
            IntUnOp::Bswap => "bswap",
        };
        write!(f, "{s}")
    }
}

/// Integer rotation. The count is taken modulo the operand's bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntRotOp {
    /// Rotate left. See IR `BinaryOp::Rotl`.
    Rotl,
    /// Rotate right. See IR `BinaryOp::Rotr`.
    Rotr,
}

impl IntRotOp {
    /// Evaluate on concrete operands, returning the canonical value of `ty`.
    pub fn evaluate(self, ty: MirType, value: i64, count: i64, pointer: PointerWidth) -> i64 {
        let bits = ty.size_bytes(pointer) as u32 * 8;
        let unsigned = ty.unsigned_integer(value, pointer);
        let count = (ty.unsigned_integer(count, pointer) % u64::from(bits)) as u32;
        let (near, far) = match self {
            Self::Rotl => (count, (bits - count) % bits),
            Self::Rotr => ((bits - count) % bits, count),
        };
        let rotated = (unsigned << near) | (unsigned >> far);
        ty.signed_integer(rotated as i64, pointer)
    }
}

impl fmt::Display for IntRotOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntRotOp::Rotl => write!(f, "rotl"),
            IntRotOp::Rotr => write!(f, "rotr"),
        }
    }
}

impl IntBinOp {
    /// Evaluate concrete operands, returning None when division would trap.
    pub fn evaluate(self, ty: MirType, lhs: i64, rhs: i64, pointer: PointerWidth) -> Option<i64> {
        let left = ty.signed_integer(lhs, pointer);
        let right = ty.signed_integer(rhs, pointer);
        let unsigned_left = ty.unsigned_integer(lhs, pointer);
        let unsigned_right = ty.unsigned_integer(rhs, pointer);
        let shift = (rhs as u32) & (ty.size_bytes(pointer) as u32 * 8 - 1);
        let minimum = ty.signed_integer(1i64 << (ty.size_bytes(pointer) * 8 - 1), pointer);
        let result = match self {
            Self::Add => left.wrapping_add(right),
            Self::Sub => left.wrapping_sub(right),
            Self::Mul => left.wrapping_mul(right),
            Self::SDiv | Self::SRem if right == 0 || (left == minimum && right == -1) => {
                return None;
            }
            Self::SDiv => left / right,
            Self::SRem => left % right,
            Self::UDiv => unsigned_left.checked_div(unsigned_right)? as i64,
            Self::URem => unsigned_left.checked_rem(unsigned_right)? as i64,
            Self::And => left & right,
            Self::Or => left | right,
            Self::Xor => left ^ right,
            Self::Shl => left.wrapping_shl(shift),
            Self::LShr => (unsigned_left >> shift) as i64,
            Self::AShr => left >> shift,
        };
        Some(ty.signed_integer(result, pointer))
    }

    pub fn division_checks(
        self,
        ty: MirType,
        lhs: Option<i64>,
        rhs: Option<i64>,
        pointer: PointerWidth,
    ) -> (bool, bool) {
        if !matches!(self, Self::SDiv | Self::SRem | Self::UDiv | Self::URem) {
            return (false, false);
        }
        let minimum = ty.signed_integer(1i64 << (ty.size_bytes(pointer) * 8 - 1), pointer);
        let zero = rhs.is_none_or(|value| ty.signed_integer(value, pointer) == 0);
        let overflow = matches!(self, Self::SDiv | Self::SRem)
            && lhs.is_none_or(|value| ty.signed_integer(value, pointer) == minimum)
            && rhs.is_none_or(|value| ty.signed_integer(value, pointer) == -1);
        (zero, overflow)
    }
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
        write!(f, "{s}")
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
        write!(f, "{s}")
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
        write!(f, "{s}")
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
        write!(f, "{s}")
    }
}

/// Floating-point comparisons: equality and relational predicates are ordered;
/// inequality is true for unordered operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for FloatCmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatCmpOp::Eq => "eq",
            FloatCmpOp::Ne => "ne",
            FloatCmpOp::Lt => "lt",
            FloatCmpOp::Le => "le",
            FloatCmpOp::Gt => "gt",
            FloatCmpOp::Ge => "ge",
        };
        write!(f, "{s}")
    }
}

/// Vector operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorOp {
    VAdd,
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
            VectorOp::VAdd => "vadd",
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
        write!(f, "{s}")
    }
}

/// SIMD (Single Instruction, Multiple Data) vector operations for MIR.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    /// Vector addition: `result = lhs + rhs` (element-wise)
    Add,
    /// Vector subtraction: `result = lhs - rhs` (element-wise)
    Sub,
    /// Vector multiplication: `result = lhs * rhs` (element-wise)
    Mul,
    /// Vector division: `result = lhs / rhs` (element-wise)
    Div,
    /// Vector minimum: `result = min(lhs, rhs)` (element-wise)
    Min,
    /// Vector maximum: `result = max(lhs, rhs)` (element-wise)
    Max,
    /// Vector absolute value: `result = abs(value)`
    Abs,
    /// Vector negation: `result = -value`
    Neg,
    /// Vector square root: `result = sqrt(value)`
    Sqrt,
    /// Vector fused multiply-add: `result = (lhs * rhs) + acc`
    Fma,
    /// Vector shuffle/rearrange elements according to mask
    Shuffle,
    /// Vector extract lane: extract single element from vector
    ExtractLane,
    /// Vector insert lane: insert single element into vector
    InsertLane,
    /// Vector splat: broadcast scalar to all vector lanes
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
        write!(f, "{s}")
    }
}

/// Memory ordering constraints for atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrdering {
    /// Relaxed ordering - no synchronization
    Relaxed,
    /// Acquire ordering - synchronizes with previous releases
    Acquire,
    /// Release ordering - synchronizes with subsequent acquires
    Release,
    /// Acquire and release ordering
    AcqRel,
    /// Sequentially consistent ordering - strongest guarantee
    SeqCst,
}

impl fmt::Display for MemoryOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MemoryOrdering::Relaxed => "relaxed",
            MemoryOrdering::Acquire => "acquire",
            MemoryOrdering::Release => "release",
            MemoryOrdering::AcqRel => "acqrel",
            MemoryOrdering::SeqCst => "seqcst",
        };
        write!(f, "{s}")
    }
}

/// Atomic binary operations for concurrent programming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicBinOp {
    /// Atomic addition: `*ptr += value`
    Add,
    /// Atomic subtraction: `*ptr -= value`
    Sub,
    /// Atomic bitwise AND: `*ptr &= value`
    And,
    /// Atomic bitwise OR: `*ptr |= value`
    Or,
    /// Atomic bitwise XOR: `*ptr ^= value`
    Xor,
    /// Atomic exchange: `old = *ptr; *ptr = value; return old`
    Exchange,
    /// Atomic compare-exchange (returns success flag)
    CompareExchange,
}

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
        write!(f, "{s}")
    }
}

/// Immediate value (constant operand)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Immediate {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl fmt::Display for Immediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Immediate::I8(v) => write!(f, "{v}"),
            Immediate::I16(v) => write!(f, "{v}"),
            Immediate::I32(v) => write!(f, "{v}"),
            Immediate::I64(v) => write!(f, "{v}"),
            Immediate::F32(v) => write!(f, "{v}"),
            Immediate::F64(v) => write!(f, "{v}"),
        }
    }
}

/// Operand (register or immediate)
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(Register),
    Immediate(Immediate),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Register(r) => write!(f, "{r}"),
            Operand::Immediate(i) => write!(f, "{i}"),
        }
    }
}

impl From<Register> for Operand {
    fn from(r: Register) -> Self {
        Operand::Register(r)
    }
}

impl From<Immediate> for Operand {
    fn from(i: Immediate) -> Self {
        Operand::Immediate(i)
    }
}

impl Operand {
    pub fn normalize_integer_immediate(&mut self, ty: MirType) {
        normalize_immediate(self, ty);
    }

    pub fn integer_constant(&self) -> Option<i64> {
        match self {
            Self::Immediate(Immediate::I8(value)) => Some(i64::from(*value)),
            Self::Immediate(Immediate::I16(value)) => Some(i64::from(*value)),
            Self::Immediate(Immediate::I32(value)) => Some(i64::from(*value)),
            Self::Immediate(Immediate::I64(value)) => Some(*value),
            _ => None,
        }
    }

    /// Create an immediate `i8` operand.
    pub fn imm_i8(v: i8) -> Self {
        Operand::Immediate(Immediate::I8(v))
    }

    /// Create an immediate `i32` operand.
    pub fn imm_i32(v: i32) -> Self {
        Operand::Immediate(Immediate::I32(v))
    }

    /// Create an immediate `i64` operand.
    pub fn imm_i64(v: i64) -> Self {
        Operand::Immediate(Immediate::I64(v))
    }

    /// Create an immediate `f32` operand.
    pub fn imm_f32(v: f32) -> Self {
        Operand::Immediate(Immediate::F32(v))
    }

    /// Create an immediate `f64` operand.
    pub fn imm_f64(v: f64) -> Self {
        Operand::Immediate(Immediate::F64(v))
    }

    /// Create a register operand from a `Register`.
    pub fn reg(r: Register) -> Self {
        Operand::Register(r)
    }

    /// Create a virtual GPR operand with the given id.
    pub fn v_gpr(id: u32) -> Self {
        Operand::Register(Register::Virtual(super::register::VirtualReg::new(
            id,
            super::register::RegisterClass::Gpr,
        )))
    }

    /// Create a virtual FPR operand with the given id.
    pub fn v_fpr(id: u32) -> Self {
        Operand::Register(Register::Virtual(super::register::VirtualReg::new(
            id,
            super::register::RegisterClass::Fpr,
        )))
    }

    /// Return `true` if this operand is an immediate value.
    pub fn is_immediate(&self) -> bool {
        matches!(self, Operand::Immediate(_))
    }

    /// Return `true` if this operand is a register.
    pub fn is_register(&self) -> bool {
        matches!(self, Operand::Register(_))
    }

    /// Extract the immediate value, if any.
    pub fn as_immediate(&self) -> Option<&Immediate> {
        match self {
            Operand::Immediate(i) => Some(i),
            _ => None,
        }
    }

    /// Extract the register, if any.
    pub fn as_register(&self) -> Option<&Register> {
        match self {
            Operand::Register(r) => Some(r),
            _ => None,
        }
    }
}

/// Memory addressing mode
#[derive(Debug, Clone, PartialEq)]
pub enum AddressMode {
    /// [base + imm12]
    BaseOffset {
        address_space: AddressSpace,
        base: Register,
        offset: i16, // 12-bit signed, extended to i16
    },
    /// [base + index<<scale + imm4]
    BaseIndexScale {
        address_space: AddressSpace,
        base: Register,
        index: Register,
        scale: u8,  // 1, 2, 4, or 8
        offset: i8, // 4-bit signed, extended to i8
    },
}

impl AddressMode {
    pub fn address_space(&self) -> AddressSpace {
        match self {
            Self::BaseOffset { address_space, .. } | Self::BaseIndexScale { address_space, .. } => {
                *address_space
            }
        }
    }
}

impl fmt::Display for AddressMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressMode::BaseOffset { base, offset, .. } => {
                if *offset == 0 {
                    write!(f, "[{base}]")
                } else {
                    write!(f, "[{base} + {offset}]")
                }
            }
            AddressMode::BaseIndexScale {
                base,
                index,
                scale,
                offset,
                ..
            } => {
                if *offset == 0 {
                    write!(f, "[{base} + {index}<<{scale}]")
                } else {
                    write!(f, "[{base} + {index}<<{scale} + {offset}]")
                }
            }
        }
    }
}

/// Memory operation attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAttrs {
    pub align: u8,
    pub volatile: bool,
}

impl Default for MemoryAttrs {
    fn default() -> Self {
        Self {
            align: 1,
            volatile: false,
        }
    }
}

/// LUMIR instruction
/// One operand of an inline assembly instruction, in declaration order. `$n` in the template
/// names the n-th operand.
#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand {
    /// `in(location) value`: the value is placed in `location` before the template runs.
    Input { location: String, value: Operand },
    /// `out(location)`: the template writes `location`. The instruction's result, when it has
    /// one, is read from it afterwards.
    Output { location: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Typed value copy
    Copy {
        ty: MirType,
        dst: Register,
        src: Operand,
    },

    // Address of a global symbol, materialized into a pointer register
    SymbolAddress {
        dst: Register,
        symbol: String,
    },

    // Integer arithmetic
    IntBinary {
        op: IntBinOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    // Floating-point arithmetic
    FloatBinary {
        op: FloatBinOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    FloatUnary {
        op: FloatUnOp,
        ty: MirType,
        dst: Register,
        src: Operand,
    },

    /// Integer bit operation on `src`; `dst` has the same integer type `ty`.
    /// See IR `UnaryOp::Popcount`, `Ctlz`, `Cttz` and `Bswap` for the contract.
    IntUnary {
        op: IntUnOp,
        ty: MirType,
        dst: Register,
        src: Operand,
    },

    /// Integer rotation of `lhs` by `rhs`; both operands and `dst` have type `ty`.
    /// See IR `BinaryOp::Rotl` and `BinaryOp::Rotr` for the contract.
    IntRotate {
        op: IntRotOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    // Comparisons
    IntCmp {
        op: IntCmpOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    FloatCmp {
        op: FloatCmpOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    // Select (conditional move)
    Select {
        ty: MirType,
        dst: Register,
        cond: Register,
        true_val: Operand,
        false_val: Operand,
    },

    // Memory operations
    Load {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        attrs: MemoryAttrs,
    },

    Store {
        ty: MirType,
        src: Operand,
        addr: AddressMode,
        attrs: MemoryAttrs,
    },

    Bitcast {
        from: MirType,
        to: MirType,
        dst: Register,
        src: Operand,
    },

    IntConvert {
        from: MirType,
        to: MirType,
        signed: bool,
        dst: Register,
        src: Operand,
    },

    /// Integer to float. `signed` selects sign or zero extension of `from`.
    /// Rounds to nearest, ties to even. See IR `IntToFloat` for the contract.
    IntToFloat {
        from: MirType,
        to: MirType,
        signed: bool,
        dst: Register,
        src: Operand,
    },

    /// Float to float. Widening is exact; narrowing rounds to nearest, ties to even.
    /// See IR `FloatConvert` for the contract.
    FloatConvert {
        from: MirType,
        to: MirType,
        dst: Register,
        src: Operand,
    },

    /// Float to integer, truncating toward zero with saturation. NaN yields 0.
    /// See IR `FloatToInt` for the contract.
    FloatToInt {
        from: MirType,
        to: MirType,
        signed: bool,
        dst: Register,
        src: Operand,
    },

    StackAddr {
        dst: Register,
        slot: VirtualReg,
    },

    // LEA (Load Effective Address)
    Lea {
        dst: Register,
        base: Register,
        offset: i32,
    },

    // Vector operations
    VectorOp {
        op: VectorOp,
        ty: MirType,
        dst: Register,
        operands: Vec<Operand>,
    },

    // Control flow
    Jmp {
        target: String, // Block label
    },

    Br {
        cond: Register,
        true_target: String,
        false_target: String,
    },

    Switch {
        value: Register,
        cases: Vec<(i64, String)>,
        default: String,
    },

    Call {
        name: String,
        args: Vec<Operand>,
        ret: Option<Register>,
    },

    /// Call through a pointer. The argument and return types travel with the
    /// instruction because there is no callee name to look a signature up by.
    CallIndirect {
        target: Operand,
        arg_types: Vec<MirType>,
        args: Vec<Operand>,
        /// Index of the first argument the convention treats as variadic, when the
        /// callee is variadic. `None` when it is not.
        variadic_fixed: Option<usize>,
        ret: Option<Register>,
        ret_ty: Option<MirType>,
    },

    // Tail call - function call in tail position that can be optimized to a jump
    TailCall {
        name: String,
        args: Vec<Operand>,
    },

    Ret {
        value: Option<Operand>,
    },

    // Meta operations
    Unreachable,

    SafePoint,

    StackMap {
        id: u32,
    },

    PatchPoint {
        id: u32,
    },

    // --- SIMD Operations (Nightly) ---
    #[cfg(feature = "nightly")]
    SimdBinary {
        op: SimdOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdUnary {
        op: SimdOp,
        ty: MirType,
        dst: Register,
        src: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdTernary {
        op: SimdOp,
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
        acc: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdShuffle {
        ty: MirType,
        dst: Register,
        lhs: Operand,
        rhs: Operand,
        mask: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdExtract {
        ty: MirType,
        dst: Register,
        vector: Operand,
        lane_index: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdInsert {
        ty: MirType,
        dst: Register,
        vector: Operand,
        scalar: Operand,
        lane_index: Operand,
    },

    #[cfg(feature = "nightly")]
    SimdLoad {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        attrs: MemoryAttrs,
    },

    #[cfg(feature = "nightly")]
    SimdStore {
        ty: MirType,
        src: Operand,
        addr: AddressMode,
        attrs: MemoryAttrs,
    },

    // --- Atomic Operations (Nightly) ---
    AtomicLoad {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        ordering: MemoryOrdering,
    },

    AtomicStore {
        ty: MirType,
        src: Operand,
        addr: AddressMode,
        ordering: MemoryOrdering,
    },

    AtomicBinary {
        op: AtomicBinOp,
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        value: Operand,
        ordering: MemoryOrdering,
    },

    AtomicCompareExchange {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        expected: Operand,
        desired: Operand,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },

    Fence {
        ordering: MemoryOrdering,
    },

    /// Inline assembly. The template is handed to the assembler after operand substitution.
    /// It always has side effects and is never moved, duplicated or removed.
    InlineAsm {
        template: String,
        operands: Vec<AsmOperand>,
        /// Named registers the template destroys, excluding operands.
        clobbers: Vec<String>,
        /// `clobber(memory)`: a barrier for memory forwarding and reordering.
        clobber_memory: bool,
        /// The result, and its type, when the instruction is `asm.T`.
        result: Option<(Register, MirType)>,
    },

    /// Initialize the `va_list` at the address `ap` from the enclosing variadic
    /// function's register save area and stack arguments.
    VaStart {
        ap: Operand,
    },

    /// Read the next variadic argument of type `ty` through the `va_list` at `ap`
    /// into `dst`, advancing `ap` past it.
    VaArg {
        ty: MirType,
        dst: Register,
        ap: Operand,
    },

    /// End the lifetime of the `va_list` at `ap`.
    VaEnd {
        ap: Operand,
    },

    /// Copy the `va_list` at `src` into the `va_list` at `dst`.
    VaCopy {
        dst: Operand,
        src: Operand,
    },

    // Pseudo-instruction for comments/debugging
    Comment {
        text: String,
    },
}

impl Instruction {
    pub fn may_trap_integer(&self, pointer: PointerWidth) -> bool {
        if let Self::IntBinary {
            op, ty, lhs, rhs, ..
        } = self
        {
            let (zero, overflow) =
                op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant(), pointer);
            zero || overflow
        } else {
            false
        }
    }

    pub fn normalize_integer_immediates(&mut self, return_type: Option<MirType>) {
        match self {
            Instruction::IntBinary { ty, lhs, rhs, .. }
            | Instruction::IntRotate { ty, lhs, rhs, .. }
            | Instruction::IntCmp { ty, lhs, rhs, .. } => {
                normalize_immediate(lhs, *ty);
                normalize_immediate(rhs, *ty);
            }
            Instruction::Store { ty, src, .. }
            | Instruction::Copy { ty, src, .. }
            | Instruction::IntConvert { from: ty, src, .. }
            | Instruction::IntUnary { ty, src, .. }
            | Instruction::IntToFloat { from: ty, src, .. }
            | Instruction::FloatToInt { from: ty, src, .. }
            | Instruction::FloatConvert { from: ty, src, .. }
            | Instruction::Bitcast { from: ty, src, .. } => {
                normalize_immediate(src, *ty);
            }
            Instruction::Select {
                ty,
                true_val,
                false_val,
                ..
            } => {
                normalize_immediate(true_val, *ty);
                normalize_immediate(false_val, *ty);
            }
            Instruction::Ret { value: Some(value) } => {
                if let Some(ty) = return_type {
                    normalize_immediate(value, ty);
                }
            }
            _ => {}
        }
    }

    /// Mutable successor labels, in the order `Block::successors` reports them.
    pub fn successors_mut(&mut self) -> Vec<&mut String> {
        match self {
            Instruction::Jmp { target } => vec![target],
            Instruction::Br {
                true_target,
                false_target,
                ..
            } => vec![true_target, false_target],
            Instruction::Switch { cases, default, .. } => {
                let mut targets = vec![default];
                targets.extend(cases.iter_mut().map(|(_, target)| target));
                targets
            }
            _ => Vec::new(),
        }
    }

    /// Get the destination register if this instruction defines one
    pub fn def_reg(&self) -> Option<&Register> {
        match self {
            Instruction::Copy { dst, .. }
            | Instruction::SymbolAddress { dst, .. }
            | Instruction::IntBinary { dst, .. }
            | Instruction::FloatBinary { dst, .. }
            | Instruction::FloatUnary { dst, .. }
            | Instruction::IntUnary { dst, .. }
            | Instruction::IntRotate { dst, .. }
            | Instruction::IntCmp { dst, .. }
            | Instruction::FloatCmp { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::IntConvert { dst, .. }
            | Instruction::IntToFloat { dst, .. }
            | Instruction::FloatToInt { dst, .. }
            | Instruction::FloatConvert { dst, .. }
            | Instruction::Bitcast { dst, .. }
            | Instruction::StackAddr { dst, .. }
            | Instruction::Lea { dst, .. }
            | Instruction::VectorOp { dst, .. }
            | Instruction::VaArg { dst, .. } => Some(dst),
            Instruction::AtomicLoad { dst, .. }
            | Instruction::AtomicBinary { dst, .. }
            | Instruction::AtomicCompareExchange { dst, .. } => Some(dst),
            Instruction::InlineAsm {
                result: Some((dst, _)),
                ..
            } => Some(dst),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { dst, .. }
            | Instruction::SimdUnary { dst, .. }
            | Instruction::SimdTernary { dst, .. }
            | Instruction::SimdShuffle { dst, .. }
            | Instruction::SimdExtract { dst, .. }
            | Instruction::SimdInsert { dst, .. }
            | Instruction::SimdLoad { dst, .. } => Some(dst),
            Instruction::Call { ret: Some(dst), .. }
            | Instruction::CallIndirect { ret: Some(dst), .. } => Some(dst),
            _ => None,
        }
    }

    /// Get all registers used by this instruction
    pub fn use_regs(&self) -> Vec<&Register> {
        let mut regs = Vec::new();

        match self {
            Instruction::Bitcast {
                src: Operand::Register(r),
                ..
            }
            | Instruction::IntConvert {
                src: Operand::Register(r),
                ..
            }
            | Instruction::IntToFloat {
                src: Operand::Register(r),
                ..
            }
            | Instruction::FloatToInt {
                src: Operand::Register(r),
                ..
            }
            | Instruction::FloatConvert {
                src: Operand::Register(r),
                ..
            }
            | Instruction::Copy {
                src: Operand::Register(r),
                ..
            } => regs.push(r),
            Instruction::Copy { .. } => {}
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::IntRotate { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => {
                if let Operand::Register(r) = lhs {
                    regs.push(r);
                }
                if let Operand::Register(r) = rhs {
                    regs.push(r);
                }
            }
            Instruction::Lea { base, .. } => {
                // LEA uses its base register as an input
                regs.push(base);
            }
            Instruction::FloatUnary {
                src: Operand::Register(r),
                ..
            }
            | Instruction::IntUnary {
                src: Operand::Register(r),
                ..
            } => {
                regs.push(r);
            }
            Instruction::Select {
                cond,
                true_val,
                false_val,
                ..
            } => {
                regs.push(cond);
                if let Operand::Register(r) = true_val {
                    regs.push(r);
                }
                if let Operand::Register(r) = false_val {
                    regs.push(r);
                }
            }
            Instruction::Load { addr, .. } => match addr {
                AddressMode::BaseOffset { base, .. } => regs.push(base),
                AddressMode::BaseIndexScale { base, index, .. } => {
                    regs.push(base);
                    regs.push(index);
                }
            },
            Instruction::Store { src, addr, .. } => {
                if let Operand::Register(r) = src {
                    regs.push(r);
                }
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::VaStart { ap }
            | Instruction::VaArg { ap, .. }
            | Instruction::VaEnd { ap } => {
                if let Operand::Register(r) = ap {
                    regs.push(r);
                }
            }
            Instruction::InlineAsm { operands, .. } => {
                for operand in operands {
                    if let AsmOperand::Input {
                        value: Operand::Register(r),
                        ..
                    } = operand
                    {
                        regs.push(r);
                    }
                }
            }
            Instruction::VaCopy { dst, src } => {
                if let Operand::Register(r) = dst {
                    regs.push(r);
                }
                if let Operand::Register(r) = src {
                    regs.push(r);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { lhs, rhs, .. }
            | Instruction::SimdShuffle { lhs, rhs, .. } => {
                if let Operand::Register(r) = lhs {
                    regs.push(r);
                }
                if let Operand::Register(r) = rhs {
                    regs.push(r);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary { src, .. } | Instruction::SimdExtract { vector: src, .. } => {
                if let Operand::Register(r) = src {
                    regs.push(r);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary { lhs, rhs, acc, .. } => {
                if let Operand::Register(r) = lhs {
                    regs.push(r);
                }
                if let Operand::Register(r) = rhs {
                    regs.push(r);
                }
                if let Operand::Register(r) = acc {
                    regs.push(r);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdInsert {
                vector,
                scalar,
                lane_index,
                ..
            } => {
                if let Operand::Register(r) = vector {
                    regs.push(r);
                }
                if let Operand::Register(r) = scalar {
                    regs.push(r);
                }
                if let Operand::Register(r) = lane_index {
                    regs.push(r);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad { addr, .. } => match addr {
                AddressMode::BaseOffset { base, .. } => regs.push(base),
                AddressMode::BaseIndexScale { base, index, .. } => {
                    regs.push(base);
                    regs.push(index);
                }
            },
            #[cfg(feature = "nightly")]
            Instruction::SimdStore { src, addr, .. } => {
                if let Operand::Register(r) = src {
                    regs.push(r);
                }
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::AtomicLoad { addr, .. } => match addr {
                AddressMode::BaseOffset { base, .. } => regs.push(base),
                AddressMode::BaseIndexScale { base, index, .. } => {
                    regs.push(base);
                    regs.push(index);
                }
            },
            Instruction::AtomicStore { src, addr, .. } => {
                if let Operand::Register(r) = src {
                    regs.push(r);
                }
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::AtomicBinary { value, addr, .. } => {
                if let Operand::Register(r) = value {
                    regs.push(r);
                }
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::AtomicCompareExchange {
                expected,
                desired,
                addr,
                ..
            } => {
                if let Operand::Register(r) = expected {
                    regs.push(r);
                }
                if let Operand::Register(r) = desired {
                    regs.push(r);
                }
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            #[allow(unreachable_patterns)]
            Instruction::Lea { base, .. } => regs.push(base),
            Instruction::VectorOp { operands, .. } => {
                for op in operands {
                    if let Operand::Register(r) = op {
                        regs.push(r);
                    }
                }
            }
            Instruction::Br { cond, .. } => regs.push(cond),
            Instruction::Switch { value, .. } => regs.push(value),
            Instruction::Call { args, .. } | Instruction::TailCall { args, .. } => {
                for arg in args {
                    if let Operand::Register(r) = arg {
                        regs.push(r);
                    }
                }
            }
            Instruction::CallIndirect { target, args, .. } => {
                if let Operand::Register(r) = target {
                    regs.push(r);
                }
                for arg in args {
                    if let Operand::Register(r) = arg {
                        regs.push(r);
                    }
                }
            }
            Instruction::Ret {
                value: Some(Operand::Register(r)),
                ..
            } => {
                regs.push(r);
            }
            _ => {}
        }

        regs
    }

    /// Returns true if this is a terminator instruction.
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Jmp { .. }
                | Instruction::Br { .. }
                | Instruction::Switch { .. }
                | Instruction::Ret { .. }
                | Instruction::TailCall { .. }
                | Instruction::Unreachable
        )
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Copy { ty, dst, src } => {
                write!(f, "{dst} = copy.{ty} {src}")
            }
            Instruction::SymbolAddress { dst, symbol } => {
                write!(f, "{dst} = symbol_addr @{symbol}")
            }
            Instruction::IntBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}")
            }
            Instruction::FloatBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}")
            }
            Instruction::FloatUnary { op, ty, dst, src } => {
                write!(f, "{dst} = {op}.{ty} {src}")
            }
            Instruction::IntUnary { op, ty, dst, src } => {
                write!(f, "{dst} = {op}.{ty} {src}")
            }
            Instruction::IntRotate {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}")
            }
            Instruction::IntCmp {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{dst} = cmp.{op}.{ty} {lhs}, {rhs}")
            }
            Instruction::FloatCmp {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{dst} = fcmp.{op}.{ty} {lhs}, {rhs}")
            }
            Instruction::Select {
                ty,
                dst,
                cond,
                true_val,
                false_val,
            } => {
                write!(f, "{dst} = select.{ty} {cond}, {true_val}, {false_val}")
            }
            Instruction::Load {
                ty,
                dst,
                addr,
                attrs,
            } => {
                write!(
                    f,
                    "ld.{} {}, {} {{align={}, volatile={}}}",
                    ty, dst, addr, attrs.align, attrs.volatile
                )
            }
            Instruction::Store {
                ty,
                src,
                addr,
                attrs,
            } => {
                write!(
                    f,
                    "st.{} {}, {} {{align={}, volatile={}}}",
                    ty, src, addr, attrs.align, attrs.volatile
                )
            }
            Instruction::Bitcast { from, to, dst, src } => {
                write!(f, "{dst} = bitcast.{from}.{to} {src}")
            }
            Instruction::IntConvert {
                from,
                to,
                signed,
                dst,
                src,
            } => {
                write!(f, "{dst} = intconvert.{from}.{to} signed={signed} {src}")
            }
            Instruction::IntToFloat {
                from,
                to,
                signed,
                dst,
                src,
            } => write!(f, "{dst} = inttofloat.{from}.{to} signed={signed} {src}"),
            Instruction::FloatToInt {
                from,
                to,
                signed,
                dst,
                src,
            } => write!(f, "{dst} = floattoint.{from}.{to} signed={signed} {src}"),
            Instruction::FloatConvert { from, to, dst, src } => {
                write!(f, "{dst} = floatconvert.{from}.{to} {src}")
            }
            Instruction::StackAddr { dst, slot } => write!(f, "{dst} = stackaddr {slot}"),
            Instruction::Lea { dst, base, offset } => {
                write!(f, "lea {dst}, {base}, {offset}")
            }
            Instruction::VectorOp {
                op,
                ty,
                dst,
                operands,
            } => {
                write!(f, "{dst} = {op}.{ty}")?;
                for (i, opnd) in operands.iter().enumerate() {
                    if i == 0 {
                        write!(f, " {opnd}")?;
                    } else {
                        write!(f, ", {opnd}")?;
                    }
                }
                Ok(())
            }
            Instruction::Jmp { target } => write!(f, "jmp {target}"),
            Instruction::Br {
                cond,
                true_target,
                false_target,
            } => {
                write!(f, "br {cond}, {true_target}, {false_target}")
            }
            Instruction::Switch {
                value,
                cases,
                default,
            } => {
                write!(f, "switch {value} [")?;
                for (i, (val, label)) in cases.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val} -> {label}")?;
                }
                write!(f, "] default {default}")
            }
            Instruction::Call { name, args, ret } => {
                if let Some(r) = ret {
                    write!(f, "{r} = call {name}(")?;
                } else {
                    write!(f, "call {name}(")?;
                }
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Instruction::CallIndirect {
                target, args, ret, ..
            } => {
                let args: Vec<String> = args.iter().map(ToString::to_string).collect();
                match ret {
                    Some(dst) => write!(f, "{dst} = callptr {target}({})", args.join(", ")),
                    None => write!(f, "callptr {target}({})", args.join(", ")),
                }
            }
            Instruction::TailCall { name, args } => {
                write!(f, "tailcall {name}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Instruction::Ret { value } => match value {
                Some(v) => write!(f, "ret {v}"),
                None => write!(f, "ret"),
            },
            Instruction::Unreachable => write!(f, "unreachable"),
            Instruction::SafePoint => write!(f, "safepoint"),
            Instruction::StackMap { id } => write!(f, "stackmap {id}"),
            Instruction::PatchPoint { id } => write!(f, "patchpoint {id}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary { op, ty, dst, src } => {
                write!(f, "{dst} = {op}.{ty} {src}")
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary {
                op,
                ty,
                dst,
                lhs,
                rhs,
                acc,
            } => write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}, {acc}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdShuffle {
                ty,
                dst,
                lhs,
                rhs,
                mask,
            } => write!(f, "{dst} = {ty}.shuffle {lhs} {rhs}, {mask}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdExtract {
                ty,
                dst,
                vector,
                lane_index,
            } => write!(f, "{dst} = extract_lane.{ty} {vector}, {lane_index}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdInsert {
                ty,
                dst,
                vector,
                scalar,
                lane_index,
            } => write!(
                f,
                "{dst} = insert_lane.{ty} {vector}, {scalar}, {lane_index}"
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad {
                ty,
                dst,
                addr,
                attrs: _,
            } => write!(f, "{dst} = load_simd.{ty} {addr}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdStore {
                ty,
                src,
                addr,
                attrs: _,
            } => write!(f, "store_simd.{ty} {src}, {addr}"),
            Instruction::AtomicLoad {
                ty,
                dst,
                addr,
                ordering,
            } => write!(f, "{dst} = atomic_load.{ty} [{addr}] {ordering}"),
            Instruction::AtomicStore {
                ty,
                src,
                addr,
                ordering,
            } => write!(f, "atomic_store.{ty} {src}, [{addr}] {ordering}"),
            Instruction::AtomicBinary {
                op,
                ty,
                dst,
                addr,
                value,
                ordering,
            } => write!(f, "{dst} = {op}.{ty} [{addr}] {value}, {ordering}"),
            Instruction::AtomicCompareExchange {
                ty,
                dst,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => write!(
                f,
                "{dst} = atomic_cmpxchg.{ty} [{addr}] {expected}, {desired}, {success_ordering}, {failure_ordering}"
            ),
            Instruction::Fence { ordering } => write!(f, "fence {ordering}"),
            Instruction::InlineAsm {
                template,
                operands,
                clobbers,
                clobber_memory,
                result,
            } => {
                match result {
                    Some((dst, ty)) => write!(f, "{dst} = asm.{ty} \"{template}\"")?,
                    None => write!(f, "asm.void \"{template}\"")?,
                }
                for operand in operands {
                    match operand {
                        AsmOperand::Input { location, value } => {
                            write!(f, ", in({location}) {value}")?
                        }
                        AsmOperand::Output { location } => write!(f, ", out({location})")?,
                    }
                }
                if !clobbers.is_empty() || *clobber_memory {
                    write!(f, ", clobber(")?;
                    let names: Vec<&str> = clobbers.iter().map(String::as_str).collect();
                    let mut all = names;
                    if *clobber_memory {
                        all.push("memory");
                    }
                    write!(f, "{}", all.join(", "))?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Instruction::VaStart { ap } => write!(f, "va_start {ap}"),
            Instruction::VaArg { ty, dst, ap } => write!(f, "{dst} = va_arg.{ty} {ap}"),
            Instruction::VaEnd { ap } => write!(f, "va_end {ap}"),
            Instruction::VaCopy { dst, src } => write!(f, "va_copy {dst}, {src}"),
            Instruction::Comment { text } => write!(f, "; {text}"),
        }
    }
}

fn normalize_immediate(operand: &mut Operand, ty: MirType) {
    let Operand::Immediate(immediate) = operand else {
        return;
    };
    let value = match *immediate {
        Immediate::I8(v) => i64::from(v),
        Immediate::I16(v) => i64::from(v),
        Immediate::I32(v) => i64::from(v),
        Immediate::I64(v) => v,
        _ => return,
    };
    *immediate = match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Immediate::I8(value as i8),
        MirType::Scalar(ScalarType::I16) => Immediate::I16(value as i16),
        MirType::Scalar(ScalarType::I32) => Immediate::I32(value as i32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
            Immediate::I64(value)
        }
        _ => return,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{register::VirtualReg, types::ScalarType};

    #[test]
    fn test_instruction_def_reg() {
        let v0 = Register::Virtual(VirtualReg::gpr(0));
        let v1 = Register::Virtual(VirtualReg::gpr(1));

        let add = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: v0.clone(),
            lhs: Operand::Register(v1.clone()),
            rhs: Operand::Immediate(Immediate::I64(42)),
        };

        assert_eq!(add.def_reg(), Some(&v0));
    }

    #[test]
    fn copy_defines_destination_and_uses_source() {
        let dst = Register::Virtual(VirtualReg::fpr(0));
        let src = Register::Virtual(VirtualReg::fpr(1));
        let copy = Instruction::Copy {
            ty: MirType::Scalar(ScalarType::F64),
            dst: dst.clone(),
            src: Operand::Register(src.clone()),
        };

        assert_eq!(copy.def_reg(), Some(&dst));
        assert_eq!(copy.use_regs(), vec![&src]);
        assert!(!copy.is_terminator());
        assert_eq!(format!("{copy}"), "v0 = copy.f64 v1");
    }

    #[test]
    fn test_instruction_is_terminator() {
        let ret = Instruction::Ret { value: None };
        let jmp = Instruction::Jmp {
            target: "bb1".to_string(),
        };
        let add = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I32),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I32(1)),
            rhs: Operand::Immediate(Immediate::I32(2)),
        };

        assert!(ret.is_terminator());
        assert!(jmp.is_terminator());
        assert!(!add.is_terminator());
    }

    #[test]
    fn bit_unary_evaluates_within_operand_width() {
        let popcount =
            |ty: MirType, v: i64| IntUnOp::Popcount.evaluate(ty, v, PointerWidth::Bits64);
        assert_eq!(popcount(MirType::Scalar(ScalarType::I8), -1), 8);
        assert_eq!(popcount(MirType::Scalar(ScalarType::I64), -1), 64);
        assert_eq!(popcount(MirType::Scalar(ScalarType::I64), 0), 0);
        assert_eq!(
            IntUnOp::Ctlz.evaluate(MirType::Scalar(ScalarType::I8), 1, PointerWidth::Bits64),
            7
        );
        assert_eq!(
            IntUnOp::Ctlz.evaluate(MirType::Scalar(ScalarType::I8), 0, PointerWidth::Bits64),
            8
        );
        assert_eq!(
            IntUnOp::Ctlz.evaluate(MirType::Scalar(ScalarType::I64), 0, PointerWidth::Bits64),
            64
        );
        assert_eq!(
            IntUnOp::Cttz.evaluate(MirType::Scalar(ScalarType::I16), 0, PointerWidth::Bits64),
            16
        );
        assert_eq!(
            IntUnOp::Cttz.evaluate(MirType::Scalar(ScalarType::I64), 0, PointerWidth::Bits64),
            64
        );
        assert_eq!(
            IntUnOp::Cttz.evaluate(MirType::Scalar(ScalarType::I32), 8, PointerWidth::Bits64),
            3
        );
        assert_eq!(
            IntUnOp::Bswap.evaluate(
                MirType::Scalar(ScalarType::I16),
                0x1234,
                PointerWidth::Bits64
            ),
            0x3412
        );
        assert_eq!(
            IntUnOp::Bswap.evaluate(
                MirType::Scalar(ScalarType::I32),
                0x0102_0304,
                PointerWidth::Bits64
            ),
            0x0403_0201
        );
        assert_eq!(
            IntUnOp::Bswap.evaluate(MirType::Scalar(ScalarType::I8), -127, PointerWidth::Bits64),
            -127
        );
    }

    #[test]
    fn rotate_evaluates_with_count_modulo_width() {
        let rotl =
            |ty: MirType, v: i64, n: i64| IntRotOp::Rotl.evaluate(ty, v, n, PointerWidth::Bits64);
        let rotr =
            |ty: MirType, v: i64, n: i64| IntRotOp::Rotr.evaluate(ty, v, n, PointerWidth::Bits64);
        assert_eq!(rotl(MirType::Scalar(ScalarType::I8), -127, 1), 3);
        assert_eq!(rotr(MirType::Scalar(ScalarType::I8), -127, 1), -64);
        assert_eq!(rotl(MirType::Scalar(ScalarType::I32), 0x8000_0001, 4), 0x18);
        assert_eq!(rotl(MirType::Scalar(ScalarType::I64), 1, 63), i64::MIN);
        assert_eq!(rotr(MirType::Scalar(ScalarType::I64), 1, 1), i64::MIN);
        assert_eq!(rotl(MirType::Scalar(ScalarType::I64), 77, 64), 77);
        assert_eq!(
            rotr(MirType::Scalar(ScalarType::I16), 1, 1),
            i64::from(i16::MIN)
        );
        assert_eq!(rotl(MirType::Scalar(ScalarType::I8), 5, 9), 10);
    }
}
