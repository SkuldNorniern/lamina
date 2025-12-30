///
/// Low-level, machine-friendly instructions that map closely to actual assembly.
use super::register::Register;
use super::types::MirType;
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
        write!(f, "{}", s)
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
        write!(f, "{}", s)
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
        write!(f, "{}", s)
    }
}

/// Memory ordering constraints for atomic operations.
#[cfg(feature = "nightly")]
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
            Immediate::I8(v) => write!(f, "{}", v),
            Immediate::I16(v) => write!(f, "{}", v),
            Immediate::I32(v) => write!(f, "{}", v),
            Immediate::I64(v) => write!(f, "{}", v),
            Immediate::F32(v) => write!(f, "{}", v),
            Immediate::F64(v) => write!(f, "{}", v),
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
            Operand::Register(r) => write!(f, "{}", r),
            Operand::Immediate(i) => write!(f, "{}", i),
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

/// Memory addressing mode
#[derive(Debug, Clone, PartialEq)]
pub enum AddressMode {
    /// [base + imm12]
    BaseOffset {
        base: Register,
        offset: i16, // 12-bit signed, extended to i16
    },
    /// [base + index<<scale + imm4]
    BaseIndexScale {
        base: Register,
        index: Register,
        scale: u8,  // 1, 2, 4, or 8
        offset: i8, // 4-bit signed, extended to i8
    },
}

impl fmt::Display for AddressMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressMode::BaseOffset { base, offset } => {
                if *offset == 0 {
                    write!(f, "[{}]", base)
                } else {
                    write!(f, "[{} + {}]", base, offset)
                }
            }
            AddressMode::BaseIndexScale {
                base,
                index,
                scale,
                offset,
            } => {
                if *offset == 0 {
                    write!(f, "[{} + {}<<{}]", base, index, scale)
                } else {
                    write!(f, "[{} + {}<<{} + {}]", base, index, scale, offset)
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
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
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
    #[cfg(feature = "nightly")]
    AtomicLoad {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        ordering: MemoryOrdering,
    },

    #[cfg(feature = "nightly")]
    AtomicStore {
        ty: MirType,
        src: Operand,
        addr: AddressMode,
        ordering: MemoryOrdering,
    },

    #[cfg(feature = "nightly")]
    AtomicBinary {
        op: AtomicBinOp,
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        value: Operand,
        ordering: MemoryOrdering,
    },

    #[cfg(feature = "nightly")]
    AtomicCompareExchange {
        ty: MirType,
        dst: Register,
        addr: AddressMode,
        expected: Operand,
        desired: Operand,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },

    #[cfg(feature = "nightly")]
    Fence {
        ordering: MemoryOrdering,
    },

    // Pseudo-instruction for comments/debugging
    Comment {
        text: String,
    },
}

impl Instruction {
    /// Get the destination register if this instruction defines one
    pub fn def_reg(&self) -> Option<&Register> {
        match self {
            Instruction::IntBinary { dst, .. }
            | Instruction::FloatBinary { dst, .. }
            | Instruction::FloatUnary { dst, .. }
            | Instruction::IntCmp { dst, .. }
            | Instruction::FloatCmp { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Lea { dst, .. }
            | Instruction::VectorOp { dst, .. } => Some(dst),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { dst, .. }
            | Instruction::SimdUnary { dst, .. }
            | Instruction::SimdTernary { dst, .. }
            | Instruction::SimdShuffle { dst, .. }
            | Instruction::SimdExtract { dst, .. }
            | Instruction::SimdInsert { dst, .. }
            | Instruction::SimdLoad { dst, .. }
            | Instruction::AtomicLoad { dst, .. }
            | Instruction::AtomicBinary { dst, .. }
            | Instruction::AtomicCompareExchange { dst, .. } => Some(dst),
            Instruction::Call { ret: Some(dst), .. } => Some(dst),
            _ => None,
        }
    }

    /// Get all registers used by this instruction
    pub fn use_regs(&self) -> Vec<&Register> {
        let mut regs = Vec::new();

        match self {
            Instruction::IntBinary { lhs, rhs, .. }
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
            #[cfg(feature = "nightly")]
            Instruction::AtomicLoad { addr, .. } => match addr {
                AddressMode::BaseOffset { base, .. } => regs.push(base),
                AddressMode::BaseIndexScale { base, index, .. } => {
                    regs.push(base);
                    regs.push(index);
                }
            },
            #[cfg(feature = "nightly")]
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
            #[cfg(feature = "nightly")]
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
            #[cfg(feature = "nightly")]
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
            Instruction::IntBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{} = {}.{} {}, {}", dst, op, ty, lhs, rhs)
            }
            Instruction::FloatBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{} = {}.{} {}, {}", dst, op, ty, lhs, rhs)
            }
            Instruction::FloatUnary { op, ty, dst, src } => {
                write!(f, "{} = {}.{} {}", dst, op, ty, src)
            }
            Instruction::IntCmp {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{} = cmp.{}.{} {}, {}", dst, op, ty, lhs, rhs)
            }
            Instruction::FloatCmp {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                write!(f, "{} = fcmp.{}.{} {}, {}", dst, op, ty, lhs, rhs)
            }
            Instruction::Select {
                ty,
                dst,
                cond,
                true_val,
                false_val,
            } => {
                write!(
                    f,
                    "{} = select.{} {}, {}, {}",
                    dst, ty, cond, true_val, false_val
                )
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
            Instruction::Lea { dst, base, offset } => {
                write!(f, "lea {}, {}, {}", dst, base, offset)
            }
            Instruction::VectorOp {
                op,
                ty,
                dst,
                operands,
            } => {
                write!(f, "{} = {}.{}", dst, op, ty)?;
                for (i, opnd) in operands.iter().enumerate() {
                    if i == 0 {
                        write!(f, " {}", opnd)?;
                    } else {
                        write!(f, ", {}", opnd)?;
                    }
                }
                Ok(())
            }
            Instruction::Jmp { target } => write!(f, "jmp {}", target),
            Instruction::Br {
                cond,
                true_target,
                false_target,
            } => {
                write!(f, "br {}, {}, {}", cond, true_target, false_target)
            }
            Instruction::Switch {
                value,
                cases,
                default,
            } => {
                write!(f, "switch {} [", value)?;
                for (i, (val, label)) in cases.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} -> {}", val, label)?;
                }
                write!(f, "] default {}", default)
            }
            Instruction::Call { name, args, ret } => {
                if let Some(r) = ret {
                    write!(f, "{} = call {}(", r, name)?;
                } else {
                    write!(f, "call {}(", name)?;
                }
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            Instruction::TailCall { name, args } => {
                write!(f, "tailcall {}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            Instruction::Ret { value } => match value {
                Some(v) => write!(f, "ret {}", v),
                None => write!(f, "ret"),
            },
            Instruction::Unreachable => write!(f, "unreachable"),
            Instruction::SafePoint => write!(f, "safepoint"),
            Instruction::StackMap { id } => write!(f, "stackmap {}", id),
            Instruction::PatchPoint { id } => write!(f, "patchpoint {}", id),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => write!(f, "{} = {}.{} {}, {}", dst, op, ty, lhs, rhs),
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary { op, ty, dst, src } => {
                write!(f, "{} = {}.{} {}", dst, op, ty, src)
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary {
                op,
                ty,
                dst,
                lhs,
                rhs,
                acc,
            } => write!(f, "{} = {}.{} {}, {}, {}", dst, op, ty, lhs, rhs, acc),
            #[cfg(feature = "nightly")]
            Instruction::SimdShuffle {
                ty,
                dst,
                lhs,
                rhs,
                mask,
            } => write!(f, "{} = {}.shuffle {} {}, {}", dst, ty, lhs, rhs, mask),
            #[cfg(feature = "nightly")]
            Instruction::SimdExtract {
                ty,
                dst,
                vector,
                lane_index,
            } => write!(
                f,
                "{} = extract_lane.{} {}, {}",
                dst, ty, vector, lane_index
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdInsert {
                ty,
                dst,
                vector,
                scalar,
                lane_index,
            } => write!(
                f,
                "{} = insert_lane.{} {}, {}, {}",
                dst, ty, vector, scalar, lane_index
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad {
                ty,
                dst,
                addr,
                attrs,
            } => write!(f, "{} = load_simd.{} {}", dst, ty, addr),
            #[cfg(feature = "nightly")]
            Instruction::SimdStore {
                ty,
                src,
                addr,
                attrs,
            } => write!(f, "store_simd.{} {}, {}", ty, src, addr),
            #[cfg(feature = "nightly")]
            Instruction::AtomicLoad {
                ty,
                dst,
                addr,
                ordering,
            } => write!(f, "{} = atomic_load.{} [{}] {}", dst, ty, addr, ordering),
            #[cfg(feature = "nightly")]
            Instruction::AtomicStore {
                ty,
                src,
                addr,
                ordering,
            } => write!(f, "atomic_store.{} {}, [{}] {}", ty, src, addr, ordering),
            #[cfg(feature = "nightly")]
            Instruction::AtomicBinary {
                op,
                ty,
                dst,
                addr,
                value,
                ordering,
            } => write!(
                f,
                "{} = {}.{} [{}] {}, {}",
                dst, op, ty, addr, value, ordering
            ),
            #[cfg(feature = "nightly")]
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
                "{} = atomic_cmpxchg.{} [{}] {}, {}, {}, {}",
                dst, ty, addr, expected, desired, success_ordering, failure_ordering
            ),
            #[cfg(feature = "nightly")]
            Instruction::Fence { ordering } => write!(f, "fence {}", ordering),
            Instruction::Comment { text } => write!(f, "; {}", text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::register::VirtualReg;
    use crate::mir::types::ScalarType;

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
}
