//! Instruction representation for LUMIR.
//!
//! LUMIR instructions are low-level, machine-friendly operations that map
//! closely to actual assembly instructions. This design supports
//! code generation and optimization.

pub mod ops;
pub use ops::{
    FloatBinOp, FloatCmpOp, FloatUnOp, IntBinOp, IntCmpOp, VectorOp,
};
#[cfg(feature = "nightly")]
pub use ops::{AtomicBinOp, MemoryOrdering, SimdOp};

use super::register::Register;
use super::types::MirType;
use std::fmt;


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

impl Operand {
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
        Operand::Register(Register::Virtual(
            super::register::VirtualReg::new(id, super::register::RegisterClass::Gpr),
        ))
    }

    /// Create a virtual FPR operand with the given id.
    pub fn v_fpr(id: u32) -> Self {
        Operand::Register(Register::Virtual(
            super::register::VirtualReg::new(id, super::register::RegisterClass::Fpr),
        ))
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
    use crate::mir::types::{ScalarType, VectorLane, VectorType};

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

    #[test]
    fn test_integer_sizes_i8() {
        // Test I8 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_i8 = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I8),
            dst: Register::Virtual(v2),
            lhs: Operand::Immediate(Immediate::I8(10)),
            rhs: Operand::Immediate(Immediate::I8(20)),
        };

        assert_eq!(add_i8.def_reg(), Some(&v2));
        assert_eq!(add_i8.ty(), &MirType::Scalar(ScalarType::I8));

        let cmp_i8 = Instruction::IntCmp {
            op: IntCmpOp::Eq,
            ty: MirType::Scalar(ScalarType::I8),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(cmp_i8.ty(), &MirType::Scalar(ScalarType::I8));
    }

    #[test]
    fn test_integer_sizes_i16() {
        // Test I16 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_i16 = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I16),
            dst: Register::Virtual(v2),
            lhs: Operand::Immediate(Immediate::I16(100)),
            rhs: Operand::Immediate(Immediate::I16(200)),
        };

        assert_eq!(add_i16.def_reg(), Some(&v2));
        assert_eq!(add_i16.ty(), &MirType::Scalar(ScalarType::I16));

        let mul_i16 = Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I16),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(mul_i16.ty(), &MirType::Scalar(ScalarType::I16));
    }

    #[test]
    fn test_integer_sizes_i32() {
        // Test I32 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_i32 = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I32),
            dst: Register::Virtual(v2),
            lhs: Operand::Immediate(Immediate::I32(1000)),
            rhs: Operand::Immediate(Immediate::I32(2000)),
        };

        assert_eq!(add_i32.def_reg(), Some(&v2));
        assert_eq!(add_i32.ty(), &MirType::Scalar(ScalarType::I32));

        let div_i32 = Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I32),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(div_i32.ty(), &MirType::Scalar(ScalarType::I32));
    }

    #[test]
    fn test_integer_sizes_i64() {
        // Test I64 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_i64 = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(v2),
            lhs: Operand::Immediate(Immediate::I64(100000)),
            rhs: Operand::Immediate(Immediate::I64(200000)),
        };

        assert_eq!(add_i64.def_reg(), Some(&v2));
        assert_eq!(add_i64.ty(), &MirType::Scalar(ScalarType::I64));

        let rem_i64 = Instruction::IntBinary {
            op: IntBinOp::SRem,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(rem_i64.ty(), &MirType::Scalar(ScalarType::I64));
    }

    #[test]
    fn test_floating_point_f32() {
        // Test F32 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_f32 = Instruction::FloatBinary {
            op: FloatBinOp::FAdd,
            ty: MirType::Scalar(ScalarType::F32),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(add_f32.def_reg(), Some(&v2));
        assert_eq!(add_f32.ty(), &MirType::Scalar(ScalarType::F32));

        let mul_f32 = Instruction::FloatBinary {
            op: FloatBinOp::FMul,
            ty: MirType::Scalar(ScalarType::F32),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(mul_f32.ty(), &MirType::Scalar(ScalarType::F32));

        let neg_f32 = Instruction::FloatUnary {
            op: FloatUnOp::FNeg,
            ty: MirType::Scalar(ScalarType::F32),
            dst: Register::Virtual(v2),
            src: Operand::Register(Register::Virtual(v0)),
        };

        assert_eq!(neg_f32.ty(), &MirType::Scalar(ScalarType::F32));
    }

    #[test]
    fn test_floating_point_f64() {
        // Test F64 operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let add_f64 = Instruction::FloatBinary {
            op: FloatBinOp::FAdd,
            ty: MirType::Scalar(ScalarType::F64),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(add_f64.def_reg(), Some(&v2));
        assert_eq!(add_f64.ty(), &MirType::Scalar(ScalarType::F64));

        let div_f64 = Instruction::FloatBinary {
            op: FloatBinOp::FDiv,
            ty: MirType::Scalar(ScalarType::F64),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(div_f64.ty(), &MirType::Scalar(ScalarType::F64));

        let sqrt_f64 = Instruction::FloatUnary {
            op: FloatUnOp::FSqrt,
            ty: MirType::Scalar(ScalarType::F64),
            dst: Register::Virtual(v2),
            src: Operand::Register(Register::Virtual(v0)),
        };

        assert_eq!(sqrt_f64.ty(), &MirType::Scalar(ScalarType::F64));
    }

    #[test]
    fn test_floating_point_comparisons() {
        // Test floating point comparisons
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let cmp_f32 = Instruction::FloatCmp {
            op: FloatCmpOp::Eq,
            ty: MirType::Scalar(ScalarType::F32),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(cmp_f32.ty(), &MirType::Scalar(ScalarType::F32));

        let cmp_f64 = Instruction::FloatCmp {
            op: FloatCmpOp::Lt,
            ty: MirType::Scalar(ScalarType::F64),
            dst: Register::Virtual(v2),
            lhs: Operand::Register(Register::Virtual(v0)),
            rhs: Operand::Register(Register::Virtual(v1)),
        };

        assert_eq!(cmp_f64.ty(), &MirType::Scalar(ScalarType::F64));
    }

    #[test]
    fn test_vector_types() {
        // Test vector operations
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let v2 = VirtualReg::gpr(2);

        let vec_i32 = Instruction::VectorOp {
            op: VectorOp::VAdd,
            ty: MirType::Vector(VectorType::V128(VectorLane::I32)),
            dst: Register::Virtual(v2),
            operands: vec![
                Operand::Register(Register::Virtual(v0)),
                Operand::Register(Register::Virtual(v1)),
            ],
        };

        assert_eq!(vec_i32.def_reg(), Some(&v2));
        assert!(vec_i32.ty().is_vector());

        let vec_f32 = Instruction::VectorOp {
            op: VectorOp::VAdd,
            ty: MirType::Vector(VectorType::V128(VectorLane::F32)),
            dst: Register::Virtual(v2),
            operands: vec![
                Operand::Register(Register::Virtual(v0)),
                Operand::Register(Register::Virtual(v1)),
            ],
        };

        assert!(vec_f32.ty().is_vector());
    }
}
