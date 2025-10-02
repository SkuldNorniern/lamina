/// LUMIR instruction set
///
/// Low-level, machine-friendly instructions that map closely to actual assembly.

use super::register::Register;
use super::types::MirType;

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

/// Floating-point binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatBinOp {
    FAdd,
    FSub,
    FMul,
    FDiv,
}

/// Floating-point unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatUnOp {
    FNeg,
    FSqrt,
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

/// Operand (register or immediate)
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(Register),
    Immediate(Immediate),
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
        scale: u8, // 1, 2, 4, or 8
        offset: i8, // 4-bit signed, extended to i8
    },
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
            Instruction::Call { ret: Some(dst), .. } => Some(dst),
            _ => None,
        }
    }

    /// Get all registers used by this instruction
    pub fn use_regs(&self) -> Vec<&Register> {
        let mut regs = Vec::new();
        
        let add_operand = |regs: &mut Vec<&Register>, op: &Operand| {
            if let Operand::Register(r) = op {
                regs.push(r);
            }
        };

        match self {
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => {
                add_operand(&mut regs, lhs);
                add_operand(&mut regs, rhs);
            }
            Instruction::FloatUnary { src, .. } => add_operand(&mut regs, src),
            Instruction::Select { cond, true_val, false_val, .. } => {
                regs.push(cond);
                add_operand(&mut regs, true_val);
                add_operand(&mut regs, false_val);
            }
            Instruction::Load { addr, .. } => {
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::Store { src, addr, .. } => {
                add_operand(&mut regs, src);
                match addr {
                    AddressMode::BaseOffset { base, .. } => regs.push(base),
                    AddressMode::BaseIndexScale { base, index, .. } => {
                        regs.push(base);
                        regs.push(index);
                    }
                }
            }
            Instruction::Lea { base, .. } => regs.push(base),
            Instruction::VectorOp { operands, .. } => {
                for op in operands {
                    add_operand(&mut regs, op);
                }
            }
            Instruction::Br { cond, .. } => regs.push(cond),
            Instruction::Switch { value, .. } => regs.push(value),
            Instruction::Call { args, .. } => {
                for arg in args {
                    add_operand(&mut regs, arg);
                }
            }
            Instruction::Ret { value: Some(val), .. } => add_operand(&mut regs, val),
            _ => {}
        }

        regs
    }

    /// Check if this is a terminator instruction
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Jmp { .. }
                | Instruction::Br { .. }
                | Instruction::Switch { .. }
                | Instruction::Ret { .. }
                | Instruction::Unreachable
        )
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
        let jmp = Instruction::Jmp { target: "bb1".to_string() };
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
