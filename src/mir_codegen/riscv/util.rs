use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Write},
};

use crate::mir::{
    IntCmpOp, Operand,
    instruction::Immediate,
    register::{Register, VirtualReg},
};
use lamina_codegen::{
    LocalRegisterAllocator as RegisterAllocator,
    riscv::{RiscVFrame, RiscVRegAlloc, Xlen},
};

/// Load a virtual register into a destination register
pub fn load_register_to_register<W: Write>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    mv {dest_reg}, {phys}")?;
    } else if let Some(offset) = stack_slots.get(src) {
        RiscVFrame::fp_load(
            writer,
            reg_alloc.target().load_word(),
            dest_reg,
            *offset,
            dest_reg,
        )?;
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {src:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Store a register to a virtual register
pub fn store_register_to_register<W: Write>(
    src_reg: &str,
    dst: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        writeln!(writer, "    mv {phys}, {src_reg}")?;
    } else if let Some(offset) = stack_slots.get(dst) {
        RiscVFrame::fp_store(
            writer,
            reg_alloc.target().store_word(),
            src_reg,
            *offset,
            "t0",
        )?;
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {dst:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Load an operand into a destination register
pub fn load_operand_to_register<W: Write>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), Error> {
    match operand {
        Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, dest_reg)
            }
            Register::Physical(p) => {
                writeln!(writer, "    mv {}, {}", dest_reg, p.name)?;
                Ok(())
            }
        },
        Operand::Immediate(imm) => {
            match imm {
                Immediate::I8(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                Immediate::I16(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                Immediate::I32(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                Immediate::I64(v) => {
                    if *v == 0 {
                        writeln!(writer, "    mv {dest_reg}, zero")?;
                    } else {
                        writeln!(writer, "    li {dest_reg}, {v}")?;
                    }
                }
                Immediate::F32(_) | Immediate::F64(_) => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "RISC-V: Floating-point immediates not yet implemented. Use integer types or load from memory instead.",
                    ));
                }
            }
            Ok(())
        }
    }
}

/// Load a floating-point operand into a floating-point register
/// For RISC-V F/D extensions, we use fa0/fa1 as scratch FP registers
pub fn load_fp_operand_to_register<W: Write>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_fp_reg: &str,
    is_f32: bool,
) -> Result<(), Error> {
    match operand {
        Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                // Load from stack to integer register, then move to FP register
                if let Some(offset) = stack_slots.get(v) {
                    let load = if is_f32 { "flw" } else { "fld" };
                    RiscVFrame::fp_load(writer, load, dest_fp_reg, *offset, "t0")?;
                } else if let Some(phys) = reg_alloc.get_mapping(v) {
                    // Move from integer register to FP register
                    if is_f32 {
                        writeln!(writer, "    fmv.w.x {dest_fp_reg}, {phys}")?;
                    } else if reg_alloc.target().xlen == Xlen::Rv32 {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "RV32 f64 requires a stack slot",
                        ));
                    } else {
                        writeln!(writer, "    fmv.d.x {dest_fp_reg}, {phys}")?;
                    }
                } else {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Virtual register {v:?} has no mapping or stack slot"),
                    ));
                }
            }
            Register::Physical(p) => {
                // Move from physical integer register to FP register
                if is_f32 {
                    writeln!(writer, "    fmv.w.x {}, {}", dest_fp_reg, p.name)?;
                } else if reg_alloc.target().xlen == Xlen::Rv32 {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "RV32 f64 requires a stack slot",
                    ));
                } else {
                    writeln!(writer, "    fmv.d.x {}, {}", dest_fp_reg, p.name)?;
                }
            }
        },
        Operand::Immediate(imm) => {
            match imm {
                Immediate::F32(v) => {
                    // Load F32 immediate via integer register
                    let bits = v.to_bits();
                    writeln!(writer, "    li t0, {bits}")?;
                    writeln!(writer, "    fmv.w.x {dest_fp_reg}, t0")?;
                }
                Immediate::F64(v) => {
                    // Load F64 immediate via integer register
                    let bits = v.to_bits();
                    if reg_alloc.target().xlen == Xlen::Rv32 {
                        writeln!(writer, "    addi sp, sp, -16")?;
                        writeln!(writer, "    li t0, {}", bits as u32)?;
                        writeln!(writer, "    sw t0, 0(sp)")?;
                        writeln!(writer, "    li t0, {}", (bits >> 32) as u32)?;
                        writeln!(writer, "    sw t0, 4(sp)")?;
                        writeln!(writer, "    fld {dest_fp_reg}, 0(sp)")?;
                        writeln!(writer, "    addi sp, sp, 16")?;
                    } else {
                        writeln!(writer, "    li t0, {bits}")?;
                        writeln!(writer, "    fmv.d.x {dest_fp_reg}, t0")?;
                    }
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "Expected floating-point immediate for FP operation",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Store a floating-point register to a virtual register
pub fn store_fp_register_to_register<W: Write>(
    src_fp_reg: &str,
    dst: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    is_f32: bool,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        // Move from FP register to integer register
        if is_f32 {
            writeln!(writer, "    fmv.x.w {phys}, {src_fp_reg}")?;
        } else if reg_alloc.target().xlen == Xlen::Rv32 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "RV32 f64 requires a stack slot",
            ));
        } else {
            writeln!(writer, "    fmv.x.d {phys}, {src_fp_reg}")?;
        }
    } else if let Some(offset) = stack_slots.get(dst) {
        // Store directly from FP register to stack
        let store = if is_f32 { "fsw" } else { "fsd" };
        RiscVFrame::fp_store(writer, store, src_fp_reg, *offset, "t0")?;
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {dst:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Emit RISC-V instruction for integer comparison operations
pub fn emit_int_cmp_op<W: Write>(op: &IntCmpOp, writer: &mut W) -> Result<(), Error> {
    match op {
        IntCmpOp::Eq => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    seqz a0, a0")?; // Set if equal to zero
        }
        IntCmpOp::Ne => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    snez a0, a0")?; // Set if not equal to zero
        }
        IntCmpOp::SLt => writeln!(writer, "    slt a0, a0, a1")?,
        IntCmpOp::SLe => {
            writeln!(writer, "    sgt a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        IntCmpOp::SGt => writeln!(writer, "    sgt a0, a0, a1")?,
        IntCmpOp::SGe => {
            writeln!(writer, "    slt a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        IntCmpOp::ULt => writeln!(writer, "    sltu a0, a0, a1")?,
        IntCmpOp::ULe => {
            writeln!(writer, "    sgtu a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        IntCmpOp::UGt => writeln!(writer, "    sgtu a0, a0, a1")?,
        IntCmpOp::UGe => {
            writeln!(writer, "    sltu a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
    }
    Ok(())
}

/// Replaces a0 with `limit` when a0 is above it, without branching. `compare` is
/// `slt` for a signed limit and `sltu` for an unsigned one.
pub fn clamp_above<W: Write>(writer: &mut W, limit: i64, compare: &str) -> Result<(), Error> {
    writeln!(writer, "    li t1, {limit}")?;
    writeln!(writer, "    {compare} t2, t1, a0")?;
    writeln!(writer, "    neg t2, t2")?;
    writeln!(writer, "    xor t3, a0, t1")?;
    writeln!(writer, "    and t3, t3, t2")?;
    writeln!(writer, "    xor a0, a0, t3")?;
    Ok(())
}

/// Replaces a0 with `limit` when a0 is below it, signed.
pub fn clamp_below<W: Write>(writer: &mut W, limit: i64) -> Result<(), Error> {
    writeln!(writer, "    li t1, {limit}")?;
    writeln!(writer, "    slt t2, a0, t1")?;
    writeln!(writer, "    neg t2, t2")?;
    writeln!(writer, "    xor t3, a0, t1")?;
    writeln!(writer, "    and t3, t3, t2")?;
    writeln!(writer, "    xor a0, a0, t3")?;
    Ok(())
}

/// Rewrites a0 as the sign-extended form of its low `bits`, the shape a 64-bit
/// register holds a narrower value in.
pub fn sign_extend_to_xlen<W: Write>(writer: &mut W, bits: usize) -> Result<(), Error> {
    if bits >= 64 {
        return Ok(());
    }
    if bits == 32 {
        writeln!(writer, "    sext.w a0, a0")?;
    } else {
        writeln!(writer, "    slli a0, a0, {}", 64 - bits)?;
        writeln!(writer, "    srai a0, a0, {}", 64 - bits)?;
    }
    Ok(())
}
