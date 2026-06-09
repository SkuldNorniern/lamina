use std::collections::HashMap;
use std::io::{Error, ErrorKind};

use crate::mir::instruction::Immediate;
use crate::mir::register::{Register, VirtualReg};
use crate::mir::{IntCmpOp, Operand};
use lamina_codegen::LocalRegisterAllocator as RegisterAllocator;
use lamina_codegen::riscv::RiscVRegAlloc;

/// Load a virtual register into a destination register
pub fn load_register_to_register<W: std::io::Write>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    mv {}, {}", dest_reg, phys)?;
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    ld {}, {}(fp)", dest_reg, offset)?;
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", src),
        ));
    }
    Ok(())
}

/// Store a register to a virtual register
pub fn store_register_to_register<W: std::io::Write>(
    src_reg: &str,
    dst: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        writeln!(writer, "    mv {}, {}", phys, src_reg)?;
    } else if let Some(offset) = stack_slots.get(dst) {
        writeln!(writer, "    sd {}, {}(fp)", src_reg, offset)?;
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", dst),
        ));
    }
    Ok(())
}

/// Load an operand into a destination register
pub fn load_operand_to_register<W: std::io::Write>(
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
                        writeln!(writer, "    mv {}, zero", dest_reg)?;
                    } else {
                        writeln!(writer, "    li {}, {}", dest_reg, v)?;
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
pub fn load_fp_operand_to_register<W: std::io::Write>(
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
                    if is_f32 {
                        writeln!(writer, "    flw {}, {}(fp)", dest_fp_reg, offset)?;
                    } else {
                        writeln!(writer, "    fld {}, {}(fp)", dest_fp_reg, offset)?;
                    }
                } else if let Some(phys) = reg_alloc.get_mapping(v) {
                    // Move from integer register to FP register
                    if is_f32 {
                        writeln!(writer, "    fmv.w.x {}, {}", dest_fp_reg, phys)?;
                    } else {
                        writeln!(writer, "    fmv.d.x {}, {}", dest_fp_reg, phys)?;
                    }
                } else {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Virtual register {:?} has no mapping or stack slot", v),
                    ));
                }
            }
            Register::Physical(p) => {
                // Move from physical integer register to FP register
                if is_f32 {
                    writeln!(writer, "    fmv.w.x {}, {}", dest_fp_reg, p.name)?;
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
                    writeln!(writer, "    li t0, {}", bits)?;
                    writeln!(writer, "    fmv.w.x {}, t0", dest_fp_reg)?;
                }
                Immediate::F64(v) => {
                    // Load F64 immediate via integer register
                    let bits = v.to_bits();
                    writeln!(writer, "    li t0, {}", bits)?;
                    writeln!(writer, "    fmv.d.x {}, t0", dest_fp_reg)?;
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
pub fn store_fp_register_to_register<W: std::io::Write>(
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
            writeln!(writer, "    fmv.x.w {}, {}", phys, src_fp_reg)?;
        } else {
            writeln!(writer, "    fmv.x.d {}, {}", phys, src_fp_reg)?;
        }
    } else if let Some(offset) = stack_slots.get(dst) {
        // Store directly from FP register to stack
        if is_f32 {
            writeln!(writer, "    fsw {}, {}(fp)", src_fp_reg, offset)?;
        } else {
            writeln!(writer, "    fsd {}, {}(fp)", src_fp_reg, offset)?;
        }
    } else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", dst),
        ));
    }
    Ok(())
}

/// Emit RISC-V instruction for integer comparison operations
pub fn emit_int_cmp_op<W: std::io::Write>(op: &IntCmpOp, writer: &mut W) -> Result<(), Error> {
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
