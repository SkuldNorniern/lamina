use crate::mir::register::{Register, VirtualReg};
use lamina_codegen::LocalRegisterAllocator as RegisterAllocator;

/// Load a virtual register into a destination register
pub fn load_register_to_register<W: std::io::Write>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &lamina_codegen::riscv::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    mv {dest_reg}, {phys}")?;
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    ld {dest_reg}, {offset}(fp)")?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {src:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Store a register to a virtual register
pub fn store_register_to_register<W: std::io::Write>(
    src_reg: &str,
    dst: &VirtualReg,
    writer: &mut W,
    reg_alloc: &lamina_codegen::riscv::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        writeln!(writer, "    mv {phys}, {src_reg}")?;
    } else if let Some(offset) = stack_slots.get(dst) {
        writeln!(writer, "    sd {src_reg}, {offset}(fp)")?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {dst:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Load an operand into a destination register
pub fn load_operand_to_register<W: std::io::Write>(
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &lamina_codegen::riscv::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), std::io::Error> {
    match operand {
        crate::mir::Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, dest_reg)
            }
            Register::Physical(p) => {
                writeln!(writer, "    mv {}, {}", dest_reg, p.name)?;
                Ok(())
            }
        },
        crate::mir::Operand::Immediate(imm) => {
            match imm {
                crate::mir::instruction::Immediate::I8(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I16(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I32(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I64(v) => {
                    if *v == 0 {
                        writeln!(writer, "    mv {dest_reg}, zero")?;
                    } else {
                        writeln!(writer, "    li {dest_reg}, {v}")?;
                    }
                }
                crate::mir::instruction::Immediate::F32(_)
                | crate::mir::instruction::Immediate::F64(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
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
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &lamina_codegen::riscv::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    dest_fp_reg: &str,
    is_f32: bool,
) -> Result<(), std::io::Error> {
    match operand {
        crate::mir::Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                // Load from stack to integer register, then move to FP register
                if let Some(offset) = stack_slots.get(v) {
                    if is_f32 {
                        writeln!(writer, "    flw {dest_fp_reg}, {offset}(fp)")?;
                    } else {
                        writeln!(writer, "    fld {dest_fp_reg}, {offset}(fp)")?;
                    }
                } else if let Some(phys) = reg_alloc.get_mapping(v) {
                    // Move from integer register to FP register
                    if is_f32 {
                        writeln!(writer, "    fmv.w.x {dest_fp_reg}, {phys}")?;
                    } else {
                        writeln!(writer, "    fmv.d.x {dest_fp_reg}, {phys}")?;
                    }
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Virtual register {v:?} has no mapping or stack slot"),
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
        crate::mir::Operand::Immediate(imm) => {
            match imm {
                crate::mir::instruction::Immediate::F32(v) => {
                    // Load F32 immediate via integer register
                    let bits = v.to_bits();
                    writeln!(writer, "    li t0, {bits}")?;
                    writeln!(writer, "    fmv.w.x {dest_fp_reg}, t0")?;
                }
                crate::mir::instruction::Immediate::F64(v) => {
                    // Load F64 immediate via integer register
                    let bits = v.to_bits();
                    writeln!(writer, "    li t0, {bits}")?;
                    writeln!(writer, "    fmv.d.x {dest_fp_reg}, t0")?;
                }
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
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
    reg_alloc: &lamina_codegen::riscv::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    is_f32: bool,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        // Move from FP register to integer register
        if is_f32 {
            writeln!(writer, "    fmv.x.w {phys}, {src_fp_reg}")?;
        } else {
            writeln!(writer, "    fmv.x.d {phys}, {src_fp_reg}")?;
        }
    } else if let Some(offset) = stack_slots.get(dst) {
        // Store directly from FP register to stack
        if is_f32 {
            writeln!(writer, "    fsw {src_fp_reg}, {offset}(fp)")?;
        } else {
            writeln!(writer, "    fsd {src_fp_reg}, {offset}(fp)")?;
        }
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {dst:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Emit RISC-V instruction for integer comparison operations
pub fn emit_int_cmp_op<W: std::io::Write>(
    op: &crate::mir::IntCmpOp,
    writer: &mut W,
) -> Result<(), std::io::Error> {
    match op {
        crate::mir::IntCmpOp::Eq => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    seqz a0, a0")?; // Set if equal to zero
        }
        crate::mir::IntCmpOp::Ne => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    snez a0, a0")?; // Set if not equal to zero
        }
        crate::mir::IntCmpOp::SLt => writeln!(writer, "    slt a0, a0, a1")?,
        crate::mir::IntCmpOp::SLe => {
            writeln!(writer, "    sgt a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::SGt => writeln!(writer, "    sgt a0, a0, a1")?,
        crate::mir::IntCmpOp::SGe => {
            writeln!(writer, "    slt a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::ULt => writeln!(writer, "    sltu a0, a0, a1")?,
        crate::mir::IntCmpOp::ULe => {
            writeln!(writer, "    sgtu a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::UGt => writeln!(writer, "    sgtu a0, a0, a1")?,
        crate::mir::IntCmpOp::UGe => {
            writeln!(writer, "    sltu a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
    }
    Ok(())
}
