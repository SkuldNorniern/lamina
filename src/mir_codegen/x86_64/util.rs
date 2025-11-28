//! Utility functions for x86_64 code generation.

use crate::mir::register::{Register, VirtualReg};

/// Load a virtual register into RAX
pub fn load_register_to_rax<
    W: std::io::Write,
    RA: crate::mir_codegen::regalloc::RegisterAllocator<PhysReg = &'static str>,
>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    movq %{}, %rax", phys)?;
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    movq {}(%rbp), %rax", offset)?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", reg),
        ));
    }
    Ok(())
}

/// Store RAX to a virtual register
pub fn store_rax_to_register<
    W: std::io::Write,
    RA: crate::mir_codegen::regalloc::RegisterAllocator<PhysReg = &'static str>,
>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    movq %rax, %{}", phys)?;
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    movq %rax, {}(%rbp)", offset)?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", reg),
        ));
    }
    Ok(())
}

/// Load a register to another register
pub fn load_register_to_register<
    W: std::io::Write,
    RA: crate::mir_codegen::regalloc::RegisterAllocator<PhysReg = &'static str>,
>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    movq %{}, %{}", phys, target_reg)?;
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    movq {}(%rbp), %{}", offset, target_reg)?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", src),
        ));
    }
    Ok(())
}

/// Load an operand into RAX
pub fn load_operand_to_rax<
    W: std::io::Write,
    RA: crate::mir_codegen::regalloc::RegisterAllocator<PhysReg = &'static str>,
>(
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
) -> Result<(), std::io::Error> {
    match operand {
        crate::mir::Operand::Register(reg) => match reg {
            Register::Virtual(v) => load_register_to_rax(v, writer, reg_alloc, stack_slots),
            Register::Physical(p) => {
                writeln!(writer, "    movq %{}, %rax", p.name)?;
                Ok(())
            }
        },
        crate::mir::Operand::Immediate(imm) => match imm {
            crate::mir::instruction::Immediate::I8(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            crate::mir::instruction::Immediate::I16(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            crate::mir::instruction::Immediate::I32(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            crate::mir::instruction::Immediate::I64(v) => writeln!(writer, "    movq ${}, %rax", v),
            crate::mir::instruction::Immediate::F32(_)
            | crate::mir::instruction::Immediate::F64(_) => {
                writeln!(
                    writer,
                    "    # Floating point immediates not yet implemented"
                )?;
                Ok(())
            }
        },
    }
}

/// Load an operand into a specific register
pub fn load_operand_to_register<
    W: std::io::Write,
    RA: crate::mir_codegen::regalloc::RegisterAllocator<PhysReg = &'static str>,
>(
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), std::io::Error> {
    match operand {
        crate::mir::Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, target_reg)
            }
            Register::Physical(p) => {
                writeln!(writer, "    movq %{}, %{}", p.name, target_reg)?;
                Ok(())
            }
        },
        crate::mir::Operand::Immediate(imm) => match imm {
            crate::mir::instruction::Immediate::I8(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            crate::mir::instruction::Immediate::I16(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            crate::mir::instruction::Immediate::I32(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            crate::mir::instruction::Immediate::I64(v) => {
                writeln!(writer, "    movq ${}, %{}", v, target_reg)
            }
            crate::mir::instruction::Immediate::F32(_)
            | crate::mir::instruction::Immediate::F64(_) => {
                writeln!(
                    writer,
                    "    # Floating point immediates not yet implemented"
                )?;
                Ok(())
            }
        },
    }
}
