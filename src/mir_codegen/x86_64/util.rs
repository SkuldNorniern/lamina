//! Utility functions for x86_64 code generation.

use std::collections::HashMap;
use std::io::{Error, Write};

use crate::mir::Operand;
use crate::mir::instruction::Immediate;
use crate::mir::register::{Register, VirtualReg};
use crate::mir::types::{MirType, ScalarType};

/// Whether a MirType is a 32-bit float (`f32`).
pub fn is_f32(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::F32))
}

/// Whether a MirType is a floating-point type (`f32` or `f64`).
pub fn is_float(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64))
}

/// Load a float operand (stored as integer bits in a GPR stack slot) into an XMM register.
///
/// Strategy: integers hold float bit patterns via `movd` (f32) or `movq` (f64) into xmm.
pub fn load_float_operand_to_xmm<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    xmm: &str,
    ty: &MirType,
) -> Result<(), Error> {
    let mov_to_xmm = if is_f32(ty) { "movd" } else { "movq" };
    match operand {
        Operand::Register(_) => {
            load_operand_to_rax(operand, writer, reg_alloc, stack_slots)?;
            writeln!(writer, "    {mov_to_xmm} %rax, %{xmm}")
        }
        Operand::Immediate(imm) => match imm {
            Immediate::F32(v) => {
                let bits = v.to_bits() as i64;
                writeln!(writer, "    movq ${bits}, %rax")?;
                writeln!(writer, "    {mov_to_xmm} %rax, %{xmm}")
            }
            Immediate::F64(v) => {
                let bits = v.to_bits() as i64;
                writeln!(writer, "    movq ${bits}, %rax")?;
                writeln!(writer, "    {mov_to_xmm} %rax, %{xmm}")
            }
            _ => {
                // Integer bits treated as float bits
                load_operand_to_rax(operand, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    {mov_to_xmm} %rax, %{xmm}")
            }
        },
    }
}

/// Store the float result from an XMM register back into a virtual register (as integer bits).
pub fn store_xmm_to_register<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    xmm: &str,
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    ty: &MirType,
) -> Result<(), Error> {
    let mov_from_xmm = if is_f32(ty) { "movd" } else { "movq" };
    writeln!(writer, "    {mov_from_xmm} %{xmm}, %rax")?;
    store_rax_to_register(reg, writer, reg_alloc, stack_slots)
}

/// Load a virtual register into RAX
pub fn load_register_to_rax<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    movq %{phys}, %rax")?;
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    movq {offset}(%rbp), %rax")?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {reg:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Store RAX to a virtual register
pub fn store_rax_to_register<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    movq %rax, %{phys}")?;
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    movq %rax, {offset}(%rbp)")?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {reg:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Load a register to another register
pub fn load_register_to_register<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    movq %{phys}, %{target_reg}")?;
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    movq {offset}(%rbp), %{target_reg}")?;
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Virtual register {src:?} has no mapping or stack slot"),
        ));
    }
    Ok(())
}

/// Load an operand into RAX
pub fn load_operand_to_rax<
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    match operand {
        Operand::Register(reg) => match reg {
            Register::Virtual(v) => load_register_to_rax(v, writer, reg_alloc, stack_slots),
            Register::Physical(p) => {
                writeln!(writer, "    movq %{}, %rax", p.name)?;
                Ok(())
            }
        },
        Operand::Immediate(imm) => match imm {
            Immediate::I8(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            Immediate::I16(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            Immediate::I32(v) => {
                writeln!(writer, "    movq ${}, %rax", *v as i64)
            }
            Immediate::I64(v) => writeln!(writer, "    movq ${v}, %rax"),
            Immediate::F32(_)
            | Immediate::F64(_) => {
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
    W: Write,
    RA: lamina_codegen::LocalRegisterAllocator<PhysReg = &'static str>,
>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), Error> {
    match operand {
        Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, target_reg)
            }
            Register::Physical(p) => {
                writeln!(writer, "    movq %{}, %{}", p.name, target_reg)?;
                Ok(())
            }
        },
        Operand::Immediate(imm) => match imm {
            Immediate::I8(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            Immediate::I16(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            Immediate::I32(v) => {
                writeln!(writer, "    movq ${}, %{}", *v as i64, target_reg)
            }
            Immediate::I64(v) => {
                writeln!(writer, "    movq ${v}, %{target_reg}")
            }
            Immediate::F32(_) | Immediate::F64(_) => {
                writeln!(
                    writer,
                    "    # Floating point immediates not yet implemented"
                )?;
                Ok(())
            }
        },
    }
}
