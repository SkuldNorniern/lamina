//! Utility functions for PowerPC64 code generation.

use std::collections::HashMap;
use std::io::{Error, ErrorKind, Write};

use crate::mir::Operand;
use crate::mir::instruction::Immediate;
use crate::mir::register::{Register, VirtualReg};
use lamina_codegen::LocalRegisterAllocator as RegisterAllocator;

/// Load a virtual register into GPR `r3` (the primary scratch/return reg).
pub fn load_register_to_r3<W: Write, RA: RegisterAllocator<PhysReg = &'static str>>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    mr 3, {}", phys)
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    ld 3, {}(1)", offset)
    } else {
        Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", reg),
        ))
    }
}

/// Store GPR `r3` into a virtual register's location.
pub fn store_r3_to_register<W: Write, RA: RegisterAllocator<PhysReg = &'static str>>(
    reg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(reg) {
        writeln!(writer, "    mr {}, 3", phys)
    } else if let Some(offset) = stack_slots.get(reg) {
        writeln!(writer, "    std 3, {}(1)", offset)
    } else {
        Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", reg),
        ))
    }
}

/// Load a register into `dest_reg`.
pub fn load_register_to_register<W: Write, RA: RegisterAllocator<PhysReg = &'static str>>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    mr {}, {}", dest_reg, phys)
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    ld {}, {}(1)", dest_reg, offset)
    } else {
        Err(Error::new(
            ErrorKind::InvalidData,
            format!("Virtual register {:?} has no mapping or stack slot", src),
        ))
    }
}

/// Load a MIR operand into GPR `dest_reg`.
pub fn load_operand_to_register<W: Write, RA: RegisterAllocator<PhysReg = &'static str>>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), Error> {
    match operand {
        Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, dest_reg)
            }
            Register::Physical(p) => writeln!(writer, "    mr {}, {}", dest_reg, p.name),
        },
        Operand::Immediate(imm) => {
            let val: i64 = match imm {
                Immediate::I8(v) => *v as i64,
                Immediate::I16(v) => *v as i64,
                Immediate::I32(v) => *v as i64,
                Immediate::I64(v) => *v,
                Immediate::F32(v) => v.to_bits() as i64,
                Immediate::F64(v) => v.to_bits() as i64,
            };
            // PowerPC uses `li` for 16-bit signed immediates, `lis`+`ori` for 32-bit,
            // and a full 4-instruction sequence for 64-bit.  We use the assembler
            // pseudo-instruction `li` which GAS handles as needed.
            if (-32768..=32767).contains(&val) {
                writeln!(writer, "    li {}, {}", dest_reg, val)
            } else if (-2147483648..=2147483647).contains(&val) {
                writeln!(writer, "    lis {}, {}@ha", dest_reg, val >> 16)?;
                writeln!(writer, "    addi {}, {}, {}@l", dest_reg, dest_reg, val)
            } else {
                // 64-bit: lis + ori + rldicr + oris + ori
                writeln!(writer, "    lis {}, {}@highest", dest_reg, val >> 48)?;
                writeln!(
                    writer,
                    "    ori {}, {}, {}@higher",
                    dest_reg,
                    dest_reg,
                    (val >> 32) & 0xFFFF
                )?;
                writeln!(writer, "    rldicr {0}, {0}, 32, 31", dest_reg)?;
                writeln!(
                    writer,
                    "    oris {0}, {0}, {1}",
                    dest_reg,
                    (val >> 16) & 0xFFFF
                )?;
                writeln!(writer, "    ori {0}, {0}, {1}", dest_reg, val & 0xFFFF)
            }
        }
    }
}
