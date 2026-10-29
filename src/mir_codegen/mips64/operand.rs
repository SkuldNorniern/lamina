//! Moves between MIR operands, frame slots and the scratch registers the emitter uses.

use super::abi::{T0, fp_load, fp_store, load_imm};
use crate::{
    error::LaminaError,
    mir::{Operand, Register, VirtualReg, instruction::Immediate},
    mir_codegen::CodegenError,
};
use std::{collections::HashMap, io, io::Write};

pub(super) type Slots = HashMap<VirtualReg, i32>;

pub(super) fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

/// Physical register names come from MIR as ABI names, sometimes without the `$`.
pub(super) fn phys(name: &str) -> String {
    if name.starts_with('$') {
        name.to_string()
    } else {
        format!("${name}")
    }
}

pub(super) fn slot(slots: &Slots, vreg: &VirtualReg) -> Result<i32, LaminaError> {
    slots
        .get(vreg)
        .copied()
        .ok_or_else(|| unsupported(format!("mips64 value {vreg:?} has no frame slot")))
}

/// Puts an integer operand into `dst`. Every slot holds a whole doubleword, so a
/// narrow value is read at the width it was written, whatever the byte order.
pub(super) fn load_operand<W: Write>(
    operand: &Operand,
    writer: &mut W,
    slots: &Slots,
    dst: &str,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = slot(slots, vreg)?;
            fp_load(writer, "ld", dst, offset, dst)?;
        }
        Operand::Register(Register::Physical(physical)) => {
            writeln!(writer, "    move {dst}, {}", phys(physical.name))?;
        }
        Operand::Immediate(immediate) => {
            let bits = match immediate {
                Immediate::I8(value) => i64::from(*value),
                Immediate::I16(value) => i64::from(*value),
                Immediate::I32(value) => i64::from(*value),
                Immediate::I64(value) => *value,
                Immediate::F32(value) => i64::from(value.to_bits() as i32),
                Immediate::F64(value) => value.to_bits() as i64,
            };
            load_imm(writer, dst, bits)?;
        }
    }
    Ok(())
}

/// Stores the full 64-bit `src` register into the frame slot of `dst`.
pub(super) fn store_register<W: Write>(
    src: &str,
    dst: &VirtualReg,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    let offset = slot(slots, dst)?;
    fp_store(writer, "sd", src, offset, T0)?;
    Ok(())
}

/// Puts a float operand into the floating-point register `dst`, at single or double
/// width. A single lives in the low half of the register whichever way the target
/// orders bytes, so the integer forms below never shift.
pub(super) fn load_float_operand<W: Write>(
    operand: &Operand,
    writer: &mut W,
    slots: &Slots,
    dst: &str,
    single: bool,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = slot(slots, vreg)?;
            let load = if single { "lwc1" } else { "ldc1" };
            fp_load(writer, load, dst, offset, T0)?;
        }
        Operand::Register(Register::Physical(physical)) => {
            let name = phys(physical.name);
            if name.starts_with("$f") {
                let move_fp = if single { "mov.s" } else { "mov.d" };
                writeln!(writer, "    {move_fp} {dst}, {name}")?;
            } else {
                int_to_fpr(writer, dst, &name, single)?;
            }
        }
        Operand::Immediate(Immediate::F32(value)) => {
            load_imm(writer, T0, i64::from(value.to_bits() as i32))?;
            int_to_fpr(writer, dst, T0, true)?;
        }
        Operand::Immediate(Immediate::F64(value)) => {
            load_imm(writer, T0, value.to_bits() as i64)?;
            int_to_fpr(writer, dst, T0, false)?;
        }
        Operand::Immediate(_) => {
            return Err(unsupported(
                "mips64 float operation given a non-float immediate",
            ));
        }
    }
    Ok(())
}

/// Stores floating-point register `src` into the frame slot of `dst`.
pub(super) fn store_float<W: Write>(
    src: &str,
    dst: &VirtualReg,
    writer: &mut W,
    slots: &Slots,
    single: bool,
) -> Result<(), LaminaError> {
    let offset = slot(slots, dst)?;
    let store = if single { "swc1" } else { "sdc1" };
    fp_store(writer, store, src, offset, T0)?;
    Ok(())
}

/// Moves the bits of an integer register into a floating-point one. The transfer is
/// between registers, so neither half of the pair depends on the byte order.
pub(super) fn int_to_fpr<W: Write>(
    writer: &mut W,
    dst: &str,
    src: &str,
    single: bool,
) -> io::Result<()> {
    if single {
        writeln!(writer, "    mtc1 {src}, {dst}")
    } else {
        writeln!(writer, "    dmtc1 {src}, {dst}")
    }
}

/// The reverse. A single comes back zero-extended, the way a 32-bit integer is held.
pub(super) fn fpr_to_int<W: Write>(
    writer: &mut W,
    dst: &str,
    src: &str,
    single: bool,
) -> io::Result<()> {
    if single {
        writeln!(writer, "    mfc1 {dst}, {src}")?;
        writeln!(writer, "    dsll {dst}, {dst}, 32")?;
        writeln!(writer, "    dsrl {dst}, {dst}, 32")
    } else {
        writeln!(writer, "    dmfc1 {dst}, {src}")
    }
}
