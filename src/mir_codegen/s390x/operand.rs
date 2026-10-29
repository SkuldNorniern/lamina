//! Moves between MIR operands, frame slots and the scratch registers the emitter uses.

use super::abi::{T0, fp_load, fp_store, load_imm};
use crate::{
    error::LaminaError,
    mir::{Operand, Register, VirtualReg, instruction::Immediate},
    mir_codegen::CodegenError,
};
use std::{collections::HashMap, io::Write};

pub(super) type Slots = HashMap<VirtualReg, i32>;

pub(super) fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

/// Physical register names come from MIR as ABI names, sometimes without the `%`.
pub(super) fn phys(name: &str) -> String {
    if name.starts_with('%') {
        name.to_string()
    } else {
        format!("%{name}")
    }
}

pub(super) fn slot(slots: &Slots, vreg: &VirtualReg) -> Result<i32, LaminaError> {
    slots
        .get(vreg)
        .copied()
        .ok_or_else(|| unsupported(format!("s390x value {vreg:?} has no frame slot")))
}

/// Puts an integer operand into `dst`. Immediates load whole, floats travel as bits.
pub(super) fn load_operand<W: Write>(
    operand: &Operand,
    writer: &mut W,
    slots: &Slots,
    dst: &str,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = slot(slots, vreg)?;
            fp_load(writer, "lg", dst, offset, dst)?;
        }
        Operand::Register(Register::Physical(physical)) => {
            writeln!(writer, "    lgr {dst}, {}", phys(physical.name))?;
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
    fp_store(writer, "stg", src, offset, T0)?;
    Ok(())
}

/// Puts a float operand into the floating-point register `dst`, at single or double
/// width. A single occupies the left half of its slot, which is where `ste` wrote it.
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
            let load = if single { "ley" } else { "ldy" };
            fp_load(writer, load, dst, offset, T0)?;
        }
        Operand::Register(Register::Physical(physical)) => {
            let name = phys(physical.name);
            if name.starts_with("%f") {
                writeln!(writer, "    ldr {dst}, {name}")?;
            } else if single {
                writeln!(writer, "    sllg {T0}, {name}, 32")?;
                writeln!(writer, "    ldgr {dst}, {T0}")?;
            } else {
                writeln!(writer, "    ldgr {dst}, {name}")?;
            }
        }
        Operand::Immediate(Immediate::F32(value)) => {
            load_imm(writer, T0, i64::from(value.to_bits()) << 32)?;
            writeln!(writer, "    ldgr {dst}, {T0}")?;
        }
        Operand::Immediate(Immediate::F64(value)) => {
            load_imm(writer, T0, value.to_bits() as i64)?;
            writeln!(writer, "    ldgr {dst}, {T0}")?;
        }
        Operand::Immediate(_) => {
            return Err(unsupported(
                "s390x float operation given a non-float immediate",
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
    let store = if single { "stey" } else { "stdy" };
    fp_store(writer, store, src, offset, T0)?;
    Ok(())
}
