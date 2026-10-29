//! Moves between MIR operands, frame slots, the three scratch registers and the x87 stack.

use crate::{
    error::LaminaError,
    mir::{MirType, Operand, Register, ScalarType, VirtualReg, instruction::Immediate},
    mir_codegen::CodegenError,
};
use std::{collections::HashMap, io::Write};

pub(super) type Slots = HashMap<VirtualReg, i32>;

/// The emitter keeps every value in a frame slot and moves it through these three
/// caller-saved registers, so `%ebx`, `%esi` and `%edi` are never touched and the
/// prologue has nothing to save.
pub(super) const EAX: &str = "%eax";
pub(super) const ECX: &str = "%ecx";
pub(super) const EDX: &str = "%edx";

pub(super) fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

pub(super) fn byte_reg(reg: &str) -> Result<&'static str, LaminaError> {
    match reg {
        EAX => Ok("%al"),
        ECX => Ok("%cl"),
        EDX => Ok("%dl"),
        other => Err(unsupported(format!("i686 has no byte form of {other}"))),
    }
}

pub(super) fn word_reg(reg: &str) -> Result<&'static str, LaminaError> {
    match reg {
        EAX => Ok("%ax"),
        ECX => Ok("%cx"),
        EDX => Ok("%dx"),
        other => Err(unsupported(format!("i686 has no word form of {other}"))),
    }
}

pub(super) fn slot(slots: &Slots, vreg: &VirtualReg) -> Result<i32, LaminaError> {
    slots
        .get(vreg)
        .copied()
        .ok_or_else(|| unsupported(format!("i686 value {vreg:?} has no frame slot")))
}

/// The 32 bits an immediate contributes. A wider value is refused rather than
/// truncated, since a silently wrong constant is worse than a failed compile.
pub(super) fn immediate_word(immediate: &Immediate) -> Result<i32, LaminaError> {
    match immediate {
        Immediate::I8(value) => Ok(i32::from(*value)),
        Immediate::I16(value) => Ok(i32::from(*value)),
        Immediate::I32(value) => Ok(*value),
        Immediate::F32(value) => Ok(value.to_bits() as i32),
        Immediate::I64(value) => i32::try_from(*value)
            .or_else(|_| u32::try_from(*value).map(|bits| bits as i32))
            .map_err(|_| {
                unsupported(format!(
                    "i686 cannot hold the 64-bit constant {value} in one register: 64-bit integers are not legalized into register pairs yet"
                ))
            }),
        Immediate::F64(value) => Err(unsupported(format!(
            "i686 cannot hold the double constant {value} in one general register"
        ))),
    }
}

/// Puts an integer or pointer operand into `dst`.
pub(super) fn load_operand<W: Write>(
    operand: &Operand,
    writer: &mut W,
    slots: &Slots,
    dst: &str,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = slot(slots, vreg)?;
            writeln!(writer, "    movl {offset}(%ebp), {dst}")?;
        }
        Operand::Register(Register::Physical(reg)) => {
            return Err(unsupported(format!(
                "i686 does not pin values to the physical register {}",
                reg.name
            )));
        }
        Operand::Immediate(immediate) => {
            let value = immediate_word(immediate)?;
            if value == 0 {
                writeln!(writer, "    xorl {dst}, {dst}")?;
            } else {
                writeln!(writer, "    movl ${value}, {dst}")?;
            }
        }
    }
    Ok(())
}

pub(super) fn store_register<W: Write>(
    src: &str,
    dst: &Register,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(vreg) => {
            let offset = slot(slots, vreg)?;
            writeln!(writer, "    movl {src}, {offset}(%ebp)")?;
            Ok(())
        }
        Register::Physical(reg) => Err(unsupported(format!(
            "i686 does not pin results to the physical register {}",
            reg.name
        ))),
    }
}

pub(super) fn is_single(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::F32))
}

/// Pushes a float operand onto the x87 stack.
pub(super) fn push_float<W: Write>(
    operand: &Operand,
    writer: &mut W,
    slots: &Slots,
    single: bool,
) -> Result<(), LaminaError> {
    let load = if single { "flds" } else { "fldl" };
    match operand {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = slot(slots, vreg)?;
            writeln!(writer, "    {load} {offset}(%ebp)")?;
        }
        Operand::Register(Register::Physical(reg)) => {
            return Err(unsupported(format!(
                "i686 does not pin floats to the physical register {}",
                reg.name
            )));
        }
        Operand::Immediate(Immediate::F32(value)) => {
            writeln!(writer, "    pushl ${}", value.to_bits() as i32)?;
            writeln!(writer, "    flds (%esp)")?;
            writeln!(writer, "    addl $4, %esp")?;
        }
        Operand::Immediate(Immediate::F64(value)) => {
            let bits = value.to_bits();
            writeln!(writer, "    pushl ${}", (bits >> 32) as u32 as i32)?;
            writeln!(writer, "    pushl ${}", bits as u32 as i32)?;
            writeln!(writer, "    fldl (%esp)")?;
            writeln!(writer, "    addl $8, %esp")?;
        }
        Operand::Immediate(other) => {
            return Err(unsupported(format!(
                "i686 float operation given the integer immediate {other}"
            )));
        }
    }
    Ok(())
}

/// Pops the x87 top into the frame slot of `dst`.
pub(super) fn pop_float<W: Write>(
    dst: &Register,
    writer: &mut W,
    slots: &Slots,
    single: bool,
) -> Result<(), LaminaError> {
    let store = if single { "fstps" } else { "fstpl" };
    match dst {
        Register::Virtual(vreg) => {
            let offset = slot(slots, vreg)?;
            writeln!(writer, "    {store} {offset}(%ebp)")?;
            Ok(())
        }
        Register::Physical(reg) => Err(unsupported(format!(
            "i686 does not pin floats to the physical register {}",
            reg.name
        ))),
    }
}
