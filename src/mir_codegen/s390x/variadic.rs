//! Variadic functions: the register save area a variadic callee spills on entry and
//! the `va_list` operations that walk it.
//!
//! The `va_list` is the twenty-four bytes the IR reserves: a byte cursor into the
//! general-register half of the save area, a byte cursor into its floating-point
//! half, the overflow pointer and the save area pointer. Both cursors index the one
//! save area, so the floating-point cursor starts where the general half ends.
//!
//! An unnamed argument always occupies a whole doubleword, in a register or in the
//! caller's outgoing area, so every `va_arg` advances by eight bytes whatever its
//! type, and a narrow integer is read from the doubleword the caller extended it in.

use super::abi::{
    A0, A1, ARG_FPRS, ARG_GPRS, ArgPlace, FA0, FP, T0, T1, WORD, address, entry_base, fp_store,
    incoming_stack_offset, memory,
};
use super::operand::{Slots, load_operand, store_float, unsupported};
use super::{EmitCtx, extend_integer, store_result};
use crate::{
    error::LaminaError,
    mir::{Instruction as MirInst, MirType, Register, ScalarType},
};
use std::io::Write;

/// Doublewords the save area holds: `%r2`-`%r6`, then `%f0`, `%f2`, `%f4` and `%f6`.
const GPR_SAVE_BYTES: i32 = ARG_GPRS.len() as i32 * WORD;
const FPR_SAVE_BYTES: i32 = ARG_FPRS.len() as i32 * WORD;
pub(super) const SAVE_BYTES: i32 = GPR_SAVE_BYTES + FPR_SAVE_BYTES;

/// Where a variadic callee's `va_list` starts reading.
#[derive(Clone, Copy, Debug)]
pub(super) struct VaArea {
    /// Frame offset of the save area's first byte.
    pub(super) save: i32,
    /// Offset from the entry stack pointer of the first unnamed stack argument.
    pub(super) overflow: i32,
    pub(super) gpr: i32,
    pub(super) fpr: i32,
}

impl VaArea {
    pub(super) fn new(save: i32, places: &[ArgPlace], first_gpr: usize) -> Self {
        let count = |want: fn(&ArgPlace) -> bool| places.iter().filter(|p| want(p)).count() as i32;
        let gprs = first_gpr as i32 + count(|p| matches!(p, ArgPlace::Gpr(_)));
        let fprs = count(|p| matches!(p, ArgPlace::Fpr(_)));
        let stack = count(|p| matches!(p, ArgPlace::Stack(_))) as usize;
        Self {
            save,
            overflow: incoming_stack_offset(stack),
            gpr: gprs * WORD,
            fpr: GPR_SAVE_BYTES + fprs * WORD,
        }
    }
}

/// Spills every argument register into the save area on entry.
pub(super) fn spill_save_area<W: Write>(writer: &mut W, area: &VaArea) -> Result<(), LaminaError> {
    for (index, reg) in ARG_GPRS.iter().enumerate() {
        fp_store(writer, "stg", reg, area.save + index as i32 * WORD, T0)?;
    }
    for (index, reg) in ARG_FPRS.iter().enumerate() {
        fp_store(
            writer,
            "stdy",
            reg,
            area.save + GPR_SAVE_BYTES + index as i32 * WORD,
            T0,
        )?;
    }
    Ok(())
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    let Some(area) = ctx.va else {
        return Err(unsupported(
            "s390x variadic arguments are only available inside a variadic function",
        ));
    };
    match inst {
        MirInst::VaStart { ap } => {
            load_operand(ap, writer, slots, T1)?;
            writeln!(writer, "    lghi {A0}, {}", area.gpr)?;
            writeln!(writer, "    st {A0}, 0({T1})")?;
            writeln!(writer, "    lghi {A0}, {}", area.fpr)?;
            writeln!(writer, "    st {A0}, 4({T1})")?;
            address(
                writer,
                A0,
                entry_base(ctx.aligned),
                i64::from(area.overflow),
            )?;
            writeln!(writer, "    stg {A0}, 8({T1})")?;
            address(writer, A0, FP, i64::from(area.save))?;
            writeln!(writer, "    stg {A0}, 16({T1})")?;
            Ok(())
        }
        MirInst::VaArg { ty, dst, ap } => {
            let float = match ty {
                MirType::Scalar(ScalarType::F64) => true,
                MirType::Scalar(
                    ScalarType::I1
                    | ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::Ptr
                    | ScalarType::PtrIn(_),
                ) => false,
                MirType::Scalar(ScalarType::F32) | MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "s390x va_arg of {ty} is not supported; use an integer, pointer or f64"
                    )));
                }
            };
            let (cursor, limit) = if float {
                (4, SAVE_BYTES)
            } else {
                (0, GPR_SAVE_BYTES)
            };
            let stack = ctx.fresh_label("va_stack");
            let done = ctx.fresh_label("va_done");
            load_operand(ap, writer, slots, T1)?;
            writeln!(writer, "    llgf {A1}, {cursor}({T1})")?;
            writeln!(writer, "    clgfi {A1}, {limit}")?;
            writeln!(writer, "    brcl 10, {stack}")?;
            writeln!(writer, "    lg {A0}, 16({T1})")?;
            writeln!(writer, "    agr {A0}, {A1}")?;
            writeln!(writer, "    aghi {A1}, {WORD}")?;
            writeln!(writer, "    st {A1}, {cursor}({T1})")?;
            writeln!(writer, "    jg {done}")?;
            writeln!(writer, "{stack}:")?;
            writeln!(writer, "    lg {A0}, 8({T1})")?;
            writeln!(writer, "    lay {T0}, {WORD}({A0})")?;
            writeln!(writer, "    stg {T0}, 8({T1})")?;
            writeln!(writer, "{done}:")?;
            match (float, dst) {
                (true, Register::Virtual(vreg)) => {
                    writeln!(writer, "    ld {FA0}, 0({A0})")?;
                    store_float(FA0, vreg, writer, slots, false)
                }
                (true, dst) => {
                    memory(writer, "lg", A0, A0, 0, T0)?;
                    store_result(writer, A0, dst, slots)
                }
                (false, dst) => {
                    memory(writer, "lg", A0, A0, 0, T0)?;
                    extend_integer(writer, A0, ty, true)?;
                    store_result(writer, A0, dst, slots)
                }
            }
        }
        MirInst::VaEnd { ap: _ } => Ok(()),
        MirInst::VaCopy { dst, src } => {
            load_operand(src, writer, slots, T1)?;
            load_operand(dst, writer, slots, A1)?;
            for offset in [0, 8, 16] {
                writeln!(writer, "    lg {A0}, {offset}({T1})")?;
                writeln!(writer, "    stg {A0}, {offset}({A1})")?;
            }
            Ok(())
        }
        other => Err(unsupported(format!(
            "s390x received {other} as a variadic instruction"
        ))),
    }
}
