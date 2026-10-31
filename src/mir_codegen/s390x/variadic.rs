//! Variadic functions: the register save area a variadic callee spills on entry and
//! the `va_list` operations that walk it.
//!
//! The shapes are the ones the zSeries ELF ABI supplement states, so a `va_list` built
//! here is the one a C function compiled for this target expects:
//!
//! ```c
//! typedef struct {
//!     long __gpr;
//!     long __fpr;
//!     void *__overflow_arg_area;
//!     void *__reg_save_area;
//! } va_list[1];
//! ```
//!
//! `__gpr` and `__fpr` count the registers already consumed, not bytes. The save area
//! is indexed by register number times eight, which puts `%r2`-`%r6` at offsets 16
//! through 56 and `%f0`, `%f2`, `%f4`, `%f6` at 128 through 160.
//!
//! An unnamed argument always occupies a whole doubleword, in a register or in the
//! caller's outgoing area, so every `va_arg` advances by one slot whatever its type,
//! and a narrow integer is read from the doubleword the caller extended it in.

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
use lamina_platform::{TargetArchitecture, VaListAbi};
use std::io::Write;

/// Byte offsets of the four `va_list` fields.
const GPR_FIELD: i32 = 0;
const FPR_FIELD: i32 = 8;
const OVERFLOW_FIELD: i32 = 16;
const SAVE_FIELD: i32 = 24;

/// The size the target table states for this ABI's `va_list`. The four fields fill it
/// exactly, so widening one without the other is a build error rather than an overrun.
const VA_LIST_BYTES: i32 = match TargetArchitecture::S390x.va_list() {
    VaListAbi::Defined { size, .. } => size as i32,
    VaListAbi::Undefined => panic!("s390x lowers va_list but the target table defines none"),
};
const _: () = assert!(SAVE_FIELD + WORD == VA_LIST_BYTES);

/// Where register `n` lands in a save area indexed by register number.
const GPR_SAVE_AT: i32 = 2 * WORD;
const FPR_SAVE_AT: i32 = 16 * WORD;

/// A save area addressed by register number spans twenty doublewords.
pub(super) const SAVE_BYTES: i32 = 20 * WORD;

/// Where a variadic callee's `va_list` starts reading.
#[derive(Clone, Copy, Debug)]
pub(super) struct VaArea {
    /// Frame offset of the save area's first byte.
    pub(super) save: i32,
    /// Offset from the entry stack pointer of the first unnamed stack argument.
    pub(super) overflow: i32,
    /// Argument registers the named parameters consumed.
    pub(super) gpr: i32,
    pub(super) fpr: i32,
}

impl VaArea {
    pub(super) fn new(save: i32, places: &[ArgPlace], first_gpr: usize) -> Self {
        let count = |want: fn(&ArgPlace) -> bool| places.iter().filter(|p| want(p)).count() as i32;
        let stack = count(|p| matches!(p, ArgPlace::Stack(_))) as usize;
        Self {
            save,
            overflow: incoming_stack_offset(stack),
            gpr: first_gpr as i32 + count(|p| matches!(p, ArgPlace::Gpr(_))),
            fpr: count(|p| matches!(p, ArgPlace::Fpr(_))),
        }
    }
}

/// Spills every argument register into the save area on entry.
pub(super) fn spill_save_area<W: Write>(writer: &mut W, area: &VaArea) -> Result<(), LaminaError> {
    for (index, reg) in ARG_GPRS.iter().enumerate() {
        fp_store(
            writer,
            "stg",
            reg,
            area.save + GPR_SAVE_AT + index as i32 * WORD,
            T0,
        )?;
    }
    for (index, reg) in ARG_FPRS.iter().enumerate() {
        fp_store(
            writer,
            "stdy",
            reg,
            area.save + FPR_SAVE_AT + index as i32 * WORD,
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
            writeln!(writer, "    stg {A0}, {GPR_FIELD}({T1})")?;
            writeln!(writer, "    lghi {A0}, {}", area.fpr)?;
            writeln!(writer, "    stg {A0}, {FPR_FIELD}({T1})")?;
            address(
                writer,
                A0,
                entry_base(ctx.aligned),
                i64::from(area.overflow),
            )?;
            writeln!(writer, "    stg {A0}, {OVERFLOW_FIELD}({T1})")?;
            address(writer, A0, FP, i64::from(area.save))?;
            writeln!(writer, "    stg {A0}, {SAVE_FIELD}({T1})")?;
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
            let (field, available, base) = if float {
                (FPR_FIELD, ARG_FPRS.len(), FPR_SAVE_AT)
            } else {
                (GPR_FIELD, ARG_GPRS.len(), GPR_SAVE_AT)
            };
            let stack = ctx.fresh_label("va_stack");
            let done = ctx.fresh_label("va_done");
            load_operand(ap, writer, slots, T1)?;
            writeln!(writer, "    lg {A1}, {field}({T1})")?;
            writeln!(writer, "    clgfi {A1}, {available}")?;
            writeln!(writer, "    brcl 10, {stack}")?;
            writeln!(writer, "    lg {A0}, {SAVE_FIELD}({T1})")?;
            writeln!(writer, "    sllg {T0}, {A1}, 3")?;
            writeln!(writer, "    lay {A0}, {base}({T0},{A0})")?;
            writeln!(writer, "    aghi {A1}, 1")?;
            writeln!(writer, "    stg {A1}, {field}({T1})")?;
            writeln!(writer, "    jg {done}")?;
            writeln!(writer, "{stack}:")?;
            writeln!(writer, "    lg {A0}, {OVERFLOW_FIELD}({T1})")?;
            writeln!(writer, "    lay {T0}, {WORD}({A0})")?;
            writeln!(writer, "    stg {T0}, {OVERFLOW_FIELD}({T1})")?;
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
            for offset in (0..VA_LIST_BYTES).step_by(WORD as usize) {
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
