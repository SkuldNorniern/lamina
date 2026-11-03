//! Variadic functions under the Intel386 psABI.
//!
//! cdecl passes every argument on the stack, so the unnamed ones are already one
//! packed run above the named ones and no register save area is needed. A `va_list`
//! is the `char *` the psABI names, pointing at the next unnamed argument, and
//! `va_arg` reads from it and steps it on by the argument's slot size. Nothing in the
//! area is aligned beyond a word: a `long long` is two consecutive words, low word
//! first, which is also why `alignof(long long)` is four here.

use super::abi::{self, INCOMING_BASE, WORD};
use super::extend_in_place;
use super::operand::{EAX, ECX, EDX, Slots, load_operand, slot, unsupported};
use crate::error::LaminaError;
use crate::mir::{Function, Instruction as MirInst, MirType, Register, ScalarType};
use lamina_platform::{TargetArchitecture, VaListAbi};
use std::io::Write;

/// Offset from `%ebp` at which a variadic callee finds its first unnamed argument:
/// past the named parameters, which `classify` has already placed.
pub(super) fn area_base(func: &Function, sret: bool) -> i32 {
    let types: Vec<MirType> = func.sig.params.iter().map(|p| p.ty).collect();
    let aggregates: Vec<Option<usize>> = func
        .sig
        .params
        .iter()
        .map(|p| p.aggregate.map(|a| a.size))
        .collect();
    let named = abi::classify(&types, &aggregates, sret);
    let used = named
        .last()
        .map(|place| place.offset + place.bytes)
        .unwrap_or(if sret { WORD } else { 0 });
    INCOMING_BASE + used as i32
}

/// Bytes `va_arg` of `ty` takes out of the area, or an error naming a type the area
/// holds in a shape this backend does not read.
fn slot_bytes(ty: &MirType, wide: bool) -> Result<usize, LaminaError> {
    if wide {
        return Ok(2 * WORD);
    }
    match ty {
        MirType::Scalar(ScalarType::F64) => Ok(2 * WORD),
        MirType::Scalar(
            ScalarType::I1 | ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::Ptr,
        ) => Ok(WORD),
        MirType::Scalar(ScalarType::I64) => Err(unsupported(
            "i686 va_arg of i64 reached the backend as one register: the legalizer did not split it into a pair",
        )),
        other => Err(unsupported(format!(
            "i686 va_arg of {other} is not supported; use an integer, a pointer or a double"
        ))),
    }
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    slots: &Slots,
    func: &Function,
    base: Option<i32>,
) -> Result<(), LaminaError> {
    let Some(base) = base else {
        return Err(unsupported(
            "i686 va_list access outside a variadic function",
        ));
    };
    match inst {
        MirInst::VaStart { ap } => {
            writeln!(writer, "    leal {base}(%ebp), {EAX}")?;
            load_operand(ap, writer, slots, ECX)?;
            writeln!(writer, "    movl {EAX}, ({ECX})")?;
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            load_operand(src, writer, slots, ECX)?;
            writeln!(writer, "    movl ({ECX}), {EAX}")?;
            load_operand(dst, writer, slots, ECX)?;
            writeln!(writer, "    movl {EAX}, ({ECX})")?;
        }
        MirInst::VaArg { ty, dst, ap } => {
            let Register::Virtual(target) = dst else {
                return Err(unsupported("i686 va_arg defines a frame slot"));
            };
            let to = slot(slots, target)?;
            let wide = func.variadic_wide_result(dst).is_some();
            let bytes = slot_bytes(ty, wide)?;
            load_operand(ap, writer, slots, ECX)?;
            writeln!(writer, "    movl ({ECX}), {EDX}")?;
            if bytes == 2 * WORD {
                writeln!(writer, "    movl ({EDX}), {EAX}")?;
                writeln!(writer, "    movl {EAX}, {to}(%ebp)")?;
                writeln!(writer, "    movl {}({EDX}), {EAX}", WORD)?;
                writeln!(writer, "    movl {EAX}, {}(%ebp)", to + WORD as i32)?;
            } else {
                writeln!(writer, "    movl ({EDX}), {EAX}")?;
                extend_in_place(writer, EAX, ty, true)?;
                writeln!(writer, "    movl {EAX}, {to}(%ebp)")?;
            }
            writeln!(writer, "    addl ${bytes}, {EDX}")?;
            writeln!(writer, "    movl {EDX}, ({ECX})")?;
        }
        other => {
            return Err(unsupported(format!(
                "i686 variadic emitter received {other}"
            )));
        }
    }
    Ok(())
}

/// The `va_list` this backend writes is one pointer, which is what the target table
/// states for i686.
const _: () = {
    let size = match TargetArchitecture::I686.va_list() {
        VaListAbi::Defined { size, .. } => size,
        VaListAbi::Undefined => {
            panic!("i686 lowers va_list but the target table defines none")
        }
    };
    assert!(size == 4);
};
