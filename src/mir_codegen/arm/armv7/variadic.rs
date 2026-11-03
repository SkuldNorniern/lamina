//! Variadic functions under AAPCS (IHI 0042), base standard.
//!
//! Past the named parameters the base standard applies, so an unnamed argument
//! travels in r0-r3 and then on the stack, a `double` included. A variadic callee
//! pushes r0-r3 before anything else, which puts them directly below the caller's
//! stack arguments: the four saved words and the stack area then form one run, and a
//! `va_list` is the `void *` the ABI names, walking that run.
//!
//! The caller's stack pointer is eight-byte aligned at the call, so the save area
//! starts eight-byte aligned too. An argument of eight-byte alignment is placed on an
//! even core register or an eight-byte-aligned stack slot, which in that one run is
//! the same thing: an offset that is a multiple of eight. `va_arg` of a pair
//! therefore rounds the cursor up to eight before it reads, and the rounding covers
//! the register half and the stack half alike.

use lamina_platform::{TargetArchitecture, VaListAbi};
use std::io::Write;

use lamina_codegen::aarch64::FrameMap;

use super::abi::{ArgKind, ArgPlace, classify, returns_in_memory};
use super::operand::{
    ADDR, F64_SCRATCH, SAVED_BYTES, Scratch, frame_store, frame_vstore, materialize, store_result,
    unsupported,
};
use crate::error::LaminaError;
use crate::mir::{Function, Instruction as MirInst, MirType, ScalarType};

/// Core registers a variadic callee saves on entry, and the bytes they take.
pub(crate) const SAVE_AREA_BYTES: i32 = 16;

/// Bytes the prologue pushes before it takes the frame pointer, which is also the
/// distance from the entry stack pointer to the caller's first stack argument.
pub(crate) fn saved_bytes(func: &Function) -> i32 {
    SAVED_BYTES
        + if func.sig.variadic {
            SAVE_AREA_BYTES
        } else {
            0
        }
}

/// Writes r0-r3 below the caller's stack arguments, before the prologue touches them.
pub(crate) fn save_core_registers<W: Write>(w: &mut W) -> Result<(), LaminaError> {
    writeln!(w, "    push {{r0, r1, r2, r3}}")?;
    writeln!(w, "    .cfi_def_cfa_offset {SAVE_AREA_BYTES}")?;
    Ok(())
}

/// AAPCS marshals a variadic function entirely by the base standard, so its named
/// floating-point parameters and its floating-point result travel in core registers
/// rather than the co-processor bank. This backend places both in the bank, which
/// agrees with itself but not with C, so a variadic function that has either is
/// refused rather than linked against a C caller that disagrees.
pub(crate) fn reject_co_processor_boundary(name: &str, func: &Function) -> Result<(), LaminaError> {
    if !func.sig.variadic {
        return Ok(());
    }
    let what = if func.sig.ret_ty.is_some_and(|ty| ty.is_float()) {
        "the result"
    } else if func
        .sig
        .params
        .iter()
        .any(|p| p.aggregate.is_none() && p.ty.is_float())
    {
        "a named parameter"
    } else {
        return Ok(());
    };
    Err(unsupported(format!(
        "armv7 variadic function @{name} takes {what} in floating-point: AAPCS marshals a variadic function by the base standard, which this backend does not place"
    )))
}

/// Bytes of the run the named parameters take, which is where the unnamed ones start.
///
/// Once a named parameter is on the stack the core registers are spent, so the run
/// carries on past the save area. The one placement that breaks that reading is a
/// named float that missed the co-processor bank: AAPCS sends it to the stack while
/// leaving the core registers free, so the unnamed ones start back in the save area
/// and the two halves are no longer one run. That is refused by name.
fn named_bytes(places: &[ArgPlace]) -> Result<i32, LaminaError> {
    let core = places
        .iter()
        .filter_map(|place| place.core.map(|span| (span.first + span.words) * 4))
        .max()
        .unwrap_or(0) as i32;
    let stack = places
        .iter()
        .filter_map(|place| place.stack.map(|span| span.offset + span.bytes.max(4)))
        .max();
    match stack {
        Some(_) if core < SAVE_AREA_BYTES => Err(unsupported(
            "armv7 va_arg in a function whose named parameters reach the stack with core registers to spare: a floating-point parameter that missed the co-processor bank splits the variadic area in two".to_string(),
        )),
        Some(end) => Ok(SAVE_AREA_BYTES + end.div_ceil(4) as i32 * 4),
        None => Ok(core),
    }
}

/// Offset from the entry stack pointer of the first unnamed argument.
pub(crate) fn area_base(func: &Function) -> Result<i32, LaminaError> {
    let mut kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind::of(p.ty, p.aggregate))
        .collect();
    if returns_in_memory(func.sig.ret_aggregate) {
        kinds.insert(0, ArgKind::pointer());
    }
    Ok(SAVED_BYTES + named_bytes(&classify(&kinds, None, false))?)
}

/// Bytes `va_arg` of `ty` takes out of the run, and the alignment the run places it
/// on. A type the run holds in a shape this backend does not read is refused by name.
fn slot(ty: &MirType, wide: bool) -> Result<(usize, i32), LaminaError> {
    if wide {
        return Ok((8, 8));
    }
    match ty {
        MirType::Scalar(ScalarType::F64) => Ok((8, 8)),
        MirType::Scalar(
            ScalarType::I1 | ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::Ptr,
        ) => Ok((4, 4)),
        MirType::Scalar(ScalarType::I64) => Err(unsupported(
            "armv7 va_arg of i64 reached the backend as one register: the legalizer did not split it into a pair".to_string(),
        )),
        other => Err(unsupported(format!(
            "armv7 va_arg of {other} is not supported; use an integer, a pointer or a double"
        ))),
    }
}

pub(crate) fn emit<W: Write>(
    w: &mut W,
    inst: &MirInst,
    func: &Function,
    frame: &FrameMap,
    base: Option<i32>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let Some(base) = base else {
        return Err(unsupported(
            "armv7 va_list access outside a variadic function".to_string(),
        ));
    };
    match inst {
        MirInst::VaStart { ap } => {
            let list = scratch.alloc()?;
            let cursor = scratch.alloc()?;
            materialize(w, ap, list, frame)?;
            entry_sp_address(w, cursor, base)?;
            writeln!(w, "    str {cursor}, [{list}]")?;
            scratch.free(cursor);
            scratch.free(list);
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            let from = scratch.alloc()?;
            let to = scratch.alloc()?;
            materialize(w, src, from, frame)?;
            materialize(w, dst, to, frame)?;
            writeln!(w, "    ldr {from}, [{from}]")?;
            writeln!(w, "    str {from}, [{to}]")?;
            scratch.free(to);
            scratch.free(from);
        }
        MirInst::VaArg { ty, dst, ap } => {
            let wide = func.variadic_wide_result(dst).is_some();
            let (bytes, align) = slot(ty, wide)?;
            let offset = frame
                .slot_of(dst)
                .ok_or_else(|| unsupported("armv7 va_arg has no frame slot".to_string()))?;
            let list = scratch.alloc()?;
            let cursor = scratch.alloc()?;
            materialize(w, ap, list, frame)?;
            writeln!(w, "    ldr {cursor}, [{list}]")?;
            if align == 8 {
                writeln!(w, "    add {cursor}, {cursor}, #7")?;
                writeln!(w, "    bic {cursor}, {cursor}, #7")?;
            }
            if wide {
                let value = scratch.alloc()?;
                writeln!(w, "    ldr {value}, [{cursor}]")?;
                frame_store(w, value, offset)?;
                writeln!(w, "    ldr {value}, [{cursor}, #4]")?;
                frame_store(w, value, offset + 4)?;
                scratch.free(value);
            } else if bytes == 8 {
                writeln!(w, "    vldr {}, [{cursor}]", F64_SCRATCH[0])?;
                frame_vstore(w, F64_SCRATCH[0], offset)?;
            } else {
                let value = scratch.alloc()?;
                writeln!(w, "    ldr {value}, [{cursor}]")?;
                super::operand::extend_integer(w, value, ty, true)?;
                store_result(w, dst, value, frame)?;
                scratch.free(value);
            }
            writeln!(w, "    add {cursor}, {cursor}, #{bytes}")?;
            writeln!(w, "    str {cursor}, [{list}]")?;
            scratch.free(cursor);
            scratch.free(list);
        }
        other => {
            return Err(unsupported(format!(
                "armv7 variadic emitter received {other}"
            )));
        }
    }
    Ok(())
}

/// `dest = entry stack pointer + offset`.
fn entry_sp_address<W: Write>(w: &mut W, dest: &str, offset: i32) -> Result<(), LaminaError> {
    use super::operand::{ENTRY_SP, is_modified_immediate, mov_imm32};
    if is_modified_immediate(offset.unsigned_abs()) && offset >= 0 {
        writeln!(w, "    add {dest}, {ENTRY_SP}, #{offset}")?;
    } else {
        mov_imm32(w, ADDR, offset as u32)?;
        writeln!(w, "    add {dest}, {ENTRY_SP}, {ADDR}")?;
    }
    Ok(())
}

/// The `va_list` this backend writes is one pointer, which is what the target table
/// states for 32-bit ARM.
const _: () = {
    let size = match TargetArchitecture::Armv7.va_list() {
        VaListAbi::Defined { size, .. } => size,
        VaListAbi::Undefined => {
            panic!("armv7 lowers va_list but the target table defines none")
        }
    };
    assert!(size == 4);
};
