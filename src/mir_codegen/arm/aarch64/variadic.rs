//! Darwin variadic arguments. Every variadic argument sits in the caller's stack argument
//! area, one eight-byte slot each, and `va_list` is a single pointer to the next slot.

use super::*;

/// Darwin's `va_list` is a `char *` at offset zero of the object, so only that word is
/// read or written; the rest of the 32-byte object stays unused.
const CURSOR_BYTES: i32 = 8;

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

/// `area_start` is the offset from x29 of the first variadic slot, which is `None` for a
/// function that is not variadic.
pub(super) fn emit<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    target_os: TargetOperatingSystem,
    area_start: Option<i32>,
) -> Result<(), LaminaError> {
    if !os_uses_macho(target_os) {
        return Err(unsupported(format!(
            "variadic argument `{inst}` is not yet supported by the AArch64 backend on {target_os:?}; it is implemented for Darwin"
        )));
    }
    let Some(start) = area_start else {
        return Err(unsupported(
            "variadic arguments are only available inside a variadic function",
        ));
    };
    match inst {
        MirInst::VaStart { ap } => {
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            emit_frame_slot_address(w, cursor, start)?;
            writeln!(w, "    str {cursor}, [{list}]")?;
            ra.free_scratch(cursor);
            ra.free_scratch(list);
        }
        MirInst::VaArg { ty, dst, ap } => {
            match ty {
                MirType::Scalar(
                    ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::Ptr
                    | ScalarType::F64,
                ) => {}
                _ => {
                    return Err(unsupported(format!(
                        "va_arg of {ty:?} is not supported; use an integer, pointer or f64"
                    )));
                }
            }
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let next = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            writeln!(w, "    ldr {cursor}, [{list}]")?;
            writeln!(w, "    add {next}, {cursor}, #{CURSOR_BYTES}")?;
            writeln!(w, "    str {next}, [{list}]")?;
            writeln!(w, "    ldr {cursor}, [{cursor}]")?;
            emit_narrow_param_extension(w, cursor, ty)?;
            store_result(w, dst, cursor, frame, ra)?;
            ra.free_scratch(next);
            ra.free_scratch(cursor);
            ra.free_scratch(list);
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            let from = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let to = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, src, from, frame, ra)?;
            emit_materialize_operand(w, dst, to, frame, ra)?;
            writeln!(w, "    ldr {cursor}, [{from}]")?;
            writeln!(w, "    str {cursor}, [{to}]")?;
            ra.free_scratch(cursor);
            ra.free_scratch(to);
            ra.free_scratch(from);
        }
        _ => return Err(unsupported("not a variadic instruction")),
    }
    Ok(())
}
