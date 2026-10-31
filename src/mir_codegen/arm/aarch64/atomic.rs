use super::*;
use crate::mir::{AtomicBinOp, MemoryOrdering};
use lamina_platform::PointerWidth;

const STATUS: &str = "w16";

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::AtomicLoad {
            ty,
            dst,
            addr,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            let a = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
            materialize_address(w, addr, a, frame, ra)?;
            let op = match ordering {
                MemoryOrdering::Relaxed => "ldr",
                MemoryOrdering::Acquire | MemoryOrdering::SeqCst => "ldar",
                MemoryOrdering::Release | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic load", ordering));
                }
            };
            writeln!(
                w,
                "    {op}{} {}, [{a}]",
                size_suffix(size),
                reg_of(size, t)
            )?;
            ra.free_scratch(a);
            extend_integer(w, t, ty, true, pointer_width)?;
            store_result(w, dst, &x_alias(t), frame, ra)?;
            ra.free_scratch(t);
        }
        MirInst::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            let a = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
            materialize_address(w, addr, a, frame, ra)?;
            emit_materialize_operand(w, src, t, frame, ra)?;
            let op = match ordering {
                MemoryOrdering::Relaxed => "str",
                MemoryOrdering::Release | MemoryOrdering::SeqCst => "stlr",
                MemoryOrdering::Acquire | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic store", ordering));
                }
            };
            writeln!(
                w,
                "    {op}{} {}, [{a}]",
                size_suffix(size),
                reg_of(size, t)
            )?;
            ra.free_scratch(a);
            ra.free_scratch(t);
        }
        MirInst::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            let a = ra.alloc_scratch().ok_or_else(scratch_error)?;
            materialize_address(w, addr, a, frame, ra)?;
            let v = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, value, v, frame, ra)?;
            let old = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let new = if *op == AtomicBinOp::Exchange {
                v
            } else {
                ra.alloc_scratch().ok_or_else(scratch_error)?
            };
            let (ldx, stx) = exclusive_ops(ordering, size);
            writeln!(w, "1:")?;
            writeln!(w, "    {ldx} {}, [{a}]", reg_of(size, old))?;
            match op {
                AtomicBinOp::Exchange => {}
                AtomicBinOp::Add => writeln!(w, "    add {new}, {old}, {v}")?,
                AtomicBinOp::Sub => writeln!(w, "    sub {new}, {old}, {v}")?,
                AtomicBinOp::And => writeln!(w, "    and {new}, {old}, {v}")?,
                AtomicBinOp::Or => writeln!(w, "    orr {new}, {old}, {v}")?,
                AtomicBinOp::Xor => writeln!(w, "    eor {new}, {old}, {v}")?,
            }
            writeln!(w, "    {stx} {STATUS}, {}, [{a}]", reg_of(size, new))?;
            writeln!(w, "    cbnz {STATUS}, 1b")?;
            if new != v {
                ra.free_scratch(new);
            }
            ra.free_scratch(v);
            ra.free_scratch(a);
            extend_integer(w, old, ty, true, pointer_width)?;
            store_result(w, dst, &x_alias(old), frame, ra)?;
            ra.free_scratch(old);
        }
        MirInst::AtomicCompareExchange {
            ty,
            dst,
            addr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            let a = ra.alloc_scratch().ok_or_else(scratch_error)?;
            materialize_address(w, addr, a, frame, ra)?;
            let e = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, expected, e, frame, ra)?;
            let d = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, desired, d, frame, ra)?;
            let old = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let acquire_fail = is_acquire(failure_ordering);
            let (ldx, stx) = exclusive_ops(success_ordering, size);
            let ldx = if acquire_fail && !ldx.starts_with("ldaxr") {
                format!("ldaxr{}", size_suffix(size))
            } else {
                ldx
            };
            let compare = match size {
                8 => format!("cmp {}, {}", reg_of(size, old), reg_of(size, e)),
                4 => format!("cmp {}, {}", reg_of(size, old), reg_of(size, e)),
                1 => format!("cmp {}, {}, uxtb", reg_of(size, old), reg_of(size, e)),
                _ => format!("cmp {}, {}, uxth", reg_of(size, old), reg_of(size, e)),
            };
            writeln!(w, "1:")?;
            writeln!(w, "    {ldx} {}, [{a}]", reg_of(size, old))?;
            writeln!(w, "    {compare}")?;
            writeln!(w, "    b.ne 2f")?;
            writeln!(w, "    {stx} {STATUS}, {}, [{a}]", reg_of(size, d))?;
            writeln!(w, "    cbnz {STATUS}, 1b")?;
            writeln!(w, "    b 3f")?;
            writeln!(w, "2:")?;
            writeln!(w, "    clrex")?;
            writeln!(w, "3:")?;
            ra.free_scratch(d);
            ra.free_scratch(e);
            ra.free_scratch(a);
            extend_integer(w, old, ty, true, pointer_width)?;
            store_result(w, dst, &x_alias(old), frame, ra)?;
            ra.free_scratch(old);
        }
        MirInst::Fence { ordering } => match ordering {
            MemoryOrdering::Relaxed => {}
            MemoryOrdering::Acquire => writeln!(w, "    dmb ishld")?,
            MemoryOrdering::Release | MemoryOrdering::AcqRel | MemoryOrdering::SeqCst => {
                writeln!(w, "    dmb ish")?;
            }
        },
        other => {
            return Err(LaminaError::ValidationError(format!(
                "AArch64 backend does not handle atomic instruction {other}"
            )));
        }
    }
    Ok(())
}

fn is_acquire(ordering: &MemoryOrdering) -> bool {
    matches!(
        ordering,
        MemoryOrdering::Acquire | MemoryOrdering::AcqRel | MemoryOrdering::SeqCst
    )
}

fn is_release(ordering: &MemoryOrdering) -> bool {
    matches!(
        ordering,
        MemoryOrdering::Release | MemoryOrdering::AcqRel | MemoryOrdering::SeqCst
    )
}

fn exclusive_ops(ordering: &MemoryOrdering, size: usize) -> (String, String) {
    let sfx = size_suffix(size);
    let ld = if is_acquire(ordering) {
        "ldaxr"
    } else {
        "ldxr"
    };
    let st = if is_release(ordering) {
        "stlxr"
    } else {
        "stxr"
    };
    (format!("{ld}{sfx}"), format!("{st}{sfx}"))
}

fn access_size(ty: &MirType, pointer_width: PointerWidth) -> Result<usize, LaminaError> {
    match ty.size_bytes(pointer_width) {
        size @ (1 | 2 | 4 | 8) => Ok(size),
        _ => Err(LaminaError::ValidationError(
            "AArch64 atomic operation has an unsupported width".into(),
        )),
    }
}

fn size_suffix(size: usize) -> &'static str {
    match size {
        1 => "b",
        2 => "h",
        _ => "",
    }
}

fn reg_of(size: usize, xreg: &str) -> String {
    if size == 8 {
        xreg.to_string()
    } else {
        w_alias(xreg)
    }
}

fn ordering_error(what: &str, ordering: &MemoryOrdering) -> LaminaError {
    LaminaError::ValidationError(format!(
        "AArch64 backend: {what} cannot use {ordering} ordering"
    ))
}
