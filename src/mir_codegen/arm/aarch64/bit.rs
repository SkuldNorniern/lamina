use super::*;
use crate::mir::{IntRotOp, IntUnOp};

pub(super) fn emit_int_unary<W: Write>(
    w: &mut W,
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    let bits = ty.size_bytes() * 8;
    let x = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, src, x, frame, ra)?;
    extend_integer(w, x, ty, false)?;
    match op {
        IntUnOp::Popcount => {
            writeln!(w, "    fmov d0, {x}")?;
            writeln!(w, "    cnt v0.8b, v0.8b")?;
            writeln!(w, "    addv b0, v0.8b")?;
            writeln!(w, "    fmov {}, s0", w_alias(t))?;
        }
        IntUnOp::Ctlz => {
            writeln!(w, "    clz {t}, {x}")?;
            if bits < 64 {
                writeln!(w, "    sub {t}, {t}, #{}", 64 - bits)?;
            }
        }
        IntUnOp::Cttz => {
            if bits < 64 {
                writeln!(w, "    orr {x}, {x}, #{}", 1u64 << bits)?;
            }
            writeln!(w, "    rbit {t}, {x}")?;
            writeln!(w, "    clz {t}, {t}")?;
        }
        IntUnOp::Bswap => match bits {
            8 => writeln!(w, "    mov {t}, {x}")?,
            16 => writeln!(w, "    rev16 {}, {}", w_alias(t), w_alias(x))?,
            32 => writeln!(w, "    rev {}, {}", w_alias(t), w_alias(x))?,
            _ => writeln!(w, "    rev {t}, {x}")?,
        },
    }
    ra.free_scratch(x);
    extend_integer(w, t, ty, true)?;
    store_result(w, dst, &x_alias(t), frame, ra)?;
    ra.free_scratch(t);
    Ok(())
}

pub(super) fn emit_int_rotate<W: Write>(
    w: &mut W,
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    (lhs, rhs): (&Operand, &Operand),
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    let bits = ty.size_bytes() * 8;
    let mask = bits - 1;
    let x = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let c = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, lhs, x, frame, ra)?;
    emit_materialize_operand(w, rhs, c, frame, ra)?;
    extend_integer(w, x, ty, false)?;
    let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
    match bits {
        32 | 64 => {
            let (xr, cr, tr) = if bits == 64 {
                (x.to_string(), c.to_string(), t.to_string())
            } else {
                (w_alias(x), w_alias(c), w_alias(t))
            };
            if op == IntRotOp::Rotl {
                writeln!(w, "    neg {cr}, {cr}")?;
            }
            writeln!(w, "    ror {tr}, {xr}, {cr}")?;
        }
        _ => {
            let u = ra.alloc_scratch().ok_or_else(scratch_error)?;
            writeln!(w, "    and {c}, {c}, #{mask}")?;
            writeln!(w, "    neg {u}, {c}")?;
            writeln!(w, "    and {u}, {u}, #{mask}")?;
            let (near, far) = match op {
                IntRotOp::Rotl => ("lsl", "lsr"),
                IntRotOp::Rotr => ("lsr", "lsl"),
            };
            writeln!(w, "    {near} {t}, {x}, {c}")?;
            writeln!(w, "    {far} {u}, {x}, {u}")?;
            writeln!(w, "    orr {t}, {t}, {u}")?;
            writeln!(w, "    ubfx {t}, {t}, #0, #{bits}")?;
            ra.free_scratch(u);
        }
    }
    ra.free_scratch(x);
    ra.free_scratch(c);
    extend_integer(w, t, ty, true)?;
    store_result(w, dst, &x_alias(t), frame, ra)?;
    ra.free_scratch(t);
    Ok(())
}
