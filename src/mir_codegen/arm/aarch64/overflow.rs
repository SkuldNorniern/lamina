//! Checked arithmetic. `<op>.ovf` is true when the wrapping result leaves the range of the
//! type. A narrow result is compared with its own extension, which is exact because the
//! operands are extended first and a 32-bit product fits in 64 bits. A 64-bit result reads
//! the carry, overflow or high-half flags the operation leaves behind.

use super::*;
use crate::mir::IntOverflowOp;

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

/// Extends the low `bits` of `source` into all of `dest` according to `signed`.
fn extend_into<W: Write>(
    w: &mut W,
    dest: &str,
    source: &str,
    bits: u32,
    signed: bool,
) -> Result<(), LaminaError> {
    let instruction = if signed { "sbfx" } else { "ubfx" };
    writeln!(w, "    {instruction} {dest}, {source}, #0, #{bits}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit<W: Write>(
    w: &mut W,
    op: IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let bits = match ty {
        MirType::Scalar(ScalarType::I8) => 8,
        MirType::Scalar(ScalarType::I16) => 16,
        MirType::Scalar(ScalarType::I32) => 32,
        MirType::Scalar(ScalarType::I64) => 64,
        MirType::Scalar(ScalarType::Ptr) => 8 * ty.size_bytes(pointer_width) as u32,
        _ => {
            return Err(unsupported(format!(
                "checked arithmetic on {ty:?} is not supported by the AArch64 backend; use an integer or pointer"
            )));
        }
    };
    let a = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let b = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, lhs, a, frame, ra)?;
    emit_materialize_operand(w, rhs, b, frame, ra)?;
    let r = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let h = ra.alloc_scratch().ok_or_else(scratch_error)?;
    if bits < 64 {
        extend_into(w, a, a, bits, signed)?;
        extend_into(w, b, b, bits, signed)?;
        let mnemonic = match op {
            IntOverflowOp::Add => "add",
            IntOverflowOp::Sub => "sub",
            IntOverflowOp::Mul => "mul",
        };
        writeln!(w, "    {mnemonic} {r}, {a}, {b}")?;
        extend_into(w, h, r, bits, signed)?;
        writeln!(w, "    cmp {r}, {h}")?;
        writeln!(w, "    cset {r}, ne")?;
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                writeln!(w, "    cmn {a}, {b}")?;
                writeln!(w, "    cset {r}, vs")?;
            }
            (IntOverflowOp::Add, false) => {
                writeln!(w, "    cmn {a}, {b}")?;
                writeln!(w, "    cset {r}, cs")?;
            }
            (IntOverflowOp::Sub, true) => {
                writeln!(w, "    cmp {a}, {b}")?;
                writeln!(w, "    cset {r}, vs")?;
            }
            (IntOverflowOp::Sub, false) => {
                writeln!(w, "    cmp {a}, {b}")?;
                writeln!(w, "    cset {r}, cc")?;
            }
            (IntOverflowOp::Mul, true) => {
                writeln!(w, "    mul {r}, {a}, {b}")?;
                writeln!(w, "    smulh {h}, {a}, {b}")?;
                writeln!(w, "    cmp {h}, {r}, asr #63")?;
                writeln!(w, "    cset {r}, ne")?;
            }
            (IntOverflowOp::Mul, false) => {
                writeln!(w, "    umulh {h}, {a}, {b}")?;
                writeln!(w, "    cmp {h}, #0")?;
                writeln!(w, "    cset {r}, ne")?;
            }
        }
    }
    ra.free_scratch(h);
    ra.free_scratch(b);
    ra.free_scratch(a);
    store_result(w, dst, r, frame, ra)?;
    ra.free_scratch(r);
    Ok(())
}
