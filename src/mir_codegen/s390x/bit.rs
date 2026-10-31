//! Integer bit intrinsics and rotates.
//!
//! `POPCNT` counts each byte on its own, so the byte counts are summed by multiplying
//! by `0x0101010101010101` and taking the top byte. `FLOGR` reports the bit position
//! of the leftmost one, or 64 when the operand is zero, which is leading-zero count
//! at full width; trailing zeros come from the population count of the bits below the
//! lowest set one. `LRVGR` reverses all eight bytes, so a narrower type shifts the
//! reversed low bytes back down.

use super::abi::{A0, A1, DIV_HI, T0, T1, T2, load_imm};
use super::operand::{Slots, load_operand, unsupported};
use super::{extend_integer, store_result};
use crate::{
    error::LaminaError,
    mir::{IntRotOp, IntUnOp, MirType, Operand, Register},
};
use lamina_platform::PointerWidth;
use std::io::Write;

/// Reduces the per-byte counts `POPCNT` leaves in `reg` to one total in `reg`.
fn sum_byte_counts<W: Write>(writer: &mut W, reg: &str) -> Result<(), LaminaError> {
    writeln!(writer, "    popcnt {T0}, {reg}")?;
    load_imm(writer, T1, 0x0101_0101_0101_0101)?;
    writeln!(writer, "    msgr {T0}, {T1}")?;
    writeln!(writer, "    srlg {reg}, {T0}, 56")?;
    Ok(())
}

fn width_bits(ty: &MirType) -> Result<usize, LaminaError> {
    if ty.is_vector() {
        return Err(unsupported(format!(
            "s390x integer bit operation on vector type {ty}"
        )));
    }
    Ok(ty.size_bytes(PointerWidth::Bits64) * 8)
}

pub(super) fn emit_int_unary<W: Write>(
    writer: &mut W,
    slots: &Slots,
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
) -> Result<(), LaminaError> {
    let bits = width_bits(ty)?;
    load_operand(src, writer, slots, A0)?;
    extend_integer(writer, A0, ty, false)?;
    match op {
        IntUnOp::Popcount => sum_byte_counts(writer, A0)?,
        IntUnOp::Ctlz => {
            writeln!(writer, "    flogr {DIV_HI}, {A0}")?;
            writeln!(writer, "    lgr {A0}, {DIV_HI}")?;
            if bits < 64 {
                writeln!(writer, "    aghi {A0}, -{}", 64 - bits)?;
            }
        }
        IntUnOp::Cttz => {
            if bits < 64 {
                load_imm(writer, T1, 1i64 << bits)?;
                writeln!(writer, "    ogr {A0}, {T1}")?;
            }
            writeln!(writer, "    lcgr {T1}, {A0}")?;
            writeln!(writer, "    ngr {A0}, {T1}")?;
            writeln!(writer, "    aghi {A0}, -1")?;
            sum_byte_counts(writer, A0)?;
        }
        IntUnOp::Bswap => {
            if bits > 8 {
                writeln!(writer, "    lrvgr {A0}, {A0}")?;
                if bits < 64 {
                    writeln!(writer, "    srlg {A0}, {A0}, {}", 64 - bits)?;
                }
            }
        }
    }
    extend_integer(writer, A0, ty, true)?;
    store_result(writer, A0, dst, slots)
}

pub(super) fn emit_int_rotate<W: Write>(
    writer: &mut W,
    slots: &Slots,
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
) -> Result<(), LaminaError> {
    let bits = width_bits(ty)?;
    load_operand(lhs, writer, slots, A0)?;
    load_operand(rhs, writer, slots, A1)?;
    extend_integer(writer, A0, ty, false)?;
    writeln!(writer, "    lghi {T0}, {}", bits - 1)?;
    writeln!(writer, "    ngr {A1}, {T0}")?;
    writeln!(writer, "    lcgr {T1}, {A1}")?;
    writeln!(writer, "    ngr {T1}, {T0}")?;
    let (near, far) = match op {
        IntRotOp::Rotl => ("sllg", "srlg"),
        IntRotOp::Rotr => ("srlg", "sllg"),
    };
    writeln!(writer, "    {near} {T2}, {A0}, 0({A1})")?;
    writeln!(writer, "    {far} {A0}, {A0}, 0({T1})")?;
    writeln!(writer, "    ogr {A0}, {T2}")?;
    extend_integer(writer, A0, ty, false)?;
    extend_integer(writer, A0, ty, true)?;
    store_result(writer, A0, dst, slots)
}
