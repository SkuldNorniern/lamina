//! Checked arithmetic: whether the wrapping result leaves the range of its type.
//!
//! Below 64 bits the exact result fits in a register, so the question is whether
//! re-reading it at the operand width gives the same number. At 64 bits the
//! condition code answers it for addition and subtraction, and multiplication
//! takes the 128-bit product `MLGR` leaves: unsigned overflows when the high half
//! is nonzero, and the signed high half is the unsigned one less the multiplicand
//! of each negative operand.

use super::abi::{A0, A1, DIV_HI, DIV_LO, T0, T1};
use super::operand::{Slots, load_operand};
use super::{EmitCtx, extend_integer, store_result};
use crate::{
    error::LaminaError,
    mir::{IntOverflowOp, MirType, Operand, Register},
};
use lamina_platform::PointerWidth;
use std::io::Write;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit<W: Write>(
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
    op: &IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
) -> Result<(), LaminaError> {
    load_operand(lhs, writer, slots, A0)?;
    load_operand(rhs, writer, slots, A1)?;
    let mask = if ty.size_bytes(PointerWidth::Bits64) < 8 {
        extend_integer(writer, A0, ty, signed)?;
        extend_integer(writer, A1, ty, signed)?;
        let mnemonic = match op {
            IntOverflowOp::Add => "agr",
            IntOverflowOp::Sub => "sgr",
            IntOverflowOp::Mul => "msgr",
        };
        writeln!(writer, "    {mnemonic} {A0}, {A1}")?;
        writeln!(writer, "    lgr {T1}, {A0}")?;
        extend_integer(writer, T1, ty, signed)?;
        writeln!(writer, "    cgr {A0}, {T1}")?;
        6
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                writeln!(writer, "    agr {A0}, {A1}")?;
                1
            }
            (IntOverflowOp::Add, false) => {
                writeln!(writer, "    algr {A0}, {A1}")?;
                3
            }
            (IntOverflowOp::Sub, true) => {
                writeln!(writer, "    sgr {A0}, {A1}")?;
                1
            }
            (IntOverflowOp::Sub, false) => {
                writeln!(writer, "    slgr {A0}, {A1}")?;
                4
            }
            (IntOverflowOp::Mul, true) => {
                writeln!(writer, "    lgr {DIV_LO}, {A0}")?;
                writeln!(writer, "    mlgr {DIV_HI}, {A1}")?;
                writeln!(writer, "    srag {T0}, {A0}, 63")?;
                writeln!(writer, "    ngr {T0}, {A1}")?;
                writeln!(writer, "    sgr {DIV_HI}, {T0}")?;
                writeln!(writer, "    srag {T0}, {A1}, 63")?;
                writeln!(writer, "    ngr {T0}, {A0}")?;
                writeln!(writer, "    sgr {DIV_HI}, {T0}")?;
                writeln!(writer, "    srag {T0}, {DIV_LO}, 63")?;
                writeln!(writer, "    cgr {DIV_HI}, {T0}")?;
                6
            }
            (IntOverflowOp::Mul, false) => {
                writeln!(writer, "    lgr {DIV_LO}, {A0}")?;
                writeln!(writer, "    mlgr {DIV_HI}, {A1}")?;
                writeln!(writer, "    ltgr {DIV_HI}, {DIV_HI}")?;
                6
            }
        }
    };
    let done = ctx.fresh_label("ovf");
    writeln!(writer, "    lghi {A0}, 1")?;
    writeln!(writer, "    brcl {mask}, {done}")?;
    writeln!(writer, "    lghi {A0}, 0")?;
    writeln!(writer, "{done}:")?;
    store_result(writer, A0, dst, slots)
}
