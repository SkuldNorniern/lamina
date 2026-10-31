//! Atomic loads, stores, read-modify-write operations and fences.
//!
//! z/Architecture keeps stores in program order and makes an aligned load or store
//! of up to a doubleword atomic on its own, so only a sequentially consistent store
//! needs the serializing `BCR`. Everything that reads and writes runs as a `CS` or
//! `CSG` loop, which is a full barrier, so the orderings below that do not change
//! the code. There is no sub-word compare-and-swap, so a byte or halfword operation
//! works on the containing word and shifts its field into place; on this big-endian
//! target the field of a byte at word offset `k` sits `(4 - size - k) * 8` bits up.
//!
//! The accept and refuse set for each ordering matches `x86_64::atomic`.

use super::abi::{A0, A1, DIV_HI, DIV_LO, T0, T1, T2, address, load_imm, memory};
use super::operand::{Slots, load_operand};
use super::{EmitCtx, extend_integer, load_base, store_result};
use crate::{
    error::LaminaError,
    mir::{
        AtomicBinOp, Instruction as MirInst, MemoryOrdering, MirType, ScalarType,
        instruction::AddressMode,
    },
};
use lamina_platform::PointerWidth;
use std::io::Write;

/// Serializing instruction: a branch on a mask of fifteen to register zero, which
/// never branches and orders every access across it.
const SERIALIZE: &str = "    bcr 15, %r0";

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::AtomicLoad {
            ty,
            dst,
            addr,
            ordering,
        } => {
            match ordering {
                MemoryOrdering::Relaxed | MemoryOrdering::Acquire | MemoryOrdering::SeqCst => {}
                MemoryOrdering::Release | MemoryOrdering::AcqRel => {
                    return Err(unsupported_ordering("atomic_load", ordering));
                }
            }
            let mnemonic = load_mnemonic(ty)?;
            pointer(writer, addr, slots, T1)?;
            memory(writer, mnemonic, A0, T1, 0, T0)?;
            store_result(writer, A0, dst, slots)
        }
        MirInst::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            match ordering {
                MemoryOrdering::Relaxed | MemoryOrdering::Release | MemoryOrdering::SeqCst => {}
                MemoryOrdering::Acquire | MemoryOrdering::AcqRel => {
                    return Err(unsupported_ordering("atomic_store", ordering));
                }
            }
            let mnemonic = store_mnemonic(ty)?;
            load_operand(src, writer, slots, A0)?;
            pointer(writer, addr, slots, T1)?;
            memory(writer, mnemonic, A0, T1, 0, T0)?;
            if matches!(ordering, MemoryOrdering::SeqCst) {
                writeln!(writer, "{SERIALIZE}")?;
            }
            Ok(())
        }
        MirInst::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ordering: _,
        } => {
            let size = width(ty)?;
            load_operand(value, writer, slots, T2)?;
            pointer(writer, addr, slots, T1)?;
            if size < 4 {
                narrow_setup(writer, size)?;
                let retry = ctx.fresh_label("atomic_rmw");
                writeln!(writer, "    llgf {A1}, 0({T1})")?;
                writeln!(writer, "{retry}:")?;
                field_mask(writer, size)?;
                writeln!(writer, "    srlg {A0}, {A1}, 0({DIV_LO})")?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
                combine(writer, op, A0, T2)?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
                writeln!(writer, "    sllg {A0}, {A0}, 0({DIV_LO})")?;
                writeln!(writer, "    sllg {T0}, {T0}, 0({DIV_LO})")?;
                writeln!(writer, "    xgr {A0}, {A1}")?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
                writeln!(writer, "    xgr {A0}, {A1}")?;
                writeln!(writer, "    cs {A1}, {A0}, 0({T1})")?;
                writeln!(writer, "    brcl 4, {retry}")?;
                field_mask(writer, size)?;
                writeln!(writer, "    srlg {A0}, {A1}, 0({DIV_LO})")?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
            } else {
                let (load, swap) = word_mnemonics(size);
                let retry = ctx.fresh_label("atomic_rmw");
                writeln!(writer, "    {load} {A1}, 0({T1})")?;
                writeln!(writer, "{retry}:")?;
                writeln!(writer, "    lgr {A0}, {A1}")?;
                combine(writer, op, A0, T2)?;
                writeln!(writer, "    {swap} {A1}, {A0}, 0({T1})")?;
                writeln!(writer, "    brcl 4, {retry}")?;
                writeln!(writer, "    lgr {A0}, {A1}")?;
            }
            extend_integer(writer, A0, ty, true)?;
            store_result(writer, A0, dst, slots)
        }
        MirInst::AtomicCompareExchange {
            ty,
            dst,
            addr,
            expected,
            desired,
            success_ordering: _,
            failure_ordering: _,
        } => {
            let size = width(ty)?;
            if size < 4 {
                load_operand(desired, writer, slots, T2)?;
                load_operand(expected, writer, slots, DIV_HI)?;
                pointer(writer, addr, slots, T1)?;
                narrow_setup(writer, size)?;
                field_mask(writer, size)?;
                writeln!(writer, "    ngr {T2}, {T0}")?;
                writeln!(writer, "    ngr {DIV_HI}, {T0}")?;
                let retry = ctx.fresh_label("atomic_cas");
                let done = ctx.fresh_label("atomic_cas_done");
                writeln!(writer, "    llgf {A1}, 0({T1})")?;
                writeln!(writer, "{retry}:")?;
                field_mask(writer, size)?;
                writeln!(writer, "    srlg {A0}, {A1}, 0({DIV_LO})")?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
                writeln!(writer, "    cgr {A0}, {DIV_HI}")?;
                writeln!(writer, "    brcl 6, {done}")?;
                writeln!(writer, "    lgr {A0}, {T2}")?;
                writeln!(writer, "    sllg {A0}, {A0}, 0({DIV_LO})")?;
                writeln!(writer, "    sllg {T0}, {T0}, 0({DIV_LO})")?;
                writeln!(writer, "    xgr {A0}, {A1}")?;
                writeln!(writer, "    ngr {A0}, {T0}")?;
                writeln!(writer, "    xgr {A0}, {A1}")?;
                writeln!(writer, "    cs {A1}, {A0}, 0({T1})")?;
                writeln!(writer, "    brcl 4, {retry}")?;
                writeln!(writer, "    lgr {A0}, {DIV_HI}")?;
                writeln!(writer, "{done}:")?;
            } else {
                let (_, swap) = word_mnemonics(size);
                load_operand(desired, writer, slots, A0)?;
                load_operand(expected, writer, slots, A1)?;
                pointer(writer, addr, slots, T1)?;
                writeln!(writer, "    {swap} {A1}, {A0}, 0({T1})")?;
                writeln!(writer, "    lgr {A0}, {A1}")?;
            }
            extend_integer(writer, A0, ty, true)?;
            store_result(writer, A0, dst, slots)
        }
        MirInst::Fence { ordering } => {
            match ordering {
                MemoryOrdering::SeqCst => writeln!(writer, "{SERIALIZE}")?,
                MemoryOrdering::Relaxed
                | MemoryOrdering::Acquire
                | MemoryOrdering::Release
                | MemoryOrdering::AcqRel => {}
            }
            Ok(())
        }
        other => Err(LaminaError::ValidationError(format!(
            "s390x backend does not support atomic instruction {other}"
        ))),
    }
}

/// Splits the byte address in `T1` into the containing word address, left in `T1`,
/// and the field's shift within that word, left in `DIV_LO`.
fn narrow_setup<W: Write>(writer: &mut W, size: usize) -> Result<(), LaminaError> {
    writeln!(writer, "    lghi {T0}, 3")?;
    writeln!(writer, "    lgr {DIV_LO}, {T1}")?;
    writeln!(writer, "    ngr {DIV_LO}, {T0}")?;
    writeln!(writer, "    lghi {T0}, -4")?;
    writeln!(writer, "    ngr {T1}, {T0}")?;
    writeln!(writer, "    lghi {T0}, {}", 4 - size)?;
    writeln!(writer, "    sgr {T0}, {DIV_LO}")?;
    writeln!(writer, "    sllg {DIV_LO}, {T0}, 3")?;
    Ok(())
}

/// Loads the unshifted mask of a field `size` bytes wide into `T0`.
fn field_mask<W: Write>(writer: &mut W, size: usize) -> Result<(), LaminaError> {
    load_imm(writer, T0, (1i64 << (size * 8)) - 1)?;
    Ok(())
}

fn combine<W: Write>(
    writer: &mut W,
    op: &AtomicBinOp,
    acc: &str,
    value: &str,
) -> Result<(), LaminaError> {
    let mnemonic = match op {
        AtomicBinOp::Add => "agr",
        AtomicBinOp::Sub => "sgr",
        AtomicBinOp::And => "ngr",
        AtomicBinOp::Or => "ogr",
        AtomicBinOp::Xor => "xgr",
        AtomicBinOp::Exchange => "lgr",
    };
    writeln!(writer, "    {mnemonic} {acc}, {value}")?;
    Ok(())
}

fn word_mnemonics(size: usize) -> (&'static str, &'static str) {
    if size == 4 {
        ("llgf", "cs")
    } else {
        ("lg", "csg")
    }
}

/// Materializes the address an atomic operates on, folding in the displacement.
fn pointer<W: Write>(
    writer: &mut W,
    addr: &AddressMode,
    slots: &Slots,
    dst: &str,
) -> Result<(), LaminaError> {
    let AddressMode::BaseOffset { base, offset, .. } = addr else {
        return Err(LaminaError::ValidationError(format!(
            "s390x backend does not support atomic address mode {addr:?}"
        )));
    };
    load_base(writer, base, slots, dst)?;
    if *offset != 0 {
        address(writer, dst, dst, i64::from(*offset))?;
    }
    Ok(())
}

fn width(ty: &MirType) -> Result<usize, LaminaError> {
    match ty {
        MirType::Scalar(
            ScalarType::I1
            | ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::Ptr
            | ScalarType::PtrIn(_),
        ) => Ok(ty.size_bytes(PointerWidth::Bits64)),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) | MirType::Vector(_) => {
            Err(LaminaError::ValidationError(format!(
                "s390x atomic operation on type {ty} is not supported"
            )))
        }
    }
}

fn load_mnemonic(ty: &MirType) -> Result<&'static str, LaminaError> {
    match width(ty)? {
        1 => Ok("lgb"),
        2 => Ok("lgh"),
        4 => Ok("lgf"),
        _ => Ok("lg"),
    }
}

fn store_mnemonic(ty: &MirType) -> Result<&'static str, LaminaError> {
    match width(ty)? {
        1 => Ok("stcy"),
        2 => Ok("sthy"),
        4 => Ok("sty"),
        _ => Ok("stg"),
    }
}

fn unsupported_ordering(op: &str, ordering: &MemoryOrdering) -> LaminaError {
    LaminaError::ValidationError(format!(
        "s390x backend does not support {ordering} ordering for {op}"
    ))
}
