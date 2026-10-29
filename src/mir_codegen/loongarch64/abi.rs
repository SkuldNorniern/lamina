//! LoongArch64 LP64D calling convention: register names, frame layout and argument places.

use crate::mir::MirType;
use std::io::{self, Write};

pub(super) const ARG_GPRS: [&str; 8] = ["$a0", "$a1", "$a2", "$a3", "$a4", "$a5", "$a6", "$a7"];
pub(super) const ARG_FPRS: [&str; 8] = [
    "$fa0", "$fa1", "$fa2", "$fa3", "$fa4", "$fa5", "$fa6", "$fa7",
];

pub(super) const WORD: i32 = 8;

/// Bytes the prologue reserves for the saved `ra` and `fp` pair.
const SAVED_PAIR_BYTES: i32 = 16;

const IMM12_MIN: i64 = -2048;
const IMM12_MAX: i64 = 2047;

pub(super) fn fits_si12(value: i64) -> bool {
    (IMM12_MIN..=IMM12_MAX).contains(&value)
}

/// Offset of local slot `index` from `$fp`. The prologue leaves `$fp` at the caller's
/// `$sp`, with `$ra` at `fp-8` and the caller's `$fp` at `fp-16`.
pub(super) fn slot_offset(index: usize) -> i32 {
    -(SAVED_PAIR_BYTES + (index as i32 + 1) * WORD)
}

/// Bytes of locals the frame reserves below the saved pair, keeping `$sp` 16-byte aligned.
pub(super) fn locals_bytes(stack_size: usize) -> usize {
    stack_size.div_ceil(16) * 16
}

pub(super) fn prologue<W: Write>(writer: &mut W, stack_size: usize) -> io::Result<()> {
    writeln!(writer, "    addi.d $sp, $sp, -{SAVED_PAIR_BYTES}")?;
    writeln!(writer, "    st.d $ra, $sp, {}", SAVED_PAIR_BYTES - WORD)?;
    writeln!(writer, "    st.d $fp, $sp, {}", SAVED_PAIR_BYTES - 2 * WORD)?;
    writeln!(writer, "    addi.d $fp, $sp, {SAVED_PAIR_BYTES}")?;
    let locals = locals_bytes(stack_size);
    if locals > 0 {
        adjust_sp(writer, -(locals as i64))?;
    }
    Ok(())
}

/// Pops the frame. A tail epilogue jumps to `tail` instead of returning.
pub(super) fn epilogue<W: Write>(
    writer: &mut W,
    stack_size: usize,
    tail: Option<&str>,
) -> io::Result<()> {
    let locals = locals_bytes(stack_size);
    if locals > 0 {
        adjust_sp(writer, locals as i64)?;
    }
    writeln!(writer, "    ld.d $ra, $fp, -{WORD}")?;
    writeln!(writer, "    ld.d $fp, $fp, -{}", 2 * WORD)?;
    writeln!(writer, "    addi.d $sp, $sp, {SAVED_PAIR_BYTES}")?;
    match tail {
        Some(target) => writeln!(writer, "    b {target}"),
        None => writeln!(writer, "    jirl $zero, $ra, 0"),
    }
}

/// Moves `$sp` by `delta` bytes, through `$t0` when the delta does not fit 12 bits.
pub(super) fn adjust_sp<W: Write>(writer: &mut W, delta: i64) -> io::Result<()> {
    if fits_si12(delta) {
        writeln!(writer, "    addi.d $sp, $sp, {delta}")
    } else {
        load_imm(writer, "$t0", delta)?;
        writeln!(writer, "    add.d $sp, $sp, $t0")
    }
}

/// Emits `{mnemonic} {reg}, $fp, {offset}`, building the address in `scratch` when the
/// offset does not fit. A load may use `reg` as its own scratch.
pub(super) fn fp_load<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    offset: i32,
    scratch: &str,
) -> io::Result<()> {
    if fits_si12(i64::from(offset)) {
        writeln!(writer, "    {mnemonic} {reg}, $fp, {offset}")
    } else {
        load_imm(writer, scratch, i64::from(offset))?;
        writeln!(writer, "    add.d {scratch}, $fp, {scratch}")?;
        writeln!(writer, "    {mnemonic} {reg}, {scratch}, 0")
    }
}

/// Stores `reg` at `$fp + offset`. When the offset is out of range and `reg` is the
/// scratch itself, `$fp` is walked to the slot and back so `reg` survives.
pub(super) fn fp_store<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    offset: i32,
    scratch: &str,
) -> io::Result<()> {
    if fits_si12(i64::from(offset)) {
        writeln!(writer, "    {mnemonic} {reg}, $fp, {offset}")
    } else if reg != scratch {
        load_imm(writer, scratch, i64::from(offset))?;
        writeln!(writer, "    add.d {scratch}, $fp, {scratch}")?;
        writeln!(writer, "    {mnemonic} {reg}, {scratch}, 0")
    } else {
        let walk = i64::from(offset) + 2048;
        walk_fp(writer, walk)?;
        writeln!(writer, "    {mnemonic} {reg}, $fp, -2048")?;
        walk_fp(writer, -walk)
    }
}

fn walk_fp<W: Write>(writer: &mut W, mut delta: i64) -> io::Result<()> {
    while delta != 0 {
        let step = delta.clamp(IMM12_MIN, IMM12_MAX);
        writeln!(writer, "    addi.d $fp, $fp, {step}")?;
        delta -= step;
    }
    Ok(())
}

/// Materialises `$fp + offset` in `dst`.
pub(super) fn fp_address<W: Write>(writer: &mut W, dst: &str, offset: i32) -> io::Result<()> {
    if fits_si12(i64::from(offset)) {
        writeln!(writer, "    addi.d {dst}, $fp, {offset}")
    } else {
        load_imm(writer, dst, i64::from(offset))?;
        writeln!(writer, "    add.d {dst}, $fp, {dst}")
    }
}

/// `$sp + offset` in `dst`, for outgoing argument areas.
pub(super) fn sp_address<W: Write>(writer: &mut W, dst: &str, offset: usize) -> io::Result<()> {
    if offset <= 2047 {
        writeln!(writer, "    addi.d {dst}, $sp, {offset}")
    } else {
        load_imm(writer, dst, offset as i64)?;
        writeln!(writer, "    add.d {dst}, $sp, {dst}")
    }
}

/// Loads a 64-bit constant into `reg`, using only `reg`.
///
/// `lu12i.w` fills bits 12..=31 and sign-extends into 32..=63, so a value that fits
/// in 32 bits needs no more than that and an `ori` for the low 12 bits. A wider value
/// also sets bits 32..=51 with `lu32i.d` and bits 52..=63 with `lu52i.d`, which
/// overwrite whatever the earlier steps left there.
pub(super) fn load_imm<W: Write>(writer: &mut W, reg: &str, value: i64) -> io::Result<()> {
    if fits_si12(value) {
        return writeln!(writer, "    addi.d {reg}, $zero, {value}");
    }
    writeln!(
        writer,
        "    lu12i.w {reg}, {}",
        signed_bits(value >> 12, 20)
    )?;
    let low = value & 0xfff;
    if low != 0 {
        writeln!(writer, "    ori {reg}, {reg}, {low}")?;
    }
    if value != i64::from(value as i32) {
        writeln!(
            writer,
            "    lu32i.d {reg}, {}",
            signed_bits(value >> 32, 20)
        )?;
        writeln!(
            writer,
            "    lu52i.d {reg}, {reg}, {}",
            signed_bits(value >> 52, 12)
        )?;
    }
    Ok(())
}

/// The low `bits` of `value`, read as a two's complement immediate.
pub(super) fn signed_bits(value: i64, bits: u32) -> i64 {
    let mask = (1i64 << bits) - 1;
    let half = 1i64 << (bits - 1);
    ((value & mask) ^ half) - half
}

/// Where one argument travels under LP64D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArgPlace {
    Gpr(usize),
    Fpr(usize),
    /// Eight-byte slot number in the outgoing argument area.
    Stack(usize),
}

/// Places scalar arguments: floats take `fa0`-`fa7`, everything else `a0`-`a7`, and the
/// rest go to the stack in order.
pub(super) fn classify(types: &[MirType]) -> Vec<ArgPlace> {
    let (mut gprs, mut fprs, mut stack) = (0, 0, 0);
    types
        .iter()
        .map(|ty| {
            if ty.is_float() && fprs < ARG_FPRS.len() {
                fprs += 1;
                ArgPlace::Fpr(fprs - 1)
            } else if !ty.is_float() && gprs < ARG_GPRS.len() {
                gprs += 1;
                ArgPlace::Gpr(gprs - 1)
            } else {
                stack += 1;
                ArgPlace::Stack(stack - 1)
            }
        })
        .collect()
}

/// Stack slots the classified arguments need, rounded to the 16-byte call alignment.
pub(super) fn outgoing_bytes(places: &[ArgPlace]) -> usize {
    let words = places
        .iter()
        .filter(|place| matches!(place, ArgPlace::Stack(_)))
        .count();
    (words * WORD as usize).div_ceil(16) * 16
}
