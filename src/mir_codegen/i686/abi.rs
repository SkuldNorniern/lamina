//! cdecl (32-bit System V) frame layout and argument places.

use crate::mir::MirType;
use lamina_platform::PointerWidth;
use std::io::{self, Write};

pub(super) const PTR: PointerWidth = PointerWidth::Bits32;

/// Bytes one stack argument word takes.
pub(super) const WORD: usize = 4;

/// Bytes one frame slot takes. A `stackalloc` reservation counts eight-byte words,
/// so a slot is eight bytes here too and a 32-bit value uses its low half.
pub(super) const SLOT: i32 = 8;

/// `%ebp` is `8 mod 16`, so a slot at offset `o` has address `8 + o` modulo 16.
pub(super) const FRAME_RESIDUE: i64 = 8;

/// Offset of local slot `index` from `%ebp`.
pub(super) fn slot_offset(index: usize) -> i32 {
    -((index as i32 + 1) * SLOT)
}

/// Bytes the frame reserves below `%ebp`.
///
/// `call` pushes four bytes onto a 16-byte-aligned `%esp` and the prologue pushes
/// `%ebp`, leaving `%esp` eight past a boundary. A frame of `8 mod 16` bytes puts it
/// back on one, and every call site restores it before returning.
pub(super) fn frame_bytes(locals: usize) -> usize {
    locals.div_ceil(16) * 16 + 8
}

pub(super) fn prologue<W: Write>(writer: &mut W, frame: usize) -> io::Result<()> {
    writeln!(writer, "    pushl %ebp")?;
    writeln!(writer, "    movl %esp, %ebp")?;
    writeln!(writer, "    subl ${frame}, %esp")
}

/// Pops the frame and returns, releasing `pop` extra bytes of the caller's argument
/// area on the way out. A struct-returning callee releases the hidden pointer that
/// way, which is what the i386 convention asks of it.
pub(super) fn epilogue<W: Write>(writer: &mut W, pop: usize) -> io::Result<()> {
    writeln!(writer, "    leave")?;
    if pop > 0 {
        writeln!(writer, "    ret ${pop}")
    } else {
        writeln!(writer, "    ret")
    }
}

/// Where one argument sits, as a byte offset into the argument area and a size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ArgSlot {
    pub offset: usize,
    pub bytes: usize,
}

/// Bytes an argument of `ty` occupies, every one rounded up to a four-byte word.
pub(super) fn argument_bytes(ty: MirType) -> usize {
    ty.size_bytes(PTR).max(WORD).div_ceil(WORD) * WORD
}

/// Bytes an aggregate argument occupies when passed by value.
pub(super) fn aggregate_bytes(size: usize) -> usize {
    size.div_ceil(WORD) * WORD
}

/// Places arguments left to right from the base of the area. cdecl passes every
/// argument on the stack, so the only question is each one's offset. `sret` reserves
/// the first word for the hidden pointer a struct return is written through.
pub(super) fn classify(
    types: &[MirType],
    aggregates: &[Option<usize>],
    sret: bool,
) -> Vec<ArgSlot> {
    let mut offset = if sret { WORD } else { 0 };
    types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            let bytes = match aggregates.get(index).copied().flatten() {
                Some(size) => aggregate_bytes(size),
                None => argument_bytes(*ty),
            };
            let slot = ArgSlot { offset, bytes };
            offset += bytes;
            slot
        })
        .collect()
}

/// Bytes of outgoing argument area, padded so `%esp` is 16-byte aligned at the call.
pub(super) fn outgoing_bytes(slots: &[ArgSlot], sret: bool) -> usize {
    let used = slots
        .last()
        .map(|slot| slot.offset + slot.bytes)
        .unwrap_or(if sret { WORD } else { 0 });
    used.div_ceil(16) * 16
}

/// Offset from `%ebp` at which the callee finds the argument area. The saved `%ebp`
/// is at `0(%ebp)` and the return address at `4(%ebp)`.
pub(super) const INCOMING_BASE: i32 = 8;
