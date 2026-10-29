//! MIPS64 N64 ABI: register names, frame layout and argument places.

use crate::mir::{Aggregate, MirType};
use std::io::{self, Write};

/// Argument doublewords `$4`-`$11`. An argument takes the register matching its first
/// doubleword, so the counter is positional rather than per-class.
pub(super) const ARG_GPRS: [&str; 8] = ["$4", "$5", "$6", "$7", "$8", "$9", "$10", "$11"];
/// `$f12`-`$f19`, positional alongside the integer registers.
pub(super) const ARG_FPRS: [&str; 8] = [
    "$f12", "$f13", "$f14", "$f15", "$f16", "$f17", "$f18", "$f19",
];

pub(super) const A0: &str = "$2";
pub(super) const A1: &str = "$3";
pub(super) const T0: &str = "$12";
pub(super) const T1: &str = "$13";
pub(super) const T2: &str = "$14";
pub(super) const T3: &str = "$15";
/// Holds the address of the function being called; a position-independent callee
/// derives its own global pointer from it.
pub(super) const T9: &str = "$25";
pub(super) const FA0: &str = "$f0";
pub(super) const FA1: &str = "$f2";
pub(super) const FA2: &str = "$f4";
pub(super) const FP: &str = "$30";
pub(super) const SP: &str = "$29";
pub(super) const RA: &str = "$31";
pub(super) const ZERO: &str = "$0";

pub(super) const WORD: i32 = 8;

/// Saved `$ra` and `$fp` sit at the top of the frame, directly below the stack pointer
/// the function was entered with.
pub(super) const SAVED: i32 = 16;

/// A trap, which is what `unreachable` and a failed division check lower to.
pub(super) const TRAP: &str = "    teq $0, $0";

fn fits_i16(value: i64) -> bool {
    i64::from(i16::MIN) <= value && value <= i64::from(i16::MAX)
}

/// Offset of local slot `index` from the frame pointer. Locals sit below the two
/// saved registers the frame pointer points at.
pub(super) fn slot_offset(index: usize) -> i32 {
    -((index as i32 + 1) * WORD)
}

pub(super) fn align_up(bytes: usize, to: usize) -> usize {
    bytes.div_ceil(to) * to
}

/// The whole frame: the outgoing argument area at the bottom, the locals above it,
/// one doubleword the byte intrinsics buffer in, and the two saved registers at the
/// top. Kept to the sixteen-byte alignment N64 asks of the stack pointer.
pub(super) fn frame_bytes(locals: usize, outgoing: usize) -> usize {
    align_up(
        align_up(outgoing, WORD as usize)
            + align_up(locals, WORD as usize)
            + WORD as usize
            + SAVED as usize,
        16,
    )
}

/// Offset from the frame pointer of that buffered doubleword.
pub(super) fn io_scratch_offset(locals: usize) -> i32 {
    -(align_up(locals, WORD as usize) as i32 + WORD)
}

pub(super) fn prologue<W: Write>(writer: &mut W, locals: usize, outgoing: usize) -> io::Result<()> {
    writeln!(writer, "    move {T0}, {SP}")?;
    adjust_sp(writer, -(frame_bytes(locals, outgoing) as i64))?;
    writeln!(writer, "    sd {RA}, -8({T0})")?;
    writeln!(writer, "    sd {FP}, -16({T0})")?;
    writeln!(writer, "    daddiu {FP}, {T0}, -{SAVED}")
}

/// Restores `$ra` and `$fp` and puts the stack pointer back where the function was
/// entered with it. `tail` jumps instead of returning.
pub(super) fn epilogue<W: Write>(writer: &mut W, tail: Option<&str>) -> io::Result<()> {
    writeln!(writer, "    daddiu {SP}, {FP}, {SAVED}")?;
    writeln!(writer, "    ld {RA}, 8({FP})")?;
    writeln!(writer, "    ld {FP}, 0({FP})")?;
    match tail {
        Some(target) => writeln!(writer, "    j {target}"),
        None => writeln!(writer, "    jr {RA}"),
    }?;
    writeln!(writer, "    nop")
}

pub(super) fn adjust_sp<W: Write>(writer: &mut W, delta: i64) -> io::Result<()> {
    if delta == 0 {
        Ok(())
    } else if fits_i16(delta) {
        writeln!(writer, "    daddiu {SP}, {SP}, {delta}")
    } else {
        load_imm(writer, T1, delta)?;
        writeln!(writer, "    daddu {SP}, {SP}, {T1}")
    }
}

/// Emits `{mnemonic} {reg}, {offset}({base})`, building the address in `scratch` when
/// the displacement does not fit the sixteen-bit field. `scratch` must differ from
/// `base` and from `reg`.
pub(super) fn memory<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    base: &str,
    offset: i64,
    scratch: &str,
) -> io::Result<()> {
    if fits_i16(offset) {
        writeln!(writer, "    {mnemonic} {reg}, {offset}({base})")
    } else {
        load_imm(writer, scratch, offset)?;
        writeln!(writer, "    daddu {scratch}, {scratch}, {base}")?;
        writeln!(writer, "    {mnemonic} {reg}, 0({scratch})")
    }
}

pub(super) fn fp_load<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    offset: i32,
    scratch: &str,
) -> io::Result<()> {
    memory(writer, mnemonic, reg, FP, i64::from(offset), scratch)
}

pub(super) fn fp_store<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    offset: i32,
    scratch: &str,
) -> io::Result<()> {
    memory(writer, mnemonic, reg, FP, i64::from(offset), scratch)
}

/// Materialises `{base} + offset` in `dst`.
pub(super) fn address<W: Write>(
    writer: &mut W,
    dst: &str,
    base: &str,
    offset: i64,
) -> io::Result<()> {
    if fits_i16(offset) {
        writeln!(writer, "    daddiu {dst}, {base}, {offset}")
    } else {
        load_imm(writer, dst, offset)?;
        writeln!(writer, "    daddu {dst}, {dst}, {base}")
    }
}

pub(super) fn load_imm<W: Write>(writer: &mut W, reg: &str, value: i64) -> io::Result<()> {
    writeln!(writer, "    dli {reg}, {value}")
}

/// Puts the address of `symbol` in `dst`, through `scratch`.
///
/// A static MIPS64 executable loads above the four-gigabyte line, so the two-instruction
/// `lui`/`daddiu` pair a 32-bit address would take truncates. The four relocations
/// build the whole doubleword.
pub(super) fn symbol_address<W: Write>(
    writer: &mut W,
    dst: &str,
    scratch: &str,
    symbol: &str,
) -> io::Result<()> {
    writeln!(writer, "    lui {dst}, %highest({symbol})")?;
    writeln!(writer, "    lui {scratch}, %hi({symbol})")?;
    writeln!(writer, "    daddiu {dst}, {dst}, %higher({symbol})")?;
    writeln!(writer, "    daddiu {scratch}, {scratch}, %lo({symbol})")?;
    writeln!(writer, "    dsll32 {dst}, {dst}, 0")?;
    writeln!(writer, "    daddu {dst}, {dst}, {scratch}")
}

/// What one argument is: a scalar of some type, or an aggregate of some size.
#[derive(Clone, Copy, Debug)]
pub(super) struct ArgKind {
    pub(super) ty: MirType,
    pub(super) aggregate: Option<Aggregate>,
}

impl ArgKind {
    pub(super) fn scalar(ty: MirType) -> Self {
        Self {
            ty,
            aggregate: None,
        }
    }

    pub(super) fn words(&self) -> usize {
        self.aggregate.map_or(1, |a| a.words())
    }

    fn float(&self) -> bool {
        self.aggregate.is_none() && self.ty.is_float()
    }
}

/// Where one argument travels. N64 numbers argument doublewords across both register
/// files, so an argument's place follows from how many doublewords came before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArgPlace {
    Gpr(usize),
    Fpr(usize),
    /// An aggregate whose doublewords all fit in registers, starting at `first`.
    Words {
        first: usize,
        words: usize,
    },
    /// An aggregate that starts in registers and runs past `$11`.
    Split {
        first: usize,
        words: usize,
    },
    /// Doubleword slot number in the outgoing argument area.
    Stack(usize),
    /// An aggregate wholly in the outgoing area, starting at slot `slot`.
    StackWords {
        slot: usize,
        words: usize,
    },
}

/// Places arguments by doubleword. A float takes the floating-point register matching
/// its doubleword, everything else the general one; an aggregate takes as many
/// consecutive doublewords as it occupies and is split when it crosses `$11`.
pub(super) fn classify_kinds(kinds: &[ArgKind], first_gpr: usize) -> Vec<ArgPlace> {
    let banks = ARG_GPRS.len();
    let mut dwords = first_gpr;
    kinds
        .iter()
        .map(|kind| {
            let words = kind.words();
            let place = match kind.aggregate {
                Some(_) if dwords >= banks => ArgPlace::StackWords {
                    slot: dwords - banks,
                    words,
                },
                Some(_) if dwords + words <= banks => ArgPlace::Words {
                    first: dwords,
                    words,
                },
                Some(_) => ArgPlace::Split {
                    first: dwords,
                    words,
                },
                None if dwords >= banks => ArgPlace::Stack(dwords - banks),
                None if kind.float() => ArgPlace::Fpr(dwords),
                None => ArgPlace::Gpr(dwords),
            };
            dwords += words;
            place
        })
        .collect()
}

/// Doublewords of outgoing area the call needs past the eight that travel in
/// registers.
pub(super) fn stack_words(kinds: &[ArgKind], first_gpr: usize) -> usize {
    let total: usize = first_gpr + kinds.iter().map(ArgKind::words).sum::<usize>();
    total.saturating_sub(ARG_GPRS.len())
}

pub(super) fn call_area(kinds: &[ArgKind], first_gpr: usize) -> usize {
    stack_words(kinds, first_gpr) * WORD as usize
}

/// Where an incoming stack argument sits, relative to the frame pointer: at the
/// bottom of the caller's outgoing area, which starts where this frame does.
pub(super) fn incoming_stack_offset(index: usize) -> i32 {
    SAVED + index as i32 * WORD
}

/// Where an outgoing stack argument sits, relative to the stack pointer.
pub(super) fn outgoing_stack_offset(index: usize) -> i64 {
    index as i64 * i64::from(WORD)
}
