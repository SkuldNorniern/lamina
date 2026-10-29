//! s390x ELF ABI: register names, frame layout and argument places.

use crate::mir::{Aggregate, MirType};
use std::io::{self, Write};

pub(super) const ARG_GPRS: [&str; 5] = ["%r2", "%r3", "%r4", "%r5", "%r6"];
pub(super) const ARG_FPRS: [&str; 4] = ["%f0", "%f2", "%f4", "%f6"];

pub(super) const A0: &str = "%r2";
pub(super) const A1: &str = "%r3";
pub(super) const T0: &str = "%r1";
pub(super) const T1: &str = "%r12";
pub(super) const T2: &str = "%r13";
pub(super) const FA0: &str = "%f0";
pub(super) const FA1: &str = "%f2";
pub(super) const FP: &str = "%r11";
/// Holds the stack pointer the function was entered with, when the frame pointer has
/// been aligned away from it.
pub(super) const ENTRY: &str = "%r9";
pub(super) const SP: &str = "%r15";
/// Even half of the even-odd pair the divide instructions require.
pub(super) const DIV_HI: &str = "%r4";
/// Odd half: dividend before the divide, quotient after.
pub(super) const DIV_LO: &str = "%r5";

pub(super) const WORD: i32 = 8;

/// Bytes at the bottom of every frame that a callee saves registers into.
pub(super) const SAVE_AREA: i32 = 160;

/// Offset of the saved `r6` within the register save area; `r6`..`r15` follow.
const SAVED_GPRS_AT: i32 = 48;

/// Operation exception: two bytes into a four-byte branch. What `__builtin_trap`
/// lowers to, there being no trap instruction.
pub(super) const TRAP: &str = "    j .+2";

/// Long-displacement forms take a signed 20-bit displacement.
pub(super) fn fits_disp20(value: i64) -> bool {
    (-(1 << 19)..(1 << 19)).contains(&value)
}

fn fits_i16(value: i64) -> bool {
    i64::from(i16::MIN) <= value && value <= i64::from(i16::MAX)
}

fn fits_i32(value: i64) -> bool {
    i64::from(i32::MIN) <= value && value <= i64::from(i32::MAX)
}

/// Offset of local slot `index` from the frame pointer, which holds the stack pointer
/// the function was entered with. Locals sit directly below it.
pub(super) fn slot_offset(index: usize) -> i32 {
    -((index as i32 + 1) * WORD)
}

pub(super) fn align_up(bytes: usize) -> usize {
    bytes.div_ceil(WORD as usize) * WORD as usize
}

/// Sixteen bytes of headroom cover the frame pointer being rounded down.
pub(super) const ALIGN_SLACK: usize = 16;

/// The whole frame: the callee save area, the outgoing argument area above it, the
/// locals above that, and one doubleword at the top the byte intrinsics buffer in.
pub(super) fn frame_bytes(locals: usize, outgoing: usize) -> usize {
    SAVE_AREA as usize + align_up(outgoing) + align_up(locals) + WORD as usize
}

/// Offset from the frame pointer of that buffered doubleword.
pub(super) fn io_scratch_offset(locals: usize) -> i32 {
    -(align_up(locals) as i32 + WORD)
}

/// The stack pointer is only doubleword aligned on entry, so a frame holding a slot
/// that wants more rounds its frame pointer down and keeps the entry value elsewhere.
pub(super) fn entry_base(aligned: bool) -> &'static str {
    if aligned { ENTRY } else { FP }
}

pub(super) fn prologue<W: Write>(
    writer: &mut W,
    locals: usize,
    outgoing: usize,
    aligned: bool,
) -> io::Result<()> {
    writeln!(writer, "    stmg %r6, %r15, {SAVED_GPRS_AT}({SP})")?;
    if aligned {
        writeln!(writer, "    lgr {ENTRY}, {SP}")?;
    }
    writeln!(writer, "    lgr {FP}, {SP}")?;
    if aligned {
        writeln!(writer, "    nill {FP}, 65520")?;
        writeln!(writer, "    lgr {SP}, {FP}")?;
    }
    let frame = frame_bytes(locals, outgoing) as i64;
    adjust_sp(writer, -frame)
}

/// Restores every saved register, `r15` among them, which leaves the stack pointer at
/// the value the function was entered with.
pub(super) fn epilogue<W: Write>(
    writer: &mut W,
    aligned: bool,
    tail: Option<&str>,
) -> io::Result<()> {
    writeln!(
        writer,
        "    lmg %r6, %r15, {SAVED_GPRS_AT}({})",
        entry_base(aligned)
    )?;
    match tail {
        Some(target) => writeln!(writer, "    jg {target}"),
        None => writeln!(writer, "    br %r14"),
    }
}

pub(super) fn adjust_sp<W: Write>(writer: &mut W, delta: i64) -> io::Result<()> {
    if delta == 0 {
        Ok(())
    } else if fits_i16(delta) {
        writeln!(writer, "    aghi {SP}, {delta}")
    } else if fits_i32(delta) {
        writeln!(writer, "    agfi {SP}, {delta}")
    } else {
        load_imm(writer, T0, delta)?;
        writeln!(writer, "    agr {SP}, {T0}")
    }
}

/// Emits `{mnemonic} {reg}, {offset}({base})`, building the address in `scratch` when
/// the displacement does not fit. `scratch` must differ from `base`.
pub(super) fn memory<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    base: &str,
    offset: i64,
    scratch: &str,
) -> io::Result<()> {
    if fits_disp20(offset) {
        writeln!(writer, "    {mnemonic} {reg}, {offset}({base})")
    } else {
        load_imm(writer, scratch, offset)?;
        writeln!(writer, "    agr {scratch}, {base}")?;
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
    if fits_disp20(offset) {
        writeln!(writer, "    lay {dst}, {offset}({base})")
    } else {
        load_imm(writer, dst, offset)?;
        writeln!(writer, "    agr {dst}, {base}")
    }
}

/// Loads a 64-bit constant into `reg`, using only `reg`.
///
/// `lghi` sign-extends 16 bits and `lgfi` 32; wider needs `llilf` for bits 32..=63 and
/// `iihf` for bits 0..=31.
pub(super) fn load_imm<W: Write>(writer: &mut W, reg: &str, value: i64) -> io::Result<()> {
    if fits_i16(value) {
        writeln!(writer, "    lghi {reg}, {value}")
    } else if fits_i32(value) {
        writeln!(writer, "    lgfi {reg}, {value}")
    } else {
        writeln!(writer, "    llilf {reg}, {}", (value as u64) & 0xffff_ffff)?;
        writeln!(writer, "    iihf {reg}, {}", (value as u64) >> 32)
    }
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

    /// An aggregate wider than one doubleword travels as the address of a copy the
    /// caller makes. The z/Architecture ABI keeps only a single word by value, unlike
    /// the two that `Aggregate::max_in_registers` assumes of every target.
    pub(super) fn by_reference(&self) -> bool {
        self.aggregate.is_some_and(|a| a.size > WORD as usize)
    }

    fn float(&self) -> bool {
        self.aggregate.is_none() && self.ty.is_float()
    }
}

/// Where one argument travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArgPlace {
    Gpr(usize),
    Fpr(usize),
    /// Eight-byte slot number in the outgoing argument area.
    Stack(usize),
}

/// Places arguments: floats take `f0`, `f2`, `f4`, `f6`, everything else — aggregates
/// among them, by value or by reference — takes one of `r2`-`r6`, and the rest go to
/// the stack in order. `first_gpr` is 1 when a hidden result pointer took `r2`.
pub(super) fn classify_kinds(kinds: &[ArgKind], first_gpr: usize) -> Vec<ArgPlace> {
    let (mut gprs, mut fprs, mut stack) = (first_gpr, 0, 0);
    kinds
        .iter()
        .map(|kind| {
            if kind.float() && fprs < ARG_FPRS.len() {
                fprs += 1;
                ArgPlace::Fpr(fprs - 1)
            } else if !kind.float() && gprs < ARG_GPRS.len() {
                gprs += 1;
                ArgPlace::Gpr(gprs - 1)
            } else {
                stack += 1;
                ArgPlace::Stack(stack - 1)
            }
        })
        .collect()
}

/// Bytes of outgoing area one call needs: its stack argument words, then a copy of
/// every aggregate that travels by reference.
pub(super) fn call_area(kinds: &[ArgKind], first_gpr: usize) -> usize {
    let places = classify_kinds(kinds, first_gpr);
    let stack = places
        .iter()
        .filter(|place| matches!(place, ArgPlace::Stack(_)))
        .count()
        * WORD as usize;
    let copies: usize = kinds
        .iter()
        .filter(|kind| kind.by_reference())
        .map(|kind| kind.aggregate.map_or(0, |a| a.words()) * WORD as usize)
        .sum();
    stack + copies
}

/// Where the copy of the `index`th by-reference aggregate starts, relative to the
/// stack pointer, after the stack argument words.
pub(super) fn copy_base(kinds: &[ArgKind], first_gpr: usize) -> i64 {
    let places = classify_kinds(kinds, first_gpr);
    let stack = places
        .iter()
        .filter(|place| matches!(place, ArgPlace::Stack(_)))
        .count();
    outgoing_stack_offset(stack)
}

/// Where an incoming stack argument sits, relative to the frame pointer: in the
/// caller's outgoing area, above its own save area.
pub(super) fn incoming_stack_offset(index: usize) -> i32 {
    SAVE_AREA + index as i32 * WORD
}

/// Where an outgoing stack argument sits, relative to the stack pointer.
pub(super) fn outgoing_stack_offset(index: usize) -> i64 {
    i64::from(SAVE_AREA) + index as i64 * i64::from(WORD)
}
