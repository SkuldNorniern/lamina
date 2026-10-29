//! The two Thumb encodings this backend emits, and the places they differ.
//!
//! Thumb-1 (ARMv6-M) has no `IT`, no `movw`/`movt`, no hardware divide, no `clz`, and
//! three-operand forms only for `add`, `sub` and the immediate shifts. Thumb-2
//! (ARMv7-M) has all of them. Every sequence below is written so that the Thumb-1 form
//! also assembles on ARMv7-M, and only the places where that would be wrong or wasteful
//! branch on the ISA.

use std::io::Write;

use crate::{error::LaminaError, mir_codegen::arm::invert_condition};

use super::operand::unsupported;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Isa {
    /// ARMv6-M: Thumb-1 with the ARMv6 extensions.
    Thumb1,
    /// ARMv7-M: Thumb-2.
    Thumb2,
}

impl Isa {
    pub(crate) const fn arch_name(self) -> &'static str {
        match self {
            Self::Thumb1 => "armv6-m",
            Self::Thumb2 => "armv7-m",
        }
    }

    pub(crate) const fn target_name(self) -> &'static str {
        match self {
            Self::Thumb1 => "thumbv6m",
            Self::Thumb2 => "thumbv7m",
        }
    }

    pub(crate) const fn has_hardware_divide(self) -> bool {
        matches!(self, Self::Thumb2)
    }

    /// Reach of the `#imm` in `ldr`/`str` against the stack pointer.
    pub(crate) const fn sp_offset_limit(self) -> i32 {
        match self {
            Self::Thumb1 => 1020,
            Self::Thumb2 => 4092,
        }
    }
}

/// The three-operand integer operations the backend asks for.
#[derive(Clone, Copy)]
pub(crate) enum Alu {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

impl Alu {
    const fn thumb2(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::And => "and",
            Self::Or => "orr",
            Self::Xor => "eor",
            Self::Shl => "lsl",
            Self::LShr => "lsr",
            Self::AShr => "asr",
        }
    }

    /// The two-operand Thumb-1 mnemonic, for the operations that have no
    /// three-operand encoding.
    const fn thumb1_two_operand(self) -> Option<&'static str> {
        match self {
            Self::Add | Self::Sub => None,
            Self::Mul => Some("muls"),
            Self::And => Some("ands"),
            Self::Or => Some("orrs"),
            Self::Xor => Some("eors"),
            Self::Shl => Some("lsls"),
            Self::LShr => Some("lsrs"),
            Self::AShr => Some("asrs"),
        }
    }
}

/// `dest = lhs <op> rhs`. `dest` may alias `lhs`; it must not alias `rhs` on Thumb-1,
/// which the callers honour by taking three distinct scratch registers.
pub(crate) fn alu<W: Write>(
    w: &mut W,
    isa: Isa,
    op: Alu,
    dest: &str,
    lhs: &str,
    rhs: &str,
) -> Result<(), LaminaError> {
    match isa {
        Isa::Thumb2 => writeln!(w, "    {} {dest}, {lhs}, {rhs}", op.thumb2())?,
        Isa::Thumb1 => match op.thumb1_two_operand() {
            None => writeln!(
                w,
                "    {}s {dest}, {lhs}, {rhs}",
                if matches!(op, Alu::Add) { "add" } else { "sub" }
            )?,
            Some(mnemonic) => {
                if dest == rhs {
                    return Err(unsupported(
                        "thumbv6m two-operand form needs a destination distinct from the right \
                         operand"
                            .to_string(),
                    ));
                }
                if dest != lhs {
                    writeln!(w, "    mov {dest}, {lhs}")?;
                }
                if matches!(op, Alu::Mul) {
                    writeln!(w, "    muls {dest}, {rhs}, {dest}")?;
                } else {
                    writeln!(w, "    {mnemonic} {dest}, {rhs}")?;
                }
            }
        },
    }
    Ok(())
}

/// Materializes a full 32-bit word. Thumb-2 has `movw`/`movt`; Thumb-1 has neither, and
/// a PC-relative literal only reaches 1020 bytes forward, so the word is built from an
/// eight-bit field and shifts instead. Nothing here reads or writes the flags that a
/// caller cares about: on Thumb-1 the sequence sets them, so it is always emitted
/// before the `cmp` it precedes.
pub(crate) fn mov_imm<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    value: u32,
) -> Result<(), LaminaError> {
    if isa == Isa::Thumb2 {
        writeln!(w, "    movw {dest}, #{}", value & 0xffff)?;
        if value >> 16 != 0 {
            writeln!(w, "    movt {dest}, #{}", value >> 16)?;
        }
        return Ok(());
    }
    if value <= 0xff {
        writeln!(w, "    movs {dest}, #{value}")?;
        return Ok(());
    }
    let shift = value.trailing_zeros();
    if value >> shift <= 0xff {
        writeln!(w, "    movs {dest}, #{}", value >> shift)?;
        writeln!(w, "    lsls {dest}, {dest}, #{shift}")?;
        return Ok(());
    }
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|b| *b != 0).unwrap_or(3);
    writeln!(w, "    movs {dest}, #{}", bytes[first])?;
    for byte in &bytes[first + 1..] {
        writeln!(w, "    lsls {dest}, {dest}, #8")?;
        if *byte != 0 {
            writeln!(w, "    adds {dest}, #{byte}")?;
        }
    }
    Ok(())
}

/// `dest = dest & mask`, with `temp` distinct from `dest`.
pub(crate) fn and_imm<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    mask: u32,
    temp: &str,
) -> Result<(), LaminaError> {
    match isa {
        Isa::Thumb2 => writeln!(w, "    and {dest}, {dest}, #{mask}")?,
        Isa::Thumb1 => {
            mov_imm(w, isa, temp, mask)?;
            writeln!(w, "    ands {dest}, {temp}")?;
        }
    }
    Ok(())
}

/// `dest = value - source`, with `dest` distinct from `source`.
pub(crate) fn reverse_subtract<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    source: &str,
    value: u32,
) -> Result<(), LaminaError> {
    mov_imm(w, isa, dest, value)?;
    writeln!(w, "    subs {dest}, {dest}, {source}")?;
    Ok(())
}

/// Zero in `dest`, emitted before the comparison whose flags [`condition_set`] reads.
/// Thumb-1 has no encoding of `mov` immediate that leaves the flags alone, so the zero
/// cannot come after the `cmp`.
pub(crate) fn clear_before_compare<W: Write>(w: &mut W, dest: &str) -> Result<(), LaminaError> {
    writeln!(w, "    movs {dest}, #0")?;
    Ok(())
}

/// Sets `dest` to one when `cond` holds, having been zeroed by [`clear_before_compare`]
/// before the comparison.
///
/// On Thumb-2 this is the backend's only `IT` block, and it is one instruction long by
/// construction: the block and the instruction it guards are written together from the
/// same `cond`, so the condition pattern cannot drift from the instruction that follows
/// it. On Thumb-1, where there is no `IT`, the move is skipped by a branch over exactly
/// one instruction, which is always inside the +-256 byte reach of a Thumb-1
/// conditional branch.
pub(crate) fn condition_set<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    cond: &str,
) -> Result<(), LaminaError> {
    // Thumb-1 has no encoding of `mov` immediate that leaves the flags alone, so the
    // Thumb-1 arm spells it `movs`; the flags are dead by then either way.
    conditional(w, isa, cond, "mov", "movs", &format!("{dest}, #1"))
}

/// `dest = source` when `cond` holds, under the same rules as [`condition_set`].
pub(crate) fn condition_move<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    source: &str,
    cond: &str,
) -> Result<(), LaminaError> {
    conditional(w, isa, cond, "mov", "mov", &format!("{dest}, {source}"))
}

/// Emits exactly one conditional instruction, and on Thumb-2 the `IT` that guards it.
/// The condition is appended to the mnemonic here, from the same `cond` that wrote the
/// `IT`, so the block is one instruction long and its pattern always matches.
fn conditional<W: Write>(
    w: &mut W,
    isa: Isa,
    cond: &str,
    thumb2: &str,
    thumb1: &str,
    operands: &str,
) -> Result<(), LaminaError> {
    match isa {
        Isa::Thumb2 => {
            writeln!(w, "    it {cond}")?;
            writeln!(w, "    {thumb2}{cond} {operands}")?;
        }
        Isa::Thumb1 => {
            let skip = invert_condition(cond).ok_or_else(|| {
                unsupported(format!("thumbv6m has no inverse of the condition `{cond}`"))
            })?;
            writeln!(w, "    b{skip} 7f")?;
            writeln!(w, "    {thumb1} {operands}")?;
            writeln!(w, "7:")?;
        }
    }
    Ok(())
}
