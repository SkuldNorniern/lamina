//! Scratch registers, the frame layout and operand access for the Cortex-M backend.
//!
//! The frame is addressed from the stack pointer, which does not move once the
//! prologue has set it: the outgoing argument area of the widest call the function
//! makes is reserved there as well, so no call site has to adjust `sp`. That is what
//! makes Thumb-1 workable, because `ldr`/`str` against `sp` reach 1020 bytes while
//! `ldr`/`str` against a general register reach 124.

use std::io::Write;

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, Function, Immediate, MirType, Operand, Register, ScalarType, index_scale_shift,
    },
    mir_codegen::CodegenError,
};
use lamina_codegen::aarch64::FrameMap;

use super::isa::{Alu, Isa, alu, mov_imm};

/// Registers the backend evaluates expressions in. All four are callee-saved, so a
/// value held in one survives a call made from the same instruction's lowering, and
/// all four are low registers, which Thumb-1 data processing requires.
pub(crate) const SCRATCH: [&str; 4] = ["r4", "r5", "r6", "r7"];

/// What the prologue pushes, which is also the distance from the frame's top to the
/// caller's first stack argument.
pub(crate) const SAVED_LIST: &str = "{r4, r5, r6, r7, lr}";
pub(crate) const SAVED_BYTES: i32 = 20;

#[derive(Default)]
pub(crate) struct Scratch {
    used: [bool; SCRATCH.len()],
}

impl Scratch {
    pub(crate) fn alloc(&mut self) -> Result<&'static str, LaminaError> {
        let index = self
            .used
            .iter()
            .position(|used| !used)
            .ok_or_else(exhausted)?;
        self.used[index] = true;
        Ok(SCRATCH[index])
    }

    pub(crate) fn free(&mut self, register: &str) {
        if let Some(index) = SCRATCH.iter().position(|name| *name == register) {
            self.used[index] = false;
        }
    }
}

fn exhausted() -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(
        "cortex-m scratch registers exhausted".to_string(),
    ))
}

pub(crate) fn unsupported(message: String) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message))
}

/// Where everything sits relative to the stack pointer the prologue leaves behind.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    /// Bytes reserved at `[sp, #0]` for the widest call's stack arguments.
    pub(crate) outgoing: i32,
    /// Bytes above that holding the virtual registers' slots.
    pub(crate) locals: i32,
    /// Everything the prologue subtracts, including the word of padding that keeps the
    /// stack pointer eight-byte aligned across the five-register push.
    pub(crate) total: i32,
}

impl Layout {
    pub(crate) fn new(outgoing: usize, locals: i32) -> Self {
        let outgoing = outgoing.div_ceil(8) as i32 * 8;
        let locals = (locals.max(0) as usize).div_ceil(8) as i32 * 8;
        // The push leaves `sp` four bytes off an eight-byte boundary, so the body of
        // the frame carries the fourth word that puts it back.
        Self {
            outgoing,
            locals,
            total: outgoing + locals + 4,
        }
    }

    /// Stack-pointer offset of the slot a `FrameMap` places at `offset` below the top
    /// of the locals area.
    pub(crate) fn slot(&self, offset: i32) -> i32 {
        self.outgoing + self.locals + offset
    }

    /// Stack-pointer offset of the caller's stack argument area.
    pub(crate) fn incoming(&self) -> i32 {
        self.total + SAVED_BYTES
    }
}

/// Strongest alignment any slot asks for. The frame stands on eight bytes, which is
/// what AAPCS guarantees at a public interface and all the stack pointer can be
/// trusted to hold on entry, so a stronger request is refused rather than silently
/// under-aligned.
pub(crate) fn check_frame_alignment(name: &str, func: &Function) -> Result<(), LaminaError> {
    if let Some(align) = func.stack_alignments.values().copied().max()
        && align > 8
    {
        return Err(unsupported(format!(
            "cortex-m frames are eight-byte aligned, which is all AAPCS guarantees of the \
             incoming stack pointer; @{name} asks for a {align}-byte aligned stack slot"
        )));
    }
    Ok(())
}

/// `dest = sp + offset`.
pub(crate) fn sp_address<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    if (0..=isa.sp_offset_limit()).contains(&offset) && offset % 4 == 0 {
        writeln!(w, "    add {dest}, sp, #{offset}")?;
        return Ok(());
    }
    let magnitude = u32::try_from(offset)
        .map_err(|_| unsupported("cortex-m frame offset is negative".to_string()))?;
    mov_imm(w, isa, dest, magnitude)?;
    // ADD (SP plus register) wants the same register as destination and operand.
    writeln!(w, "    add {dest}, sp, {dest}")?;
    Ok(())
}

pub(crate) fn frame_load<W: Write>(
    w: &mut W,
    isa: Isa,
    dest: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    if (0..=isa.sp_offset_limit()).contains(&offset) && offset % 4 == 0 {
        writeln!(w, "    ldr {dest}, [sp, #{offset}]")?;
        return Ok(());
    }
    sp_address(w, isa, dest, offset)?;
    writeln!(w, "    ldr {dest}, [{dest}]")?;
    Ok(())
}

pub(crate) fn frame_store<W: Write>(
    w: &mut W,
    isa: Isa,
    source: &str,
    offset: i32,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    if (0..=isa.sp_offset_limit()).contains(&offset) && offset % 4 == 0 {
        writeln!(w, "    str {source}, [sp, #{offset}]")?;
        return Ok(());
    }
    let address = scratch.alloc()?;
    sp_address(w, isa, address, offset)?;
    writeln!(w, "    str {source}, [{address}]")?;
    scratch.free(address);
    Ok(())
}

pub(crate) fn imm_bits32(immediate: &Immediate) -> u32 {
    match immediate {
        Immediate::I8(v) => *v as i32 as u32,
        Immediate::I16(v) => *v as i32 as u32,
        Immediate::I32(v) => *v as u32,
        Immediate::I64(v) => *v as u32,
        Immediate::F32(v) => v.to_bits(),
        Immediate::F64(v) => v.to_bits() as u32,
    }
}

pub(crate) struct Access<'a> {
    pub(crate) isa: Isa,
    pub(crate) frame: &'a FrameMap,
    pub(crate) layout: Layout,
}

impl Access<'_> {
    pub(crate) fn slot_of(&self, register: &Register) -> Option<i32> {
        self.frame
            .slot_of(register)
            .map(|off| self.layout.slot(off))
    }

    pub(crate) fn load_reg<W: Write>(
        &self,
        w: &mut W,
        register: &Register,
        dest: &str,
    ) -> Result<(), LaminaError> {
        match register {
            Register::Virtual(_) => match self.slot_of(register) {
                Some(offset) => frame_load(w, self.isa, dest, offset),
                None => Err(unsupported(format!(
                    "cortex-m operand {register} has no frame slot"
                ))),
            },
            Register::Physical(physical) => {
                if physical.name != dest {
                    writeln!(w, "    mov {dest}, {}", physical.name)?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn materialize<W: Write>(
        &self,
        w: &mut W,
        operand: &Operand,
        dest: &str,
    ) -> Result<(), LaminaError> {
        match operand {
            Operand::Immediate(immediate) => mov_imm(w, self.isa, dest, imm_bits32(immediate)),
            Operand::Register(register) => self.load_reg(w, register, dest),
        }
    }

    pub(crate) fn store_result<W: Write>(
        &self,
        w: &mut W,
        dst: &Register,
        source: &str,
        scratch: &mut Scratch,
    ) -> Result<(), LaminaError> {
        match dst {
            Register::Virtual(_) => match self.slot_of(dst) {
                Some(offset) => frame_store(w, self.isa, source, offset, scratch),
                None => Err(unsupported(format!(
                    "cortex-m result {dst} has no frame slot"
                ))),
            },
            Register::Physical(physical) => {
                if physical.name != source {
                    writeln!(w, "    mov {}, {source}", physical.name)?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn frame_address<W: Write>(
        &self,
        w: &mut W,
        dest: &str,
        offset: i32,
    ) -> Result<(), LaminaError> {
        sp_address(w, self.isa, dest, offset)
    }

    pub(crate) fn add_offset<W: Write>(
        &self,
        w: &mut W,
        dest: &str,
        offset: i64,
        scratch: &mut Scratch,
    ) -> Result<(), LaminaError> {
        if offset == 0 {
            return Ok(());
        }
        let magnitude = offset.unsigned_abs();
        if magnitude <= 255 {
            let op = if offset > 0 { "adds" } else { "subs" };
            writeln!(w, "    {op} {dest}, #{magnitude}")?;
            return Ok(());
        }
        let temp = scratch.alloc()?;
        mov_imm(w, self.isa, temp, magnitude as u32)?;
        let op = if offset > 0 { Alu::Add } else { Alu::Sub };
        alu(w, self.isa, op, dest, dest, temp)?;
        scratch.free(temp);
        Ok(())
    }

    pub(crate) fn materialize_address<W: Write>(
        &self,
        w: &mut W,
        addr: &AddressMode,
        dest: &str,
        scratch: &mut Scratch,
    ) -> Result<(), LaminaError> {
        match addr {
            AddressMode::BaseOffset { base, offset, .. } => {
                self.load_reg(w, base, dest)?;
                self.add_offset(w, dest, i64::from(*offset), scratch)?;
            }
            AddressMode::BaseIndexScale {
                base,
                index,
                scale,
                offset,
                ..
            } => {
                self.load_reg(w, base, dest)?;
                let idx = scratch.alloc()?;
                self.load_reg(w, index, idx)?;
                let shift = index_scale_shift(*scale).ok_or_else(|| {
                    LaminaError::ValidationError(format!(
                        "address scale {scale} is not 1, 2, 4 or 8"
                    ))
                })?;
                if shift > 0 {
                    writeln!(w, "    lsls {idx}, {idx}, #{shift}")?;
                }
                alu(w, self.isa, Alu::Add, dest, dest, idx)?;
                scratch.free(idx);
                self.add_offset(w, dest, i64::from(*offset), scratch)?;
            }
        }
        Ok(())
    }
}

/// Sign- or zero-extends a value narrower than a word in place.
pub(crate) fn extend_integer<W: Write>(
    w: &mut W,
    register: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    let instruction = match ty {
        MirType::Scalar(ScalarType::I8) if signed => "sxtb",
        MirType::Scalar(ScalarType::I8) => "uxtb",
        MirType::Scalar(ScalarType::I16) if signed => "sxth",
        MirType::Scalar(ScalarType::I16) => "uxth",
        _ => return Ok(()),
    };
    writeln!(w, "    {instruction} {register}, {register}")?;
    Ok(())
}

/// Moves `bytes` a word at a time from the address in `source` to the address in
/// `destination`, through a scratch register.
pub(crate) fn copy_through_pointer<W: Write>(
    w: &mut W,
    isa: Isa,
    source: &str,
    destination: &str,
    bytes: usize,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let reach = if isa == Isa::Thumb1 { 124 } else { 4092 };
    let word = scratch.alloc()?;
    for index in 0..bytes.div_ceil(4) {
        let at = index * 4;
        if at > reach {
            scratch.free(word);
            return Err(unsupported(format!(
                "{} aggregate copy of {bytes} bytes reaches past the {reach}-byte offset a Thumb \
                 load encodes",
                isa.target_name()
            )));
        }
        writeln!(w, "    ldr {word}, [{source}, #{at}]")?;
        writeln!(w, "    str {word}, [{destination}, #{at}]")?;
    }
    scratch.free(word);
    Ok(())
}
