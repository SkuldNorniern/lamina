//! Register scratch, immediates and frame-slot access for the armv7 backend.

use std::io::Write;

use crate::{
    error::LaminaError,
    mir::{AddressMode, Immediate, MirType, Operand, Register, ScalarType, index_scale_shift},
    mir_codegen::CodegenError,
};
use lamina_codegen::aarch64::FrameMap;

/// Callee-saved registers the backend evaluates expressions in, so a value held in one
/// survives a call made from the same instruction's lowering.
pub(crate) const SCRATCH: [&str; 6] = ["r4", "r5", "r6", "r7", "r8", "r9"];

/// `ip`, which only ever holds an address or a materialized immediate and never a value.
pub(crate) const ADDR: &str = "r12";

pub(crate) const FP: &str = "r11";

/// Holds the stack pointer as it stood after the prologue's pushes: the frame pointer
/// is rounded down from there to the alignment the frame asks for, so it cannot serve
/// as the restore point or as the base of the caller's stack arguments.
pub(crate) const ENTRY_SP: &str = "r10";

/// Bytes the prologue pushes before it takes the frame pointer, which is also the
/// distance from the frame pointer to the caller's first stack argument.
pub(crate) const SAVED_BYTES: i32 = 104;

pub(crate) const F32_SCRATCH: [&str; 3] = ["s16", "s17", "s18"];
pub(crate) const F64_SCRATCH: [&str; 3] = ["d12", "d13", "d14"];

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
        "armv7 scratch registers exhausted".to_string(),
    ))
}

pub(crate) fn unsupported(message: String) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message))
}

/// Whether the value is an A32 modified immediate: an eight-bit field rotated right by
/// an even amount.
pub(crate) fn is_modified_immediate(value: u32) -> bool {
    (0..16).any(|rotation| value.rotate_left(2 * rotation) <= 0xff)
}

pub(crate) fn sp_adjust<W: Write>(w: &mut W, op: &str, bytes: usize) -> Result<(), LaminaError> {
    if bytes == 0 {
        return Ok(());
    }
    let bytes = u32::try_from(bytes)
        .map_err(|_| unsupported("armv7 outgoing argument area exceeds 4 GiB".to_string()))?;
    if is_modified_immediate(bytes) {
        writeln!(w, "    {op} sp, sp, #{bytes}")?;
    } else {
        mov_imm32(w, ADDR, bytes)?;
        writeln!(w, "    {op} sp, sp, {ADDR}")?;
    }
    Ok(())
}

/// Materializes a full word with `movw`/`movt`, which armv7 always has, rather than
/// asking whether the value happens to be an A32 modified immediate.
pub(crate) fn mov_imm32<W: Write>(w: &mut W, dest: &str, value: u32) -> Result<(), LaminaError> {
    writeln!(w, "    movw {dest}, #{}", value & 0xffff)?;
    if value >> 16 != 0 {
        writeln!(w, "    movt {dest}, #{}", value >> 16)?;
    }
    Ok(())
}

/// `dest = fp + offset`.
pub(crate) fn frame_address<W: Write>(
    w: &mut W,
    dest: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    if offset >= 0 && is_modified_immediate(offset as u32) {
        writeln!(w, "    add {dest}, {FP}, #{offset}")?;
    } else if offset < 0 && is_modified_immediate(offset.unsigned_abs()) {
        writeln!(w, "    sub {dest}, {FP}, #{}", offset.unsigned_abs())?;
    } else {
        mov_imm32(w, dest, offset as u32)?;
        writeln!(w, "    add {dest}, {FP}, {dest}")?;
    }
    Ok(())
}

/// `base`/`#offset` pair naming a frame slot, going through `ip` when the slot is past
/// the reach of the load or store `limit`.
fn frame_ref<W: Write>(w: &mut W, offset: i32, limit: i32) -> Result<String, LaminaError> {
    if offset.abs() <= limit {
        return Ok(format!("[{FP}, #{offset}]"));
    }
    frame_address(w, ADDR, offset)?;
    Ok(format!("[{ADDR}]"))
}

pub(crate) fn frame_load<W: Write>(w: &mut W, dest: &str, offset: i32) -> Result<(), LaminaError> {
    let at = frame_ref(w, offset, 4095)?;
    writeln!(w, "    ldr {dest}, {at}")?;
    Ok(())
}

pub(crate) fn frame_store<W: Write>(
    w: &mut W,
    source: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    let at = frame_ref(w, offset, 4095)?;
    writeln!(w, "    str {source}, {at}")?;
    Ok(())
}

pub(crate) fn frame_vload<W: Write>(w: &mut W, dest: &str, offset: i32) -> Result<(), LaminaError> {
    let at = frame_ref(w, offset, 1020)?;
    writeln!(w, "    vldr {dest}, {at}")?;
    Ok(())
}

pub(crate) fn frame_vstore<W: Write>(
    w: &mut W,
    source: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    let at = frame_ref(w, offset, 1020)?;
    writeln!(w, "    vstr {source}, {at}")?;
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

/// Low and high words of an immediate read as eight bytes.
pub(crate) fn imm_bits64(immediate: &Immediate) -> (u32, u32) {
    let bits = match immediate {
        Immediate::I8(v) => *v as i64 as u64,
        Immediate::I16(v) => *v as i64 as u64,
        Immediate::I32(v) => *v as i64 as u64,
        Immediate::I64(v) => *v as u64,
        Immediate::F32(v) => f64::from(*v).to_bits(),
        Immediate::F64(v) => v.to_bits(),
    };
    (bits as u32, (bits >> 32) as u32)
}

pub(crate) fn load_reg<W: Write>(
    w: &mut W,
    register: &Register,
    dest: &str,
    frame: &FrameMap,
) -> Result<(), LaminaError> {
    match register {
        Register::Virtual(_) => match frame.slot_of(register) {
            Some(offset) => frame_load(w, dest, offset)?,
            None => {
                return Err(unsupported(format!(
                    "armv7 operand {register} has no frame slot"
                )));
            }
        },
        Register::Physical(physical) => {
            if physical.name != dest {
                writeln!(w, "    mov {dest}, {}", physical.name)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn materialize<W: Write>(
    w: &mut W,
    operand: &Operand,
    dest: &str,
    frame: &FrameMap,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Immediate(immediate) => mov_imm32(w, dest, imm_bits32(immediate)),
        Operand::Register(register) => load_reg(w, register, dest, frame),
    }
}

/// Loads an operand into `dest`, a VFP single or double register.
pub(crate) fn materialize_float<W: Write>(
    w: &mut W,
    operand: &Operand,
    dest: &str,
    double: bool,
    frame: &FrameMap,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Immediate(immediate) if double => {
            let (low, high) = imm_bits64(immediate);
            let other = scratch.alloc()?;
            mov_imm32(w, ADDR, low)?;
            mov_imm32(w, other, high)?;
            writeln!(w, "    vmov {dest}, {ADDR}, {other}")?;
            scratch.free(other);
        }
        Operand::Immediate(immediate) => {
            let bits = match immediate {
                Immediate::F64(value) => (*value as f32).to_bits(),
                other => imm_bits32(other),
            };
            mov_imm32(w, ADDR, bits)?;
            writeln!(w, "    vmov {dest}, {ADDR}")?;
        }
        Operand::Register(register) => match frame.slot_of(register) {
            Some(offset) => frame_vload(w, dest, offset)?,
            None => {
                return Err(unsupported(format!(
                    "armv7 float operand {register} has no frame slot"
                )));
            }
        },
    }
    Ok(())
}

pub(crate) fn store_result<W: Write>(
    w: &mut W,
    dst: &Register,
    source: &str,
    frame: &FrameMap,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(_) => match frame.slot_of(dst) {
            Some(offset) => frame_store(w, source, offset)?,
            None => return Err(unsupported(format!("armv7 result {dst} has no frame slot"))),
        },
        Register::Physical(physical) => {
            if physical.name != source {
                writeln!(w, "    mov {}, {source}", physical.name)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn store_float_result<W: Write>(
    w: &mut W,
    dst: &Register,
    source: &str,
    frame: &FrameMap,
) -> Result<(), LaminaError> {
    match frame.slot_of(dst) {
        Some(offset) => frame_vstore(w, source, offset),
        None => Err(unsupported(format!(
            "armv7 float result {dst} has no frame slot"
        ))),
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

pub(crate) fn add_offset<W: Write>(w: &mut W, dest: &str, offset: i64) -> Result<(), LaminaError> {
    if offset == 0 {
        return Ok(());
    }
    let magnitude = offset.unsigned_abs();
    if magnitude <= u64::from(u32::MAX) && is_modified_immediate(magnitude as u32) {
        let op = if offset > 0 { "add" } else { "sub" };
        writeln!(w, "    {op} {dest}, {dest}, #{magnitude}")?;
    } else {
        mov_imm32(w, ADDR, offset as u32)?;
        writeln!(w, "    add {dest}, {dest}, {ADDR}")?;
    }
    Ok(())
}

pub(crate) fn materialize_address<W: Write>(
    w: &mut W,
    addr: &AddressMode,
    dest: &str,
    frame: &FrameMap,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            load_reg(w, base, dest, frame)?;
            add_offset(w, dest, i64::from(*offset))?;
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => {
            load_reg(w, base, dest, frame)?;
            let idx = scratch.alloc()?;
            load_reg(w, index, idx, frame)?;
            let shift = index_scale_shift(*scale).ok_or_else(|| {
                LaminaError::ValidationError(format!("address scale {scale} is not 1, 2, 4 or 8"))
            })?;
            if shift > 0 {
                writeln!(w, "    add {dest}, {dest}, {idx}, lsl #{shift}")?;
            } else {
                writeln!(w, "    add {dest}, {dest}, {idx}")?;
            }
            scratch.free(idx);
            add_offset(w, dest, i64::from(*offset))?;
        }
    }
    Ok(())
}
