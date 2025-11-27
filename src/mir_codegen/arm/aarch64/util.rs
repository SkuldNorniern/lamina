//! AArch64 utility functions for code generation.

use std::io::Write;

use crate::error::LaminaError;

/// Emits instructions to materialize a 64-bit immediate into a destination register.
///
/// Uses movz/movk sequence for values that don't fit in a single mov instruction.
pub fn emit_mov_imm64<W: Write>(
    w: &mut W,
    dest: &str,
    value: u64,
) -> std::result::Result<(), LaminaError> {
    if value <= 0xFFFF {
        writeln!(w, "    mov {}, #{}", dest, value)?;
        return Ok(());
    }
    let mut first = true;
    for shift in [0u32, 16, 32, 48] {
        let part = ((value >> shift) & 0xFFFF) as u16;
        if part != 0 || first {
            if first {
                writeln!(w, "    movz {}, #{}, lsl #{}", dest, part, shift)?;
                first = false;
            } else {
                writeln!(w, "    movk {}, #{}, lsl #{}", dest, part, shift)?;
            }
        }
    }
    Ok(())
}

/// Converts a MIR immediate value to u64 representation.
pub fn imm_to_u64(i: &crate::mir::Immediate) -> u64 {
    match i {
        crate::mir::Immediate::I8(v) => *v as i64 as u64,
        crate::mir::Immediate::I16(v) => *v as i64 as u64,
        crate::mir::Immediate::I32(v) => *v as i64 as u64,
        crate::mir::Immediate::I64(v) => *v as u64,
        crate::mir::Immediate::F32(v) => v.to_bits() as u64,
        crate::mir::Immediate::F64(v) => v.to_bits(),
    }
}
