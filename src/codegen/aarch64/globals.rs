use super::state::CodegenState;
use super::util::get_type_size_directive_and_bytes;
use crate::{GlobalDeclaration, LaminaError, Literal, Module, Result, Value};
use std::io::Write;

// Emit data and bss for AArch64 (Mach-O and ELF compatible directives kept generic)
pub fn generate_global_data_section<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    let mut has_data = false;
    let mut has_bss = false;

    for global in module.global_declarations.values() {
        match global.initializer {
            Some(_) => has_data = true,
            None => has_bss = true,
        }
    }

    if has_data {
        writeln!(writer, ".data")?;
        for (name, global) in module
            .global_declarations
            .iter()
            .filter(|(_, g)| g.initializer.is_some())
        {
            let asm_label = format!("global_{}", name);
            state.global_layout.insert(name, asm_label.clone());
            writeln!(writer, ".globl {}", asm_label)?;
            // Mach-O does not support .type/.size; keep them for ELF-friendly output if assembler ignores
            writeln!(writer, "{}:", asm_label)?;
            generate_global_initializer(writer, global)?;
        }
    }

    if has_bss {
        // For Mach-O, .zerofill is preferred; fallback to .bss/.comm for generality
        writeln!(writer, "\n.bss")?;
        for (name, global) in module
            .global_declarations
            .iter()
            .filter(|(_, g)| g.initializer.is_none())
        {
            let asm_label = format!("global_{}", name);
            state.global_layout.insert(name, asm_label.clone());
            let (_, size_bytes) = get_type_size_directive_and_bytes(&global.ty)?;
            // POTENTIAL BUG: Hardcoded alignment of 8 bytes - may not be optimal for all types
            writeln!(writer, ".comm {},{},8", asm_label, size_bytes)?;
        }
    }

    Ok(())
}

fn generate_global_initializer<W: Write>(
    writer: &mut W,
    global: &GlobalDeclaration<'_>,
) -> Result<()> {
    if let Some(ref initializer) = global.initializer {
        match initializer {
            Value::Constant(literal) => match literal {
                Literal::I32(v) => writeln!(writer, "    .word {}", v)?,
                Literal::I64(v) => writeln!(writer, "    .xword {}", v)?,
                Literal::F32(v) => {
                    let bits = v.to_bits();
                    writeln!(writer, "    .word {}", bits)?;
                }
                Literal::Bool(v) => writeln!(writer, "    .byte {}", if *v { 1 } else { 0 })?,
                Literal::String(s) => {
                    // GAS-compatible .string
                    writeln!(
                        writer,
                        "    .string \"{}\"",
                        crate::codegen::common::utils::escape_asm_string(s)
                    )?;
                }
                Literal::I8(v) => writeln!(writer, "    .byte {}", v)?,
                _ => {
                    return Err(LaminaError::CodegenError(format!(
                        "Unsupported literal type in global initializer: {:?}",
                        literal
                    )));
                }
            },
            Value::Global(_) => {
                return Err(LaminaError::CodegenError(
                    "Global initializer pointing to another global not implemented yet".to_string(),
                ));
            }
            Value::Variable(_) => {
                return Err(LaminaError::CodegenError(
                    "Cannot initialize global with a variable".to_string(),
                ));
            }
        }
    } else {
        return Err(LaminaError::CodegenError(
            "generate_global_initializer called on uninitialized global".to_string(),
        ));
    }
    Ok(())
}

pub fn generate_globals<W: Write>(state: &CodegenState, writer: &mut W) -> Result<()> {
    if !state.rodata_strings.is_empty() {
        // Use Mach-O compatible section directive for AArch64
        // FEAT:TODO: Support other Host platform using AArch64 like linux / Windows
        writeln!(writer, "\n.section __TEXT,__cstring,cstring_literals")?;
        for (label, content) in &state.rodata_strings {
            let escaped = crate::codegen::common::utils::escape_asm_string(content);
            writeln!(writer, "{}: .asciz \"{}\"", label, escaped)?;
        }
    }
    Ok(())
}
