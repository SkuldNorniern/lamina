use super::state::CodegenState;
use crate::codegen::{CodegenError, LiteralType};
use crate::{GlobalDeclaration, LaminaError, Literal, Module, Type};
use std::io::Write;
use std::result::Result;

// Emit data and bss for RISC-V (ELF-friendly directives)
pub fn generate_global_data_section<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<(), LaminaError> {
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
            writeln!(writer, "{}:", asm_label)?;
            generate_global_initializer(writer, global)?;
        }
    }

    if has_bss {
        writeln!(writer, "\n.bss")?;
        for (name, global) in module
            .global_declarations
            .iter()
            .filter(|(_, g)| g.initializer.is_none())
        {
            let asm_label = format!("global_{}", name);
            state.global_layout.insert(name, asm_label.clone());
            let (_, size_bytes) = super::util::get_type_size_directive_and_bytes(&global.ty)?;
            // Use .comm name,size,align
            let alignment = crate::codegen::common::utils::get_type_alignment(&global.ty)?;
            writeln!(writer, ".comm {},{},{}", asm_label, size_bytes, alignment)?;
        }
    }

    Ok(())
}

fn generate_global_initializer<W: Write>(
    writer: &mut W,
    global: &GlobalDeclaration<'_>,
) -> Result<(), LaminaError> {
    if let Some(ref initializer) = global.initializer {
        match initializer {
            crate::Value::Constant(literal) => match literal {
                Literal::I32(v) => writeln!(writer, "    .word {}", v)?,
                Literal::I64(v) => writeln!(writer, "    .dword {}", v)?,
                Literal::F32(v) => {
                    let bits = v.to_bits();
                    writeln!(writer, "    .word {}", bits)?;
                }
                Literal::Bool(v) => writeln!(writer, "    .byte {}", if *v { 1 } else { 0 })?,
                Literal::String(s) => {
                    writeln!(
                        writer,
                        "    .asciz \"{}\"",
                        crate::codegen::common::utils::escape_asm_string(s)
                    )?;
                }
                Literal::I8(v) => writeln!(writer, "    .byte {}", v)?,
                _ => {
                    return Err(LaminaError::CodegenError(
                        CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::Unknown(
                            format!("{:?}", literal),
                        )),
                    ));
                }
            },
            crate::Value::Global(_) => {
                return Err(LaminaError::CodegenError(
                    CodegenError::GlobalToGlobalInitNotImplemented,
                ));
            }
            crate::Value::Variable(_) => {
                return Err(LaminaError::CodegenError(
                    CodegenError::GlobalVarInitNotSupported,
                ));
            }
        }
    } else {
        return Err(LaminaError::CodegenError(
            CodegenError::UninitializedGlobalInit,
        ));
    }
    Ok(())
}

pub fn generate_globals<W: Write>(state: &CodegenState, writer: &mut W) -> Result<(), LaminaError> {
    if !state.rodata_strings.is_empty() {
        writeln!(writer, "\n.section .rodata")?;
        for (label, content) in &state.rodata_strings {
            let escaped = crate::codegen::common::utils::escape_asm_string(content);
            writeln!(writer, "{}: .asciz \"{}\"", label, escaped)?;
        }
    }
    Ok(())
}
