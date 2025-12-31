use super::state::CodegenState;
use super::util::get_type_size_directive_and_bytes;
use crate::codegen::CodegenError;
use crate::{GlobalDeclaration, LaminaError, Literal, Module, Value};
use std::io::Write;
use std::result::Result;

// Helper to generate the .data and .bss sections based on globals
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
        writeln!(writer, ".section .data")?;
        for (name, global) in module
            .global_declarations
            .iter()
            .filter(|(_, g)| g.initializer.is_some())
        {
            let asm_label = format!("global_{}", name);
            state.global_layout.insert(name, asm_label.clone());
            writeln!(writer, ".globl {}", asm_label)?;
            writeln!(writer, ".type {}, @object", asm_label)?;
            let (_, size_bytes) = get_type_size_directive_and_bytes(&global.ty)?;
            writeln!(writer, ".size {}, {}", asm_label, size_bytes)?;
            // Note: Alignment directives can be added here if needed (.align)
            writeln!(writer, "{}:", asm_label)?;
            generate_global_initializer(writer, global)?;
        }
    }

    if has_bss {
        writeln!(writer, "\n.section .bss")?;
        for (name, global) in module
            .global_declarations
            .iter()
            .filter(|(_, g)| g.initializer.is_none())
        {
            let asm_label = format!("global_{}", name);
            state.global_layout.insert(name, asm_label.clone());
            writeln!(writer, ".globl {}", asm_label)?;
            writeln!(writer, ".type {}, @object", asm_label)?;
            let (_, size_bytes) = get_type_size_directive_and_bytes(&global.ty)?;
            // .comm reserves space in BSS (alignment 8 for simplicity)
            writeln!(writer, ".comm {},{},{}", asm_label, size_bytes, 8)?;
        }
    }

    Ok(())
}

// Helper to emit initializer data for a global
fn generate_global_initializer<W: Write>(
    writer: &mut W,
    global: &GlobalDeclaration<'_>,
) -> Result<(), LaminaError> {
    if let Some(ref initializer) = global.initializer {
        match initializer {
            Value::Constant(literal) => match literal {
                Literal::I8(v) => writeln!(writer, "    .byte {}", v)?,
                Literal::I16(v) => writeln!(writer, "    .word {}", v)?,
                Literal::I32(v) => writeln!(writer, "    .long {}", v)?,
                Literal::I64(v) => writeln!(writer, "    .quad {}", v)?,
                Literal::U8(v) => writeln!(writer, "    .byte {}", v)?,
                Literal::U16(v) => writeln!(writer, "    .word {}", v)?,
                Literal::U32(v) => writeln!(writer, "    .long {}", v)?,
                Literal::U64(v) => writeln!(writer, "    .quad {}", v)?,
                Literal::F32(v) => {
                    // Represent f32 as its raw u32 bits
                    let bits = v.to_bits();
                    writeln!(writer, "    .long {}", bits)?;
                }
                Literal::F64(v) => {
                    // Represent f64 as its raw u64 bits
                    let bits = v.to_bits();
                    writeln!(writer, "    .quad {}", bits)?;
                }
                Literal::Bool(v) => writeln!(writer, "    .byte {}", if *v { 1 } else { 0 })?,
                Literal::Char(c) => writeln!(writer, "    .byte {}", *c as u8)?,
                Literal::String(s) => {
                    // Use .string directive for consistent handling of string literals
                    writeln!(writer, "    .string \"{}\"", escape_asm_string(s))?;
                }
            },
            Value::Global(_) => {
                return Err(LaminaError::CodegenError(
                    CodegenError::GlobalToGlobalInitNotImplemented,
                ));
            }
            Value::Variable(_) => {
                return Err(LaminaError::CodegenError(
                    CodegenError::GlobalVarInitNotSupported,
                ));
            }
        }
    } else {
        // Should have been handled by BSS section logic
        return Err(LaminaError::CodegenError(
            CodegenError::UninitializedGlobalInit,
        ));
    }
    Ok(())
}

// Escape characters in a string literal for GAS .string directive
fn escape_asm_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"), // Escape backslash
            '"' => escaped.push_str("\\\""),  // Escape double quote
            '\n' => escaped.push_str("\\n"),  // Escape newline
            '\t' => escaped.push_str("\\t"),  // Escape tab
            '\r' => escaped.push_str("\\r"),  // Escape carriage return
            '\0' => escaped.push_str("\\0"),  // Escape null character
            c if c.is_control() => {
                // Escape other control characters as octal
                escaped.push_str(&format!("\\{:03o}", c as u8));
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

pub fn generate_globals<W: Write>(state: &CodegenState, writer: &mut W) -> Result<(), LaminaError> {
    // --- Read-Only Data ---
    if !state.rodata_strings.is_empty() {
        writeln!(writer, "\n.section .rodata")?;
        for (label, content) in &state.rodata_strings {
            // Use escaped content for consistency
            let escaped_content = escape_asm_string(content);
            writeln!(writer, "{}: .string \"{}\"", label, escaped_content)?;
        }
    }

    // --- Initialized Data ---
    if !state.global_layout.is_empty() {
        // Assuming global_layout implies .data for now
        writeln!(writer, "\n.section .data")?;
        // Logic to emit initialized globals from state would go here
        // Example: writeln!(writer, "{}: .quad {}", label, value)?;
    }

    // --- Uninitialized Data (BSS) ---
    // Logic to emit .bss section and symbols from state would go here
    // Example: writeln!(writer, ".comm {}, {}, {}", label, size, alignment)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::x86_64::state::CodegenState;
    use crate::ir::types::{PrimitiveType, Type};
    use std::io::Cursor;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_escape_asm_string() {
        assert_eq!(escape_asm_string("simple"), "simple");
        assert_eq!(escape_asm_string("with\"quote"), "with\\\"quote");
        assert_eq!(escape_asm_string("with\\backslash"), "with\\\\backslash");
        // Control characters should be properly escaped
        assert_eq!(escape_asm_string("new\nline"), "new\\nline");
        assert_eq!(escape_asm_string("tab\tstop"), "tab\\tstop");
        assert_eq!(escape_asm_string("carriage\rreturn"), "carriage\\rreturn");
        assert_eq!(
            escape_asm_string("mixed \"quotes\" and \\slashes\\"),
            "mixed \\\"quotes\\\" and \\\\slashes\\\\"
        );
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_generate_global_initializer() {
        let mut buf = Cursor::new(Vec::new());

        // I32
        let global1 = GlobalDeclaration {
            name: "g1",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Constant(Literal::I32(42))),
        };
        generate_global_initializer(&mut buf, &global1).unwrap();
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".long 42"
        );

        // I64
        buf = Cursor::new(Vec::new());
        let global2 = GlobalDeclaration {
            name: "g2",
            ty: Type::Primitive(PrimitiveType::I64),
            initializer: Some(Value::Constant(Literal::I64(-100))),
        };
        generate_global_initializer(&mut buf, &global2).unwrap();
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".quad -100"
        );

        // Bool (true)
        buf = Cursor::new(Vec::new());
        let global3 = GlobalDeclaration {
            name: "g3",
            ty: Type::Primitive(PrimitiveType::Bool),
            initializer: Some(Value::Constant(Literal::Bool(true))),
        };
        generate_global_initializer(&mut buf, &global3).unwrap();
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".byte 1"
        );

        // Bool (false)
        buf = Cursor::new(Vec::new());
        let global4 = GlobalDeclaration {
            name: "g4",
            ty: Type::Primitive(PrimitiveType::Bool),
            initializer: Some(Value::Constant(Literal::Bool(false))),
        };
        generate_global_initializer(&mut buf, &global4).unwrap();
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".byte 0"
        );

        // String
        buf = Cursor::new(Vec::new());
        let global5 = GlobalDeclaration {
            name: "g5",
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 6,
            }, // Assuming string type maps to array
            initializer: Some(Value::Constant(Literal::String("hello"))),
        };
        generate_global_initializer(&mut buf, &global5).unwrap();
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".string \"hello\""
        );

        // String with escapes
        buf = Cursor::new(Vec::new());
        let global6 = GlobalDeclaration {
            name: "g6",
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 18,
            }, // Adjusted size
            initializer: Some(Value::Constant(Literal::String("a \"quoted\" string"))),
        };
        generate_global_initializer(&mut buf, &global6).unwrap();
        // generate_global_initializer calls escape_asm_string, check against corrected escape logic
        assert_eq!(
            String::from_utf8(buf.into_inner()).unwrap().trim(),
            ".string \"a \\\"quoted\\\" string\""
        ); // Updated expectation

        // Error cases
        let global_err1 = GlobalDeclaration {
            name: "ge1",
            ty: Type::Primitive(PrimitiveType::F32),
            initializer: Some(Value::Constant(Literal::F32(1.0))),
        };
        assert!(generate_global_initializer(&mut Cursor::new(Vec::new()), &global_err1).is_ok()); // Changed to .ok() since we now handle F32 
        let global_err2 = GlobalDeclaration {
            name: "ge2",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Global("other")),
        };
        assert!(generate_global_initializer(&mut Cursor::new(Vec::new()), &global_err2).is_err());
        let global_err3 = GlobalDeclaration {
            name: "ge3",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Variable("var")),
        };
        assert!(generate_global_initializer(&mut Cursor::new(Vec::new()), &global_err3).is_err());
        let global_err4 = GlobalDeclaration {
            name: "ge4",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: None,
        }; // Uninitialized
        assert!(generate_global_initializer(&mut Cursor::new(Vec::new()), &global_err4).is_err());
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_generate_globals_rodata() {
        let mut state = CodegenState::<'static>::new();
        state
            .rodata_strings
            .push((".L.rodata_str_0".to_string(), "Hello, World!\n".to_string()));
        state.rodata_strings.push((
            ".L.rodata_str_1".to_string(),
            "Another \"string\" here".to_string(),
        ));

        let mut buf = Cursor::new(Vec::new());
        generate_globals(&state, &mut buf).unwrap();

        // Get the actual output
        let output = String::from_utf8(buf.into_inner()).unwrap();

        // Check for key elements that must be present
        assert!(
            output.contains(".section .rodata"),
            "Should include .rodata section"
        );
        assert!(
            output.contains(".L.rodata_str_0: .string \"Hello, World!"),
            "Should include first string"
        );
        assert!(
            output.contains(".L.rodata_str_1: .string \"Another \\\"string\\\" here\""),
            "Should include second string with escaped quotes"
        );
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_generate_globals_no_rodata() {
        let state = CodegenState::<'static>::new(); // Empty state
        let mut buf = Cursor::new(Vec::new());
        generate_globals(&state, &mut buf).unwrap();
        let output = String::from_utf8(buf.into_inner()).unwrap();
        assert!(!output.contains(".rodata"));
        assert!(!output.contains(".data")); // Check no sections are emitted
        assert!(!output.contains(".bss"));
        assert!(output.trim().is_empty());
    }

    // Note: Testing generate_global_data_section directly is more complex,
    // requiring module setup and state checks. Better suited for integration tests.
}
