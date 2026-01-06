use super::types::GlobalLayout;
use super::utils::{escape_asm_string, get_type_size_bytes};
use crate::{GlobalDeclaration, LaminaError, Literal, Module, Value};
use std::collections::HashMap;
use std::io::Write;
use std::result::Result;

/// Common interface for global variable generation
pub trait GlobalGenerator<'a> {
    /// Generate global variable sections for a module
    fn generate_globals<W: Write>(
        &self,
        module: &'a Module<'a>,
        writer: &mut W,
        layout: &mut GlobalLayout,
    ) -> Result<(), LaminaError>;

    /// Generate a single global variable
    fn generate_global<W: Write>(
        &self,
        name: &str,
        global: &GlobalDeclaration<'a>,
        writer: &mut W,
    ) -> Result<(), LaminaError>;
}

/// Common global variable manager
pub struct GlobalManager {
    /// Map from global name to assembly label
    pub label_map: HashMap<String, String>,
    /// Read-only string literals
    pub rodata_strings: Vec<(String, String)>,
    /// Next unique ID for labels
    next_id: u32,
}

impl GlobalManager {
    pub fn new() -> Self {
        Self {
            label_map: HashMap::new(),
            rodata_strings: Vec::new(),
            next_id: 0,
        }
    }

    /// Generate a unique label for a global
    pub fn generate_global_label(&mut self, name: &str) -> String {
        let label = format!("global_{}", name);
        self.label_map.insert(name.to_string(), label.clone());
        label
    }

    /// Add a read-only string and return its label
    pub fn add_rodata_string(&mut self, content: &str) -> String {
        // Check if this string already exists
        for (label, existing_content) in &self.rodata_strings {
            if existing_content == content {
                return label.clone();
            }
        }

        // Create new string label
        let label = format!(".L.str.{}", self.next_id);
        self.next_id += 1;
        self.rodata_strings
            .push((label.clone(), content.to_string()));
        label
    }

    /// Get the label for a global variable
    pub fn get_global_label(&self, name: &str) -> Option<&String> {
        self.label_map.get(name)
    }

    /// Get all read-only strings
    pub fn get_rodata_strings(&self) -> &[(String, String)] {
        &self.rodata_strings
    }
}

impl Default for GlobalManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard global variable generator that works for most architectures
pub struct StandardGlobalGenerator {
    /// Architecture-specific section names
    data_section: String,
    bss_section: String,
    rodata_section: String,
    /// Whether to emit type and size directives
    emit_metadata: bool,
}

impl StandardGlobalGenerator {
    pub fn new(data_section: String, bss_section: String, rodata_section: String) -> Self {
        Self {
            data_section,
            bss_section,
            rodata_section,
            emit_metadata: true,
        }
    }

    /// Create generator for x86_64 Linux
    pub fn x86_64_linux() -> Self {
        Self::new(
            ".section .data".to_string(),
            ".section .bss".to_string(),
            ".section .rodata".to_string(),
        )
    }

    /// Create generator for AArch64 macOS  
    pub fn aarch64_macos() -> Self {
        Self::new(
            ".section __DATA,__data".to_string(),
            ".section __DATA,__bss".to_string(),
            ".section __DATA,__const".to_string(),
        )
    }

    /// Set whether to emit type and size metadata
    pub fn set_emit_metadata(&mut self, emit: bool) {
        self.emit_metadata = emit;
    }
}

impl<'a> GlobalGenerator<'a> for StandardGlobalGenerator {
    fn generate_globals<W: Write>(
        &self,
        module: &'a Module<'a>,
        writer: &mut W,
        layout: &mut GlobalLayout,
    ) -> Result<(), LaminaError> {
        // Separate globals into initialized and uninitialized
        let mut initialized_globals = Vec::new();
        let mut uninitialized_globals = Vec::new();

        for (name, global) in &module.global_declarations {
            if global.initializer.is_some() {
                initialized_globals.push((name, global));
            } else {
                uninitialized_globals.push((name, global));
            }
        }

        // Generate .data section for initialized globals
        if !initialized_globals.is_empty() {
            writeln!(writer, "{}", self.data_section)?;
            for (name, global) in initialized_globals {
                let label = format!("global_{}", name);
                layout.label_map.insert(name.to_string(), label.clone());
                layout.data_globals.push(label.clone());

                self.generate_global_declaration(name, global, &label, writer)?;
                self.generate_global_initializer(global, writer)?;
            }
        }

        // Generate .bss section for uninitialized globals
        if !uninitialized_globals.is_empty() {
            writeln!(writer, "\n{}", self.bss_section)?;
            for (name, global) in uninitialized_globals {
                let label = format!("global_{}", name);
                layout.label_map.insert(name.to_string(), label.clone());
                layout.bss_globals.push(label.clone());

                self.generate_global_declaration(name, global, &label, writer)?;
                self.generate_bss_allocation(global, &label, writer)?;
            }
        }

        // Generate .rodata section for string literals
        if !layout.rodata_strings.is_empty() {
            writeln!(writer, "\n{}", self.rodata_section)?;
            for (label, content) in &layout.rodata_strings {
                writeln!(writer, "{}:", label)?;
                writeln!(writer, "    .string \"{}\"", escape_asm_string(content))?;
            }
        }

        Ok(())
    }

    fn generate_global<W: Write>(
        &self,
        name: &str,
        global: &GlobalDeclaration<'a>,
        writer: &mut W,
    ) -> Result<(), LaminaError> {
        let label = format!("global_{}", name);

        if global.initializer.is_some() {
            writeln!(writer, "{}", self.data_section)?;
        } else {
            writeln!(writer, "{}", self.bss_section)?;
        }

        self.generate_global_declaration(name, global, &label, writer)?;

        if global.initializer.is_some() {
            self.generate_global_initializer(global, writer)?;
        } else {
            self.generate_bss_allocation(global, &label, writer)?;
        }

        Ok(())
    }
}

impl StandardGlobalGenerator {
    /// Generate global declaration directives
    fn generate_global_declaration<W: Write>(
        &self,
        _name: &str,
        global: &GlobalDeclaration,
        label: &str,
        writer: &mut W,
    ) -> Result<(), LaminaError> {
        // Make symbol global
        writeln!(writer, ".globl {}", label)?;

        // Emit type directive if enabled
        if self.emit_metadata {
            writeln!(writer, ".type {}, @object", label)?;

            // Emit size directive
            let size = get_type_size_bytes(&global.ty)?;
            writeln!(writer, ".size {}, {}", label, size)?;
        }

        // Emit alignment
        let alignment = get_alignment_for_type(&global.ty)?;
        if alignment > 1 {
            writeln!(writer, ".align {}", alignment.ilog2())?;
        }

        // Emit label
        writeln!(writer, "{}:", label)?;

        Ok(())
    }

    /// Generate initializer data for a global
    fn generate_global_initializer<W: Write>(
        &self,
        global: &GlobalDeclaration,
        writer: &mut W,
    ) -> Result<(), LaminaError> {
        if let Some(ref initializer) = global.initializer {
            match initializer {
                Value::Constant(literal) => {
                    self.generate_literal_data(literal, writer)?;
                }
                Value::Global(global_name) => {
                    // Reference to another global
                    writeln!(writer, "    .quad global_{}", global_name)?;
                }
                Value::Variable(_) => {
                    return Err(LaminaError::CodegenError(
                        crate::codegen::CodegenError::GlobalVarInitNotSupported,
                    ));
                }
            }
        } else {
            return Err(LaminaError::CodegenError(
                crate::codegen::CodegenError::UninitializedGlobalInit,
            ));
        }
        Ok(())
    }

    /// Generate literal data
    fn generate_literal_data<W: Write>(
        &self,
        literal: &Literal,
        writer: &mut W,
    ) -> Result<(), LaminaError> {
        match literal {
            Literal::I8(v) => writeln!(writer, "    .byte {}", v)?,
            Literal::I16(v) => writeln!(writer, "    .word {}", v)?,
            Literal::I32(v) => writeln!(writer, "    .long {}", v)?,
            Literal::I64(v) => writeln!(writer, "    .quad {}", v)?,
            Literal::U8(v) => writeln!(writer, "    .byte {}", v)?,
            Literal::U16(v) => writeln!(writer, "    .word {}", v)?,
            Literal::U32(v) => writeln!(writer, "    .long {}", v)?,
            Literal::U64(v) => writeln!(writer, "    .quad {}", v)?,
            Literal::F32(v) => {
                let bits = v.to_bits();
                writeln!(writer, "    .long {}", bits)?;
            }
            Literal::F64(v) => {
                let bits = v.to_bits();
                writeln!(writer, "    .quad {}", bits)?;
            }
            Literal::Bool(v) => writeln!(writer, "    .byte {}", if *v { 1 } else { 0 })?,
            Literal::Char(c) => writeln!(writer, "    .byte {}", *c as u8)?,
            Literal::String(s) => {
                writeln!(writer, "    .string \"{}\"", escape_asm_string(s))?;
            }
        }
        Ok(())
    }

    /// Generate BSS allocation
    fn generate_bss_allocation<W: Write>(
        &self,
        global: &GlobalDeclaration,
        label: &str,
        writer: &mut W,
    ) -> Result<(), LaminaError> {
        let size = get_type_size_bytes(&global.ty)?;
        let alignment = get_alignment_for_type(&global.ty)?;

        // Use .comm directive for BSS allocation
        writeln!(writer, ".comm {}, {}, {}", label, size, alignment)?;

        Ok(())
    }
}

/// Get alignment for a type
fn get_alignment_for_type(ty: &crate::Type) -> Result<u64, LaminaError> {
    use crate::PrimitiveType;
    use crate::Type;

    match ty {
        Type::Primitive(pt) => Ok(match pt {
            PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 | PrimitiveType::Ptr => 8,
        }),
        Type::Array { element_type, .. } => get_alignment_for_type(element_type),
        Type::Struct(_) => Ok(8), // Default alignment for structs
        Type::Tuple(_) => Ok(8),  // Default alignment for tuples
        Type::Named(_) => Ok(8),  // Default alignment for named types
        #[cfg(feature = "nightly")]
        Type::Vector { element_type, lanes } => {
            // Vector alignment is typically the vector size (128-bit = 16 bytes, 256-bit = 32 bytes)
            let elem_size = match element_type {
                PrimitiveType::I8 | PrimitiveType::U8 => 1,
                PrimitiveType::I16 | PrimitiveType::U16 => 2,
                PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
                _ => return Err(LaminaError::CodegenError(
                    crate::codegen::CodegenError::StructNotImplemented, // Temporary
                )),
            };
            let vector_size = elem_size * (*lanes as u64);
            // Align to vector size (typically 16 or 32 bytes)
            Ok(vector_size.min(32).max(16))
        }
        Type::Void => Err(LaminaError::CodegenError(
            crate::codegen::CodegenError::VoidTypeSize,
        )),
    }
}

/// Global optimization utilities
pub struct GlobalOptimizer;

impl GlobalOptimizer {
    /// Merge identical string literals
    pub fn merge_string_literals(layout: &mut GlobalLayout) -> usize {
        let mut merged_count = 0;
        let mut unique_strings: HashMap<String, String> = HashMap::new();
        let mut new_rodata_strings = Vec::new();

        for (label, content) in &layout.rodata_strings {
            if let Some(_existing_label) = unique_strings.get(content) {
                // This string already exists, we could update references
                // For now, just count the merge
                merged_count += 1;
            } else {
                unique_strings.insert(content.clone(), label.clone());
                new_rodata_strings.push((label.clone(), content.clone()));
            }
        }

        layout.rodata_strings = new_rodata_strings;
        merged_count
    }

    /// Remove unused globals (requires usage analysis)
    pub fn remove_unused_globals(
        layout: &mut GlobalLayout,
        used_globals: &std::collections::HashSet<String>,
    ) -> usize {
        let initial_count = layout.data_globals.len() + layout.bss_globals.len();

        layout.data_globals.retain(|global| {
            // Extract global name from label
            let name = global.strip_prefix("global_").unwrap_or(global);
            used_globals.contains(name)
        });

        layout.bss_globals.retain(|global| {
            let name = global.strip_prefix("global_").unwrap_or(global);
            used_globals.contains(name)
        });

        let final_count = layout.data_globals.len() + layout.bss_globals.len();
        initial_count - final_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{PrimitiveType, Type};
    use std::io::Cursor;

    #[test]
    fn test_global_manager() {
        let mut manager = GlobalManager::new();

        let label1 = manager.generate_global_label("counter");
        assert_eq!(label1, "global_counter");
        assert_eq!(
            manager.get_global_label("counter"),
            Some(&"global_counter".to_string())
        );

        let str_label = manager.add_rodata_string("hello");
        assert_eq!(str_label, ".L.str.0");

        // Adding same string should return same label
        let str_label2 = manager.add_rodata_string("hello");
        assert_eq!(str_label, str_label2);

        assert_eq!(manager.get_rodata_strings().len(), 1);
    }

    #[test]
    fn test_standard_global_generator() {
        let generator = StandardGlobalGenerator::x86_64_linux();
        assert_eq!(generator.data_section, ".section .data");
        assert_eq!(generator.bss_section, ".section .bss");
        assert_eq!(generator.rodata_section, ".section .rodata");

        let generator = StandardGlobalGenerator::aarch64_macos();
        assert_eq!(generator.data_section, ".section __DATA,__data");
    }

    #[test]
    fn test_generate_literal_data() {
        let generator = StandardGlobalGenerator::x86_64_linux();
        let mut writer = Cursor::new(Vec::new());

        generator
            .generate_literal_data(&Literal::I32(42), &mut writer)
            .unwrap();
        let output = String::from_utf8(writer.into_inner()).unwrap();
        assert_eq!(output.trim(), ".long 42");

        let mut writer = Cursor::new(Vec::new());
        generator
            .generate_literal_data(&Literal::String("test"), &mut writer)
            .unwrap();
        let output = String::from_utf8(writer.into_inner()).unwrap();
        assert_eq!(output.trim(), ".string \"test\"");
    }

    #[test]
    fn test_get_alignment_for_type() {
        assert_eq!(
            get_alignment_for_type(&Type::Primitive(PrimitiveType::I8)).unwrap(),
            1
        );
        assert_eq!(
            get_alignment_for_type(&Type::Primitive(PrimitiveType::I32)).unwrap(),
            4
        );
        assert_eq!(
            get_alignment_for_type(&Type::Primitive(PrimitiveType::I64)).unwrap(),
            8
        );
    }

    #[test]
    fn test_global_optimizer() {
        let mut layout = GlobalLayout {
            label_map: HashMap::new(),
            rodata_strings: vec![
                (".L.str.0".to_string(), "hello".to_string()),
                (".L.str.1".to_string(), "world".to_string()),
                (".L.str.2".to_string(), "hello".to_string()), // Duplicate
            ],
            data_globals: vec!["global_a".to_string(), "global_b".to_string()],
            bss_globals: vec!["global_c".to_string()],
        };

        let merged = GlobalOptimizer::merge_string_literals(&mut layout);
        assert_eq!(merged, 1); // One duplicate was found
        assert_eq!(layout.rodata_strings.len(), 2); // Only unique strings remain

        let mut used_globals = std::collections::HashSet::new();
        used_globals.insert("a".to_string());
        // Note: "b" and "c" are not in used_globals

        let removed = GlobalOptimizer::remove_unused_globals(&mut layout, &used_globals);
        assert_eq!(removed, 2); // "global_b" and "global_c" should be removed
        assert_eq!(layout.data_globals.len(), 1);
        assert_eq!(layout.bss_globals.len(), 0);
    }
}
