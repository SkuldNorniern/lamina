use crate::codegen::CodegenError;
use crate::{
    Identifier,
    Label,
    // Value, Type,
    LaminaError,
    Result,
    // Module, GlobalDeclaration, PrimitiveType
};
use std::collections::{HashMap, HashSet}; // Re-add HashMap and HashSet
// Using HashMap for standard library compatibility
// IndexMap dependency removed for release

// x86-64 registers used for integer/pointer arguments in System V ABI
pub const ARG_REGISTERS: [&str; 6] = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
// Register used for integer/pointer return value
pub const RETURN_REGISTER: &str = "%rax";

// Represents the location of an IR value
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLocation {
    Register(String), // Stored in a specific register (e.g., "%rdi" for 1st arg)
    StackOffset(i64), // Offset relative to %rbp
}

impl ValueLocation {
    // Get the assembly operand string for this location
    pub fn to_operand_string(&self) -> String {
        match self {
            ValueLocation::Register(reg) => reg.clone(),
            ValueLocation::StackOffset(offset) => format!("{}(%rbp)", offset),
        }
    }
}

// Context specific to generating code for a single function
#[derive(Debug)]
pub struct FunctionContext<'a> {
    // Keep track of where SSA variables (%result) and arguments are stored
    pub value_locations: HashMap<Identifier<'a>, ValueLocation>,
    // Map IR block labels to assembly labels (deterministic order)
    pub block_labels: HashMap<Label<'a>, String>,
    // Track spilled argument registers and their stack offsets
    pub arg_register_spills: HashMap<String, i64>, // Register name -> stack offset (%rbp relative)
    // Track total stack space needed for locals and spills (aligned)
    pub total_stack_size: u64,
    // Used internally by layout_function_stack to track current offset
    pub current_stack_offset: i64,
    // Assembly label for the function epilogue
    pub epilogue_label: String,
    // Track which values are heap allocations (need to be freed)
    pub heap_allocations: HashSet<Identifier<'a>>,
    // Potentially useful: Keep track of types for locals if needed later
    // pub variable_types: HashMap<Identifier<'a>, Type<'a>>,
}

impl Default for FunctionContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> FunctionContext<'a> {
    pub fn new() -> Self {
        Self {
            value_locations: HashMap::new(),
            block_labels: HashMap::new(), // Use HashMap (previously IndexMap)
            arg_register_spills: HashMap::new(),
            total_stack_size: 0,
            current_stack_offset: -8, // Start allocating below RBP (first local/spill)
            epilogue_label: String::new(),
            heap_allocations: HashSet::new(),
            // variable_types: HashMap::new(),
        }
    }

    // Get the location (register, stack offset, spilled arg) of an IR value
    // This version assumes layout_function_stack has already run and populated locations.
    pub fn get_value_location(&self, name: Identifier<'a>) -> Result<ValueLocation> {
        self.value_locations.get(name).cloned().ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::ValueLocationNotFound(name.to_string()))
        })
    }

    // Get the assembly label for a basic block
    pub fn get_block_label(&self, ir_label: &Label<'a>) -> Result<String> {
        self.block_labels.get(ir_label).cloned().ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::BlockLabelNotFound(ir_label.to_string()))
        })
    }
}

// Codegen state for the entire module
#[derive(Debug, Default)]
pub struct CodegenState<'a> {
    pub global_layout: HashMap<Identifier<'a>, String>,
    pub rodata_strings: Vec<(String, String)>, // label -> content
    pub next_label_id: u32,
    pub next_rodata_id: u32,                  // For unique rodata labels
    pub inlinable_functions: HashSet<String>, // Functions that can be inlined
}

impl CodegenState<'_> {
    pub fn new() -> Self {
        CodegenState {
            global_layout: HashMap::new(),
            rodata_strings: Vec::new(),
            next_label_id: 0,
            next_rodata_id: 0,
            inlinable_functions: HashSet::new(),
        }
    }

    pub fn new_label(&mut self, prefix: &str) -> String {
        let id = self.next_label_id;
        self.next_label_id += 1;
        format!(".L_{}_{}", prefix, id)
    }

    // Generates a unique label for a read-only string, stores it, and returns the label.
    pub fn add_rodata_string(&mut self, content: &str) -> String {
        // Check if identical string already exists to reuse label
        for (label, existing_content) in &self.rodata_strings {
            if existing_content == content {
                return label.clone();
            }
        }
        // Otherwise, create a new one
        let id = self.next_rodata_id;
        self.next_rodata_id += 1;
        let label = format!(".L.rodata_str_{}", id);
        self.rodata_strings
            .push((label.clone(), content.to_string()));
        label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Remove unused test imports:
    // use crate::ir::types::{PrimitiveType, Type, Value, Literal};

    #[test]
    fn test_function_context_new() {
        let _ctx = FunctionContext::new();
        assert!(_ctx.value_locations.is_empty());
        assert!(_ctx.block_labels.is_empty());
        assert!(_ctx.arg_register_spills.is_empty());
        assert_eq!(_ctx.total_stack_size, 0);
        assert_eq!(_ctx.current_stack_offset, -8);
        assert!(_ctx.epilogue_label.is_empty());
        // assert!(ctx.variable_types.is_empty());
    }

    #[test]
    fn test_function_context_get_block_label() {
        let mut ctx = FunctionContext::new();
        let ir_label1 = "entry";
        let asm_label1 = ".L_entry_0".to_string();
        ctx.block_labels.insert(ir_label1, asm_label1.clone());

        let ir_label2 = "loop_header";
        let asm_label2 = ".L_loop_header_1".to_string();
        ctx.block_labels.insert(ir_label2, asm_label2.clone());

        // Correct the comparison: compare String with String
        assert_eq!(ctx.get_block_label(&ir_label1).unwrap(), asm_label1);
        assert_eq!(ctx.get_block_label(&ir_label2).unwrap(), asm_label2);
        assert!(ctx.get_block_label(&"non_existent").is_err());
    }

    // ... rest of tests (test_codegen_state_new, test_codegen_state_add_rodata_string, test_value_location_to_operand_string) ...
}
