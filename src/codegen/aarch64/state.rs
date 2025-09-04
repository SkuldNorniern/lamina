use crate::{Identifier, Label, LaminaError, Result};
use std::collections::{HashMap, HashSet};

// AArch64 (ARM64) integer/pointer argument registers (AAPCS64 / SysV-like)
pub const ARG_REGISTERS: [&str; 8] = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];

// AArch64 return register for integer/pointer
pub const RETURN_REGISTER: &str = "x0";

// AArch64 callee-saved registers that may need to be preserved
pub const CALLEE_SAVED_REGISTERS: [&str; 12] = ["x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30"];

// Size of each register in bytes (all x registers are 64-bit = 8 bytes)
pub const REGISTER_SIZE: i64 = 8;

// Represents the location of an IR value in generated code
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLocation {
    // Stored in a specific register (e.g., "x0" for first arg)
    Register(String),
    // Offset relative to frame pointer x29 (negative offsets for locals/spills)
    StackOffset(i64),
}

impl ValueLocation {
    // Human-friendly rendering useful for comments and simple operand formatting
    pub fn to_operand_string(&self) -> String {
        match self {
            ValueLocation::Register(reg) => reg.clone(),
            ValueLocation::StackOffset(offset) => format!("[x29, #{}]", offset),
        }
    }
}

// Per-function context tracking stack layout, value locations, labels, etc.
#[derive(Debug)]
pub struct FunctionContext<'a> {
    pub value_locations: HashMap<Identifier<'a>, ValueLocation>,
    pub block_labels: HashMap<Label<'a>, String>,
    pub arg_register_spills: HashMap<String, i64>, // reg name -> stack offset rel x29
    pub total_stack_size: u64,
    pub current_stack_offset: i64,
    pub epilogue_label: String,
    pub callee_saved_regs: Vec<&'static str>, // Track which registers need to be saved
}

impl<'a> FunctionContext<'a> {
    pub fn new() -> Self {
        Self {
            value_locations: HashMap::new(),
            block_labels: HashMap::new(),
            arg_register_spills: HashMap::new(),
            total_stack_size: 0,
            current_stack_offset: 0, // Will be calculated later based on callee-saved regs
            epilogue_label: String::new(),
            callee_saved_regs: Vec::new(),
        }
    }

    /// Calculate the initial stack offset based on callee-saved registers that need preservation
    pub fn calculate_initial_stack_offset(&mut self) {
        // Register save space is handled in prologue with 16-byte alignment
        // Frame pointer points to saved registers, so locals start at 0 offset initially
        let reg_save_space = (2 + self.callee_saved_regs.len()) as i64 * REGISTER_SIZE;
        let aligned_reg_space = (reg_save_space + 15) & !15;
        self.current_stack_offset = -aligned_reg_space;
    }

    pub fn get_value_location(&self, name: Identifier<'a>) -> Result<ValueLocation> {
        self.value_locations.get(name).cloned().ok_or_else(|| {
            LaminaError::CodegenError(format!(
                "Value '{}' location not found in function context",
                name
            ))
        })
    }

    pub fn get_block_label(&self, ir_label: &Label<'a>) -> Result<String> {
        self.block_labels.get(ir_label).cloned().ok_or_else(|| {
            LaminaError::CodegenError(format!(
                "Label '{}' not found in function context",
                ir_label
            ))
        })
    }
}

impl Default for FunctionContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// Global codegen state for the entire module
#[derive(Debug, Default)]
pub struct CodegenState<'a> {
    pub global_layout: HashMap<Identifier<'a>, String>,
    pub rodata_strings: Vec<(String, String)>, // label -> content
    pub next_label_id: u32,
    pub next_rodata_id: u32,
    pub inlinable_functions: HashSet<String>,
}

impl<'a> CodegenState<'a> {
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
        // POTENTIAL BUG: No validation that generated labels don't conflict with assembler reserved names
        let id = self.next_label_id;
        self.next_label_id += 1;
        format!(".L_{}_{}", prefix, id)
    }

    pub fn add_rodata_string(&mut self, content: &str) -> String {
        for (label, existing) in &self.rodata_strings {
            if existing == content {
                return label.clone();
            }
        }
        let id = self.next_rodata_id;
        self.next_rodata_id += 1;
        let label = format!(".L.rodata_str_{}", id);
        self.rodata_strings
            .push((label.clone(), content.to_string()));
        label
    }
}
