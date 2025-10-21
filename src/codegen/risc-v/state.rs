use crate::codegen::CodegenError;
use crate::{Identifier, Label, LaminaError, Result};
use std::collections::{HashMap, HashSet};

use super::IsaWidth;

// RISC-V integer/pointer argument registers (System V ABI)
pub const ARG_REGISTERS: [&str; 8] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"];

// RISC-V return register for integer/pointer
pub const RETURN_REGISTER: &str = "a0";

// RISC-V callee-saved registers that may need to be preserved
pub const CALLEE_SAVED_REGISTERS: [&str; 12] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
];

/// Size of a register in bytes for a given ISA width
pub fn register_size_bytes(width: IsaWidth) -> i64 {
    match width {
        IsaWidth::Rv32 => 4,
        IsaWidth::Rv64 => 8,
        IsaWidth::Rv128 => 16,
    }
}

/// Represents the location of an IR value in generated code
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLocation {
    /// Stored in a specific register (e.g., "a0" for first arg)
    Register(String),
    /// Offset relative to frame pointer s0 (negative offsets for locals/spills)
    StackOffset(i64),
}

impl ValueLocation {
    /// Render as an operand string (used in comments or simple cases)
    pub fn to_operand_string(&self) -> String {
        match self {
            ValueLocation::Register(reg) => reg.clone(),
            ValueLocation::StackOffset(offset) => format!("{}(s0)", offset),
        }
    }
}

/// Per-function context tracking stack layout, value locations, labels, etc.
#[derive(Debug)]
pub struct FunctionContext<'a> {
    pub value_locations: HashMap<Identifier<'a>, ValueLocation>,
    pub block_labels: HashMap<Label<'a>, String>,
    pub total_stack_size: u64,
    pub epilogue_label: String,
    pub callee_saved_regs: Vec<&'static str>,
    pub stack_allocated_vars: HashSet<Identifier<'a>>, // present for parity; not used yet
}

impl<'a> FunctionContext<'a> {
    pub fn new() -> Self {
        Self {
            value_locations: HashMap::new(),
            block_labels: HashMap::new(),
            total_stack_size: 0,
            epilogue_label: String::new(),
            callee_saved_regs: Vec::new(),
            stack_allocated_vars: HashSet::new(),
        }
    }

    pub fn get_block_label(&self, ir_label: &Label<'a>) -> Result<String> {
        self.block_labels.get(ir_label).cloned().ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::BlockLabelNotFound(ir_label.to_string()))
        })
    }

    pub fn get_value_location(&self, name: Identifier<'a>) -> Result<ValueLocation> {
        self.value_locations.get(name).cloned().ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::ValueLocationNotFound(name.to_string()))
        })
    }
}

impl Default for FunctionContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Global codegen state for the entire module
#[derive(Debug)]
pub struct CodegenState<'a> {
    pub global_layout: HashMap<Identifier<'a>, String>,
    pub rodata_strings: Vec<(String, String)>, // label -> content
    pub next_label_id: u32,
    pub next_rodata_id: u32,
    width: IsaWidth,
}

impl<'a> CodegenState<'a> {
    pub fn new(width: IsaWidth) -> Self {
        CodegenState {
            global_layout: HashMap::new(),
            rodata_strings: Vec::new(),
            next_label_id: 0,
            next_rodata_id: 0,
            width,
        }
    }

    pub fn width(&self) -> IsaWidth {
        self.width
    }

    pub fn new_label(&mut self, prefix: &str) -> String {
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


