use crate::{Identifier, Label, Type};
use std::collections::HashMap;

/// Represents where a value is stored in a backend-agnostic way
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLocation {
    /// Value is stored in a register
    Register(String),
    /// Value is stored on the stack at an offset from the frame pointer
    StackOffset(i64),
    /// Value is stored in memory (global variables, string literals, etc.)
    Memory(String),
    /// Value is an immediate constant
    Immediate(i64),
}

impl ValueLocation {
    /// Convert location to an assembly operand string (backend-specific implementation needed)
    pub fn to_operand_string(&self) -> String {
        match self {
            ValueLocation::Register(reg) => reg.clone(),
            ValueLocation::StackOffset(offset) => format!("{}(%rbp)", offset), // x86_64 style, needs backend override
            ValueLocation::Memory(addr) => addr.clone(),
            ValueLocation::Immediate(val) => format!("${}", val),
        }
    }
}

/// Register allocation information
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Variable name to register mapping
    pub assignments: HashMap<String, String>,
    /// Variables that need to be spilled to memory
    pub spilled_vars: std::collections::HashSet<String>,
    /// Callee-saved registers actually used
    pub callee_saved_used: std::collections::HashSet<String>,
}

/// Stack frame layout information
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Total size of the stack frame
    pub total_size: u64,
    /// Size of local variables area
    pub locals_size: u64,
    /// Size of spilled registers area
    pub spills_size: u64,
    /// Size of outgoing arguments area
    pub outgoing_args_size: u64,
    /// Alignment requirement for the frame
    pub alignment: u64,
}

/// Function signature information for codegen
#[derive(Debug, Clone)]
pub struct FunctionSignature<'a> {
    /// Function name
    pub name: &'a str,
    /// Parameter information
    pub params: Vec<Parameter<'a>>,
    /// Return type
    pub return_type: Option<Type<'a>>,
    /// Whether function is exported
    pub is_exported: bool,
}

/// Parameter information
#[derive(Debug, Clone)]
pub struct Parameter<'a> {
    /// Parameter name
    pub name: Identifier<'a>,
    /// Parameter type
    pub ty: Type<'a>,
    /// Location where parameter is passed (register or stack)
    pub location: Option<ValueLocation>,
}

/// Basic block information for codegen
#[derive(Debug, Clone)]
pub struct BasicBlockInfo<'a> {
    /// IR label for the block
    pub ir_label: Label<'a>,
    /// Assembly label for the block
    pub asm_label: String,
    /// Whether this is the entry block
    pub is_entry: bool,
    /// Predecessor blocks
    pub predecessors: Vec<Label<'a>>,
    /// Successor blocks
    pub successors: Vec<Label<'a>>,
}

/// Live range information for register allocation
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Variable name
    pub var: String,
    /// Start position (instruction index)
    pub start: usize,
    /// End position (instruction index)
    pub end: usize,
    /// All use positions
    pub uses: Vec<usize>,
    /// Spill cost (higher = more expensive to spill)
    pub spill_cost: f64,
}

/// Interference graph node for register allocation
#[derive(Debug, Clone)]
pub struct InterferenceNode {
    /// Variable name
    pub var: String,
    /// Interfering variables
    pub neighbors: std::collections::HashSet<String>,
    /// Number of neighbors (degree in interference graph)
    pub degree: usize,
    /// Cost of spilling this variable
    pub spill_cost: f64,
    /// Preferred registers for this variable
    pub preferred_registers: Vec<String>,
}

/// Global variable layout information
#[derive(Debug, Clone)]
pub struct GlobalLayout {
    /// Map from global name to assembly label
    pub label_map: HashMap<String, String>,
    /// Read-only data strings
    pub rodata_strings: Vec<(String, String)>,
    /// Initialized data globals
    pub data_globals: Vec<String>,
    /// Uninitialized data globals
    pub bss_globals: Vec<String>,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable register coalescing
    pub register_coalescing: bool,
    /// Enable peephole optimizations
    pub peephole_optimizations: bool,
    /// Enable function inlining
    pub function_inlining: bool,
    /// Optimization level (0-3)
    pub level: u8,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            register_coalescing: true,
            peephole_optimizations: true,
            function_inlining: false,
            level: 1,
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Target architecture
    pub architecture: Architecture,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Generate debug information
    pub debug_info: bool,
    /// Use position-independent code
    pub position_independent: bool,
}

/// Supported architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    AArch64,
    RiscV64,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::X86_64,
            optimization: OptimizationConfig::default(),
            debug_info: false,
            position_independent: false,
        }
    }
}

/// Error types specific to codegen
#[derive(Debug, Clone)]
pub enum CodegenError {
    /// Register allocation failed
    RegisterAllocationFailed(String),
    /// Invalid instruction for target
    InvalidInstruction(String),
    /// Stack overflow (too many locals)
    StackOverflow,
    /// Unsupported feature
    UnsupportedFeature(String),
    /// Internal error
    InternalError(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::RegisterAllocationFailed(msg) => {
                write!(f, "Register allocation failed: {}", msg)
            }
            CodegenError::InvalidInstruction(msg) => {
                write!(f, "Invalid instruction: {}", msg)
            }
            CodegenError::StackOverflow => {
                write!(f, "Stack overflow: too many local variables")
            }
            CodegenError::UnsupportedFeature(feature) => {
                write!(f, "Unsupported feature: {}", feature)
            }
            CodegenError::InternalError(msg) => {
                write!(f, "Internal codegen error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CodegenError {}

/// Result type for codegen operations
pub type CodegenResult<T> = std::result::Result<T, CodegenError>;
