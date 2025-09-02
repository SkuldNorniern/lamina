use crate::{Function, Instruction, Module, Result};
use std::collections::HashMap;
use std::io::Write;

/// Core trait that all codegen backends must implement
pub trait CodegenBackend<'a> {
    type State: CodegenState<'a>;
    type Context: FunctionContext<'a>;

    /// Generate assembly for an entire module
    fn generate_assembly<W: Write>(&mut self, module: &'a Module<'a>, writer: &mut W)
    -> Result<()>;

    /// Generate assembly for a single function
    fn generate_function<W: Write>(
        &mut self,
        func_name: &'a str,
        func: &'a Function<'a>,
        writer: &mut W,
        state: &mut Self::State,
    ) -> Result<()>;

    /// Get the calling convention for this backend
    fn calling_convention(&self) -> &dyn CallingConvention;

    /// Get register information for this backend
    fn register_info(&self) -> &dyn RegisterInfo;
}

/// Trait for managing codegen state across a module
pub trait CodegenState<'a> {
    /// Create new empty state
    fn new() -> Self;

    /// Generate a new unique label
    fn new_label(&mut self, prefix: &str) -> String;

    /// Add a read-only string and return its label
    fn add_rodata_string(&mut self, content: &str) -> String;

    /// Get global variable layout information
    fn get_global_layout(&self) -> &HashMap<&'a str, String>;
}

/// Trait for managing function-specific context
pub trait FunctionContext<'a> {
    /// Create new empty context
    fn new() -> Self;

    /// Get the location of a value (register or stack)
    fn get_value_location(&self, name: &'a str) -> Result<ValueLocation>;

    /// Get the assembly label for a basic block
    fn get_block_label(&self, ir_label: &'a str) -> Result<String>;

    /// Get the total stack size needed
    fn get_stack_size(&self) -> u64;
}

/// Represents where a value is stored
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLocation {
    Register(String),
    StackOffset(i64),
    Memory(String), // For global variables, etc.
}

/// Trait for instruction generation
pub trait InstructionGenerator<'a> {
    type State: CodegenState<'a>;
    type Context: FunctionContext<'a>;

    /// Generate assembly for a single instruction
    fn generate_instruction<W: Write>(
        &self,
        instr: &Instruction<'a>,
        writer: &mut W,
        state: &mut Self::State,
        context: &Self::Context,
    ) -> Result<()>;
}

/// Trait for calling convention handling
pub trait CallingConvention {
    /// Get argument registers in order
    fn arg_registers(&self) -> &[&'static str];

    /// Get return register
    fn return_register(&self) -> &'static str;

    /// Get callee-saved registers
    fn callee_saved_registers(&self) -> &[&'static str];

    /// Get caller-saved registers
    fn caller_saved_registers(&self) -> &[&'static str];

    /// Calculate stack offset for nth argument (0-indexed)
    fn stack_arg_offset(&self, arg_index: usize) -> i64;
}

/// Trait for register allocation
pub trait RegisterAllocator<'a> {
    type AllocationResult;

    /// Allocate registers for a function
    fn allocate_registers(&mut self, func: &'a Function<'a>) -> Result<Self::AllocationResult>;
}

/// Trait for register information
pub trait RegisterInfo {
    /// Check if a register is general purpose
    fn is_gp_register(&self, reg: &str) -> bool;

    /// Check if a register is callee-saved
    fn is_callee_saved(&self, reg: &str) -> bool;

    /// Check if a register is caller-saved
    fn is_caller_saved(&self, reg: &str) -> bool;

    /// Get register name with size suffix
    fn get_sized_register(&self, base_reg: &str, size_bytes: u64) -> String;
}

/// Trait for stack layout computation
pub trait StackLayoutComputer<'a> {
    type Context: FunctionContext<'a>;

    /// Compute stack layout for a function
    fn compute_layout(&self, func: &'a Function<'a>, context: &mut Self::Context) -> Result<()>;
}

/// Trait for global variable handling
pub trait GlobalGenerator<'a> {
    type State: CodegenState<'a>;

    /// Generate global variable sections
    fn generate_globals<W: Write>(
        &self,
        module: &'a Module<'a>,
        writer: &mut W,
        state: &mut Self::State,
    ) -> Result<()>;
}
