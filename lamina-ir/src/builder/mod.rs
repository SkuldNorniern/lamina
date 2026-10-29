//! # IR Builder
//!
//! A fluent API for programmatically constructing Lamina IR modules.

mod annotations;
mod arithmetic;
mod atomics;
mod blocks;
pub(crate) mod checks;
mod control_flow;
mod conversions;
mod functions;
mod io;
mod memory;
mod pointers;
mod simd;
mod types;
mod values;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::{
    function::{BasicBlock, Function, FunctionAnnotation, FunctionSignature},
    instruction::Instruction,
    module::Module,
};

// Re-export value factory functions
pub use values::*;

/// # IR Builder
///
/// A fluent API for programmatically constructing Lamina IR modules.
///
/// ## Overview
///
/// The `IRBuilder` allows you to construct IR code in a safe, programmatic way without
/// having to manually build instruction objects. It provides methods for all IR operations
/// and maintains the context of the current function and basic block.
///
/// Every method emits IR that the text parser accepts. Misuse that would produce IR the
/// parser rejects, or a different meaning than requested, panics with the replacement to use.
///
/// ## Basic Usage Pattern
///
/// 1. Create a builder with `IRBuilder::new()`
/// 2. Define a function with `function()` or `function_with_params()`
/// 3. Add basic blocks with `block()`
/// 4. Add instructions to the current block
/// 5. Generate the final module with `build()`
pub struct IRBuilder<'a> {
    pub(super) module: Module<'a>,
    pub(super) functions: Vec<PendingFunction<'a>>,
    pub(super) current_function: Option<usize>,
    pub(super) current_block: Option<usize>,
    pub(super) temp_var_counter: usize,
}

/// A function under construction, kept in definition order.
pub(super) struct PendingFunction<'a> {
    pub(super) name: &'a str,
    pub(super) signature: FunctionSignature<'a>,
    pub(super) annotations: Vec<FunctionAnnotation>,
    pub(super) is_extern: bool,
    pub(super) blocks: Vec<PendingBlock<'a>>,
    pub(super) entry: Option<&'a str>,
}

/// A basic block under construction. Labels are scoped to their function.
pub(super) struct PendingBlock<'a> {
    pub(super) name: &'a str,
    pub(super) instructions: Vec<Instruction<'a>>,
}

impl Default for IRBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IRBuilder<'a> {
    /// Creates a new empty IR builder
    ///
    /// Initializes a builder with no functions or blocks.
    pub fn new() -> Self {
        IRBuilder {
            module: Module::new(),
            functions: Vec::new(),
            current_function: None,
            current_block: None,
            temp_var_counter: 0,
        }
    }

    /// Generates a unique temporary variable name
    ///
    /// Returns: A fresh variable name in the format "temp_N" where N is a counter
    ///
    /// Use this when you need a variable but don't care about its specific name.
    /// Each call returns a new name that won't conflict with previous ones.
    pub fn temp_var(&mut self) -> String {
        let var = format!("temp_{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        var
    }

    /// Adds a raw instruction to the current block
    ///
    /// Panics when no block is open, when the current function is an extern declaration,
    /// or when the instruction would print a string literal, which only global initializers
    /// may contain.
    pub(super) fn inst(&mut self, instruction: Instruction<'a>) -> &mut Self {
        let text = instruction.to_string();
        assert!(
            !text.contains('"'),
            "string literals only initialize globals, not instruction operands: {text}"
        );
        let open_block = self.current_block;
        let function = self.open_body();
        let block = open_block
            .unwrap_or_else(|| panic!("no open block in '@{}'; call block() first", function.name));
        let pending = &mut function.blocks[block];
        assert!(
            pending
                .instructions
                .last()
                .is_none_or(|last| !last.is_terminator()),
            "block '{}' already ends with a terminator; start a new block with block()",
            pending.name
        );
        pending.instructions.push(instruction);
        self
    }

    /// Returns the function currently being built, or panics when none is open.
    #[allow(clippy::expect_used)]
    pub(super) fn open_function(&mut self) -> &mut PendingFunction<'a> {
        let index = self
            .current_function
            .expect("no open function; call function() or external_function() first");
        &mut self.functions[index]
    }

    /// Returns the function currently being built, or panics when none is open.
    pub(super) fn open_body(&mut self) -> &mut PendingFunction<'a> {
        let function = self.open_function();
        assert!(
            !function.is_extern,
            "extern function '@{}' has no body; declare the body with function() instead",
            function.name
        );
        function
    }

    /// Finalizes and returns the complete IR module
    ///
    /// This method converts all the accumulated function and block data
    /// into a complete Module object that can be used for code generation,
    /// optimization, or serialization.
    ///
    /// Call this method only once all functions and instructions have been added.
    ///
    /// Returns: A Module object containing all defined functions and blocks
    ///
    /// Panics when a function's entry block was set to a label that was never defined, or
    /// when a block does not end with a terminator.
    pub fn build(&mut self) -> Module<'a> {
        let mut module = self.module.clone();
        for function in &self.functions {
            for block in &function.blocks {
                assert!(
                    block
                        .instructions
                        .last()
                        .is_some_and(|last| last.is_terminator()),
                    "block '{}' of '@{}' must end with a terminator (ret, jmp, br, switch, unreachable)",
                    block.name,
                    function.name
                );
            }
            let basic_blocks: HashMap<&'a str, BasicBlock<'a>> = function
                .blocks
                .iter()
                .map(|block| {
                    (
                        block.name,
                        BasicBlock {
                            instructions: block.instructions.clone(),
                            instruction_locs: Vec::new(),
                        },
                    )
                })
                .collect();
            let entry_block = if function.is_extern {
                ""
            } else {
                let entry = function.entry.unwrap_or("entry");
                assert!(
                    basic_blocks.contains_key(entry),
                    "entry block '{entry}' of '@{}' is not defined",
                    function.name
                );
                entry
            };
            module.functions.insert(
                function.name,
                Function {
                    name: function.name,
                    signature: function.signature.clone(),
                    annotations: function.annotations.clone(),
                    basic_blocks,
                    entry_block,
                },
            );
        }
        module
    }
}
