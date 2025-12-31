//! # IR Builder
//!
//! A fluent API for programmatically constructing Lamina IR modules.

mod functions;
mod annotations;
mod blocks;
mod memory;
mod arithmetic;
mod control_flow;
mod conversions;
mod pointers;
mod io;
mod tuples;
#[cfg(feature = "nightly")]
mod atomics;
#[cfg(feature = "nightly")]
mod simd;
mod values;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use super::function::{BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature};
use super::instruction::Instruction;
use super::module::Module;
use super::types::Type;

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
/// ## Basic Usage Pattern
///
/// 1. Create a builder with `IRBuilder::new()`
/// 2. Define a function with `function()` or `function_with_params()`
/// 3. Add basic blocks with `block()`
/// 4. Add instructions to the current block
/// 5. Generate the final module with `build()`
pub struct IRBuilder<'a> {
    pub(super) module: Module<'a>,
    pub(super) current_function: Option<&'a str>,
    pub(super) current_block: Option<&'a str>,
    pub(super) block_instructions: HashMap<&'a str, Vec<Instruction<'a>>>,
    pub(super) function_blocks: HashMap<&'a str, HashMap<&'a str, BasicBlock<'a>>>,
    pub(super) function_signatures: HashMap<&'a str, FunctionSignature<'a>>,
    pub(super) function_annotations: HashMap<&'a str, Vec<FunctionAnnotation>>,
    pub(super) function_entry_blocks: HashMap<&'a str, &'a str>,
    pub(super) temp_var_counter: usize,
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
            current_function: None,
            current_block: None,
            block_instructions: HashMap::new(),
            function_blocks: HashMap::new(),
            function_signatures: HashMap::new(),
            function_annotations: HashMap::new(),
            function_entry_blocks: HashMap::new(),
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
    /// Parameters:
    /// - `instruction`: The instruction to add
    ///
    /// This is a low-level method, mainly used internally by other builder methods.
    /// You should prefer using the specialized methods unless you need to add a
    /// custom instruction type.
    pub(super) fn inst(&mut self, instruction: Instruction<'a>) -> &mut Self {
        if let (Some(_func_name), Some(block_name)) = (self.current_function, self.current_block)
            && let Some(instructions) = self.block_instructions.get_mut(block_name)
        {
            instructions.push(instruction);
        }
        self
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
    pub fn build(&mut self) -> Module<'a> {
        // Finalize the module by converting all block instructions to basicblocks
        // and all functions to their final representation

        for (func_name, block_map) in &mut self.function_blocks {
            // Convert block instructions to BasicBlocks
            for (block_name, instructions) in self.block_instructions.iter() {
                // Only process blocks for current function
                if self.block_instructions.contains_key(block_name) {
                    block_map.insert(
                        block_name,
                        BasicBlock {
                            instructions: instructions.clone(),
                        },
                    );
                }
            }

            // Create the function and add it to the module
            if let (Some(signature), Some(entry_block)) = (
                self.function_signatures.get(func_name),
                self.function_entry_blocks.get(func_name),
            ) {
                let function = Function {
                    name: func_name,
                    signature: signature.clone(),
                    annotations: self
                        .function_annotations
                        .get(func_name)
                        .cloned()
                        .unwrap_or_else(Vec::new),
                    basic_blocks: block_map.clone(),
                    entry_block,
                };

                self.module.functions.insert(func_name, function);
            }
        }

        self.module.clone()
    }
}

// Include all the method implementations from submodules
// Each module extends IRBuilder with its category of methods

// External function support
impl<'a> IRBuilder<'a> {
    /// Declares an external function (imported from another module)
    ///
    /// Parameters:
    /// - `name`: Name of the external function
    /// - `params`: Vector of parameter types and names
    /// - `return_type`: Return type of the function
    ///
    /// External functions are declarations only (no implementation) and are
    /// marked with the Export annotation. This is typically used for declaring
    /// external library functions or functions from other modules.
    pub fn external_function(
        &mut self,
        name: &'a str,
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
    ) -> &mut Self {
        let signature = FunctionSignature {
            params,
            return_type,
        };

        self.function_signatures.insert(name, signature);
        self.function_blocks.insert(name, HashMap::new());
        self.function_annotations
            .insert(name, vec![FunctionAnnotation::Extern]);
        self.current_function = Some(name);

        self
    }
}

