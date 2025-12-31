//! Basic block management for IR builder.
//!
//! This module provides methods for creating and managing basic blocks
//! within functions. Basic blocks are sequences of instructions with
//! a single entry and exit point.

use super::IRBuilder;

impl<'a> IRBuilder<'a> {
    /// Creates a new basic block in the current function
    ///
    /// Parameters:
    /// - `name`: The block label (used for branching)
    ///
    /// Creates a new basic block and sets it as the current block for
    /// subsequent instruction additions. Must be called while a function
    /// is active.
    pub fn block(&mut self, name: &'a str) -> &mut Self {
        if self.current_function.is_some() {
            self.current_block = Some(name);
            self.block_instructions.insert(name, vec![]);
        }
        self
    }

    /// Sets the entry block for the current function
    ///
    /// Parameters:
    /// - `name`: The block name to use as entry point
    ///
    /// By default, the first block created ("entry") will be the entry point.
    /// Use this method to override that behavior.
    pub fn set_entry_block(&mut self, name: &'a str) -> &mut Self {
        if let Some(func_name) = self.current_function {
            self.function_entry_blocks.insert(func_name, name);
        }
        self
    }
}

