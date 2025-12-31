//! Function definition operations for IR builder.
//!
//! This module provides methods for creating functions, defining parameters,
//! and managing function signatures in the IR builder API.

use super::IRBuilder;
use crate::ir::function::{FunctionParameter, FunctionSignature};
use crate::ir::types::Type;

impl<'a> IRBuilder<'a> {
    /// Creates a new function with no parameters
    ///
    /// Parameters:
    /// - `name`: The function name (without @ prefix)
    /// - `return_type`: The function's return type
    pub fn function(&mut self, name: &'a str, return_type: Type<'a>) -> &mut Self {
        self.function_with_params(name, vec![], return_type)
    }

    /// Creates a new function with parameters
    ///
    /// Parameters:
    /// - `name`: The function name (without @ prefix)
    /// - `params`: Vector of function parameters
    /// - `return_type`: The function's return type
    pub fn function_with_params(
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
        self.function_blocks.insert(name, std::collections::HashMap::new());
        self.function_annotations.insert(name, vec![]);
        self.current_function = Some(name);

        // Default entry block
        self.block("entry");
        self.function_entry_blocks.insert(name, "entry");

        self
    }

    /// Creates a function parameter with optional annotations
    ///
    /// Parameters:
    /// - `name`: Parameter name (without % prefix)
    /// - `ty`: Parameter type
    /// - `annotations`: Optional vector of parameter annotations
    pub fn param(
        &mut self,
        name: &'a str,
        ty: Type<'a>,
        annotations: Vec<crate::ir::function::VariableAnnotation>,
    ) -> FunctionParameter<'a> {
        FunctionParameter {
            name,
            ty,
            annotations,
        }
    }

    /// Creates a simple function parameter without annotations
    ///
    /// Parameters:
    /// - `name`: Parameter name (without % prefix)
    /// - `ty`: Parameter type
    pub fn param_simple(&mut self, name: &'a str, ty: Type<'a>) -> FunctionParameter<'a> {
        self.param(name, ty, vec![])
    }
}

