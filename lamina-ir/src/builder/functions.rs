//! Function definition operations for IR builder.
//!
//! This module provides methods for creating functions, defining parameters,
//! and managing function signatures in the IR builder API.

use crate::{
    builder::{IRBuilder, PendingFunction},
    function::{FunctionAnnotation, FunctionParameter, FunctionSignature, VariableAnnotation},
    types::Type,
};

impl<'a> IRBuilder<'a> {
    /// Defines a function with no parameters and opens its `entry` block.
    pub fn function(&mut self, name: &'a str, return_type: Type<'a>) -> &mut Self {
        self.function_with_params(name, vec![], return_type)
    }

    /// Defines a function with the given parameters and opens its `entry` block.
    ///
    /// Panics when a function with the same name is already defined.
    pub fn function_with_params(
        &mut self,
        name: &'a str,
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
    ) -> &mut Self {
        self.push_function(PendingFunction {
            name,
            signature: FunctionSignature {
                params,
                return_type,
                variadic: false,
            },
            annotations: vec![],
            is_extern: false,
            blocks: vec![],
            entry: Some("entry"),
        });
        self.block("entry")
    }

    /// Declares an external function (imported from another module)
    ///
    /// Parameters:
    /// - `name`: Name of the external function
    /// - `params`: Vector of parameter types and names
    /// - `return_type`: Return type of the function
    ///
    /// External functions are declarations only (no implementation) and are
    /// marked with the Extern annotation. This is typically used for declaring
    /// external library functions or functions from other modules.
    pub fn external_function(
        &mut self,
        name: &'a str,
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
    ) -> &mut Self {
        self.push_function(PendingFunction {
            name,
            signature: FunctionSignature {
                params,
                return_type,
                variadic: false,
            },
            annotations: vec![FunctionAnnotation::Extern],
            is_extern: true,
            blocks: vec![],
            entry: None,
        });
        self
    }

    fn push_function(&mut self, function: PendingFunction<'a>) {
        assert!(
            self.functions
                .iter()
                .all(|other| other.name != function.name),
            "function '@{}' is already defined",
            function.name
        );
        self.functions.push(function);
        self.current_function = Some(self.functions.len() - 1);
        self.current_block = None;
    }

    /// Creates a parameter for `function_with_params` or `external_function`.
    pub fn param(
        &mut self,
        name: &'a str,
        ty: Type<'a>,
        annotations: Vec<VariableAnnotation>,
    ) -> FunctionParameter<'a> {
        FunctionParameter {
            name,
            ty,
            annotations,
        }
    }

    /// Creates a parameter with no annotations.
    pub fn param_simple(&mut self, name: &'a str, ty: Type<'a>) -> FunctionParameter<'a> {
        self.param(name, ty, vec![])
    }
}
