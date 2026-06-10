//! Module representation in LUMIR.
//!
//! A module is a collection of functions and global data. Modules are the
//! top-level unit of organization in LUMIR and can be compiled independently.
use crate::function::Function;
use crate::instruction::Instruction;
use crate::types::MirType;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Global variable declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Global {
    /// Global variable name
    pub name: String,

    /// Type of the global
    pub ty: MirType,

    /// Whether this is mutable
    pub mutable: bool,

    /// Initial value (as raw bytes)
    pub initializer: Option<Vec<u8>>,
}

impl Global {
    pub fn new(name: impl Into<String>, ty: MirType) -> Self {
        Self {
            name: name.into(),
            ty,
            mutable: true,
            initializer: None,
        }
    }

    pub fn immutable(mut self) -> Self {
        self.mutable = false;
        self
    }

    pub fn with_initializer(mut self, data: Vec<u8>) -> Self {
        self.initializer = Some(data);
        self
    }
}

/// LUMIR module
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    /// Module name
    pub name: String,

    /// Functions in this module
    pub functions: HashMap<String, Function>,

    /// Global variables
    pub globals: HashMap<String, Global>,

    /// Names of external functions (functions with external linkage)
    pub external_functions: HashSet<String>,
}

impl Module {
    /// Create a new module with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: HashMap::new(),
            globals: HashMap::new(),
            external_functions: HashSet::new(),
        }
    }

    /// Add a function to this module
    pub fn add_function(&mut self, func: Function) {
        let name = func.sig.name.clone();
        self.functions.insert(name, func);
    }

    /// Mark a function as external
    pub fn mark_external(&mut self, name: impl Into<String>) {
        self.external_functions.insert(name.into());
    }

    /// Check if a function is external
    pub fn is_external(&self, name: &str) -> bool {
        self.external_functions.contains(name)
    }

    /// Add a global variable to this module
    pub fn add_global(&mut self, global: Global) {
        let name = global.name.clone();
        self.globals.insert(name, global);
    }

    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// Get a mutable reference to a function by name
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.get_mut(name)
    }

    /// Get a global by name
    pub fn get_global(&self, name: &str) -> Option<&Global> {
        self.globals.get(name)
    }

    /// Get all function names
    pub fn function_names(&self) -> Vec<&str> {
        self.functions
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Get all global names
    pub fn global_names(&self) -> Vec<&str> {
        self.globals
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Total number of instructions across all functions
    pub fn instruction_count(&self) -> usize {
        self.functions
            .values()
            .map(Function::instruction_count)
            .sum()
    }

    /// Validate the entire module
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate all functions
        for (name, func) in &self.functions {
            if let Err(e) = func.validate() {
                errors.push(format!("Function '{name}': {e}"));
            }
        }

        // Check for duplicate function/global names
        for func_name in self.functions.keys() {
            if self.globals.contains_key(func_name) {
                errors.push(format!(
                    "Name '{func_name}' used for both function and global"
                ));
            }
        }

        errors.extend(self.validate_internal_call_arities());

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// [`Call`](Instruction::Call) / [`TailCall`](Instruction::TailCall) arity vs in-module callee signatures.
    fn validate_internal_call_arities(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (caller_name, caller) in &self.functions {
            for block in &caller.blocks {
                for inst in &block.instructions {
                    let (op, callee_name, argc) = match inst {
                        Instruction::Call { name, args, .. } => ("call", name.as_str(), args.len()),
                        Instruction::TailCall { name, args } => {
                            ("tailcall", name.as_str(), args.len())
                        }
                        _ => continue,
                    };
                    if let Some(callee) = self.functions.get(callee_name) {
                        let expected = callee.sig.params.len();
                        if argc != expected {
                            out.push(format!(
                                "Function '{caller_name}' {op} '{callee_name}' with {argc} args but callee expects {expected}"
                            ));
                        }
                    }
                }
            }
        }
        out
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Functions
        let mut names: Vec<_> = self.functions.keys().collect();
        names.sort();
        for (i, name) in names.iter().enumerate() {
            if let Some(func) = self.functions.get(*name) {
                writeln!(f, "{func}")?;
                if i < names.len() - 1 {
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

/// Builder for constructing modules
pub struct ModuleBuilder {
    module: Module,
}

impl ModuleBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            module: Module::new(name),
        }
    }

    pub fn function(mut self, func: Function) -> Self {
        self.module.add_function(func);
        self
    }

    pub fn global(mut self, global: Global) -> Self {
        self.module.add_global(global);
        self
    }

    pub fn build(self) -> Module {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::function::{Function, Parameter, Signature};
    use crate::instruction::{Instruction, Operand};
    use crate::register::{Register, VirtualReg};
    use crate::types::{MirType, ScalarType};

    #[test]
    fn test_module_creation() {
        let module = Module::new("test_module");
        assert_eq!(module.name, "test_module");
        assert!(module.functions.is_empty());
        assert!(module.globals.is_empty());
    }

    #[test]
    fn test_module_add_function() {
        let mut module = Module::new("test");
        let func = Function::new(Signature::new("test_func"));

        module.add_function(func);
        assert_eq!(module.functions.len(), 1);
        assert!(module.get_function("test_func").is_some());
    }

    #[test]
    fn test_module_add_global() {
        let mut module = Module::new("test");
        let global = Global::new("my_global", MirType::Scalar(ScalarType::I32));

        module.add_global(global);
        assert_eq!(module.globals.len(), 1);
        assert!(module.get_global("my_global").is_some());
    }

    #[test]
    fn test_module_builder() {
        let func = Function::new(Signature::new("main"));
        let global = Global::new("counter", MirType::Scalar(ScalarType::I64));

        let module = ModuleBuilder::new("my_module")
            .function(func)
            .global(global)
            .build();

        assert_eq!(module.name, "my_module");
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.globals.len(), 1);
    }

    #[test]
    fn test_validate_rejects_tailcall_arity_mismatch() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut sink = Function::new(Signature::new("sink").with_return(i64_ty.clone()));
        let mut sink_entry = Block::new("entry");
        sink_entry.push(Instruction::Ret {
            value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(0)))),
        });
        sink.add_block(sink_entry);

        let mut bad = Function::new(
            Signature::new("bad")
                .with_return(i64_ty.clone())
                .with_params(vec![Parameter::new(
                    Register::Virtual(VirtualReg::gpr(0)),
                    i64_ty.clone(),
                )]),
        );
        let mut bad_entry = Block::new("entry");
        bad_entry.push(Instruction::TailCall {
            name: "sink".to_string(),
            args: vec![Operand::Register(Register::Virtual(VirtualReg::gpr(0)))],
        });
        bad.add_block(bad_entry);

        let mut module = Module::new("m");
        module.add_function(sink);
        module.add_function(bad);

        let errs = module.validate().expect_err("arity mismatch");
        assert!(
            errs.iter()
                .any(|e| e.contains("tailcall") && e.contains("sink")),
            "{errs:?}"
        );
    }
}
