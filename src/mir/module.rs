/// Module representation in LUMIR
///
/// A module is a collection of functions and global data.
use super::function::Function;
use super::types::MirType;
use std::collections::HashMap;

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
}

impl Module {
    /// Create a new module with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: HashMap::new(),
            globals: HashMap::new(),
        }
    }

    /// Add a function to this module
    pub fn add_function(&mut self, func: Function) {
        let name = func.sig.name.clone();
        self.functions.insert(name, func);
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
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Get all global names
    pub fn global_names(&self) -> Vec<&str> {
        self.globals.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of instructions across all functions
    pub fn instruction_count(&self) -> usize {
        self.functions.values().map(|f| f.instruction_count()).sum()
    }

    /// Validate the entire module
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate all functions
        for (name, func) in &self.functions {
            if let Err(e) = func.validate() {
                errors.push(format!("Function '{}': {}", name, e));
            }
        }

        // Check for duplicate function/global names
        for func_name in self.functions.keys() {
            if self.globals.contains_key(func_name) {
                errors.push(format!(
                    "Name '{}' used for both function and global",
                    func_name
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
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
    use crate::mir::function::{Function, Signature};
    use crate::mir::types::{MirType, ScalarType};

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
}

