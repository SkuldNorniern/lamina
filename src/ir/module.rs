//! # Module Representation
//!
//! This module defines the top-level structures for representing complete
//! Lamina IR modules. A module is the highest level of organization and
//! contains all functions, types, and global variables.
//!
//! ## Module Structure
//!
//! A module consists of:
//! - **Functions**: All function definitions in the module
//! - **Type Declarations**: Named type definitions and aliases
//! - **Global Variables**: Module-level data and constants
//!
//! ## Module Organization
//!
//! - **Functions**: The main code of the module, organized by name
//! - **Types**: User-defined types that can be referenced by functions
//! - **Globals**: Module-level data that persists for the lifetime of the program
//!
//! ## Examples
//!
//! ```rust
//! use lamina::{IRBuilder, Type, PrimitiveType, var, i32, string};
//!
//! let mut builder = IRBuilder::new();
//! 
//! // Define a global variable
//! builder
//!     .function("main", Type::Void)
//!     .print(string("Hello, World!"))
//!     .ret_void();
//!
//! let module = builder.build();
//! // module contains all the functions and types
//! ```

use std::collections::HashMap; // Using HashMap for functions and types
use std::fmt;

use super::function::Function;
use super::types::{Identifier, Type, Value};

/// Represents a named type declaration (e.g., `type @Vec2 = struct { ... }`).
///
/// Type declarations allow you to define custom types that can be referenced
/// throughout the module. This is useful for creating reusable type definitions
/// and making the IR more readable.
///
/// # Examples
///
/// ```rust
/// use lamina::{TypeDeclaration, Type, StructField, PrimitiveType};
///
/// let point_type = TypeDeclaration {
///     name: "Point",
///     ty: Type::Struct(vec![
///         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
///         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) },
///     ]),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Eq might be too restrictive if Type contains f32 indirectly, but should be okay here.
pub struct TypeDeclaration<'a> {
    /// The type name (without the `@` prefix)
    pub name: Identifier<'a>, // e.g., "@Vec2"
    /// The type definition
    pub ty: Type<'a>,
}

/// Represents a global variable declaration (e.g., `global @message: [5 x i8] = "hello"`).
///
/// Global variables are module-level data that persists for the lifetime of the program.
/// They can be initialized with constant values or left uninitialized (for external symbols).
///
/// # Examples
///
/// ```rust
/// use lamina::{GlobalDeclaration, Type, PrimitiveType, Value};
///
/// let message_global = GlobalDeclaration {
///     name: "message",
///     ty: Type::Array {
///         element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
///         size: 13,
///     },
///     initializer: Some(Value::Constant(Literal::String("Hello, World!"))),
/// };
/// ```
#[derive(Debug, Clone, PartialEq)] // PartialEq due to Value potentially holding f32
pub struct GlobalDeclaration<'a> {
    /// The global variable name (without the `@` prefix)
    pub name: Identifier<'a>, // e.g., "@message"
    /// The type of the global variable
    pub ty: Type<'a>,
    /// Optional initializer value (None for external symbols)
    pub initializer: Option<Value<'a>>, // Globals might be uninitialized (extern)
}

/// Represents a complete Lamina IR module (file).
///
/// A module is the top-level container for all IR code. It contains functions,
/// type declarations, and global variables that together form a complete program.
///
/// # Structure
///
/// - **Functions**: All function definitions, keyed by name
/// - **Type Declarations**: Named type definitions, keyed by name
/// - **Global Variables**: Module-level data, keyed by name
///
/// # Examples
///
/// ```rust
/// use lamina::{IRBuilder, Type, PrimitiveType, var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("add", Type::Primitive(PrimitiveType::I32))
///     .binary(BinaryOp::Add, "result", PrimitiveType::I32, i32(10), i32(20))
///     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
///
/// let module = builder.build();
/// // module.functions contains the "add" function
/// // module.type_declarations is empty
/// // module.global_declarations is empty
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Module<'a> {
    /// Named type declarations in this module
    ///
    /// Using a HashMap allows efficient lookup by type name. The order
    /// of type declarations is not preserved, but this is typically not critical.
    pub type_declarations: HashMap<Identifier<'a>, TypeDeclaration<'a>>,
    /// Global variable declarations in this module
    ///
    /// Global variables are module-level data that persists for the lifetime
    /// of the program. They can be initialized with constant values.
    pub global_declarations: HashMap<Identifier<'a>, GlobalDeclaration<'a>>,
    /// Function definitions in this module
    ///
    /// Functions contain the actual code of the module. They are organized
    /// by name for efficient lookup during function calls.
    pub functions: HashMap<Identifier<'a>, Function<'a>>,
}

impl Module<'_> {
    // Basic constructor
    pub fn new() -> Self {
        Module {
            type_declarations: HashMap::new(),
            global_declarations: HashMap::new(),
            functions: HashMap::new(),
        }
    }
}

impl Default for Module<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Display Implementations ---

impl fmt::Display for TypeDeclaration<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type @{} = {}", self.name, self.ty)
    }
}

impl fmt::Display for GlobalDeclaration<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "global @{}: {}", self.name, self.ty)?;
        if let Some(init) = &self.initializer {
            write!(f, " = {}", init)?;
        }
        Ok(())
    }
}

impl fmt::Display for Module<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print type declarations first
        for decl in self.type_declarations.values() {
            writeln!(f, "{}", decl)?;
        }
        if !self.type_declarations.is_empty() {
            writeln!(f)?;
        }

        // Print global declarations
        for decl in self.global_declarations.values() {
            writeln!(f, "{}", decl)?;
        }
        if !self.global_declarations.is_empty() {
            writeln!(f)?;
        }

        // Print functions
        let mut function_names: Vec<_> = self.functions.keys().collect();
        function_names.sort(); // Sort by name for consistent output order

        for (i, name) in function_names.iter().enumerate() {
            // Dereference name twice (&&str -> &str) to match the key type for lookup
            if let Some(func) = self.functions.get(*name) {
                write!(f, "{}", func)?;
                if i < function_names.len() - 1 {
                    writeln!(f)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::{BasicBlock, Function, FunctionParameter, FunctionSignature};
    use crate::ir::instruction::{BinaryOp, Instruction};
    use crate::ir::types::{Literal, PrimitiveType, Type, Value}; // Assuming crate root
    use std::collections::HashMap;

    #[test]
    fn test_display_type_declaration() {
        let decl = TypeDeclaration {
            name: "MyType",
            ty: Type::Struct(vec![]), // Empty struct
        };
        // Expect two spaces for empty struct {  }
        assert_eq!(format!("{}", decl), "type @MyType = struct {  }");
    }

    #[test]
    fn test_display_global_declaration() {
        // Initialized
        let decl1 = GlobalDeclaration {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Constant(Literal::I32(0))),
        };
        assert_eq!(format!("{}", decl1), "global @count: i32 = 0");

        // Uninitialized (Extern)
        let decl2 = GlobalDeclaration {
            name: "external_data",
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 256,
            },
            initializer: None,
        };
        assert_eq!(format!("{}", decl2), "global @external_data: [256 x i8]");
    }

    #[test]
    fn test_display_module() {
        let mut module = Module::new();

        // Add type declaration
        module.type_declarations.insert(
            "Vec2",
            TypeDeclaration {
                name: "Vec2",
                ty: Type::Struct(vec![
                    crate::ir::types::StructField {
                        name: "x",
                        ty: Type::Primitive(PrimitiveType::F32),
                    },
                    crate::ir::types::StructField {
                        name: "y",
                        ty: Type::Primitive(PrimitiveType::F32),
                    },
                ]),
            },
        );

        // Add global declaration
        module.global_declarations.insert(
            "PI",
            GlobalDeclaration {
                name: "PI",
                ty: Type::Primitive(PrimitiveType::F32),
                initializer: Some(Value::Constant(Literal::F32(3.14159))),
            },
        );

        // Add a simple function
        let func_sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "a",
                ty: Type::Primitive(PrimitiveType::I32),
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };
        let mut func_blocks = HashMap::new();
        func_blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "res",
                        ty: PrimitiveType::I32,
                        lhs: Value::Variable("a"),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Ret {
                        ty: Type::Primitive(PrimitiveType::I32),
                        value: Some(Value::Variable("res")),
                    },
                ],
            },
        );
        let func = Function {
            name: "add_one",
            signature: func_sig,
            annotations: vec![],
            basic_blocks: func_blocks,
            entry_block: "entry",
        };
        module.functions.insert("add_one", func);

        // Match actual output - no indentation for function instructions
        let expected_output = "type @Vec2 = struct { x: f32, y: f32 }\n\nglobal @PI: f32 = 3.14159\n\nfn @add_one(i32 %a) -> i32 {\nentry:\n  %res = add.i32 %a, 1\n  ret.i32 %res\n}\n";

        // Note: Hashmap iteration order isn't guaranteed, but Display impl sorts function keys.
        // Type/Global order isn't sorted, so this test might be fragile if more are added.
        // A more robust test might parse the output or check for substrings.
        assert_eq!(format!("{}", module), expected_output);
    }

    #[test]
    fn test_display_empty_module() {
        let module = Module::<'static>::new();
        assert_eq!(format!("{}", module), ""); // Empty module should print nothing
    }
}
