//! # Module Representation
//!
//! Top-level structures for representing complete
//! Lamina IR modules. A module is the highest level of organization and
//! contains all functions, types, and global variables that make up a program.
//!
//! ## Module Structure
//!
//! A module consists of:
//! - **Functions**: All function definitions in the module (the executable code)
//! - **Type Declarations**: Named type definitions and aliases for code reuse
//! - **Global Variables**: Module-level data that persists for the lifetime of the program
//! - **Metadata**: Additional information about the module (version, dependencies, etc.)
//!
//! ## Module Organization
//!
//! ### Functions (`HashMap<&str, Function>`)
//! The main code of the module, organized by name for fast lookup:
//! - **Entry Points**: Functions that can be called from outside the module
//! - **Internal Functions**: Helper functions used within the module
//! - **Exported Functions**: Functions marked with `@export` for external use
//!
//! ### Types (`HashMap<&str, TypeDeclaration>`)
//! User-defined types that can be referenced by functions:
//! - **Struct Types**: Custom data structures with named fields
//! - **Array Types**: Fixed-size sequences of elements
//! - **Alias Types**: Shorter names for complex type expressions
//!
//! ### Globals (`HashMap<&str, GlobalDeclaration>`)
//! Module-level data that persists for the lifetime of the program:
//! - **Constants**: Immutable values known at compile time
//! - **Static Data**: Mutable data with module lifetime
//! - **External References**: References to data in other modules
//!
//! ## Module Lifecycle
//!
//! ### Construction Phase
//! ```rust
//! use lamina_ir::IRBuilder;
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("main", lamina_ir::Type::Void)
//!     // ... add instructions
//!     .ret_void();
//! ```
//!
//! ### Build Phase
//! ```rust
//! # use lamina_ir::IRBuilder;
//! let mut builder = IRBuilder::new();
//! let module = builder.build(); // Convert to Module struct
//! ```
//!
//! ### Code Generation Phase
//! ```rust
//! # use lamina_ir::IRBuilder;
//!
//! let mut builder = IRBuilder::new();
//! let module = builder.build();
//! let mut assembly: Vec<u8> = Vec::new();
//! // Code generation would write assembly bytes to the vector
//! ```
//!
//! ## Module Validation
//!
//! Modules are validated to ensure:
//! - **Function Signatures**: All referenced functions exist and have compatible signatures
//! - **Type Consistency**: All type references are valid and properly defined
//! - **Global References**: All global variables are properly declared
//! - **SSA Form**: All variables follow Single Static Assignment rules
//! - **Control Flow**: All basic blocks have valid terminators
//!
//! ## Examples
//!
//! ### Simple Module
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType};
//! use lamina_ir::builder::i32;
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function("main", Type::Void)
//!     .print(i32(42))
//!     .ret_void();
//!
//! let module = builder.build();
//! assert!(module.functions.contains_key("main"));
//! ```
//!
//! ### Module with Custom Types
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, StructField};
//! use lamina_ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function("use_custom_type", Type::Void)
//!     // Allocate a struct with two i32 fields
//!     .alloc_stack("point", Type::Struct(vec![
//!         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
//!         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) }
//!     ]))
//!     // Use the allocated struct
//!     .ret_void();
//!
//! let module = builder.build();
//! ```
//!
//! ### Multi-Function Module
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina_ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//!
//! // Helper function
//! builder
//!     .function("double", Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Mul, "result", PrimitiveType::I32, var("value"), i32(2))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//!
//! // Main function
//! builder
//!     .function("main", Type::Void)
//!     .call(Some("result"), "double", vec![i32(21)])
//!     .print(var("result")) // Should print 42
//!     .ret_void();
//!
//! let module = builder.build();
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::function::Function;
use crate::types::{Identifier, Type, Value};

/// A named type declaration (e.g., `type @Vec2 = struct { ... }`).
///
/// Type declarations allow you to define custom types that can be referenced
/// throughout the module. This is useful for creating reusable type definitions
/// and making the IR more readable.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{TypeDeclaration, Type, StructField, PrimitiveType, Value, Literal};
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

/// A global variable declaration (e.g., `global @message: [5 x i8] = "hello"`).
///
/// Global variables are module-level data that persists for the lifetime of the program.
/// They can be initialized with constant values or left uninitialized (for external symbols).
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{GlobalDeclaration, Type, PrimitiveType, Value, Literal};
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

/// A complete Lamina IR module (file).
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
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina_ir::builder::{var, i32};
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
///
/// ## Module Annotations
///
/// Modules can have attributes that affect how the entire compilation unit is treated.
/// These annotations provide global control over optimization, linking, and code generation.
///
/// ### Available Module Annotations
/// - **`@pic`**: Generate position-independent code
/// - **`@pie`**: Generate position-independent executable
/// - **`@optimize_speed`**: Optimize for execution speed
/// - **`@optimize_size`**: Optimize for code size
/// - **`@debug`**: Include debug information
/// - **`@strip`**: Strip debug information and symbols
/// - **`@target_triple`**: Specify target triple (e.g., "x86_64-linux-gnu")
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleAnnotation {
    /// Generate position-independent code (PIC).
    ///
    /// PIC lets the code be loaded at any address in memory, which is
    /// required for shared libraries and improves security through ASLR.
    PositionIndependentCode,

    /// Generate position-independent executable (PIE).
    ///
    /// PIE creates executables that can be loaded at random addresses,
    /// providing additional security benefits.
    PositionIndependentExecutable,

    /// Optimize this module for execution speed.
    ///
    /// This may increase code size but should improve runtime performance.
    OptimizeForSpeed,

    /// Optimize this module for code size.
    ///
    /// This may reduce performance but will create smaller binaries.
    OptimizeForSize,

    /// Include debug information in the compiled output.
    ///
    /// Debug information improves debugging and profiling.
    /// but increases the size of the final binary.
    IncludeDebugInfo,

    /// Strip debug information and symbols from the compiled output.
    ///
    /// This reduces binary size and removes potentially sensitive information
    /// but makes debugging impossible.
    StripSymbols,

    /// Specify the target triple for this module.
    ///
    /// The target triple identifies the architecture, vendor, OS, and ABI
    /// for which the code should be compiled (e.g., "x86_64-unknown-linux-gnu").
    TargetTriple(String),
}

#[cfg(feature = "nightly")]
impl fmt::Display for ModuleAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleAnnotation::PositionIndependentCode => write!(f, "@pic"),
            ModuleAnnotation::PositionIndependentExecutable => write!(f, "@pie"),
            ModuleAnnotation::OptimizeForSpeed => write!(f, "@optimize_speed"),
            ModuleAnnotation::OptimizeForSize => write!(f, "@optimize_size"),
            ModuleAnnotation::IncludeDebugInfo => write!(f, "@debug"),
            ModuleAnnotation::StripSymbols => write!(f, "@strip"),
            ModuleAnnotation::TargetTriple(triple) => write!(f, "@target_triple({triple})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module<'a> {
    /// Named type declarations in this module
    ///
    /// Using a HashMap enables efficient lookup by type name. The order
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
    /// by name for fast lookup during function calls.
    pub functions: HashMap<Identifier<'a>, Function<'a>>,

    /// Module-level annotations that affect compilation.
    ///
    /// These annotations control global aspects of how the module is compiled,
    /// such as optimization level, code generation options, and linking behavior.
    #[cfg(feature = "nightly")]
    pub annotations: Vec<ModuleAnnotation>,
}

impl<'a> Module<'a> {
    // Basic constructor
    pub fn new() -> Self {
        #[cfg(feature = "nightly")]
        let annotations = Vec::new();

        Module {
            type_declarations: HashMap::new(),
            global_declarations: HashMap::new(),
            functions: HashMap::new(),
            #[cfg(feature = "nightly")]
            annotations,
        }
    }

    /// Validate all functions in the module.
    ///
    /// Runs `Function::validate()` on every function and aggregates all errors.
    /// Returns `Ok(())` if no errors are found, or a list of all violations.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut all_errors: Vec<String> = Vec::new();
        for func in self.functions.values() {
            if let Err(errs) = func.validate() {
                all_errors.extend(errs);
            }
        }
        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
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
            write!(f, " = {init}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Module<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print type declarations first
        for decl in self.type_declarations.values() {
            writeln!(f, "{decl}")?;
        }
        if !self.type_declarations.is_empty() {
            writeln!(f)?;
        }

        // Print global declarations
        for decl in self.global_declarations.values() {
            writeln!(f, "{decl}")?;
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
                write!(f, "{func}")?;
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
    use crate::function::{BasicBlock, Function, FunctionParameter, FunctionSignature};
    use crate::instruction::{BinaryOp, Instruction};
    use crate::types::{Literal, PrimitiveType, StructField, Type, Value};
    use std::collections::HashMap;

    #[test]
    fn test_display_type_declaration() {
        let decl = TypeDeclaration {
            name: "MyType",
            ty: Type::Struct(vec![]), // Empty struct
        };
        assert_eq!(format!("{}", decl), "type @MyType = struct {  }");
    }

    #[test]
    fn test_display_global_declaration() {
        let decl1 = GlobalDeclaration {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Constant(Literal::I32(0))),
        };
        assert_eq!(format!("{}", decl1), "global @count: i32 = 0");

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

        module.type_declarations.insert(
            "Vec2",
            TypeDeclaration {
                name: "Vec2",
                ty: Type::Struct(vec![
                    StructField {
                        name: "x",
                        ty: Type::Primitive(PrimitiveType::F32),
                    },
                    StructField {
                        name: "y",
                        ty: Type::Primitive(PrimitiveType::F32),
                    },
                ]),
            },
        );

        module.global_declarations.insert(
            "PI",
            GlobalDeclaration {
                name: "PI",
                ty: Type::Primitive(PrimitiveType::F32),
                initializer: Some(Value::Constant(Literal::F32(std::f32::consts::PI))),
            },
        );

        let func_sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "a",
                ty: Type::Primitive(PrimitiveType::I32),
                annotations: vec![],
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

        let expected_output = format!(
            "type @Vec2 = struct {{ x: f32, y: f32 }}\n\nglobal @PI: f32 = {}\n\nfn @add_one(i32 %a) -> i32 {{\nentry:\n  %res = add.i32 %a, 1\n  ret.i32 %res\n}}\n",
            std::f32::consts::PI
        );

        // Display sorts functions, but type/global maps still depend on insertion order.
        assert_eq!(format!("{}", module), expected_output);
    }

    #[test]
    fn test_display_empty_module() {
        let module = Module::<'static>::new();
        assert_eq!(format!("{}", module), "");
    }
}
