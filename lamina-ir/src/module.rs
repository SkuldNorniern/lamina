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
//! - **Value Types**: A name may be reassigned, but never with a different type
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
//!     .write_byte(i32(52), "w")
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
//!     .stackalloc("point", Type::Struct { fields: vec![
//!         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
//!         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) }
//!     ], packed: false })
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
//!     .write_byte(var("result"), "w") // Writes byte 42
//!     .ret_void();
//!
//! let module = builder.build();
//! ```

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

pub use lamina_platform::TlsModel;

use crate::{
    calls::{validate_calls, value_types},
    function::Function,
    instruction::{BinaryOp, Instruction},
    types::{
        Identifier, Literal, PrimitiveType, Type, Value, check_vector_shape, classify_vector_shift,
        resolve_type_names,
    },
};

/// A named type declaration (e.g., `type !Vec2 = struct { ... }`).
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
///     ty: Type::Struct { fields: vec![
///         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
///         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) },
///     ], packed: false },
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Eq might be too restrictive if Type contains f32 indirectly, but should be okay here.
pub struct TypeDeclaration<'a> {
    /// The type name (without the `!` prefix)
    pub name: Identifier<'a>, // e.g., "!Vec2"
    /// The type definition
    pub ty: Type<'a>,
}

/// How a global variable is initialized.
#[derive(Debug, Clone, PartialEq)]
pub enum GlobalInit<'a> {
    /// One value, for a global of primitive type.
    Value(Value<'a>),
    /// One literal per element, for a global of array type. Elements left out are zero.
    List(Vec<Literal<'a>>),
}

impl fmt::Display for GlobalInit<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalInit::Value(value) => write!(f, "{value}"),
            GlobalInit::List(elements) => {
                let elements: Vec<String> = elements.iter().map(ToString::to_string).collect();
                write!(f, "[{}]", elements.join(", "))
            }
        }
    }
}

/// A global variable declaration (e.g., `global $message: [5 x i8] = "hello"`).
///
/// Global variables are module-level data that persists for the lifetime of the program.
/// They can be initialized with constant values or left uninitialized (for external symbols).
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{GlobalDeclaration, GlobalInit, Type, PrimitiveType, Value, Literal};
///
/// let message_global = GlobalDeclaration {
///     name: "message",
///     ty: Type::Array {
///         element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
///         size: 13,
///     },
///     initializer: Some(GlobalInit::Value(Value::Constant(Literal::String("Hello, World!")))),
///     exported: false,
///     align: None,
///     thread_local: None,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)] // PartialEq due to Value potentially holding f32
pub struct GlobalDeclaration<'a> {
    /// The global variable name (without the `$` prefix)
    pub name: Identifier<'a>, // e.g., "$message"
    /// The type of the global variable
    pub ty: Type<'a>,
    /// Optional initializer (None for external symbols)
    pub initializer: Option<GlobalInit<'a>>,
    /// Whether the global is visible to other modules (declared with `@export`)
    pub exported: bool,
    /// Minimum alignment in bytes (declared with `@align(n)`)
    pub align: Option<u32>,
    /// The thread-local model this global is placed under (declared with
    /// `@thread_local`), or `None` for ordinary static storage.
    pub thread_local: Option<TlsModel>,
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
/// These annotations provide global control over linking and code generation.
///
/// ### Available Module Annotations
/// - **`@pic`**: Generate position-independent code
/// - **`@pie`**: Generate position-independent executable
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
}

#[cfg(feature = "nightly")]
impl fmt::Display for ModuleAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleAnnotation::PositionIndependentCode => write!(f, "@pic"),
            ModuleAnnotation::PositionIndependentExecutable => write!(f, "@pie"),
        }
    }
}

/// A parsed IR file: its type declarations, its globals and its functions. Each has
/// its own sigil and its own namespace: `!Point`, `$point` and `@point` are three names.
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
    /// An empty module.
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
            all_errors.extend(func.duplicate_definition_errors(&|name| {
                self.functions
                    .get(name)
                    .map(|callee| callee.signature.return_type.clone())
            }));
        }
        self.validate_named_types(&mut all_errors);
        self.validate_operand_types(&mut all_errors);
        self.validate_vectors(&mut all_errors);
        self.validate_field_pointers(&mut all_errors);
        validate_calls(self, &mut all_errors);
        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }

    /// Checks that each operand of an elementwise instruction is a value of the type
    /// the instruction states.
    ///
    /// A register of a different width is not an implicit conversion, and a shift count
    /// is no exception: `shl.i64` takes an `i64` count. Signedness is not a width, so
    /// `u64` stands where `i64` is stated. A vector operand is the same rule over the
    /// lane type and the lane count: a `<4 x i32>` does not stand where `<2 x i64>` is
    /// stated, and neither does a scalar.
    fn validate_operand_types(&self, errors: &mut Vec<String>) {
        let mut functions: Vec<&Function<'a>> = self.functions.values().collect();
        functions.sort_by_key(|func| func.name);
        for func in functions {
            let env = value_types(self, func);
            let mut blocks: Vec<_> = func.basic_blocks.iter().collect();
            blocks.sort_by_key(|(label, _)| **label);
            for (label, block) in blocks {
                for instr in &block.instructions {
                    self.check_operand_types(func.name, label, instr, &env, errors);
                }
            }
        }
    }

    fn check_operand_types(
        &self,
        func: &str,
        label: &str,
        instr: &Instruction<'a>,
        env: &HashMap<Identifier<'a>, Type<'a>>,
        errors: &mut Vec<String>,
    ) {
        let scalar = |ty: PrimitiveType| Type::Primitive(ty);
        let vector = |element_type: PrimitiveType, lanes: u32| Type::Vector {
            element_type,
            lanes,
        };
        let mut operands: Vec<(&str, &Value<'a>, Type<'a>)> = Vec::new();
        match instr {
            Instruction::Binary {
                ty,
                lhs,
                rhs,
                op,
                result: _,
            } => {
                operands.push(("left operand", lhs, scalar(*ty)));
                if !matches!(op, BinaryOp::Shl | BinaryOp::Shr) || !rhs_is_vector(rhs, env) {
                    operands.push(("right operand", rhs, scalar(*ty)));
                }
            }
            Instruction::Cmp { ty, lhs, rhs, .. } => {
                operands.push(("left operand", lhs, scalar(*ty)));
                operands.push(("right operand", rhs, scalar(*ty)));
            }
            Instruction::Unary {
                ty,
                value,
                op: _,
                result: _,
            } => {
                operands.push(("operand", value, scalar(*ty)));
            }
            Instruction::VectorBinary {
                op,
                element_type,
                lanes,
                lhs,
                rhs,
                ..
            } => {
                operands.push(("left operand", lhs, vector(*element_type, *lanes)));
                match op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::And
                    | BinaryOp::Or
                    | BinaryOp::Xor => {
                        operands.push(("right operand", rhs, vector(*element_type, *lanes)));
                    }
                    BinaryOp::Shl
                    | BinaryOp::Shr
                    | BinaryOp::Div
                    | BinaryOp::Rem
                    | BinaryOp::Rotl
                    | BinaryOp::Rotr => {}
                }
            }
            Instruction::VectorShuffle {
                element_type,
                lanes,
                lhs,
                rhs,
                ..
            } => {
                operands.push(("left operand", lhs, vector(*element_type, *lanes)));
                operands.push(("right operand", rhs, vector(*element_type, *lanes)));
            }
            Instruction::VectorSplat {
                element_type,
                value,
                ..
            } => {
                operands.push(("operand", value, scalar(*element_type)));
            }
            Instruction::VectorExtractLane {
                element_type,
                lanes,
                vector: source,
                ..
            } => {
                operands.push(("operand", source, vector(*element_type, *lanes)));
            }
            Instruction::VectorInsertLane {
                element_type,
                lanes,
                vector: source,
                scalar: lane_value,
                ..
            } => {
                operands.push(("operand", source, vector(*element_type, *lanes)));
                operands.push(("lane value", lane_value, scalar(*element_type)));
            }
            Instruction::Select {
                ty,
                true_val,
                false_val,
                ..
            } => {
                operands.push(("true value", true_val, ty.clone()));
                operands.push(("false value", false_val, ty.clone()));
            }
            Instruction::Store { ty, value, .. }
            | Instruction::AtomicStore { ty, value, .. }
            | Instruction::AtomicBinary { ty, value, .. } => {
                operands.push(("stored value", value, ty.clone()));
            }
            Instruction::AtomicCompareExchange {
                ty,
                expected,
                desired,
                ..
            } => {
                operands.push(("expected value", expected, ty.clone()));
                operands.push(("desired value", desired, ty.clone()));
            }
            Instruction::OverflowFlag { .. }
            | Instruction::ZeroExtend { .. }
            | Instruction::Trunc { .. }
            | Instruction::SignExtend { .. }
            | Instruction::Bitcast { .. }
            | Instruction::IntToFloat { .. }
            | Instruction::FloatToInt { .. }
            | Instruction::FloatConvert { .. }
            | Instruction::FuncAddr { .. }
            | Instruction::CallIndirect { .. }
            | Instruction::Switch { .. }
            | Instruction::Br { .. }
            | Instruction::Jmp { .. }
            | Instruction::Ret { .. }
            | Instruction::Alloc { .. }
            | Instruction::Load { .. }
            | Instruction::AtomicLoad { .. }
            | Instruction::Fence { .. }
            | Instruction::InlineAsm { .. }
            | Instruction::GetFieldPtr { .. }
            | Instruction::GetElemPtr { .. }
            | Instruction::PtrToInt { .. }
            | Instruction::IntToPtr { .. }
            | Instruction::Call { .. }
            | Instruction::Copy { .. }
            | Instruction::Write { .. }
            | Instruction::Read { .. }
            | Instruction::WriteByte { .. }
            | Instruction::ReadByte { .. }
            | Instruction::VaStart { .. }
            | Instruction::VaArg { .. }
            | Instruction::VaEnd { .. }
            | Instruction::VaCopy { .. }
            | Instruction::Unreachable => {}
        }
        for (side, value, expected) in operands {
            let Value::Variable(name) = value else {
                continue;
            };
            let Some(found) = env.get(name) else {
                continue;
            };
            let resolved = resolve_type_names(found, &self.type_declarations);
            let actual = resolved.as_ref().unwrap_or(found);
            if !operand_type_mismatch(actual, &expected) {
                continue;
            }
            let rule = match (&expected, side) {
                (Type::Vector { .. }, _) => {
                    "an operand of a vector operation has the instruction's lane type and lane count"
                }
                (_, "left operand" | "right operand") => {
                    "both operands take the instruction's type, a shift count included"
                }
                (_, _) => "the value an instruction names takes the instruction's type",
            };
            let hint = format!("\n  Hint: {rule}; convert '%{name}' to {expected} first");
            errors.push(format!(
                "the {side} of '{instr}' in block '{label}' of '@{func}' is '%{name}', which has type {found}, but the instruction states {expected}{hint}"
            ));
        }
    }

    /// Checks every use of `<N x T>`.
    ///
    /// A vector is an instruction's type suffix and the type of an SSA value. It has
    /// no memory layout, so it may not appear in a declaration: not in a global, a
    /// struct field, a named type, a parameter, a return type or a `stackalloc`.
    /// See `Plan/simd-surface.md` for why.
    fn validate_vectors(&self, errors: &mut Vec<String>) {
        let mut declarations: Vec<(String, &Type<'_>)> = Vec::new();
        for decl in self.type_declarations.values() {
            declarations.push((format!("type '!{}'", decl.name), &decl.ty));
        }
        for global in self.global_declarations.values() {
            declarations.push((format!("global '${}'", global.name), &global.ty));
        }
        for func in self.functions.values() {
            for param in &func.signature.params {
                declarations.push((
                    format!("parameter '%{}' of '@{}'", param.name, func.name),
                    &param.ty,
                ));
            }
            declarations.push((
                format!("return type of '@{}'", func.name),
                &func.signature.return_type,
            ));
            for block in func.basic_blocks.values() {
                for instr in &block.instructions {
                    if let Instruction::Alloc { allocated_ty, .. } = instr {
                        declarations
                            .push((format!("stackalloc in '@{}'", func.name), allocated_ty));
                    }
                }
            }
        }
        for (owner, ty) in declarations {
            if type_mentions_vector(ty) {
                errors.push(format!(
                    "{owner} names a vector type\n  Hint: a vector is a value in a register, not a layout in memory; it may only be an instruction's type suffix. Declare the memory as [N x T] and use 'load.<N x T>' to read it"
                ));
            }
        }

        for func in self.functions.values() {
            let env = value_types(self, func);
            for block in func.basic_blocks.values() {
                for instr in &block.instructions {
                    self.validate_vector_instruction(func.name, instr, &env, errors);
                }
            }
        }
    }

    fn validate_vector_instruction(
        &self,
        func: &str,
        instr: &Instruction<'a>,
        env: &HashMap<Identifier<'a>, Type<'a>>,
        errors: &mut Vec<String>,
    ) {
        let where_ = |what: &str| format!("{what} in '@{func}'");
        let mut shape =
            |element_type: PrimitiveType, lanes: u32, what: &str| match check_vector_shape(
                element_type,
                lanes,
                &where_(what),
            ) {
                Ok(()) => true,
                Err(message) => {
                    errors.push(message);
                    false
                }
            };
        match instr {
            Instruction::Load { ty, .. } | Instruction::Store { ty, .. } => {
                let resolved = resolve_type_names(ty, &self.type_declarations);
                if let Some(Type::Vector {
                    element_type,
                    lanes,
                }) = resolved.as_ref().or(Some(ty))
                {
                    let what = if matches!(instr, Instruction::Load { .. }) {
                        "load"
                    } else {
                        "store"
                    };
                    shape(*element_type, *lanes, what);
                }
            }
            Instruction::VectorBinary {
                op,
                element_type,
                lanes,
                rhs,
                ..
            } => {
                if !shape(*element_type, *lanes, "vector operation") {
                    return;
                }
                match op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::And
                    | BinaryOp::Or
                    | BinaryOp::Xor => {}
                    BinaryOp::Shl | BinaryOp::Shr => {
                        if let Err(reason) =
                            classify_vector_shift(*element_type, *lanes, rhs, env)
                        {
                            errors.push(format!(
                                "{} {reason}",
                                where_(&format!("'{op}.<{lanes} x {element_type}>'"))
                            ));
                        }
                    }
                    BinaryOp::Div | BinaryOp::Rem | BinaryOp::Rotl | BinaryOp::Rotr => errors
                        .push(format!(
                            "{} has no vector form\n  Hint: extract the lanes, operate on them as scalars and insert the results",
                            where_(&format!("'{op}.<{lanes} x {element_type}>'"))
                        )),
                }
            }
            Instruction::VectorSplat {
                element_type,
                lanes,
                ..
            } => {
                shape(*element_type, *lanes, "splat");
            }
            Instruction::VectorExtractLane {
                element_type,
                lanes,
                lane,
                ..
            } => {
                if shape(*element_type, *lanes, "extractlane") && lane >= lanes {
                    errors.push(format!(
                        "{} reads lane {lane} of <{lanes} x {element_type}>",
                        where_("extractlane")
                    ));
                }
            }
            Instruction::VectorInsertLane {
                element_type,
                lanes,
                lane,
                ..
            } => {
                if shape(*element_type, *lanes, "insertlane") && lane >= lanes {
                    errors.push(format!(
                        "{} writes lane {lane} of <{lanes} x {element_type}>",
                        where_("insertlane")
                    ));
                }
            }
            Instruction::VectorShuffle {
                element_type,
                lanes,
                mask,
                ..
            } => {
                if !shape(*element_type, *lanes, "shuffle") {
                    return;
                }
                if mask.len() as u64 != u64::from(*lanes) {
                    errors.push(format!(
                        "{} has {} mask entries for <{lanes} x {element_type}>\n  Hint: the mask names one source lane per result lane",
                        where_("shuffle"),
                        mask.len()
                    ));
                }
                let sources = lanes.saturating_mul(2);
                for index in mask {
                    if *index >= sources {
                        errors.push(format!(
                            "{} selects lane {index}, but its two <{lanes} x {element_type}> operands hold lanes 0 to {}",
                            where_("shuffle"),
                            sources - 1
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    fn validate_named_types(&self, errors: &mut Vec<String>) {
        let mut uses: Vec<(String, &Type<'_>)> = Vec::new();
        for decl in self.type_declarations.values() {
            uses.push((format!("type '!{}'", decl.name), &decl.ty));
        }
        for global in self.global_declarations.values() {
            uses.push((format!("global '${}'", global.name), &global.ty));
        }
        for func in self.functions.values() {
            for param in &func.signature.params {
                uses.push((
                    format!("parameter '%{}' of '@{}'", param.name, func.name),
                    &param.ty,
                ));
            }
            uses.push((
                format!("return type of '@{}'", func.name),
                &func.signature.return_type,
            ));
            for block in func.basic_blocks.values() {
                for instr in &block.instructions {
                    match instr {
                        Instruction::Alloc { allocated_ty, .. } => {
                            uses.push((format!("alloc in '@{}'", func.name), allocated_ty));
                        }
                        Instruction::GetFieldPtr { struct_type, .. } => {
                            uses.push((format!("getfieldptr in '@{}'", func.name), struct_type));
                        }
                        _ => {}
                    }
                }
            }
        }

        for (owner, ty) in &uses {
            let mut refs = Vec::new();
            named_type_refs(ty, &mut refs);
            for name in refs {
                if !self.type_declarations.contains_key(name) {
                    errors.push(format!("unknown type '!{name}' used in {owner}"));
                }
            }
        }

        let mut names: Vec<&str> = self.type_declarations.keys().copied().collect();
        names.sort_unstable();
        for name in names {
            if self.type_contains_itself(name) {
                errors.push(format!(
                    "type '!{name}' contains itself, so its size is infinite"
                ));
            }
        }
    }

    fn validate_field_pointers(&self, errors: &mut Vec<String>) {
        for func in self.functions.values() {
            for block in func.basic_blocks.values() {
                for instr in &block.instructions {
                    let Instruction::GetFieldPtr {
                        struct_type,
                        field_index,
                        ..
                    } = instr
                    else {
                        continue;
                    };
                    match resolve_type_names(struct_type, &self.type_declarations) {
                        Some(Type::Struct { fields, .. }) if *field_index < fields.len() => {}
                        Some(Type::Struct { fields, .. }) => errors.push(format!(
                            "getfieldptr in '@{}' takes field {field_index}, but its struct has {} fields",
                            func.name,
                            fields.len()
                        )),
                        Some(other) => errors.push(format!(
                            "getfieldptr in '@{}' needs a struct layout, found {other}",
                            func.name
                        )),
                        None => {}
                    }
                }
            }
        }
    }

    fn type_contains_itself(&self, name: &str) -> bool {
        let mut stack: Vec<&str> = vec![name];
        let mut seen: HashSet<&str> = HashSet::new();
        while let Some(current) = stack.pop() {
            let Some(decl) = self.type_declarations.get(current) else {
                continue;
            };
            let mut refs = Vec::new();
            named_type_refs(&decl.ty, &mut refs);
            for referenced in refs {
                if referenced == name {
                    return true;
                }
                if seen.insert(referenced) {
                    stack.push(referenced);
                }
            }
        }
        false
    }
}

fn type_mentions_vector(ty: &Type<'_>) -> bool {
    match ty {
        Type::Vector { .. } => true,
        Type::Array { element_type, .. } => type_mentions_vector(element_type),
        Type::Struct { fields, .. } => fields.iter().any(|f| type_mentions_vector(&f.ty)),
        Type::Primitive(_) | Type::Named(_) | Type::Void | Type::VaList => false,
    }
}

fn named_type_refs<'a>(ty: &Type<'a>, out: &mut Vec<&'a str>) {
    match ty {
        Type::Named(name) => out.push(name),
        Type::Array { element_type, .. } => named_type_refs(element_type, out),
        Type::Struct { fields, .. } => {
            for field in fields {
                named_type_refs(&field.ty, out);
            }
        }
        _ => {}
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
        write!(f, "type !{} = {}", self.name, self.ty)
    }
}

impl fmt::Display for GlobalDeclaration<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(align) = self.align {
            writeln!(f, "@align({align})")?;
        }
        if let Some(model) = self.thread_local {
            write!(f, "@thread_local({model}) ")?;
        }
        if self.exported {
            write!(f, "@export ")?;
        }
        write!(f, "global ${}: {}", self.name, self.ty)?;
        if let Some(init) = &self.initializer {
            write!(f, " = {init}")?;
        }
        Ok(())
    }
}

fn write_global(
    f: &mut fmt::Formatter<'_>,
    decl: &GlobalDeclaration<'_>,
    types: &HashMap<Identifier<'_>, TypeDeclaration<'_>>,
) -> fmt::Result {
    if let Some(align) = decl.align {
        writeln!(f, "@align({align})")?;
    }
    if let Some(model) = decl.thread_local {
        write!(f, "@thread_local({model}) ")?;
    }
    if decl.exported {
        write!(f, "@export ")?;
    }
    write!(f, "global ${}: {}", decl.name, decl.ty)?;
    match &decl.initializer {
        None => Ok(()),
        Some(GlobalInit::List(elements)) => {
            let (open, close) = if is_struct_type(&decl.ty, types, 0) {
                ('{', '}')
            } else {
                ('[', ']')
            };
            let items: Vec<String> = elements.iter().map(ToString::to_string).collect();
            write!(f, " = {open}{}{close}", items.join(", "))
        }
        Some(init) => write!(f, " = {init}"),
    }
}

fn is_struct_type(
    ty: &Type<'_>,
    types: &HashMap<Identifier<'_>, TypeDeclaration<'_>>,
    depth: usize,
) -> bool {
    match ty {
        Type::Struct { .. } => true,
        Type::Named(name) if depth < 32 => types
            .get(name)
            .is_some_and(|decl| is_struct_type(&decl.ty, types, depth + 1)),
        _ => false,
    }
}

impl fmt::Display for Module<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(feature = "nightly")]
        if !self.annotations.is_empty() {
            for annotation in &self.annotations {
                writeln!(f, "{annotation}")?;
            }
            writeln!(f)?;
        }

        let mut type_names: Vec<_> = self.type_declarations.keys().collect();
        type_names.sort();
        for name in &type_names {
            writeln!(f, "{}", self.type_declarations[*name])?;
        }
        if !type_names.is_empty() {
            writeln!(f)?;
        }

        let mut global_names: Vec<_> = self.global_declarations.keys().collect();
        global_names.sort();
        for name in &global_names {
            write_global(f, &self.global_declarations[*name], &self.type_declarations)?;
            writeln!(f)?;
        }
        if !global_names.is_empty() {
            writeln!(f)?;
        }

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

/// Whether a value stated at `expected` may not stand where `actual` is found.
///
/// Signedness is not a width, so `u64` stands where `i64` is stated, and a `<4 x u32>`
/// stands where `<4 x i32>` is. Anything whose shape this cannot judge — a struct, an
/// array, a named type that did not resolve — is left to the checks that own it.
fn operand_type_mismatch(actual: &Type<'_>, expected: &Type<'_>) -> bool {
    match (actual, expected) {
        (Type::Primitive(found), Type::Primitive(stated)) => {
            found.value_class() != stated.value_class()
        }
        (
            Type::Vector {
                element_type: found,
                lanes: found_lanes,
            },
            Type::Vector {
                element_type: stated,
                lanes: stated_lanes,
            },
        ) => found_lanes != stated_lanes || found.value_class() != stated.value_class(),
        (Type::Vector { .. }, Type::Primitive(_)) | (Type::Primitive(_), Type::Vector { .. }) => {
            true
        }
        (Type::Primitive(_) | Type::Vector { .. }, _)
        | (
            Type::Named(_) | Type::Array { .. } | Type::Struct { .. } | Type::Void | Type::VaList,
            _,
        ) => false,
    }
}

/// Whether a shift count names a vector, which `classify_vector_shift` judges instead.
fn rhs_is_vector<'a>(rhs: &Value<'a>, env: &HashMap<Identifier<'a>, Type<'a>>) -> bool {
    let Value::Variable(name) = rhs else {
        return false;
    };
    matches!(env.get(name), Some(Type::Vector { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        function::{BasicBlock, Function, FunctionParameter, FunctionSignature},
        instruction::{BinaryOp, Instruction},
        types::{Literal, PrimitiveType, StructField, Type, Value},
    };
    use std::{collections::HashMap, f32::consts::PI};

    #[test]
    fn test_display_type_declaration() {
        let decl = TypeDeclaration {
            name: "MyType",
            ty: Type::Struct {
                fields: vec![],
                packed: false,
            }, // Empty struct
        };
        assert_eq!(format!("{}", decl), "type !MyType = struct {  }");
    }

    #[test]
    fn test_display_global_declaration() {
        let decl1 = GlobalDeclaration {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(GlobalInit::Value(Value::Constant(Literal::I32(0)))),
            exported: false,
            align: None,
            thread_local: None,
        };
        assert_eq!(format!("{}", decl1), "global $count: i32 = 0");

        let decl2 = GlobalDeclaration {
            name: "external_data",
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 256,
            },
            initializer: None,
            exported: false,
            align: None,
            thread_local: None,
        };
        assert_eq!(format!("{}", decl2), "global $external_data: [256 x i8]");
    }

    #[test]
    fn test_display_global_declaration_with_align() {
        let decl = GlobalDeclaration {
            name: "arena",
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 1048576,
            },
            initializer: None,
            exported: true,
            align: Some(64),
            thread_local: None,
        };
        assert_eq!(
            format!("{}", decl),
            "@align(64)\n@export global $arena: [1048576 x i8]"
        );
    }

    #[test]
    fn test_display_is_independent_of_insertion_order() {
        let names = [
            "zeta", "alpha", "mu", "beta", "omega", "gamma", "kappa", "delta",
        ];
        let build = |order: &[&'static str]| {
            let mut module = Module::new();
            for &name in order {
                module.type_declarations.insert(
                    name,
                    TypeDeclaration {
                        name,
                        ty: Type::Struct {
                            fields: vec![StructField {
                                name: "x",
                                ty: Type::Primitive(PrimitiveType::I32),
                            }],
                            packed: false,
                        },
                    },
                );
                module.global_declarations.insert(
                    name,
                    GlobalDeclaration {
                        name,
                        ty: Type::Primitive(PrimitiveType::I32),
                        initializer: Some(GlobalInit::Value(Value::Constant(Literal::I32(1)))),
                        exported: false,
                        align: None,
                        thread_local: None,
                    },
                );
            }
            module
        };

        let forward = names.to_vec();
        let mut reversed = forward.clone();
        reversed.reverse();

        let first = format!("{}", build(&forward));
        let second = format!("{}", build(&reversed));
        assert_eq!(first, second);
        assert!(first.find("type !alpha").unwrap() < first.find("type !zeta").unwrap());
        assert!(first.find("global $alpha").unwrap() < first.find("global $zeta").unwrap());
    }

    #[test]
    fn test_display_module() {
        let mut module = Module::new();

        module.type_declarations.insert(
            "Vec2",
            TypeDeclaration {
                name: "Vec2",
                ty: Type::Struct {
                    fields: vec![
                        StructField {
                            name: "x",
                            ty: Type::Primitive(PrimitiveType::F32),
                        },
                        StructField {
                            name: "y",
                            ty: Type::Primitive(PrimitiveType::F32),
                        },
                    ],
                    packed: false,
                },
            },
        );

        module.global_declarations.insert(
            "PI",
            GlobalDeclaration {
                name: "PI",
                ty: Type::Primitive(PrimitiveType::F32),
                initializer: Some(GlobalInit::Value(Value::Constant(Literal::F32(PI)))),
                exported: false,
                align: None,
                thread_local: None,
            },
        );

        let func_sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "a",
                ty: Type::Primitive(PrimitiveType::I32),
                annotations: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
            variadic: false,
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
                instruction_locs: Vec::new(),
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
            "type !Vec2 = struct {{ x: f32, y: f32 }}\n\nglobal $PI: f32 = {}\n\nfn @add_one(i32 %a) -> i32 {{\nentry:\n  %res = add.i32 %a, 1\n  ret.i32 %res\n}}\n",
            PI
        );

        // Display sorts functions, but type/global maps still depend on insertion order.
        assert_eq!(format!("{}", module), expected_output);
    }

    #[test]
    fn test_display_empty_module() {
        let module = Module::<'static>::new();
        assert_eq!(format!("{}", module), "");
    }

    #[test]
    fn test_validate_unknown_named_type_fails() {
        let mut module = Module::<'static>::new();
        module.type_declarations.insert(
            "pair",
            TypeDeclaration {
                name: "pair",
                ty: Type::Struct {
                    fields: vec![StructField {
                        name: "x",
                        ty: Type::Named("missing"),
                    }],
                    packed: false,
                },
            },
        );
        let errs = module.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unknown type '!missing'")));
    }

    #[test]
    fn test_validate_self_containing_type_fails() {
        let mut module = Module::<'static>::new();
        module.type_declarations.insert(
            "node",
            TypeDeclaration {
                name: "node",
                ty: Type::Struct {
                    fields: vec![StructField {
                        name: "next",
                        ty: Type::Named("node"),
                    }],
                    packed: false,
                },
            },
        );
        let errs = module.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'!node' contains itself")));
    }
}
