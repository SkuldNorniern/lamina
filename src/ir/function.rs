//! # Function Representation
//!
//! This module defines the structures for representing functions in the Lamina IR.
//! Functions are the primary unit of code organization and contain basic blocks,
//! which in turn contain instructions.
//!
//! ## Key Concepts
//!
//! - **Functions**: Complete function definitions with signatures and implementations
//! - **Basic Blocks**: Linear sequences of instructions ending with a terminator
//! - **Function Signatures**: Parameter types and return types
//! - **Annotations**: Metadata that affects function behavior
//!
//! ## Function Structure
//!
//! A function consists of:
//! 1. **Signature**: Parameters and return type
//! 2. **Basic Blocks**: The actual implementation
//! 3. **Entry Block**: The first block to execute
//! 4. **Annotations**: Metadata like `@inline`, `@export`
//!
//! ## Basic Block Rules
//!
//! - Each basic block must end with a **terminator instruction**:
//!   - `Br` (conditional branch)
//!   - `Jmp` (unconditional jump)
//!   - `Ret` (return)
//! - Instructions within a block execute sequentially
//! - Control flow only changes at block boundaries
//!
//! ## Example
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, CmpOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("max", Type::Primitive(PrimitiveType::I32))
//!     .cmp(CmpOp::Gt, "a_greater", PrimitiveType::I32, var("a"), var("b"))
//!     .branch(var("a_greater"), "return_a", "return_b")
//!     .block("return_a")
//!     .ret(Type::Primitive(PrimitiveType::I32), var("a"))
//!     .block("return_b")
//!     .ret(Type::Primitive(PrimitiveType::I32), var("b"));
//! ```

use std::collections::HashMap; // Using HashMap for basic blocks
use std::fmt;

use super::instruction::Instruction;
use super::types::{Identifier, Label, Type};

/// Function annotations that provide metadata about function behavior.
///
/// Annotations are hints to the compiler about how to handle a function.
/// They can affect optimization decisions, calling conventions, and code generation.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, FunctionAnnotation};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("fast_calc", Type::Void)
///     .annotate(FunctionAnnotation::Inline)  // Hint to inline this function
///     .annotate(FunctionAnnotation::Cold);   // Function is rarely called
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAnnotation {
    /// Hint to the compiler to inline this function at call sites.
    ///
    /// Inlining can improve performance by eliminating function call overhead,
    /// but may increase code size. Use for small, frequently called functions.
    Inline,

    /// Mark this function as exported (visible to other modules).
    ///
    /// Exported functions can be called from other modules or linked against
    /// by external code. This is typically used for public APIs.
    Export,

    /// Indicate that this function never returns normally.
    ///
    /// Functions marked with `NoReturn` always terminate the program (e.g., `exit`, `panic`).
    /// This allows the compiler to optimize control flow and eliminate unreachable code.
    NoReturn,

    /// Hint to the compiler to never inline this function.
    ///
    /// Use for large functions or when you want to preserve the function call
    /// for debugging or profiling purposes.
    NoInline,

    /// Mark this function as "cold" (rarely executed).
    ///
    /// Cold functions are optimized for size rather than speed, and may be
    /// placed in a separate code section to improve instruction cache usage.
    Cold,
    // Add more as needed
}

/// Represents a function parameter with its name and type.
///
/// Function parameters are the inputs to a function and are bound to values
/// when the function is called. Each parameter has a unique name within the
/// function scope and a specific type.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{FunctionParameter, Type, PrimitiveType};
///
/// let param = FunctionParameter {
///     name: "x",
///     ty: Type::Primitive(PrimitiveType::I32),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParameter<'a> {
    /// The parameter name (without the `%` prefix)
    pub name: Identifier<'a>, // e.g., "%a"
    /// The parameter's type
    pub ty: Type<'a>,
}

/// Represents the signature of a function (parameters and return type).
///
/// A function signature defines the interface of a function, including what
/// parameters it accepts and what type it returns. This is used for type
/// checking and to ensure function calls are correct.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{FunctionSignature, FunctionParameter, Type, PrimitiveType};
///
/// let signature = FunctionSignature {
///     params: vec![
///         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) },
///         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32) },
///     ],
///     return_type: Type::Primitive(PrimitiveType::I32),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature<'a> {
    /// List of function parameters in order
    pub params: Vec<FunctionParameter<'a>>,
    /// The return type of the function (`Type::Void` if it doesn't return)
    pub return_type: Type<'a>, // Type::Void if it doesn't return
}

/// Represents a basic block within a function.
///
/// A basic block is a sequence of instructions that execute sequentially
/// and end with a terminator instruction (branch, jump, or return). This
/// structure enables control flow analysis and optimization.
///
/// # Invariants
///
/// - The last instruction MUST be a terminator (`Br`, `Jmp`, or `Ret`)
/// - Instructions within a block execute in order
/// - Control flow only changes at block boundaries
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("example", Type::Void)
///     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
///     .print(var("sum"))
///     .ret_void();
///
/// let module = builder.build();
/// let func = &module.functions["example"];
/// let entry_block = &func.basic_blocks["entry"];
/// // entry_block.instructions contains: [Binary, Print, Ret]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock<'a> {
    /// Instructions in this block (must end with a terminator)
    pub instructions: Vec<Instruction<'a>>,
    // Note: The last instruction MUST be a terminator (Br, Jmp, Ret)
}

/// Represents a complete function definition.
///
/// A function contains its signature, implementation (basic blocks), and
/// metadata (annotations). Functions are the primary unit of code organization
/// in the IR.
///
/// # Structure
///
/// - **Name**: Unique identifier for the function
/// - **Signature**: Parameters and return type
/// - **Basic Blocks**: The actual implementation
/// - **Entry Block**: The first block to execute
/// - **Annotations**: Metadata affecting behavior
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("add", Type::Primitive(PrimitiveType::I32))
///     .binary(BinaryOp::Add, "result", PrimitiveType::I32, i32(10), i32(20))
///     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
///
/// let module = builder.build();
/// let func = &module.functions["add"];
/// // func contains the complete function definition
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Function<'a> {
    /// The function name (without the `@` prefix)
    pub name: Identifier<'a>,
    /// The function's signature (parameters and return type)
    pub signature: FunctionSignature<'a>,
    /// Annotations that affect function behavior
    pub annotations: Vec<FunctionAnnotation>,
    /// Basic blocks in this function, keyed by their label
    ///
    /// Using a HashMap allows efficient lookup by label name. The order
    /// of blocks is implicitly defined by the control flow (branches and jumps).
    pub basic_blocks: HashMap<Label<'a>, BasicBlock<'a>>,
    /// Label of the first block to execute when the function is called
    pub entry_block: Label<'a>, // Label of the first block to execute
}

// --- Display Implementations ---

impl fmt::Display for FunctionAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "@{}",
            match self {
                FunctionAnnotation::Inline => "inline",
                FunctionAnnotation::Export => "export",
                FunctionAnnotation::NoReturn => "noreturn",
                FunctionAnnotation::NoInline => "noinline",
                FunctionAnnotation::Cold => "cold",
            }
        )
    }
}

impl fmt::Display for FunctionParameter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} %{}", self.ty, self.name)
    }
}

impl fmt::Display for FunctionSignature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.params.iter().enumerate() {
            write!(f, "{}", param)?;
            if i < self.params.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ") -> {}", self.return_type)
    }
}

impl fmt::Display for BasicBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instr in &self.instructions {
            writeln!(f, "  {}", instr)?;
        }
        Ok(())
    }
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for annotation in &self.annotations {
            writeln!(f, "{}", annotation)?;
        }
        writeln!(f, "fn @{}{} {{", self.name, self.signature)?;

        // We need a consistent order for printing blocks. Let's try to print entry first.
        if let Some(entry) = self.basic_blocks.get(self.entry_block) {
            writeln!(f, "{}:", self.entry_block)?;
            write!(f, "{}", entry)?;
        }

        // Print remaining blocks (order might not be source order, but good enough for display)
        for (label, block) in &self.basic_blocks {
            if *label != self.entry_block {
                writeln!(f, "{}:", label)?;
                write!(f, "{}", block)?;
            }
        }

        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instruction::{AllocType, BinaryOp, Instruction};
    use crate::ir::types::{Literal, PrimitiveType, Type, Value};
    use std::collections::HashMap;

    #[test]
    fn test_display_function_annotation() {
        assert_eq!(format!("{}", FunctionAnnotation::Inline), "@inline");
        assert_eq!(format!("{}", FunctionAnnotation::Export), "@export");
        assert_eq!(format!("{}", FunctionAnnotation::NoReturn), "@noreturn");
        assert_eq!(format!("{}", FunctionAnnotation::NoInline), "@noinline");
        assert_eq!(format!("{}", FunctionAnnotation::Cold), "@cold");
    }

    #[test]
    fn test_display_function_parameter() {
        let param = FunctionParameter {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_eq!(format!("{}", param), "i64 %count");
    }

    #[test]
    fn test_display_function_signature() {
        // No params, void return
        let sig1 = FunctionSignature {
            params: vec![],
            return_type: Type::Void,
        };
        assert_eq!(format!("{}", sig1), "() -> void");

        // One param, primitive return
        let sig2 = FunctionSignature {
            params: vec![FunctionParameter {
                name: "input",
                ty: Type::Primitive(PrimitiveType::F32),
            }],
            return_type: Type::Primitive(PrimitiveType::F32),
        };
        assert_eq!(format!("{}", sig2), "(f32 %input) -> f32");

        // Multiple params, named return
        let sig3 = FunctionSignature {
            params: vec![
                FunctionParameter {
                    name: "a",
                    ty: Type::Primitive(PrimitiveType::I32),
                },
                FunctionParameter {
                    name: "b",
                    ty: Type::Named("MyType"),
                },
            ],
            return_type: Type::Named("ResultType"),
        };
        assert_eq!(format!("{}", sig3), "(i32 %a, @MyType %b) -> @ResultType");
    }

    #[test]
    fn test_display_basic_block() {
        let block = BasicBlock {
            instructions: vec![
                Instruction::Alloc {
                    result: "p",
                    alloc_type: AllocType::Stack,
                    allocated_ty: Type::Primitive(PrimitiveType::I32),
                },
                Instruction::Store {
                    ty: Type::Primitive(PrimitiveType::I32),
                    ptr: Value::Variable("p"),
                    value: Value::Constant(Literal::I32(10)),
                },
                Instruction::Jmp {
                    target_label: "next_block",
                },
            ],
        };
        // Expect leading spaces on ALL instruction lines due to writeln!
        let expected_output = "  %p = alloc.ptr.stack i32\n  store.i32 %p, 10\n  jmp next_block\n";
        assert_eq!(format!("{}", block), expected_output);
    }

    #[test]
    fn test_display_function() {
        let sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "x",
                ty: Type::Primitive(PrimitiveType::I32),
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };

        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "tmp",
                        ty: PrimitiveType::I32,
                        lhs: Value::Variable("x"),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Ret {
                        ty: Type::Primitive(PrimitiveType::I32),
                        value: Some(Value::Variable("tmp")),
                    },
                ],
            },
        );

        let func = Function {
            name: "increment",
            signature: sig,
            annotations: vec![FunctionAnnotation::Inline, FunctionAnnotation::Export],
            basic_blocks: blocks,
            entry_block: "entry",
        };

        // Match actual output - no indentation for instructions
        let expected_output = "@inline\n@export\nfn @increment(i32 %x) -> i32 {\nentry:\n  %tmp = add.i32 %x, 1\n  ret.i32 %tmp\n}\n";

        assert_eq!(format!("{}", func), expected_output);
    }

    #[test]
    fn test_display_function_multiple_blocks() {
        let sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "cond",
                ty: Type::Primitive(PrimitiveType::Bool),
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };

        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Br {
                    condition: Value::Variable("cond"),
                    true_label: "if_true",
                    false_label: "if_false",
                }],
            },
        );
        blocks.insert(
            "if_true",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(1))),
                }],
            },
        );
        blocks.insert(
            "if_false",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(0))),
                }],
            },
        );

        let func = Function {
            name: "branch_test",
            signature: sig,
            annotations: vec![],
            basic_blocks: blocks,
            entry_block: "entry",
        };

        // Match actual output - no indentation for instructions
        let output = format!("{}", func);

        let expected_order1 = "fn @branch_test(bool %cond) -> i32 {\nentry:\n  br %cond, if_true, if_false\nif_false:\n  ret.i32 0\nif_true:\n  ret.i32 1\n}\n";

        let expected_order2 = "fn @branch_test(bool %cond) -> i32 {\n  entry:\n    br %cond, if_true, if_false\n  if_true:\n    ret.i32 1\n  if_false:\n    ret.i32 0\n}\n";
        let expected_order3 = "fn @branch_test(bool %cond) -> i32 {\nentry:\n  br %cond, if_true, if_false\nif_true:\n  ret.i32 1\nif_false:\n  ret.i32 0\n}\n";
        let expected_order4 = "fn @branch_test(bool %cond) -> i32 {\n  entry:\n    br %cond, if_true, if_false\n  if_true:\n    ret.i32 1\n  if_false:\n    ret.i32 0\n}\n";

        assert!(
            output == expected_order1
                || output == expected_order2
                || output == expected_order3
                || output == expected_order4,
            "Unexpected output:\n{}\nExpected one of:\n{}\nOR\n{}\n",
            output,
            expected_order1,
            expected_order2
        );
    }
}
