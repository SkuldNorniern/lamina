pub mod builder;
pub mod function;
pub mod instruction;
pub mod module;
pub mod types;

// Re-export key types
pub use function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
};
pub use instruction::{AllocType, BinaryOp, CmpOp, Instruction};
pub use module::{GlobalDeclaration, Module, TypeDeclaration};
pub use types::{Identifier, Literal, PrimitiveType, Type, Value};
