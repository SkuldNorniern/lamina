pub mod types;
pub mod instruction;
pub mod function;
pub mod module;
pub mod builder;

// Re-export key types
pub use types::{Type, PrimitiveType, Identifier, Value, Literal};
pub use instruction::{Instruction, BinaryOp, CmpOp, AllocType};
pub use function::{Function, FunctionSignature, FunctionParameter, BasicBlock, FunctionAnnotation};
pub use module::{Module, GlobalDeclaration, TypeDeclaration}; 