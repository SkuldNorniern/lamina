pub mod aarch64;
pub mod common;
pub mod wasm;
pub mod x86_64;
pub mod riscv;

// Re-export the main codegen functions for external use
pub use aarch64::generate_aarch64_assembly;
pub use x86_64::generate_x86_64_assembly;
pub use riscv::{generate_riscv32_assembly, generate_riscv64_assembly, generate_riscv128_assembly};

use crate::PrimitiveType;

// Codegen Errors for Detailed Error Handling

/// Error types specific to codegen
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    // Type System Errors
    /// Unsupported primitive type
    UnsupportedPrimitiveType(PrimitiveType),
    /// Struct operations not implemented
    StructNotImplemented,
    /// Tuple operations not implemented
    TupleNotImplemented,
    /// Named type lookup not implemented
    NamedTypeNotImplemented,
    /// Cannot get size of void type
    VoidTypeSize,
    /// Unsupported literal type in global initializer
    UnsupportedLiteralTypeInGlobal(LiteralType),
    /// Unsupported type for operation
    UnsupportedTypeForOperation(OperationType),

    // Instruction Errors
    /// Store operation not implemented for type
    StoreNotImplementedForType(TypeInfo),
    /// Load operation not implemented for type
    LoadNotImplementedForType(TypeInfo),
    /// Binary operation not supported for type
    BinaryOpNotSupportedForType(TypeInfo),
    /// Comparison operation not supported for type
    ComparisonOpNotSupportedForType(TypeInfo),
    /// Zero extension not supported
    ZeroExtensionNotSupported(ExtensionInfo),
    /// Invalid allocation location
    InvalidAllocationLocation,
    /// Heap allocation not supported
    HeapAllocationNotSupported,
    /// Unsupported type for heap allocation
    UnsupportedTypeForHeapAllocation(String),

    // Operand/Immediate Errors
    /// Invalid immediate value
    InvalidImmediateValue,
    /// F32 literal not implemented in operands
    F32LiteralNotImplemented,
    /// String literal requires global variable workaround
    StringLiteralRequiresGlobal,
    /// Global variable not found
    GlobalNotFound(Identifier),

    // Register/Stack Errors
    /// Register allocation failed
    RegisterAllocationFailed,
    /// Stack overflow (too many locals)
    StackOverflow,
    /// Value location not found
    ValueLocationNotFound(Identifier),
    /// Block label not found
    BlockLabelNotFound(Label),

    // Global/Initializer Errors
    /// Global initializer pointing to another global not implemented
    GlobalToGlobalInitNotImplemented,
    /// Cannot initialize global with variable
    GlobalVarInitNotSupported,
    /// Global initializer called on uninitialized global
    UninitializedGlobalInit,

    // Feature Support Errors
    /// Unsupported feature
    UnsupportedFeature(FeatureType),
    /// Invalid instruction for target
    InvalidInstructionForTarget(InstructionType),

    // Internal Errors
    /// Internal codegen error
    InternalError,
}

/// Helper types to minimize String usage
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiteralType {
    Unknown(String),
    F32,
    F64,
    String,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationType {
    Store,
    Load,
    BinaryOp,
    Comparison,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeInfo {
    Primitive(PrimitiveType),
    Array,
    Struct,
    Tuple,
    Named,
    Void,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtensionInfo {
    I8ToI32,
    I8ToI64,
    I32ToI64,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureType {
    FloatOperations,
    StructOperations,
    HeapAllocation,
    TailCalls,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionType {
    InvalidOpcode,
    UnsupportedTarget,
    Custom(String),
}

// Type aliases for clarity
pub type Identifier = String;
pub type Label = String;

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Type System Errors
            CodegenError::UnsupportedPrimitiveType(pt) => {
                write!(f, "Unsupported primitive type: {:?}", pt)
            }
            CodegenError::StructNotImplemented => {
                write!(f, "Struct size calculation not implemented yet")
            }
            CodegenError::TupleNotImplemented => {
                write!(f, "Tuple size calculation not implemented yet")
            }
            CodegenError::NamedTypeNotImplemented => {
                write!(f, "Named type size calculation requires lookup (not implemented yet)")
            }
            CodegenError::VoidTypeSize => {
                write!(f, "Cannot get size of void type")
            }
            CodegenError::UnsupportedLiteralTypeInGlobal(lit_type) => {
                write!(f, "Unsupported literal type in global initializer: {}", lit_type)
            }
            CodegenError::UnsupportedTypeForOperation(op_type) => {
                write!(f, "Unsupported type for {} operation", op_type)
            }

            // Instruction Errors
            CodegenError::StoreNotImplementedForType(type_info) => {
                write!(f, "Store for type '{}' not implemented yet", type_info)
            }
            CodegenError::LoadNotImplementedForType(type_info) => {
                write!(f, "Load for type '{}' not implemented yet", type_info)
            }
            CodegenError::BinaryOpNotSupportedForType(type_info) => {
                write!(f, "Binary operation for type '{}' not supported yet", type_info)
            }
            CodegenError::ComparisonOpNotSupportedForType(type_info) => {
                write!(f, "Comparison operation for type '{}' not supported yet", type_info)
            }
            CodegenError::ZeroExtensionNotSupported(ext_info) => {
                write!(f, "Unsupported zero extension: {}", ext_info)
            }
            CodegenError::InvalidAllocationLocation => {
                write!(f, "Stack allocation result location invalid")
            }
            CodegenError::HeapAllocationNotSupported => {
                write!(f, "Heap allocation requires runtime/libc (malloc)")
            }
            CodegenError::UnsupportedTypeForHeapAllocation(ty) => {
                write!(f, "Unsupported type for heap allocation: {}", ty)
            }

            // Operand/Immediate Errors
            CodegenError::InvalidImmediateValue => {
                write!(f, "Invalid immediate value provided")
            }
            CodegenError::F32LiteralNotImplemented => {
                write!(f, "f32 literal operand not implemented")
            }
            CodegenError::StringLiteralRequiresGlobal => {
                write!(f, "String literal operand requires label (use global var)")
            }
            CodegenError::GlobalNotFound(name) => {
                write!(f, "Global '{}' not found in layout map", name)
            }

            // Register/Stack Errors
            CodegenError::RegisterAllocationFailed => {
                write!(f, "Register allocation failed")
            }
            CodegenError::StackOverflow => {
                write!(f, "Stack overflow: too many local variables")
            }
            CodegenError::ValueLocationNotFound(name) => {
                write!(f, "Value '{}' location not found in function context", name)
            }
            CodegenError::BlockLabelNotFound(label) => {
                write!(f, "Label '{}' not found in function context", label)
            }

            // Global/Initializer Errors
            CodegenError::GlobalToGlobalInitNotImplemented => {
                write!(f, "Global initializer pointing to another global not implemented yet")
            }
            CodegenError::GlobalVarInitNotSupported => {
                write!(f, "Cannot initialize global with a variable")
            }
            CodegenError::UninitializedGlobalInit => {
                write!(f, "generate_global_initializer called on uninitialized global")
            }

            // Feature Support Errors
            CodegenError::UnsupportedFeature(feature_type) => {
                write!(f, "Unsupported feature: {}", feature_type)
            }
            CodegenError::InvalidInstructionForTarget(instr_type) => {
                write!(f, "Invalid instruction for target: {}", instr_type)
            }

            // Internal Errors
            CodegenError::InternalError => {
                write!(f, "Internal codegen error occurred")
            }
        }
    }
}

impl std::error::Error for CodegenError {}

impl std::fmt::Display for LiteralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralType::Unknown(s) => write!(f, "{}", s),
            LiteralType::F32 => write!(f, "F32"),
            LiteralType::F64 => write!(f, "F64"),
            LiteralType::String => write!(f, "String"),
            LiteralType::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Store => write!(f, "store"),
            OperationType::Load => write!(f, "load"),
            OperationType::BinaryOp => write!(f, "binary"),
            OperationType::Comparison => write!(f, "comparison"),
            OperationType::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for TypeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeInfo::Primitive(pt) => write!(f, "{:?}", pt),
            TypeInfo::Array => write!(f, "Array"),
            TypeInfo::Struct => write!(f, "Struct"),
            TypeInfo::Tuple => write!(f, "Tuple"),
            TypeInfo::Named => write!(f, "Named"),
            TypeInfo::Void => write!(f, "Void"),
            TypeInfo::Unknown(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for ExtensionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionInfo::I8ToI32 => write!(f, "i8 to i32"),
            ExtensionInfo::I8ToI64 => write!(f, "i8 to i64"),
            ExtensionInfo::I32ToI64 => write!(f, "i32 to i64"),
            ExtensionInfo::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureType::FloatOperations => write!(f, "Float Operations"),
            FeatureType::StructOperations => write!(f, "Struct Operations"),
            FeatureType::HeapAllocation => write!(f, "Heap Allocation"),
            FeatureType::TailCalls => write!(f, "Tail Calls"),
            FeatureType::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for InstructionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionType::InvalidOpcode => write!(f, "Invalid Opcode"),
            InstructionType::UnsupportedTarget => write!(f, "Unsupported Target"),
            InstructionType::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Example of how to use the new CodegenError types
/// This demonstrates replacing unwrap() calls with proper error handling
pub mod error_examples {
    use super::{CodegenError, LiteralType, TypeInfo};
    use crate::{PrimitiveType, Result};

    /// Example: Replace unwrap() with proper error handling
    pub fn safe_parse_immediate(value: &str) -> Result<u64> {
        value
            .parse::<u64>()
            .map_err(|_| CodegenError::InvalidImmediateValue.into())
    }

    /// Example: Replace string-based errors with typed errors
    pub fn safe_check_primitive_type(pt: PrimitiveType) -> Result<()> {
        match pt {
            PrimitiveType::I8
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::Bool
            | PrimitiveType::Ptr => Ok(()),
            _ => Err(CodegenError::UnsupportedPrimitiveType(pt).into()),
        }
    }

    /// Example: Replace generic error messages with specific error types
    pub fn safe_global_lookup(
        name: &str,
        globals: &std::collections::HashMap<String, String>,
    ) -> Result<String> {
        globals
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::GlobalNotFound(name.to_string()).into())
    }

    /// Example: Using specific type information instead of strings
    pub fn safe_store_operation(pt: PrimitiveType) -> Result<()> {
        match pt {
            PrimitiveType::F32 | PrimitiveType::F64 => {
                Err(CodegenError::StoreNotImplementedForType(TypeInfo::Primitive(pt)).into())
            }
            _ => Ok(()),
        }
    }

    /// Example: Using specific literal type information
    pub fn safe_literal_check(lit_type: &str) -> Result<()> {
        let literal_type = match lit_type {
            "f32" => LiteralType::F32,
            "f64" => LiteralType::F64,
            "string" => LiteralType::String,
            _ => LiteralType::Unknown(lit_type.to_string()),
        };

        match literal_type {
            LiteralType::F32 => {
                Err(CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::F32).into())
            }
            _ => Ok(()),
        }
    }
}
