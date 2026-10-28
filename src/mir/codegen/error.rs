//! Error types for IR-to-MIR conversion.
//!
//! Errors that can occur during the conversion from
//! high-level IR to low-level MIR.

use std::fmt;

/// Errors that can occur during IR-to-MIR conversion.
#[derive(Debug)]
pub enum FromIRError {
    /// The IR structure is invalid or malformed.
    InvalidIR,
    /// The IR type cannot be converted to MIR (e.g., unsupported composite types).
    UnsupportedType,
    /// The IR instruction is not supported in MIR.
    UnsupportedInstruction,
    /// An unsupported IR instruction, named with the block and function holding it.
    UnsupportedInstructionIn {
        /// Text of the offending instruction.
        instruction: String,
        /// Label of the block holding it.
        block: String,
        /// Name of the function holding it.
        function: String,
    },
    /// A function is missing its entry block.
    MissingEntryBlock,
    /// A variable reference cannot be resolved.
    UnknownVariable,
    /// Lowering produced MIR that failed verification.
    InvalidMir(Vec<String>),
    /// A global declaration uses a form the backend cannot lower yet.
    UnsupportedGlobal(String),
    /// An aggregate crosses a function boundary in a form lowering cannot classify yet.
    UnsupportedAggregate(String),
    /// A width conversion names a source and destination it cannot relate.
    InvalidConversion(String),
    /// A variadic argument form that lowering does not handle.
    VariadicArgument(String),
    /// A value name is defined with two different register classes or signedness.
    ConflictingDefinitions(String),
    /// A stack allocation requests an alignment the frame cannot provide yet.
    UnsupportedAlignment(String),
}

impl fmt::Display for FromIRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FromIRError::InvalidIR => write!(f, "Invalid IR structure detected during conversion"),
            FromIRError::UnsupportedType => {
                write!(
                    f,
                    "IR type cannot be converted to MIR (composite types not yet supported)"
                )
            }
            FromIRError::UnsupportedInstruction => {
                write!(f, "IR instruction is not supported in MIR")
            }
            FromIRError::UnsupportedInstructionIn {
                instruction,
                block,
                function,
            } => write!(
                f,
                "IR instruction is not supported in MIR: '{instruction}' in block '{block}' of '@{function}'"
            ),
            FromIRError::MissingEntryBlock => {
                write!(f, "Function is missing its entry block")
            }
            FromIRError::UnknownVariable => {
                write!(
                    f,
                    "Variable reference cannot be resolved in current context"
                )
            }
            FromIRError::UnsupportedGlobal(reason) => {
                write!(f, "global not yet supported: {reason}")
            }
            FromIRError::UnsupportedAggregate(reason) => {
                write!(f, "aggregate not yet supported: {reason}")
            }
            FromIRError::InvalidConversion(reason) => {
                write!(f, "invalid conversion: {reason}")
            }
            FromIRError::VariadicArgument(reason) => {
                write!(f, "variadic argument not supported: {reason}")
            }
            FromIRError::ConflictingDefinitions(name) => {
                write!(f, "'%{name}' is defined with conflicting types")
            }
            FromIRError::UnsupportedAlignment(reason) => {
                write!(f, "alignment not yet supported: {reason}")
            }
            FromIRError::InvalidMir(errors) => {
                write!(f, "lowering produced invalid MIR: {}", errors.join("; "))
            }
        }
    }
}
