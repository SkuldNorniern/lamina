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
    /// A function is missing its entry block.
    MissingEntryBlock,
    /// A variable reference cannot be resolved.
    UnknownVariable,
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
            FromIRError::MissingEntryBlock => {
                write!(f, "Function is missing its entry block")
            }
            FromIRError::UnknownVariable => {
                write!(
                    f,
                    "Variable reference cannot be resolved in current context"
                )
            }
        }
    }
}
