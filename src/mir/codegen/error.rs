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
    /// writeptr needs a 1, 2, 4 or 8 byte pointee known from the pointer's defining instruction.
    WritePtrPointeeUnknown,
    /// A function is missing its entry block.
    MissingEntryBlock,
    /// A variable reference cannot be resolved.
    UnknownVariable,
    /// print with string literal is not supported; use writebyte for portable output.
    PrintStringLiteralUnsupported,
    /// Lowering produced MIR that failed verification.
    InvalidMir(Vec<String>),
    /// A global declaration uses a form the backend cannot lower yet.
    UnsupportedGlobal(String),
    /// An aggregate crosses a function boundary in a form lowering cannot classify yet.
    UnsupportedAggregate(String),
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
            FromIRError::WritePtrPointeeUnknown => write!(
                f,
                "writeptr needs a pointer to a 1, 2, 4 or 8 byte scalar, defined by alloc, getelemptr or getfieldptr in the same function"
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
            FromIRError::PrintStringLiteralUnsupported => write!(
                f,
                "print with string literal is not supported. Use writebyte in a loop for portable output. See docs/OUTPUT_SEMANTICS.md"
            ),
            FromIRError::UnsupportedGlobal(reason) => {
                write!(f, "global not yet supported: {reason}")
            }
            FromIRError::UnsupportedAggregate(reason) => {
                write!(f, "aggregate not yet supported: {reason}")
            }
            FromIRError::InvalidMir(errors) => {
                write!(f, "lowering produced invalid MIR: {}", errors.join("; "))
            }
        }
    }
}
