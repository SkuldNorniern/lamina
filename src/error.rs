//! Error types for the Lamina compiler pipeline.

use crate::mir::codegen::FromIRError;
use crate::mir_codegen::CodegenError;
use std::error::Error;
use std::fmt;
use std::string::FromUtf8Error;

/// Top-level error type covering all compiler pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaminaError {
    /// Syntax or structural error while parsing IR text.
    ParsingError(String),
    /// Target-specific code generation failure.
    CodegenError(CodegenError),
    /// Error during IR→MIR lowering or MIR-level code generation.
    MirError(String),
    /// Semantic error: undefined types, missing declarations, type mismatches.
    ValidationError(String),
    /// File I/O failure.
    IoError(String),
    /// Input was not valid UTF-8.
    Utf8Error(String),
    /// Compiler bug — should never reach the caller.
    InternalError(String),
    /// Failure during interpreter or JIT execution.
    RuntimeError(String),
}

impl fmt::Display for LaminaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaminaError::ParsingError(msg) => write!(f, "Parsing Error: {msg}"),
            LaminaError::CodegenError(msg) => write!(f, "Codegen Error: {msg}"),
            LaminaError::MirError(msg) => write!(f, "MIR Error: {msg}"),
            LaminaError::ValidationError(msg) => write!(f, "Validation Error: {msg}"),
            LaminaError::IoError(msg) => write!(f, "IO Error: {msg}"),
            LaminaError::Utf8Error(msg) => write!(f, "UTF8 Error: {msg}"),
            LaminaError::InternalError(msg) => write!(f, "Internal Error: {msg}"),
            LaminaError::RuntimeError(msg) => write!(f, "Runtime Error: {msg}"),
        }
    }
}

impl Error for LaminaError {}

impl From<std::io::Error> for LaminaError {
    fn from(err: std::io::Error) -> Self {
        LaminaError::IoError(err.to_string())
    }
}

impl From<FromUtf8Error> for LaminaError {
    fn from(err: FromUtf8Error) -> Self {
        LaminaError::Utf8Error(err.to_string())
    }
}

impl From<CodegenError> for LaminaError {
    fn from(err: CodegenError) -> Self {
        LaminaError::CodegenError(err)
    }
}

impl From<FromIRError> for LaminaError {
    fn from(err: FromIRError) -> Self {
        LaminaError::MirError(format!("{err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_variant_prefix() {
        assert!(LaminaError::ParsingError("x".into()).to_string().starts_with("Parsing Error:"));
        assert!(LaminaError::MirError("x".into()).to_string().starts_with("MIR Error:"));
        assert!(LaminaError::ValidationError("x".into()).to_string().starts_with("Validation Error:"));
        assert!(LaminaError::IoError("x".into()).to_string().starts_with("IO Error:"));
        assert!(LaminaError::Utf8Error("x".into()).to_string().starts_with("UTF8 Error:"));
        assert!(LaminaError::InternalError("x".into()).to_string().starts_with("Internal Error:"));
        assert!(LaminaError::RuntimeError("x".into()).to_string().starts_with("Runtime Error:"));
    }

    #[test]
    fn from_io_error_wraps_message() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let lamina_err = LaminaError::from(io_err);
        assert!(matches!(lamina_err, LaminaError::IoError(_)));
        assert!(lamina_err.to_string().contains("file missing"));
    }

    #[test]
    fn from_utf8_error_wraps_message() {
        let utf8_err = String::from_utf8(vec![0xFF]).unwrap_err();
        let lamina_err = LaminaError::from(utf8_err);
        assert!(matches!(lamina_err, LaminaError::Utf8Error(_)));
    }
}
