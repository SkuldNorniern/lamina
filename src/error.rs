//! Error types for the Lamina compiler.
//!
//! This module defines the error types used throughout the compiler pipeline,
//! from parsing through code generation.
//!
//! ## Error Categories
//!
//! - **ParsingError**: Errors encountered while parsing IR text into structured data.
//!   These include syntax errors, unexpected tokens, and malformed constructs.
//!
//! - **ValidationError**: Errors that occur when validating the IR structure.
//!   These include undefined types, missing declarations, and type mismatches.
//!
//! - **CodegenError**: Errors during code generation, such as unsupported operations
//!   or target-specific limitations.
//!
//! - **MirError**: Errors during MIR (Machine Intermediate Representation) conversion
//!   or MIR-based code generation.
//!
//! - **IoError**: File I/O errors when reading or writing files.
//!
//! - **Utf8Error**: UTF-8 encoding errors when processing text.
//!
//! - **InternalError**: Internal compiler errors that indicate bugs in the compiler itself.
//!
//! ## Error Message Guidelines
//!
//! Error messages should be:
//! - **Clear**: Explain what went wrong in plain language
//! - **Actionable**: Suggest how to fix the issue when possible
//! - **Contextual**: Include relevant information like type names, line numbers, etc.
//! - **User-friendly**: Avoid technical jargon when possible
//!
//! ## Example
//!
//! ```rust
//! use lamina::LaminaError;
//!
//! // Good error message
//! let err = LaminaError::ValidationError(
//!     "Type 'MyType' is not defined. Please declare it with 'type MyType = ...' before use".to_string()
//! );
//!
//! // Bad error message (too technical, no guidance)
//! let err = LaminaError::ValidationError(
//!     "Type 'MyType' not found in type declarations - this indicates an invalid module".to_string()
//! );
//! ```

use crate::codegen::CodegenError;
use crate::mir::codegen::FromIRError;
use std::error::Error;
use std::fmt;
use std::string::FromUtf8Error;

/// Main error type for the Lamina compiler.
///
/// This enum represents all possible errors that can occur during compilation,
/// including parsing, validation, code generation, and I/O errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaminaError {
    /// Errors encountered during IR parsing.
    ParsingError(String),
    /// Errors during code generation.
    CodegenError(CodegenError),
    /// Errors during MIR conversion or MIR-based code generation.
    MirError(String),
    /// Validation errors for IR or intermediate representations.
    ValidationError(String),
    /// I/O errors when reading or writing files.
    IoError(String),
    /// UTF-8 encoding errors.
    Utf8Error(String),
    /// Internal compiler errors indicating bugs.
    InternalError(String),
}

impl fmt::Display for LaminaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaminaError::ParsingError(msg) => write!(f, "Parsing Error: {}", msg),
            LaminaError::CodegenError(msg) => write!(f, "Codegen Error: {}", msg),
            LaminaError::MirError(msg) => write!(f, "MIR Error: {}", msg),
            LaminaError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
            LaminaError::IoError(msg) => write!(f, "IO Error: {}", msg),
            LaminaError::Utf8Error(msg) => write!(f, "UTF8 Error: {}", msg),
            LaminaError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
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

/// Convert a CodegenError to LaminaError (for gradual migration)
impl From<CodegenError> for LaminaError {
    fn from(err: CodegenError) -> Self {
        LaminaError::CodegenError(err)
    }
}

/// Convert a FromIRError to LaminaError
impl From<FromIRError> for LaminaError {
    fn from(err: FromIRError) -> Self {
        LaminaError::MirError(format!("{:?}", err))
    }
}
