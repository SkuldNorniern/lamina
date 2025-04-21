use std::error::Error; // Import the Error trait
use std::fmt;
use std::string::FromUtf8Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaminaError {
    ParsingError(String),    // Placeholder for parsing errors
    CodegenError(String),    // Placeholder for codegen errors
    ValidationError(String), // Placeholder for validation errors
    IoError(String),         // Placeholder for IO errors
    Utf8Error(String),       // Added variant for UTF8 errors
}

impl fmt::Display for LaminaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaminaError::ParsingError(msg) => write!(f, "Parsing Error: {}", msg),
            LaminaError::CodegenError(msg) => write!(f, "Codegen Error: {}", msg),
            LaminaError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
            LaminaError::IoError(msg) => write!(f, "IO Error: {}", msg),
            LaminaError::Utf8Error(msg) => write!(f, "UTF8 Error: {}", msg),
        }
    }
}

// Implement the standard Error trait
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

pub type Result<T> = std::result::Result<T, LaminaError>;
