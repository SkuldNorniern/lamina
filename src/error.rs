use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaminaError {
    ParsingError(String), // Placeholder for parsing errors
    CodegenError(String), // Placeholder for codegen errors
    ValidationError(String), // Placeholder for validation errors
    IoError(String), // Placeholder for IO errors
}

impl fmt::Display for LaminaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaminaError::ParsingError(msg) => write!(f, "Parsing Error: {}", msg),
            LaminaError::CodegenError(msg) => write!(f, "Codegen Error: {}", msg),
            LaminaError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
            LaminaError::IoError(msg) => write!(f, "IO Error: {}", msg),
        }
    }
}

// Implement std::error::Error if needed, requires more detail or potentially external crates for backtraces.
// impl std::error::Error for LaminaError {}

impl From<std::io::Error> for LaminaError {
    fn from(err: std::io::Error) -> Self {
        LaminaError::IoError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LaminaError>; 