//! Error types for ras assembler

#[derive(Debug)]
pub enum RasError {
    /// I/O error
    IoError(String),
    /// Assembly parsing error
    ParseError(String),
    /// Instruction encoding error
    EncodingError(String),
    /// Object file generation error
    ObjectError(String),
    /// Unsupported target architecture
    UnsupportedTarget(String),
    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for RasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RasError::IoError(msg) => write!(f, "I/O error: {}", msg),
            RasError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            RasError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
            RasError::ObjectError(msg) => write!(f, "Object file error: {}", msg),
            RasError::UnsupportedTarget(msg) => write!(f, "Unsupported target: {}", msg),
            RasError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for RasError {}

