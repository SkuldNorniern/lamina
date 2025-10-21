#[derive(Debug)]
pub enum FromIRError {
    InvalidIR,
    UnsupportedType,
    UnsupportedInstruction,
    MissingEntryBlock,
    UnknownVariable,
}

impl std::fmt::Display for FromIRError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromIRError::InvalidIR => write!(f, "InvalidIR"),
            FromIRError::UnsupportedType => write!(f, "Unsupported Type"),
            FromIRError::UnsupportedInstruction => write!(f, "Unsupported Inst"),
            FromIRError::MissingEntryBlock => write!(f, "Missing Entry"),
            FromIRError::UnknownVariable => write!(f, "Variable Unknown"),
        }
    }
}
