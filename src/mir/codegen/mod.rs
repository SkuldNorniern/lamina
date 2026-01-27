//! IR-to-MIR conversion module.
//!
//! Functionality to convert high-level Lamina IR into
//! low-level LUMIR (Lamina Unified Machine Intermediate Representation).
//! The conversion process includes:
//!
//! - Type mapping from IR types to MIR types
//! - Instruction lowering from IR operations to MIR operations
//! - Variable binding and register assignment
//! - Control flow graph construction

mod convert;
mod error;
mod mapping;

pub use convert::from_ir;
pub use error::FromIRError;
