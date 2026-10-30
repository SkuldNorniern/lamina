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

pub use convert::{from_ir_x86_64, from_ir_for_target};
pub use error::FromIRError;
pub use mapping::MAX_REGISTER_AGGREGATE;
