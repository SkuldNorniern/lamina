//! Binary instruction encoders
//!
//! This module provides binary encoding of instructions for different architectures.
//! The encoders are shared between ras (assembler) and runtime compilation (JIT).

pub mod traits;
pub mod x86_64;

pub use traits::InstructionEncoder;

