//! Assembler module
//!
//! This module is split into submodules for better organization:
//! - `core`: Main assembler struct and basic operations
//! - `x86_64`: x86_64 architecture-specific code
//! - `aarch64`: AArch64 architecture-specific code
//! - `common`: Shared utilities (register parsing, etc.)

pub mod core;
pub mod common;

#[cfg(feature = "encoder")]
pub mod x86_64;

#[cfg(feature = "encoder")]
pub mod aarch64;

pub use core::RasAssembler;




