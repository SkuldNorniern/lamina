//! x86_64 codegen utilities
//!
//! This module provides register allocation, ABI handling, and frame management
//! utilities for x86_64 code generation.

pub mod abi;
pub mod constants;
pub mod frame;
pub mod regalloc;
pub mod util;

pub use abi::X86ABI;
pub use frame::X86Frame;
pub use regalloc::X64RegAlloc;

