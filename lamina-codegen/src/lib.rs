//! lamina-codegen - Codegen utilities for Lamina
//!
//! This crate provides code generation utilities including register allocation,
//! ABI handling, and frame management for various target architectures.
//!
//! ## Usage
//!
//! ```rust
//! use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
//! ```

pub mod abi;
pub mod regalloc;

pub mod x86_64;
pub mod aarch64;
pub mod riscv;
pub mod wasm;

// Re-exports for convenience
pub use abi::Abi;
pub use regalloc::{PhysRegHandle, PhysRegConvertible, RegisterAllocator};

// Architecture-specific re-exports
pub use x86_64::{X64RegAlloc, X86ABI, X86Frame};
