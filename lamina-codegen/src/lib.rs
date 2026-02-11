//! lamina-codegen - Codegen utilities for Lamina
//!
//! This crate provides code generation utilities including register allocation,
//! ABI handling, and frame management for various target architectures.
//!
//! Cross-compilation is supported: pass the desired target (arch, OS) when
//! creating ABI or regalloc; the generated code will follow that target's ABI.
//!
//! ## Usage
//!
//! ```rust
//! use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
//! use lamina_platform::TargetOperatingSystem;
//!
//! let abi = X86ABI::new(TargetOperatingSystem::Linux);
//! let regalloc = X64RegAlloc::new(TargetOperatingSystem::Windows);
//! ```

pub mod abi;
pub mod regalloc;
pub mod target_support;

pub mod x86_64;
pub mod aarch64;
pub mod riscv;
pub mod wasm;

// Re-exports for convenience
pub use abi::Abi;
pub use regalloc::{PhysRegHandle, PhysRegConvertible, RegisterAllocator};
pub use target_support::{
    is_assembly_supported,
    os_uses_coff,
    os_uses_elf,
    os_uses_macho,
    supported_assembly_targets_hint,
};

// Architecture-specific re-exports
pub use x86_64::{X64RegAlloc, X86ABI, X86Frame};
