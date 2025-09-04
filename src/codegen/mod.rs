pub mod aarch64;
pub mod common;
pub mod x86_64;
// pub mod riscv;

// Re-export the main codegen functions for external use
pub use aarch64::generate_aarch64_assembly;
pub use x86_64::generate_x86_64_assembly;
