pub mod aarch64;
pub mod x86_64;

// Re-export codegen functions
pub use aarch64::generate_aarch64_assembly;
pub use x86_64::generate_x86_64_assembly;
