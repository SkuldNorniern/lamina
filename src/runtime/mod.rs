//! Runtime compilation and execution
//!
//! This module provides runtime code generation (JIT compilation) using ras,
//! with optional sandboxing for secure execution.

pub mod sandbox;
pub mod compiler;

pub use compiler::RuntimeCompiler;
pub use sandbox::{Sandbox, SandboxConfig};

use crate::error::LaminaError;
use crate::mir::Module as MirModule;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Runtime compilation result
pub struct RuntimeResult {
    /// Executable memory containing compiled code
    pub memory: ras::ExecutableMemory,
    /// Function pointer (unsafe - caller must ensure signature matches)
    pub function_ptr: *const u8,
}

// Re-export ras types for convenience
pub use ras::{ExecutableMemory, RasRuntime};

/// Compile MIR module to executable memory using runtime compilation
pub fn compile_to_runtime(
    module: &MirModule,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Result<RuntimeResult, LaminaError> {
    use ras::RasRuntime;
    
    let mut runtime = RasRuntime::new(target_arch, target_os);
    let memory = runtime
        .compile_to_memory(module)
        .map_err(|e| LaminaError::ValidationError(format!("Runtime compilation failed: {}", e)))?;
    
    let function_ptr = unsafe { memory.as_function_ptr::<u8>() };
    
    Ok(RuntimeResult {
        memory,
        function_ptr,
    })
}

