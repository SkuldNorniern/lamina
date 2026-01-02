//! Runtime compiler
//!
//! Compiles MIR modules to executable memory using ras.

use crate::error::LaminaError;
use crate::mir::Module as MirModule;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::RasRuntime;
use std::collections::HashMap;

/// Runtime compiler for JIT compilation
pub struct RuntimeCompiler {
    runtime: RasRuntime,
    code_cache: HashMap<String, ras::ExecutableMemory>,
}

impl RuntimeCompiler {
    /// Create a new runtime compiler
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Self {
        Self {
            runtime: RasRuntime::new(target_arch, target_os),
            code_cache: HashMap::new(),
        }
    }

    /// Compile a MIR module to executable memory
    ///
    /// Note: Caching is not yet implemented as ExecutableMemory may not be Clone.
    /// Future implementation will use reference counting or other strategies.
    pub fn compile(
        &mut self,
        module: &MirModule,
        _function_name: Option<&str>,
    ) -> Result<ras::ExecutableMemory, LaminaError> {
        // TODO: Implement caching when ExecutableMemory supports it
        // For now, always compile fresh
        
        // Compile using ras runtime
        let memory = self
            .runtime
            .compile_to_memory(module)
            .map_err(|e| LaminaError::ValidationError(format!("Runtime compilation failed: {}", e)))?;

        Ok(memory)
    }

    /// Compile and get function pointer
    ///
    /// Returns an unsafe function pointer. The caller must ensure:
    /// 1. The signature matches the expected function signature
    /// 2. The memory remains valid for the lifetime of the function pointer
    pub unsafe fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, LaminaError> {
        let memory = self.compile(module, Some(function_name))?;
        let ptr: *const unsafe extern "C" fn() -> T = memory.as_function_ptr();
        Ok(*ptr)
    }

    /// Invalidate cached code
    pub fn invalidate(&mut self, function_name: &str) {
        self.code_cache.remove(function_name);
    }

    /// Clear all cached code
    pub fn clear_cache(&mut self) {
        self.code_cache.clear();
    }
}

