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
    #[allow(dead_code)] // Used when encoder feature is enabled
    runtime: RasRuntime,
    code_cache: HashMap<String, ras::ExecutableMemory>,
}

impl RuntimeCompiler {
    /// Create a new runtime compiler
    pub fn new(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
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
        _module: &MirModule,
        _function_name: Option<&str>,
    ) -> Result<ras::ExecutableMemory, LaminaError> {
        // TODO: Implement caching when ExecutableMemory supports it
        // For now, always compile fresh

        // Compile using ras runtime
        #[cfg(feature = "encoder")]
        {
            self.runtime.compile_to_memory(_module).map_err(|e| {
                let error_msg = format!("{}", e);
                if error_msg.contains("not yet implemented")
                    || error_msg.contains("Unsupported target")
                {
                    LaminaError::ValidationError(format!(
                        "JIT compilation is not yet supported for this architecture.\n\
                             Error: {}\n\
                             Currently only x86_64 is supported for JIT compilation.\n\
                             Consider using AOT compilation instead (remove --jit flag).",
                        error_msg
                    ))
                } else {
                    LaminaError::ValidationError(format!("Runtime compilation failed: {}", e))
                }
            })
        }
        #[cfg(not(feature = "encoder"))]
        {
            Err(LaminaError::ValidationError(
                "Runtime compilation requires the 'encoder' feature to be enabled in ras"
                    .to_string(),
            ))
        }
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
        unsafe {
            let ptr = memory.as_function_ptr::<u8>();
            if ptr.is_null() {
                return Err(LaminaError::ValidationError(
                    "ExecutableMemory has null ptr".to_string(),
                ));
            }
            let f: unsafe extern "C" fn() -> T = std::mem::transmute(ptr);
            Ok(f)
        }
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
