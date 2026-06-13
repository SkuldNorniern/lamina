//! Runtime compiler
//!
//! Compiles MIR modules to executable memory using ras.

use crate::error::LaminaError;
use crate::mir::Module as MirModule;
#[cfg(feature = "encoder")]
use crate::mir_codegen::validate_module_call_parameters;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::RasRuntime;
use std::collections::HashMap;
use std::mem;

/// Runtime compiler for JIT compilation
pub struct RuntimeCompiler {
    #[cfg_attr(not(feature = "encoder"), allow(dead_code))] // Read when encoder feature is enabled
    target_arch: TargetArchitecture,
    #[allow(dead_code)] // Used when encoder feature is enabled
    runtime: RasRuntime,
    code_cache: HashMap<String, ras::ExecutableMemory>,
}

impl RuntimeCompiler {
    /// Create a new runtime compiler
    pub fn new(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
        Self {
            target_arch,
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
            validate_module_call_parameters(_module, self.target_arch)?;
            self.runtime.compile_to_memory(_module).map_err(|e| {
                let error_msg = format!("{e}");
                if error_msg.contains("not yet implemented")
                    || error_msg.contains("Unsupported target")
                {
                    LaminaError::ValidationError(format!(
                        "JIT compilation is not supported for this target (or the MIR uses an unsupported construct).\n\
                             Error: {error_msg}\n\
                             JIT machine code is emitted only for x86_64 and AArch64.\n\
                             Consider AOT compilation instead (run without --jit)."
                    ))
                } else {
                    LaminaError::ValidationError(format!("Runtime compilation failed: {e}"))
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
    /// Returns an unsafe function pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. The signature `T` matches the actual function signature in the compiled module.
    /// 2. The returned `ExecutableMemory` (and the `RuntimeCompiler` that owns it) outlives
    ///    every invocation of the returned function pointer.
    pub unsafe fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, LaminaError> {
        let memory = self.compile(module, Some(function_name))?;
        unsafe {
            let ptr = memory.code_start();
            if ptr.is_null() {
                return Err(LaminaError::ValidationError(
                    "ExecutableMemory has null ptr".to_string(),
                ));
            }
            let f: unsafe extern "C" fn() -> T = mem::transmute(ptr);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "encoder"))]
    use crate::mir::Module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn make_compiler() -> RuntimeCompiler {
        RuntimeCompiler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
    }

    #[test]
    fn invalidate_on_empty_cache_does_not_panic() {
        let mut compiler = make_compiler();
        compiler.invalidate("nonexistent");
    }

    #[test]
    fn clear_cache_on_empty_does_not_panic() {
        let mut compiler = make_compiler();
        compiler.clear_cache();
    }

    #[cfg(not(feature = "encoder"))]
    #[test]
    fn compile_without_encoder_returns_validation_error() {
        let mut compiler = make_compiler();
        let module = Module::new("test");
        let result = compiler.compile(&module, None);
        assert!(matches!(
            result,
            Err(crate::error::LaminaError::ValidationError(_))
        ));
    }
}
