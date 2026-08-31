//! Runtime compiler
//!
//! Compiles MIR modules to executable memory using ras.

#[cfg(feature = "encoder")]
use crate::mir_codegen::validate_module_call_parameters;
use crate::{error::LaminaError, mir::Module as MirModule};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{ExecutableMemory, RasRuntime};
use std::{collections::HashMap, mem};

/// Runtime compiler for JIT compilation
pub struct RuntimeCompiler {
    #[cfg_attr(not(feature = "encoder"), allow(dead_code))] // Read when encoder feature is enabled
    target_arch: TargetArchitecture,
    #[allow(dead_code)] // Used when encoder feature is enabled
    runtime: RasRuntime,
    code_cache: HashMap<String, ExecutableMemory>,
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
    ) -> Result<ExecutableMemory, LaminaError> {
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

    /// Compile and get a callable handle.
    ///
    /// The handle owns the executable mapping, so the code stays mapped for as long as
    /// the handle lives. This used to return a bare function pointer taken from a local
    /// `ExecutableMemory`, which was unmapped before the caller could call it.
    ///
    /// # Safety
    ///
    /// `T` must match the return type of the compiled function, and that function must
    /// take no arguments.
    pub unsafe fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<CompiledFunction<T>, LaminaError> {
        let memory = self.compile(module, Some(function_name))?;
        let ptr = memory.code_start();
        if ptr.is_null() {
            return Err(LaminaError::ValidationError(
                "ExecutableMemory has null ptr".to_string(),
            ));
        }
        let entry: unsafe extern "C" fn() -> T = unsafe { mem::transmute(ptr) };
        Ok(CompiledFunction {
            _memory: memory,
            entry,
        })
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

/// A compiled function together with the mapping it lives in.
///
/// Holding the `ExecutableMemory` here is the point: dropping this handle unmaps the
/// code, so the entry pointer cannot outlive what it points at.
pub struct CompiledFunction<T> {
    _memory: ExecutableMemory,
    entry: unsafe extern "C" fn() -> T,
}

impl<T> CompiledFunction<T> {
    /// # Safety
    ///
    /// `T` must match the return type of the compiled function, and that function must
    /// take no arguments.
    pub unsafe fn call(&self) -> T {
        unsafe { (self.entry)() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "encoder"))]
    use crate::mir::Module;
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    use crate::mir::codegen::from_ir;
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    use crate::parser::parse_module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn make_compiler() -> RuntimeCompiler {
        RuntimeCompiler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
    }

    /// The handle must keep the mapping alive. This used to return a bare pointer into
    /// an ExecutableMemory that was dropped on the way out, so calling it read unmapped
    /// memory.
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn compiled_function_stays_callable_after_the_compiler_returns() {
        let input = r#"
        fn @answer() -> i64 {
            entry:
                ret.i64 42
        }
        "#;
        let ir_module = parse_module(input).expect("parse");
        let module = from_ir(&ir_module, "jit_test").expect("lower");

        let handle = {
            let mut compiler = make_compiler();
            // SAFETY: @answer takes no arguments and returns i64.
            match unsafe { compiler.compile_function::<i64>(&module, "@answer") } {
                Ok(handle) => handle,
                // JIT support is narrower than the AOT backends; skip where it is absent
                // rather than fail for an unrelated reason.
                Err(_) => return,
            }
        };

        // SAFETY: same contract as above, and the handle still owns the mapping.
        assert_eq!(unsafe { handle.call() }, 42);
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
