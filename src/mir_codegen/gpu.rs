use crate::{
    error::LaminaError,
    mir::{Function, Instruction, KernelMetadata, Module, gpu::GpuIntrinsic},
};
use std::collections::HashSet;

pub(super) fn kernels(
    module: &Module,
    allow_parameters: bool,
) -> Result<Vec<(&Function, &str, &KernelMetadata)>, LaminaError> {
    if !module.globals.is_empty()
        || module
            .external_functions
            .iter()
            .any(|name| GpuIntrinsic::from_name(name).is_none())
    {
        return Err(error("globals and external declarations are unsupported"));
    }
    if module.functions.is_empty() {
        return Err(error("at least one kernel is required"));
    }
    let mut functions: Vec<_> = module.functions.values().collect();
    functions.sort_by_key(|f| &f.sig.name);
    let mut names = HashSet::new();
    functions.into_iter().map(|function| {
        function.validate().map_err(|e| error(&e))?;
        let metadata = function.sig.kernel.as_ref().ok_or_else(|| error("functions must be marked as kernels"))?;
        let name = function.sig.name.strip_prefix('@').unwrap_or(&function.sig.name);
        if name.is_empty() || !name.bytes().enumerate().all(|(i, c)| c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit()))
            || name.starts_with("__lamina_gpu_") || !names.insert(name) {
            return Err(error("invalid or duplicate kernel symbol"));
        }
        if (!allow_parameters && !function.sig.params.is_empty()) || !function.sig.is_void()
            || !function.stack_reservations.is_empty() || function.blocks.len() != 1 {
            return Err(error("only single-block void kernels without parameters or stack storage are supported"));
        }
        if !matches!(function.blocks[0].instructions.last(), Some(Instruction::Ret { value: None })) {
            return Err(error("kernel must end with a void return"));
        }
        if let Some(size) = metadata.required_workgroup_size {
            let product = size.into_iter().try_fold(1u32, |a, b| a.checked_mul(b));
            if size.contains(&0) || product.is_none_or(|n| n > 1024 || metadata.max_threads.is_some_and(|max| n > max)) {
                return Err(error("invalid required workgroup size"));
            }
        }
        if metadata.max_threads.is_some_and(|n| n == 0 || n > 1024) {
            return Err(error("max_threads must be between 1 and 1024"));
        }
        Ok((function, name, metadata))
    }).collect()
}

/// Name of the first function taking an aggregate by value, for backends that have no
/// classification for one.
pub(super) fn aggregate_parameter_function(module: &Module) -> Option<&str> {
    let mut names: Vec<&str> = module
        .functions
        .iter()
        .filter(|(_, func)| func.sig.params.iter().any(|p| p.aggregate.is_some()))
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    names.first().copied()
}

pub(super) fn error(message: &str) -> LaminaError {
    LaminaError::ValidationError(format!("GPU code emission: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mir::{FunctionBuilder, Register, VirtualReg},
        mir_codegen::{amdgpu, ptx},
    };

    fn module(instructions: Vec<Instruction>) -> Module {
        let mut function = FunctionBuilder::new("empty").block("entry");
        for instruction in instructions {
            function = function.instr(instruction);
        }
        let mut function = function.build();
        function.sig.kernel = Some(KernelMetadata::default());
        let mut module = Module::new("gpu");
        module.add_function(function);
        module
    }

    #[test]
    fn emits_empty_kernel_and_barrier() -> Result<(), LaminaError> {
        let module = module(vec![
            Instruction::Call {
                name: "__lamina_gpu_workgroup_barrier".into(),
                args: vec![],
                ret: None,
            },
            Instruction::Ret { value: None },
        ]);
        let mut ptx_output = Vec::new();
        ptx::emit(&module, &mut ptx_output)?;
        let ptx_output = String::from_utf8(ptx_output)?;
        assert!(ptx_output.contains(".visible .entry empty()"));
        assert!(ptx_output.contains(".reg .b32"));
        assert!(ptx_output.contains("bar.sync 0;"));
        let mut amd_output = Vec::new();
        amdgpu::emit(&module, &mut amd_output)?;
        let amd_output = String::from_utf8(amd_output)?;
        assert!(amd_output.contains(".amdhsa_kernel empty"));
        assert!(amd_output.contains(".amdgpu_metadata"));
        assert!(amd_output.contains("s_barrier"));
        Ok(())
    }

    #[test]
    fn gpu_ids_use_virtual_registers() -> Result<(), LaminaError> {
        let module = module(vec![
            Instruction::Call {
                name: "__lamina_gpu_thread_id_x".into(),
                args: vec![],
                ret: Some(Register::Virtual(VirtualReg::gpr(4000))),
            },
            Instruction::Ret { value: None },
        ]);
        let mut output = Vec::new();
        ptx::emit(&module, &mut output)?;
        let output = String::from_utf8(output)?;
        assert!(output.contains(".reg .b32 %r4000;"));
        assert!(output.contains("mov.u32 %r4000, %tid.x;"));
        let mut output = Vec::new();
        amdgpu::emit(&module, &mut output)?;
        assert!(String::from_utf8(output)?.contains("v_bfe_u32"));
        Ok(())
    }

    #[test]
    fn rejects_host_calls_without_partial_output() {
        let module = module(vec![
            Instruction::Call {
                name: "print".into(),
                args: vec![],
                ret: None,
            },
            Instruction::Ret { value: None },
        ]);
        let mut output = Vec::new();
        assert!(ptx::emit(&module, &mut output).is_err());
        assert!(output.is_empty());
        assert!(amdgpu::emit(&module, &mut output).is_err());
        assert!(output.is_empty());
    }
}
