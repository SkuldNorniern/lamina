//! Common code for MIR codegen backends.

use std::collections::HashMap;
use std::io::Write;

use crate::mir::{Global, MirType, Module as MirModule, Signature};
use crate::mir_codegen::{CodegenError, CodegenOptions};
use crate::target::TargetOperatingSystem;

/// Base structure for codegen backends with common fields.
pub struct CodegenBase<'a> {
    pub target_os: TargetOperatingSystem,
    pub module: Option<&'a MirModule>,
    pub prepared: bool,
    pub verbose: bool,
    pub output: Vec<u8>,
}

impl<'a> CodegenBase<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            target_os,
            module: None,
            prepared: false,
            verbose: false,
            output: Vec::new(),
        }
    }

    pub fn set_module(&mut self, module: &'a MirModule) {
        self.module = Some(module);
    }

    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output)
    }

    pub fn prepare_base(
        &mut self,
        _types: &HashMap<String, MirType>,
        _globals: &HashMap<String, Global>,
        _funcs: &HashMap<String, Signature>,
        verbose: bool,
        _options: &[CodegenOptions],
        _input_name: &str,
    ) -> Result<(), CodegenError> {
        self.verbose = verbose;
        self.prepared = true;
        Ok(())
    }

    pub fn compile_base(&self) -> Result<(), CodegenError> {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(
                "Codegen not prepared".to_string(),
            ));
        }
        Ok(())
    }

    pub fn finalize_base(&mut self) -> Result<(), CodegenError> {
        self.module = None;
        self.prepared = false;
        Ok(())
    }

    pub fn emit_asm_base<F>(&mut self, emit_fn: F, backend_name: &str) -> Result<(), CodegenError>
    where
        F: FnOnce(
            &MirModule,
            &mut Vec<u8>,
            TargetOperatingSystem,
        ) -> Result<(), crate::error::LaminaError>,
    {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(format!(
                "emit_asm called before prepare for {}",
                backend_name
            )));
        }
        let module = self.module.ok_or_else(|| {
            CodegenError::InvalidCodegenOptions(format!(
                "No module set for emission in {} backend",
                backend_name
            ))
        })?;
        self.output.clear();
        emit_fn(module, &mut self.output, self.target_os).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("{} emission failed: {}", backend_name, e))
        })
    }
}

/// Helper to emit rodata section with format string for print intrinsics.
pub fn emit_print_format_section<W: Write>(
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), crate::error::LaminaError> {
    match target_os {
        TargetOperatingSystem::MacOS => {
            writeln!(writer, ".section __TEXT,__cstring,cstring_literals")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        TargetOperatingSystem::Windows => {
            writeln!(writer, ".section .rdata,\"dr\"")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        TargetOperatingSystem::Linux => {
            writeln!(writer, ".section .rodata")?;
            writeln!(writer, ".L_mir_fmt_int: .string \"%lld\\n\"")?;
        }
        _ => {
            writeln!(writer, ".section .rodata")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
    }
    Ok(())
}

/// Convert LaminaError to CodegenError with consistent error type.
pub fn lamina_to_codegen_error(err: crate::error::LaminaError) -> CodegenError {
    match err {
        crate::error::LaminaError::InternalError(msg) => {
            CodegenError::InvalidCodegenOptions(format!("Internal error: {}", msg))
        }
        crate::error::LaminaError::CodegenError(inner) => {
            CodegenError::InvalidCodegenOptions(inner.to_string())
        }
        crate::error::LaminaError::ParsingError(msg)
        | crate::error::LaminaError::ValidationError(msg)
        | crate::error::LaminaError::MirError(msg)
        | crate::error::LaminaError::IoError(msg)
        | crate::error::LaminaError::Utf8Error(msg) => CodegenError::InvalidCodegenOptions(msg),
    }
}
