//! MIR-based code generation for multiple target architectures.

pub mod assemble;
pub mod link;
pub mod regalloc;

pub mod arm;
pub mod riscv;
pub mod wasm;
pub mod x86_64;

use std::collections::HashMap;
use std::io::Write;

use crate::error::LaminaError;
use crate::mir::{Global, MirType, Signature};
use crate::target::{TargetArchitecture, TargetOperatingSystem};

/// Generates assembly from MIR for the requested target architecture and OS.
pub fn generate_mir_to_target<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), LaminaError> {
    match target_arch {
        TargetArchitecture::Aarch64 => {
            let mut backend = arm::aarch64::AArch64Codegen::new(target_os);
            backend.set_module(module);
            backend.emit_into(module, writer)?;
        }
        TargetArchitecture::X86_64 => {
            let mut codegen = x86_64::X86Codegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer)?;
        }
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
            let mut codegen = wasm::WasmCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer)?;
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            let mut codegen = riscv::RiscVCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer)?;
        }
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 => {
            let mut codegen = riscv::RiscVCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer)?;
        }
        _ => {
            return Err(LaminaError::ValidationError(format!(
                "Unsupported target architecture: {:?}",
                target_arch
            )));
        }
    }

    Ok(())
}

/// Generates AArch64 assembly from MIR for the requested host OS.
#[deprecated(
    since = "0.0.9",
    note = "Use generate_mir_to_target with TargetArchitecture::Aarch64 instead"
)]
pub fn generate_mir_to_aarch64<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), LaminaError> {
    generate_mir_to_target(module, writer, TargetArchitecture::Aarch64, target_os)
}

/// Generates x86_64 assembly from MIR for the requested host OS.
#[deprecated(
    since = "0.0.9",
    note = "Use generate_mir_to_target with TargetArchitecture::X86_64 instead"
)]
pub fn generate_mir_to_x86_64<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), LaminaError> {
    generate_mir_to_target(module, writer, TargetArchitecture::X86_64, target_os)
}

/// Generates WASM from MIR for the requested host OS.
#[deprecated(
    since = "0.0.9",
    note = "Use generate_mir_to_target with TargetArchitecture::Wasm32/Wasm64 instead"
)]
pub fn generate_mir_to_wasm<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), LaminaError> {
    generate_mir_to_target(module, writer, TargetArchitecture::Wasm32, target_os)
}

/// Generates RISC-V assembly from MIR for the requested host OS.
#[deprecated(
    since = "0.0.9",
    note = "Use generate_mir_to_target with TargetArchitecture::Riscv32/Riscv64/Riscv128 instead"
)]
pub fn generate_mir_to_riscv<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> std::result::Result<(), LaminaError> {
    generate_mir_to_target(module, writer, TargetArchitecture::Riscv64, target_os)
}

/// Wraps a codegen error into a LaminaError.
fn wrap_codegen_error(err: CodegenError) -> crate::error::LaminaError {
    use crate::codegen::{CodegenError as CoreCodegenError, FeatureType};

    let message = match err {
        CodegenError::UnsupportedFeature(msg) => msg,
        CodegenError::InvalidCodegenOptions(msg) => {
            format!("Invalid codegen options: {}", msg)
        }
        CodegenError::InvalidTargetOs(msg) => format!("Invalid target OS: {}", msg),
        CodegenError::InvalidMaxBitWidth(bits) => {
            format!("Invalid max bit width: {}", bits)
        }
        CodegenError::InvalidInputName(name) => format!("Invalid input name: {}", name),
        CodegenError::InvalidVerbose(flag) => {
            format!("Invalid verbose flag supplied: {}", flag)
        }
        CodegenError::InvalidOptions(opts) => {
            format!("Invalid options provided ({} entries)", opts.len())
        }
        CodegenError::InvalidTypes(types) => {
            format!("Invalid types referenced: {}", types.join(", "))
        }
        CodegenError::InvalidGlobals(globals) => {
            format!("Invalid globals referenced: {}", globals.join(", "))
        }
        CodegenError::InvalidFuncs(funcs) => {
            format!("Invalid functions referenced: {}", funcs.join(", "))
        }
    };

    crate::error::LaminaError::CodegenError(CoreCodegenError::UnsupportedFeature(
        FeatureType::Custom(message),
    ))
}

/// Code generation options.
#[derive(Debug)]
pub enum CodegenOptions {
    /// Debug mode outputs full debug information.
    Debug,
    /// Release mode outputs optimized code without debug information.
    Release,
    /// Custom codegen options.
    Custom((String, String)),
}

/// Trait for code generation backends.
pub trait Codegen {
    /// The binary extension of the target architecture
    const BIN_EXT: &'static str;
    /// Whether this codegen can output assembly.
    const CAN_OUTPUT_ASM: bool;
    /// Whether this codegen can output binary.
    const CAN_OUTPUT_BIN: bool;

    /// The Supported codegen options
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions];
    /// The target operating system
    const TARGET_OS: TargetOperatingSystem;
    /// The max bit width of the target architecture
    const MAX_BIT_WIDTH: u8;

    fn prepare(
        &mut self,
        types: &HashMap<String, MirType>,
        globals: &HashMap<String, Global>,
        funcs: &HashMap<String, Signature>,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError>;

    fn compile(&mut self) -> Result<(), CodegenError>;
    fn finalize(&mut self) -> Result<(), CodegenError>;
    fn emit_asm(&mut self) -> Result<(), CodegenError>;
    fn emit_bin(&mut self) -> Result<(), CodegenError>;
}

/// Code generation errors.
#[derive(Debug)]
pub enum CodegenError {
    UnsupportedFeature(String),
    InvalidCodegenOptions(String),
    InvalidTargetOs(String),
    InvalidMaxBitWidth(u8),
    InvalidInputName(String),
    InvalidVerbose(bool),
    InvalidOptions(Vec<CodegenOptions>),
    InvalidTypes(Vec<String>),
    InvalidGlobals(Vec<String>),
    InvalidFuncs(Vec<String>),
}
