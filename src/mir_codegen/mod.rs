//! MIR-based code generation for multiple target architectures.

pub mod abi;
pub mod assemble;
pub mod capability;
pub mod common;
pub mod link;
pub mod regalloc;

pub mod arm;
pub mod riscv;
pub mod wasm;
pub mod x86_64;

pub use abi::Abi;
pub use capability::{CapabilitySet, CodegenCapability};

use std::collections::HashMap;
use std::fmt;
use std::io::Write;

use crate::error::LaminaError;
use crate::mir::{Global, MirType, Signature};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Generates assembly from MIR for the requested target architecture and OS.
pub fn generate_mir_to_target<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> std::result::Result<(), LaminaError> {
    match target_arch {
        TargetArchitecture::Aarch64 => {
            let mut backend = arm::aarch64::AArch64Codegen::new(target_os);
            backend.set_module(module);
            backend.emit_into(module, writer, codegen_units)?;
        }
        TargetArchitecture::X86_64 => {
            let mut codegen = x86_64::X86Codegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
        }
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
            let mut codegen = wasm::WasmCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            let mut codegen = riscv::RiscVCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
        }
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 => {
            let mut codegen = riscv::RiscVCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
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
    generate_mir_to_target(module, writer, TargetArchitecture::Aarch64, target_os, 1)
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
    generate_mir_to_target(module, writer, TargetArchitecture::X86_64, target_os, 1)
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
    generate_mir_to_target(module, writer, TargetArchitecture::Wasm32, target_os, 1)
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
    generate_mir_to_target(module, writer, TargetArchitecture::Riscv64, target_os, 1)
}

/// Code generation options.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Returns the capabilities supported by this backend
    fn capabilities() -> CapabilitySet
    where
        Self: Sized,
    {
        // Default: core capabilities only
        CapabilitySet::core()
    }

    /// Check if a specific capability is supported
    fn supports(cap: &CodegenCapability) -> bool
    where
        Self: Sized,
    {
        Self::capabilities().supports(cap)
    }

    fn prepare(
        &mut self,
        types: &HashMap<String, MirType>,
        globals: &HashMap<String, Global>,
        funcs: &HashMap<String, Signature>,
        codegen_units: usize,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::UnsupportedFeature(message) => {
                write!(f, "Unsupported feature: {}", message)
            }
            CodegenError::InvalidCodegenOptions(message) => {
                write!(f, "Invalid codegen options: {}", message)
            }
            CodegenError::InvalidTargetOs(message) => write!(f, "Invalid target OS: {}", message),
            CodegenError::InvalidMaxBitWidth(width) => {
                write!(f, "Invalid max bit width: {}", width)
            }
            CodegenError::InvalidInputName(message) => {
                write!(f, "Invalid input name: {}", message)
            }
            CodegenError::InvalidVerbose(value) => write!(f, "Invalid verbose flag: {}", value),
            CodegenError::InvalidOptions(options) => {
                write!(f, "Invalid options: {:?}", options)
            }
            CodegenError::InvalidTypes(types) => write!(f, "Invalid types: {:?}", types),
            CodegenError::InvalidGlobals(globals) => write!(f, "Invalid globals: {:?}", globals),
            CodegenError::InvalidFuncs(funcs) => write!(f, "Invalid functions: {:?}", funcs),
        }
    }
}
