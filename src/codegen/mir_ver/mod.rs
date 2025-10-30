pub mod arm;

use crate::mir::{Global, MirType, Signature};

use std::collections::HashMap;
use std::io::Write;

/// Generate AArch64 assembly from MIR for the requested host OS.
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_aarch64<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> std::result::Result<(), crate::error::LaminaError> {
    match host_os {
        "macos" | "darwin" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::MacOs),
        "linux" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Linux),
        "windows" | "win" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Windows),
        "bsd" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::BSD),
        _ => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Linux),
    }
}

/// Mark the target operating system
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetOs {
    MacOs,
    Linux,
    Windows,
    BSD,
    Redox,
}

/// The options for the codegen
pub enum CodegenOptions {
    /// Debug mode is the default mode, it will output with the full debug information (example)
    Debug,
    /// Release mode is the optimized mode, it will output without debug information (example)
    Release,
    /// FEAT: TODO: Add more options for codegen
    Custom((String, String)),
}

// FEAT: TODO: Support multithreaded codegen

/// The trait for the codegen
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
    const TARGET_OS: TargetOs;
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

// The Strings are the placeholder for the types
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
