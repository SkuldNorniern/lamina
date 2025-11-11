pub mod regalloc;

pub mod arm;
pub mod riscv;
pub mod wasm;
pub mod x86_64;

use std::collections::HashMap;
use std::io::Write;

use crate::mir::{Global, MirType, Signature};

/// Generate AArch64 assembly from MIR for the requested host OS.
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_aarch64<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> std::result::Result<(), crate::error::LaminaError> {
    use crate::error::LaminaError;

    let target_os = match host_os {
        "macos" | "darwin" => TargetOs::MacOs,
        "linux" => TargetOs::Linux,
        "windows" | "win" => TargetOs::Windows,
        "bsd" => TargetOs::BSD,
        _ => TargetOs::Linux,
    };

    let types: HashMap<String, MirType> = HashMap::new();
    let globals: HashMap<String, Global> = module
        .globals
        .iter()
        .map(|(name, global)| (name.clone(), global.clone()))
        .collect();
    let funcs: HashMap<String, Signature> = module
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), func.sig.clone()))
        .collect();

    let mut backend = arm::aarch64::AArch64Codegen::new(target_os);
    backend
        .prepare(&types, &globals, &funcs, false, &[], &module.name)
        .map_err(wrap_codegen_error)?;
    backend.set_module(module);
    backend.compile().map_err(wrap_codegen_error)?;
    backend.emit_asm().map_err(wrap_codegen_error)?;
    let asm_output = backend.drain_output();
    writer.write_all(&asm_output).map_err(LaminaError::from)?;
    backend.finalize().map_err(wrap_codegen_error)?;

    Ok(())
}

/// Generate x86_64 assembly from MIR for the requested host OS.
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_x86_64<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> std::result::Result<(), crate::error::LaminaError> {
    use crate::error::LaminaError;

    let target_os = match host_os {
        "macos" | "darwin" => TargetOs::MacOs,
        "linux" => TargetOs::Linux,
        "windows" | "win" => TargetOs::Windows,
        "bsd" => TargetOs::BSD,
        _ => TargetOs::Linux,
    };

    let mut codegen = x86_64::X86Codegen::new(target_os);
    codegen.set_module(module);
    codegen.emit_into(module, writer)?;

    Ok(())
}

/// Generate WASM from MIR for the requested host OS.
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_wasm<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> std::result::Result<(), crate::error::LaminaError> {
    use crate::error::LaminaError;

    let target_os = match host_os {
        "macos" | "darwin" => TargetOs::MacOs,
        "linux" => TargetOs::Linux,
        "windows" | "win" => TargetOs::Windows,
        "bsd" => TargetOs::BSD,
        _ => TargetOs::Linux,
    };

    let mut codegen = wasm::WasmCodegen::new(target_os);
    codegen.set_module(module);
    codegen.emit_into(module, writer)?;

    Ok(())
}

/// Generate RISC-V assembly from MIR for the requested host OS.
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_riscv<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> std::result::Result<(), crate::error::LaminaError> {
    use crate::error::LaminaError;

    let target_os = match host_os {
        "macos" | "darwin" => TargetOs::MacOs,
        "linux" => TargetOs::Linux,
        "windows" | "win" => TargetOs::Windows,
        "bsd" => TargetOs::BSD,
        _ => TargetOs::Linux,
    };

    let mut codegen = riscv::RiscVCodegen::new(target_os);
    codegen.set_module(module);
    codegen.emit_into(module, writer)?;

    Ok(())
}

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

/// Mark the target operating system
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetOs {
    MacOs,
    Linux,
    Windows,
    BSD,
    Redox,
    Artery,
    Unknown,
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
