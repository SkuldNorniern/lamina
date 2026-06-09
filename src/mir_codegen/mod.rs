//! MIR-based code generation for multiple target architectures.

pub mod arx64;
pub mod assemble;
pub mod capability;
pub mod common;
pub mod limits;
pub mod link;
pub mod settings;

pub mod arm;
pub mod powerpc;
pub mod riscv;
pub mod wasm;
pub mod x86_64;

pub use capability::{CapabilitySet, CodegenCapability};
pub use limits::{MAX_MIR_CALL_PARAMETERS, validate_module_call_parameters};
pub use settings::{MirCodegenSettings, RegallocStrategy};

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
) -> Result<(), LaminaError> {
    generate_mir_to_target_with_settings(
        module,
        writer,
        target_arch,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

/// Like [`generate_mir_to_target`], but honors [`MirCodegenSettings`] for register allocation
/// and optional assembly debug directives.
pub fn generate_mir_to_target_with_settings<W: Write>(
    module: &crate::mir::Module,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    if settings.emit_asm_debug_lines
        && !CapabilitySet::for_architecture(target_arch).supports(&CodegenCapability::DebugInfo)
    {
        return Err(LaminaError::ValidationError(
            "emit_asm_debug_lines requires DebugInfo capability on this target".to_string(),
        ));
    }
    let global_regalloc_supported = matches!(
        target_arch,
        TargetArchitecture::X86_64
            | TargetArchitecture::Aarch64
            | TargetArchitecture::Arx64
            | TargetArchitecture::Riscv32
            | TargetArchitecture::Riscv64
            | TargetArchitecture::PowerPC64
    ) || {
        #[cfg(feature = "nightly")]
        {
            matches!(target_arch, TargetArchitecture::Riscv128)
        }
        #[cfg(not(feature = "nightly"))]
        {
            false
        }
    };

    if settings.regalloc != RegallocStrategy::Incremental && !global_regalloc_supported {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for this target"
                .to_string(),
        ));
    }

    match target_arch {
        TargetArchitecture::Aarch64 => {
            arm::aarch64::generate_mir_aarch64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Arx64 => {
            arx64::generate_mir_arx64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::X86_64 => {
            x86_64::generate_mir_x86_64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
            let mut codegen = wasm::WasmCodegen::new(target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            riscv::generate_mir_riscv_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::PowerPC64 => {
            powerpc::generate_mir_ppc64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 => {
            riscv::generate_mir_riscv_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
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
) -> Result<(), LaminaError> {
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
) -> Result<(), LaminaError> {
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
) -> Result<(), LaminaError> {
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
) -> Result<(), LaminaError> {
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

    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::mir::codegen;
    use crate::parser;

    fn create_simple_add_function() -> crate::mir::Module {
        let input = r#"
        fn @add(i64 %a, i64 %b) -> i64 {
            entry:
                %res = add.i64 %a, %b
                ret.i64 %res
        }
        "#;

        let ir_module = parser::parse_module(input).expect("Failed to parse module");
        codegen::from_ir(&ir_module, "test_input").expect("Failed to lower to MIR")
    }

    #[test]
    fn test_cross_target_x86_64() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            1,
        )
        .expect("Failed to generate x86_64 assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(!asm.is_empty(), "x86_64 output should not be empty");
        assert!(
            asm.contains("add") || asm.contains("ADD"),
            "x86_64 output should contain add instruction"
        );
    }

    #[test]
    fn test_x86_64_graph_color_and_asm_debug_lines() {
        let module = create_simple_add_function();
        let settings = MirCodegenSettings {
            regalloc: RegallocStrategy::GraphColorGlobal,
            emit_asm_debug_lines: true,
            debug_file_tag: "add.lamina".to_string(),
            ..Default::default()
        };
        let mut output = Vec::new();
        generate_mir_to_target_with_settings(
            &module,
            &mut output,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )
        .expect("graph-color + debug codegen");

        let asm = String::from_utf8(output).expect("UTF-8");
        assert!(
            asm.contains(".file 1 \"add.lamina\""),
            "expected .file: {asm}"
        );
        assert!(asm.contains(".loc 1"), "expected .loc directives: {asm}");
    }

    #[test]
    fn test_cross_target_aarch64() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux,
            1,
        )
        .expect("Failed to generate AArch64 assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(!asm.is_empty(), "AArch64 output should not be empty");
        assert!(
            asm.contains("add") || asm.contains("ADD"),
            "AArch64 output should contain add instruction"
        );
    }

    #[test]
    fn test_cross_target_riscv64() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Linux,
            1,
        )
        .expect("Failed to generate RISC-V assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(!asm.is_empty(), "RISC-V output should not be empty");
        assert!(
            asm.contains("add") || asm.contains("ADD"),
            "RISC-V output should contain add instruction"
        );
    }

    #[test]
    fn test_cross_target_wasm32() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            TargetArchitecture::Wasm32,
            TargetOperatingSystem::Linux,
            1,
        )
        .expect("Failed to generate WASM");

        let wat = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(!wat.is_empty(), "WASM output should not be empty");
        assert!(
            wat.contains("i64.add") || wat.contains("add"),
            "WASM output should contain add instruction"
        );
    }

    #[test]
    fn test_cross_target_same_ir_different_backends() {
        // Test that the same IR produces valid output for all backends
        let module = create_simple_add_function();
        let targets = vec![
            (TargetArchitecture::X86_64, "x86_64"),
            (TargetArchitecture::Aarch64, "aarch64"),
            (TargetArchitecture::Riscv64, "riscv64"),
            (TargetArchitecture::Wasm32, "wasm32"),
        ];

        for (arch, name) in targets {
            let mut output = Vec::new();
            let result =
                generate_mir_to_target(&module, &mut output, arch, TargetOperatingSystem::Linux, 1);

            assert!(
                result.is_ok(),
                "Failed to generate code for {}: {:?}",
                name,
                result.err()
            );

            let output_str = String::from_utf8(output).expect("Invalid UTF-8");
            assert!(
                !output_str.is_empty(),
                "{} output should not be empty",
                name
            );
        }
    }

    #[test]
    fn test_cross_target_integer_arithmetic() {
        // Test integer arithmetic operations across all targets
        let input = r#"
        fn @test_arithmetic(i64 %a, i64 %b) -> i64 {
            entry:
                %sum = add.i64 %a, %b
                %diff = sub.i64 %a, %b
                %prod = mul.i64 %sum, %diff
                ret.i64 %prod
        }
        "#;

        let ir_module = parser::parse_module(input).expect("Failed to parse");
        let mir_module = codegen::from_ir(&ir_module, "test").expect("Failed to lower to MIR");

        let targets = vec![
            TargetArchitecture::X86_64,
            TargetArchitecture::Aarch64,
            TargetArchitecture::Riscv64,
            TargetArchitecture::Wasm32,
        ];

        for arch in targets {
            let mut output = Vec::new();
            let result = generate_mir_to_target(
                &mir_module,
                &mut output,
                arch,
                TargetOperatingSystem::Linux,
                1,
            );

            assert!(
                result.is_ok(),
                "Failed to generate arithmetic code for {:?}: {:?}",
                arch,
                result.err()
            );
        }
    }
}
