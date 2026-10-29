//! MIR-based code generation for multiple target architectures.

pub mod amdgpu;
#[cfg(feature = "nightly")]
pub mod arx64;
pub mod assemble;
pub mod capability;
pub mod common;
mod gpu;
pub mod i686;
pub mod limits;
pub mod link;
pub mod loongarch64;
pub mod mips64;
pub mod ptx;
pub mod s390x;
pub mod settings;
mod simd;

pub mod arm;
pub mod powerpc;
pub mod riscv;
pub mod wasm;
pub mod x86_64;

pub use capability::{CapabilitySet, CodegenCapability};
pub use limits::{MAX_MIR_CALL_PARAMETERS, validate_module_call_parameters};
pub use settings::{MirCodegenSettings, RegallocStrategy};

use std::{borrow::Cow, collections::HashMap, fmt, io::Write, mem::discriminant};

use crate::{
    error::LaminaError,
    mir::{Global, Instruction, MirType, Module, RelocationModel, Signature},
};
use lamina_codegen::{
    abi::{BuiltinLibrary, RuntimeSymbol},
    riscv::{RiscVTarget, Xlen},
};
use lamina_platform::{ArchVariant, TargetArchitecture, TargetOperatingSystem, X86CpuLevel};

/// Generates assembly from MIR for the requested target architecture and OS.
pub fn generate_mir_to_target<W: Write>(
    module: &Module,
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

/// Rewrites runtime helper calls to the names `LAMINA_BUILTINS` selects.
///
/// Only calls whose names match a [`RuntimeSymbol`] are touched, and only when an
/// override is set, so the default path borrows the module unchanged.
pub fn retarget_runtime_calls<'a>(
    module: &'a Module,
    builtins: &BuiltinLibrary,
) -> Cow<'a, Module> {
    if !builtins.has_overrides() {
        return Cow::Borrowed(module);
    }
    let retarget = |name: &mut String| {
        if let Some(symbol) = RuntimeSymbol::from_c_name(name) {
            *name = builtins.c_name(symbol).to_string();
        }
    };
    let mut retargeted = module.clone();
    let declared: Vec<(String, String)> = retargeted
        .external_functions
        .iter()
        .filter_map(|name| {
            let symbol = RuntimeSymbol::from_c_name(name)?;
            Some((name.clone(), builtins.c_name(symbol).to_string()))
        })
        .collect();
    for (old, new) in declared {
        if old == new {
            continue;
        }
        retargeted.external_functions.remove(&old);
        retargeted.external_functions.insert(new.clone());
        if let Some(mut function) = retargeted.functions.remove(&old) {
            function.sig.name = new.clone();
            retargeted.functions.insert(new, function);
        }
    }
    for function in retargeted.functions.values_mut() {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if let Instruction::Call { name, .. } | Instruction::TailCall { name, .. } =
                    &mut instruction.instr
                {
                    retarget(name);
                }
            }
        }
    }
    Cow::Owned(retargeted)
}

/// A target with no C library has no compiler runtime to link against, so a 128-bit
/// helper the module calls must be defined in the module or redirected through
/// `LAMINA_BUILTINS`. Reported here, naming the symbol, rather than at link time.
fn check_compiler_runtime_symbols(
    module: &Module,
    target_os: TargetOperatingSystem,
    builtins: &BuiltinLibrary,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Unknown {
        return Ok(());
    }
    for symbol in RuntimeSymbol::ALL {
        if !symbol.c_name().starts_with("__") {
            continue;
        }
        let name = builtins.c_name(symbol);
        if module
            .functions
            .get(name)
            .is_some_and(|f| f.blocks.is_empty())
        {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!(
                    "`{name}` is a compiler runtime helper that a target without a C library does not provide; \
                 define it in the module, or name an implementation with LAMINA_BUILTINS={}=<symbol>",
                    symbol.c_name()
                ),
            )));
        }
    }
    Ok(())
}

/// The variant a backend runs with: the one the target named, or the architecture's
/// default when the settings carry none or carry another family's.
fn effective_variant(arch: TargetArchitecture, chosen: ArchVariant) -> ArchVariant {
    let default = arch.default_variant();
    if discriminant(&chosen) == discriminant(&default) {
        chosen
    } else {
        default
    }
}

/// The RISC-V the backend emits for: the ISA the target named.
fn riscv_target(variant: ArchVariant) -> RiscVTarget {
    variant
        .riscv()
        .map_or_else(|| RiscVTarget::general(Xlen::Rv64), RiscVTarget::from_isa)
}

fn not_implemented(arch: TargetArchitecture, os: TargetOperatingSystem) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "{arch} code generation is not implemented: target {arch}_{os} produces no output"
    )))
}

/// Like [`generate_mir_to_target`], but honors [`MirCodegenSettings`] for register allocation
/// and optional assembly debug directives.
pub fn generate_mir_to_target_with_settings<W: Write>(
    input: &Module,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    let pointer_width = target_arch.pointer_width();
    let resolved_settings = settings
        .clone()
        .with_arch_variant(effective_variant(target_arch, settings.arch_variant));
    let settings = &resolved_settings;
    let builtins = BuiltinLibrary::from_env(target_os);
    let retargeted = retarget_runtime_calls(input, &builtins);
    let module: &Module = &retargeted;
    check_compiler_runtime_symbols(module, target_os, &builtins)?;
    if module.relocation != RelocationModel::Static
        && !(target_arch == TargetArchitecture::X86_64 && target_os == TargetOperatingSystem::Linux)
    {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "@pic and @pie are only implemented for x86_64 Linux (ELF); target {target_arch:?}-{target_os:?} is not supported"
            ),
        )));
    }
    if !matches!(
        target_arch,
        TargetArchitecture::X86_64 | TargetArchitecture::I686
    ) {
        if let Some(name) = module
            .functions
            .iter()
            .filter(|(_, function)| function.sig.naked)
            .map(|(name, _)| name)
            .min()
        {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!(
                    "@naked function '{name}' is not supported on target {target_arch:?}-{target_os:?}; only x86_64 Linux honours @naked"
                ),
            )));
        }
    }
    let effective_caps = {
        #[cfg(feature = "nightly")]
        {
            CapabilitySet::for_architecture_with_simd(target_arch, settings.simd.as_ref())
        }
        #[cfg(not(feature = "nightly"))]
        {
            CapabilitySet::for_architecture(target_arch)
        }
    };

    if settings.emit_asm_debug_lines && !effective_caps.supports(&CodegenCapability::DebugInfo) {
        return Err(LaminaError::ValidationError(
            "emit_asm_debug_lines requires DebugInfo capability on this target".to_string(),
        ));
    }
    if settings.regalloc != RegallocStrategy::Incremental
        && !effective_caps.supports(&CodegenCapability::GlobalRegisterAllocation)
    {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for this target"
                .to_string(),
        ));
    }

    let gpu_target = matches!(
        target_arch,
        TargetArchitecture::Nvptx64 | TargetArchitecture::Amdgcn
    );
    if !gpu_target {
        for function in module.functions.values() {
            if function.sig.kernel.is_some() || function.blocks.iter().flat_map(|b| &b.instructions).any(|i| {
                matches!(&i.instr, Instruction::Call { name, .. } if name.trim_start_matches('@').starts_with("__lamina_gpu_"))
            }) {
                return Err(LaminaError::ValidationError("GPU kernels and intrinsics require a GPU target".into()));
            }
        }
    }

    match target_arch {
        TargetArchitecture::Nvptx64 if target_os == TargetOperatingSystem::Cuda => {
            ptx::emit(module, writer)?
        }
        TargetArchitecture::Amdgcn if target_os == TargetOperatingSystem::AmdHsa => {
            amdgpu::emit_with_target(module, writer, settings.amdgpu_target, pointer_width)?
        }
        TargetArchitecture::Aarch64 => {
            arm::aarch64::generate_mir_aarch64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
                pointer_width,
            )?;
        }
        #[cfg(feature = "nightly")]
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
            let mut codegen = wasm::WasmCodegen::new(target_arch, target_os);
            codegen.set_module(module);
            codegen.emit_into(module, writer, codegen_units)?;
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            riscv::generate_mir_riscv_with_units_and_settings(
                module,
                writer,
                riscv_target(settings.arch_variant),
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::LoongArch64 => {
            loongarch64::generate_mir_loongarch64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::S390x => {
            s390x::generate_mir_s390x_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Mips64 | TargetArchitecture::Mips64El => {
            mips64::generate_mir_mips64_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
                target_arch,
            )?;
        }
        TargetArchitecture::PowerPC64Be => {
            powerpc::powerpc64::generate_mir_ppc64_for_arch(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
                target_arch,
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
                riscv_target(settings.arch_variant),
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::I686 => {
            i686::generate_mir_i686_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Armv7 => {
            arm::armv7::generate_mir_armv7_with_units_and_settings(
                module,
                writer,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Thumbv6m | TargetArchitecture::Thumbv7m => {
            arm::cortexm::generate_mir_cortexm_with_units_and_settings(
                module,
                writer,
                target_arch,
                target_os,
                codegen_units,
                settings,
            )?;
        }
        TargetArchitecture::Arm32 => {
            return Err(not_implemented(target_arch, target_os));
        }
        #[cfg(feature = "nightly")]
        TargetArchitecture::Lisa => return Err(not_implemented(target_arch, target_os)),
        TargetArchitecture::Nvptx64 | TargetArchitecture::Amdgcn => {
            return Err(LaminaError::ValidationError(format!(
                "Unsupported target architecture: {target_arch:?}"
            )));
        }
    }

    Ok(())
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

    /// Capabilities this backend declares. Required, so a backend cannot inherit a default.
    fn capabilities() -> CapabilitySet
    where
        Self: Sized;

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
    InvalidMir(Vec<String>),
    CpuLevelRequired {
        instruction: String,
        required: X86CpuLevel,
        selected: X86CpuLevel,
    },
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::UnsupportedFeature(message) => {
                write!(f, "Unsupported feature: {message}")
            }
            CodegenError::CpuLevelRequired {
                instruction,
                required,
                selected,
            } => write!(
                f,
                "instruction `{instruction}` requires {required}, but the selected CPU level is {selected} (raise it with --cpu-level)"
            ),
            CodegenError::InvalidCodegenOptions(message) => {
                write!(f, "Invalid codegen options: {message}")
            }
            CodegenError::InvalidTargetOs(message) => write!(f, "Invalid target OS: {message}"),
            CodegenError::InvalidMaxBitWidth(width) => {
                write!(f, "Invalid max bit width: {width}")
            }
            CodegenError::InvalidInputName(message) => {
                write!(f, "Invalid input name: {message}")
            }
            CodegenError::InvalidVerbose(value) => write!(f, "Invalid verbose flag: {value}"),
            CodegenError::InvalidOptions(options) => {
                write!(f, "Invalid options: {options:?}")
            }
            CodegenError::InvalidTypes(types) => write!(f, "Invalid types: {types:?}"),
            CodegenError::InvalidGlobals(globals) => write!(f, "Invalid globals: {globals:?}"),
            CodegenError::InvalidFuncs(funcs) => write!(f, "Invalid functions: {funcs:?}"),
            CodegenError::InvalidMir(errors) => {
                write!(f, "Invalid MIR: {}", errors.join("; "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::{mir::codegen, parser};

    fn create_simple_add_function() -> Module {
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
        // The source above puts `add` on line 4 and `ret` on line 5.
        let lines: Vec<&str> = asm
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with(".loc "))
            .collect();
        assert_eq!(
            lines
                .iter()
                .map(|line| line.split_whitespace().nth(2).unwrap_or(""))
                .collect::<Vec<_>>(),
            ["4", "5"],
            "expected .loc rows on the real source lines: {asm}"
        );
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

#[cfg(test)]
pub(crate) mod simd_support;

#[cfg(test)]
pub(crate) mod test_support;

// Every test in it builds and runs a native program with the host toolchain.
#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod float_tests;

#[cfg(test)]
mod allocation_tests;

#[cfg(test)]
mod unary_tests;

#[cfg(test)]
mod atomic_target_tests {
    #![allow(clippy::expect_used)]

    use super::generate_mir_to_target;
    use crate::{mir::codegen, parser::parse_module};
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    #[test]
    fn wasm_reports_atomics_as_a_clean_error() {
        let ir = parse_module(
            "fn @f(ptr %p) -> i64 {\nentry:\n %old = atomic_add.i64 %p, 1, seqcst\n ret.i64 %old\n}",
        )
        .expect("parse atomic add");
        let mir = codegen::from_ir(&ir, "atomics").expect("lower atomic add");
        let mut output = Vec::new();
        let err = generate_mir_to_target(
            &mir,
            &mut output,
            TargetArchitecture::Wasm32,
            TargetOperatingSystem::Wasi,
            1,
        )
        .expect_err("atomics must not reach the wasm encoder");
        let message = err.to_string();
        assert!(message.contains("atomic_add"), "{message}");
        assert!(message.contains("WASM"), "{message}");
    }

    #[test]
    fn naked_is_refused_on_targets_that_do_not_honour_it() {
        let ir = parse_module(
            "@naked\nfn @isr() -> void {\nentry:\n    asm.void \"cli; hlt\"\n    unreachable\n}\n",
        )
        .expect("parse naked");
        let mir = codegen::from_ir(&ir, "naked").expect("lower naked");
        for (arch, os) in [
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Wasm32, TargetOperatingSystem::Wasi),
        ] {
            let mut output = Vec::new();
            let err = generate_mir_to_target(&mir, &mut output, arch, os, 1)
                .expect_err("naked must be refused");
            let message = err.to_string();
            assert!(message.contains("@naked function 'isr'"), "{message}");
            assert!(message.contains(&format!("{arch:?}-{os:?}")), "{message}");
        }
    }
}

#[cfg(test)]
mod runtime_symbol_tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::{mir::codegen, parser};
    use std::borrow::Cow;

    const RUNTIME_PROGRAM: &str = r#"
    @extern
    fn @malloc(i64 %n) -> ptr
    @extern
    fn @free(ptr %p) -> void
    @extern
    fn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr
    @extern
    fn @memmove(ptr %d, ptr %s, i64 %n) -> ptr
    @extern
    fn @memset(ptr %d, i32 %v, i64 %n) -> ptr

    fn @main() -> i32 {
      entry:
        %p = call @malloc(8)
        %src = call @malloc(8)
        call @memcpy(%p, %src, 8)
        call @memmove(%p, %src, 8)
        call @memset(%p, 0, 8)
        call @free(%p)
        ret.i32 0
    }
    "#;

    fn lower(input: &str) -> Module {
        let ir = parser::parse_module(input).expect("parse");
        codegen::from_ir(&ir, "runtime_symbols").expect("lower")
    }

    fn emit_x86(module: &Module, target_os: TargetOperatingSystem) -> String {
        let mut out = Vec::new();
        generate_mir_to_target(module, &mut out, TargetArchitecture::X86_64, target_os, 1)
            .expect("emit");
        String::from_utf8(out).expect("utf8")
    }

    fn overrides_for(symbols: &[RuntimeSymbol]) -> HashMap<RuntimeSymbol, String> {
        symbols
            .iter()
            .map(|symbol| (*symbol, format!("k_{}", symbol.c_name())))
            .collect()
    }

    const MIR_SYMBOLS: [RuntimeSymbol; 5] = [
        RuntimeSymbol::Malloc,
        RuntimeSymbol::Free,
        RuntimeSymbol::Memcpy,
        RuntimeSymbol::Memmove,
        RuntimeSymbol::Memset,
    ];

    #[test]
    fn default_calls_keep_c_names_and_decoration() {
        let module = lower(RUNTIME_PROGRAM);
        let linux = emit_x86(&module, TargetOperatingSystem::Linux);
        for name in ["malloc", "free", "memcpy", "memmove", "memset"] {
            assert!(
                linux.contains(&format!("call {name}\n")),
                "{name}:\n{linux}"
            );
        }
        let macos = emit_x86(&module, TargetOperatingSystem::MacOS);
        for name in ["malloc", "free", "memcpy", "memmove", "memset"] {
            assert!(
                macos.contains(&format!("call _{name}\n")),
                "{name}:\n{macos}"
            );
        }
    }

    #[test]
    fn retarget_is_a_no_op_without_overrides() {
        let module = lower(RUNTIME_PROGRAM);
        let lib = BuiltinLibrary::new(TargetOperatingSystem::Linux, HashMap::new());
        assert!(matches!(
            retarget_runtime_calls(&module, &lib),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn overrides_reach_every_mir_call_site() {
        let module = lower(RUNTIME_PROGRAM);
        let lib = BuiltinLibrary::new(TargetOperatingSystem::Linux, overrides_for(&MIR_SYMBOLS));
        let retargeted = retarget_runtime_calls(&module, &lib);
        assert!(matches!(retargeted, Cow::Owned(_)));
        let asm = emit_x86(&retargeted, TargetOperatingSystem::Linux);
        for symbol in MIR_SYMBOLS {
            let call = format!("call k_{}\n", symbol.c_name());
            assert!(asm.contains(&call), "missing {call}:\n{asm}");
            assert!(
                !asm.contains(&format!("call {}\n", symbol.c_name())),
                "{asm}"
            );
        }
    }

    #[test]
    fn overrides_get_platform_decoration_like_any_c_symbol() {
        let module = lower(RUNTIME_PROGRAM);
        let lib = BuiltinLibrary::new(TargetOperatingSystem::MacOS, overrides_for(&MIR_SYMBOLS));
        let asm = emit_x86(
            &retarget_runtime_calls(&module, &lib),
            TargetOperatingSystem::MacOS,
        );
        assert!(asm.contains("call _k_malloc\n"), "{asm}");
        assert!(asm.contains("call _k_memset\n"), "{asm}");
    }
}
