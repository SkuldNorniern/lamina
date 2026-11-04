//! MIR-based code generation for multiple target architectures.

#[cfg(test)]
mod abi_claim_tests;
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
pub mod macho_unwind;
pub mod mips64;
pub mod ptx;
pub mod s390x;
pub mod settings;
mod simd;

pub mod arm;
pub mod powerpc;
pub mod riscv;
pub mod runtime_builtins;
pub mod wasm;
pub mod x86_64;

pub use capability::{CapabilitySet, CodegenCapability};
pub use limits::{MAX_MIR_CALL_PARAMETERS, validate_module_call_parameters};
pub use settings::{MirCodegenSettings, RegallocStrategy};

use std::{borrow::Cow, collections::HashMap, fmt, io::Write};

use crate::{
    error::LaminaError,
    mir::{Global, Instruction, MirType, Module, RelocationModel, Signature},
};
use lamina_codegen::{
    abi::{BuiltinLibrary, RuntimeSymbol},
    riscv::{RiscVTarget, Xlen},
};
use lamina_platform::{
    ArchVariant, CompilerRuntime, Target, TargetArchitecture, TargetEnvironment,
    TargetOperatingSystem, X86CpuLevel,
};

/// Generates assembly from MIR for the requested target.
pub fn generate_mir_to_target<W: Write>(
    module: &Module,
    writer: &mut W,
    target: &Target,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_to_target_with_settings(
        module,
        writer,
        target,
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

/// Refuses a compiler-runtime helper the module calls but nothing will define.
///
/// Which runtime a target gets is the environment's answer, not the operating system's:
/// [`CompilerRuntime::LaminaBuiltins`] means a definition arrives, and
/// [`CompilerRuntime::None`] means none does. An operating system that supplies its own
/// runtime leaves the environment axis nothing to choose and so reads as
/// [`TargetEnvironment::Freestanding`] without meaning it, which
/// [`TargetOperatingSystem::links_a_c_runtime`] distinguishes.
///
/// Reported at codegen time, naming the symbol, rather than at link time, because a
/// freestanding user often takes the object and links it themselves.
fn check_compiler_runtime_symbols(
    module: &Module,
    target_os: TargetOperatingSystem,
    environment: TargetEnvironment,
    builtins: &BuiltinLibrary,
) -> Result<(), LaminaError> {
    match environment.compiler_runtime() {
        CompilerRuntime::LaminaBuiltins => return Ok(()),
        CompilerRuntime::None => {}
    }
    if !target_os.links_a_c_runtime() {
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
                    "`{name}` is a compiler runtime helper that the `{}` environment does not provide; \
                 define it in the module, or name an implementation with LAMINA_BUILTINS={}=<symbol>",
                    environment.token(),
                    symbol.c_name()
                ),
            )));
        }
    }
    Ok(())
}

/// The RISC-V the backend emits for: the ISA the target named.
fn riscv_target(variant: ArchVariant) -> RiscVTarget {
    variant
        .riscv()
        .map_or_else(|| RiscVTarget::general(Xlen::Rv64), RiscVTarget::from_isa)
}

/// Backends that check a stated `sret` or `byval` against where they actually place
/// the value. Every other backend refuses the attribute by name, because honouring it
/// unchecked would turn a frontend's wrong classification into a wrong program.
const ABI_CLAIM_BACKENDS: [TargetArchitecture; 2] =
    [TargetArchitecture::X86_64, TargetArchitecture::Aarch64];

fn refuse_unverifiable_abi_claims(
    module: &Module,
    arch: TargetArchitecture,
    os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    if ABI_CLAIM_BACKENDS.contains(&arch) {
        return Ok(());
    }
    let refuse = |what: String| {
        Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "{what}, and the {arch}_{os} backend does not check stated argument placement; remove the attribute or build for x86_64 or aarch64"
            ),
        )))
    };
    for (name, function) in &module.functions {
        if function.sig.sret_claimed {
            return refuse(format!("@{name} states sret on its return"));
        }
        for param in &function.sig.params {
            if param.byval_claimed {
                let named = param
                    .name
                    .clone()
                    .unwrap_or_else(|| "a parameter".to_string());
                return refuse(format!("@{name} states byval on %{named}"));
            }
        }
    }
    Ok(())
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
    target: &Target,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    let target_arch = target.architecture;
    let target_os = target.operating_system;
    let pointer_width = target_arch.pointer_width();
    let chosen_variant = if settings.arch_variant == ArchVariant::None {
        target.variant
    } else {
        settings.arch_variant
    };
    let resolved_settings = settings
        .clone()
        .with_arch_variant(target_arch.spec().variants.accept(chosen_variant))
        .with_environment(settings.environment.unwrap_or(target.environment));
    let settings = &resolved_settings;
    let builtins = BuiltinLibrary::from_env(target_os)
        .map_err(|e| LaminaError::ValidationError(e.to_string()))?;
    let retargeted = retarget_runtime_calls(input, &builtins);
    let module: &Module = &retargeted;
    check_compiler_runtime_symbols(
        module,
        target_os,
        settings.environment.unwrap_or(target.environment),
        &builtins,
    )?;
    common::reject_invalid_cmpxchg_orderings(module)?;
    common::reject_underaligned_atomics(module, pointer_width)?;
    if module.relocation != RelocationModel::Static
        && !(target_arch == TargetArchitecture::X86_64 && target_os == TargetOperatingSystem::Linux)
    {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "@pic and @pie are only implemented for x86_64 Linux (ELF); target {target_arch:?}-{target_os:?} is not supported"
            ),
        )));
    }
    if target_arch != TargetArchitecture::X86_64
        && let Some(name) = module
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

    refuse_unverifiable_abi_claims(module, target_arch, target_os)?;

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
        TargetArchitecture::Riscv128 => return Err(not_implemented(target_arch, target_os)),
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
///
/// Internal: the backends in this crate are its only implementors. Nothing calls
/// through it today — the driver reaches each backend's free functions directly — so
/// it stands as the shape a backend is expected to have.
#[allow(dead_code)]
pub(crate) trait Codegen {
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
///
/// Non-exhaustive: a backend that learns to refuse something new adds a variant, and
/// that must not break a downstream `match`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodegenError {
    /// The target cannot express what the module asks for.
    UnsupportedFeature(String),
    /// The options handed to a backend do not describe a build it can run.
    InvalidCodegenOptions(String),
    /// The MIR reaching a backend did not verify.
    InvalidMir(Vec<String>),
    /// An x86-64 instruction the module needs is above the selected CPU level.
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
        codegen::from_ir_x86_64(&ir_module, "test_input").expect("Failed to lower to MIR")
    }

    #[test]
    fn test_cross_target_x86_64() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            &Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
            1,
        )
        .expect("Failed to generate x86_64 assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(
            asm.lines()
                .any(|line| line.trim_start().starts_with("addq %")),
            "x86_64 output should contain a register addq: {asm}"
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
            &Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
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
            &Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux),
            1,
        )
        .expect("Failed to generate AArch64 assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(
            asm.lines()
                .any(|line| line.trim_start().starts_with("add x")),
            "AArch64 output should contain an add x register instruction: {asm}"
        );
    }

    #[test]
    fn test_cross_target_riscv64() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            &Target::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux),
            1,
        )
        .expect("Failed to generate RISC-V assembly");

        let asm = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(
            asm.lines()
                .any(|line| line.trim_start().starts_with("add a")),
            "RISC-V output should contain an add a register instruction: {asm}"
        );
    }

    #[test]
    fn test_cross_target_wasm32() {
        let module = create_simple_add_function();
        let mut output = Vec::new();

        generate_mir_to_target(
            &module,
            &mut output,
            &Target::new(TargetArchitecture::Wasm32, TargetOperatingSystem::Linux),
            1,
        )
        .expect("Failed to generate WASM");

        let wat = String::from_utf8(output).expect("Invalid UTF-8");
        assert!(
            wat.contains("i64.add"),
            "WASM output should contain i64.add: {wat}"
        );
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
    use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};

    #[test]
    fn wasm_reports_atomics_as_a_clean_error() {
        let ir = parse_module(
            "fn @f(ptr %p) -> i64 {\nentry:\n %old = atomic_add.i64 %p, 1, seqcst\n ret.i64 %old\n}",
        )
        .expect("parse atomic add");
        let mir = codegen::from_ir_x86_64(&ir, "atomics").expect("lower atomic add");
        let mut output = Vec::new();
        let err = generate_mir_to_target(
            &mir,
            &mut output,
            &Target::new(TargetArchitecture::Wasm32, TargetOperatingSystem::Wasi),
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
        let mir = codegen::from_ir_x86_64(&ir, "naked").expect("lower naked");
        for (arch, os) in [
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Wasm32, TargetOperatingSystem::Wasi),
            (TargetArchitecture::I686, TargetOperatingSystem::Linux),
        ] {
            let mut output = Vec::new();
            let err = generate_mir_to_target(&mir, &mut output, &Target::new(arch, os), 1)
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
        codegen::from_ir_x86_64(&ir, "runtime_symbols").expect("lower")
    }

    fn emit_x86(module: &Module, target_os: TargetOperatingSystem) -> String {
        let mut out = Vec::new();
        generate_mir_to_target(
            module,
            &mut out,
            &Target::new(TargetArchitecture::X86_64, target_os),
            1,
        )
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

    const HELPER_PROGRAM: &str = r#"
    fn @divide(i128 %a, i128 %b) -> i128 {
      entry:
        %q = div.i128 %a, %b
        ret.i128 %q
    }
    "#;

    #[test]
    fn the_compiler_runtime_check_follows_the_environment_not_the_operating_system() {
        let module = lower(HELPER_PROGRAM);
        let lib = BuiltinLibrary::new(TargetOperatingSystem::Linux, HashMap::new());
        for environment in TargetEnvironment::ALL {
            let result = check_compiler_runtime_symbols(
                &module,
                TargetOperatingSystem::Linux,
                environment,
                &lib,
            );
            match environment.compiler_runtime() {
                CompilerRuntime::LaminaBuiltins => {
                    assert!(result.is_ok(), "{environment}: {result:?}");
                }
                CompilerRuntime::None => {
                    let message = result
                        .expect_err("a runtime with no helpers must refuse")
                        .to_string();
                    assert!(message.contains("__divti3"), "{message}");
                    assert!(message.contains(environment.token()), "{message}");
                }
            }
        }
    }

    #[test]
    fn an_operating_system_with_its_own_runtime_is_not_a_freestanding_environment() {
        let module = lower(HELPER_PROGRAM);
        let lib = BuiltinLibrary::new(TargetOperatingSystem::MacOS, HashMap::new());
        for target_os in [
            TargetOperatingSystem::MacOS,
            TargetOperatingSystem::Windows,
            TargetOperatingSystem::Wasi,
        ] {
            let result = check_compiler_runtime_symbols(
                &module,
                target_os,
                TargetEnvironment::Freestanding,
                &lib,
            );
            assert!(result.is_ok(), "{target_os:?}: {result:?}");
        }
    }

    #[test]
    fn a_freestanding_environment_on_a_hosted_operating_system_is_checked() {
        let module = lower(HELPER_PROGRAM);
        let hosted: Target = "x86_64_linux_gnu".parse().expect("parse hosted target");
        let freestanding: Target = "x86_64_linux_none".parse().expect("parse none target");
        let mut out = Vec::new();
        generate_mir_to_target(&module, &mut out, &hosted, 1).expect("gnu supplies the helper");
        let message = generate_mir_to_target(&module, &mut Vec::new(), &freestanding, 1)
            .expect_err("none supplies no helper")
            .to_string();
        assert!(message.contains("__divti3"), "{message}");
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
