//! Assembly/compilation backends for converting intermediate formats to binaries.
//!
//! Supports multiple assemblers:
//! - **Ras**: Drop-in replacement for as/gas; x86_64 and AArch64, ELF (library or `ras` binary).
//! - gas (GNU Assembler), clang's as
//! - WASM: wat2wasm

use crate::error::LaminaError;
use lamina_codegen::os_uses_macho;
use lamina_platform::{
    ArchVariant, OutputArtifact, RiscvIsa, Target, TargetArchitecture, TargetOperatingSystem,
};
use ras::{ObjectWriteOptions, Ras, is_assembly_target_supported, is_object_file_supported};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// Assembler backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblerBackend {
    /// Ras (drop-in replacement for as/gas); x86_64 and AArch64, ELF object output
    Ras,
    /// GNU Assembler (gas/as)
    Gas,
    /// Clang's integrated assembler
    Lld,
    /// wat2wasm for WebAssembly
    Wat2Wasm,
    /// Custom assembler (specified by name)
    Custom(&'static str),
}

/// Result of assembly operation
pub struct AssembleResult {
    /// Path to the output file (object file or WASM binary)
    pub output_path: PathBuf,
    /// Whether linking is needed after assembly
    pub needs_linking: bool,
}

/// Assemble intermediate format to binary
pub fn assemble(
    input_path: &Path,
    output_path: &Path,
    target: Target,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<AssembleResult, LaminaError> {
    assemble_with_ras_object_options(
        input_path,
        output_path,
        target,
        backend,
        additional_flags,
        verbose,
        ObjectWriteOptions::default(),
    )
}

/// Like [`assemble`], but passes [`ObjectWriteOptions`] when the backend is Ras (ignored for gas/lld/wat2wasm).
#[allow(clippy::too_many_arguments)]
pub fn assemble_with_ras_object_options(
    input_path: &Path,
    output_path: &Path,
    target: Target,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
    ras_object_write_options: ObjectWriteOptions,
) -> Result<AssembleResult, LaminaError> {
    let target_arch = target.architecture;
    let target_os = target.operating_system;
    if !input_path.exists() {
        return Err(LaminaError::ValidationError(format!(
            "Input file does not exist: {}",
            input_path.display()
        )));
    }

    match target_arch.output_artifact() {
        OutputArtifact::Text => {
            let expected_os = if target_arch == TargetArchitecture::Nvptx64 {
                TargetOperatingSystem::Cuda
            } else {
                TargetOperatingSystem::AmdHsa
            };
            if target_os != expected_os || backend.is_some() || !additional_flags.is_empty() {
                return Err(LaminaError::ValidationError(
                    "GPU text artifacts require the matching GPU OS and no assembler options"
                        .into(),
                ));
            }
            if input_path != output_path {
                fs::copy(input_path, output_path)?;
            }
            Ok(AssembleResult {
                output_path: output_path.to_path_buf(),
                needs_linking: false,
            })
        }
        OutputArtifact::Module => {
            assemble_wasm(input_path, output_path, backend, additional_flags, verbose)
        }
        OutputArtifact::Executable => assemble_native(
            input_path,
            output_path,
            target,
            backend,
            additional_flags,
            verbose,
            ras_object_write_options,
        ),
    }
}

/// Assemble WASM (WAT -> WASM binary)
fn assemble_wasm(
    input_path: &Path,
    output_path: &Path,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<AssembleResult, LaminaError> {
    let backend = backend.unwrap_or(AssemblerBackend::Wat2Wasm);

    match backend {
        AssemblerBackend::Wat2Wasm => {
            let mut cmd = Command::new("wat2wasm");
            cmd.arg(input_path);
            cmd.arg("-o");
            cmd.arg(output_path);
            cmd.args(additional_flags);
            let output = cmd.output().map_err(|e| {
                LaminaError::ValidationError(format!(
                    "Failed to spawn wat2wasm: {e}. Make sure wat2wasm is installed and in PATH."
                ))
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(LaminaError::ValidationError(format!(
                    "wat2wasm failed: {stderr}"
                )));
            }

            if verbose {
                println!(
                    "[VERBOSE] wat2wasm: {} -> {}",
                    input_path.display(),
                    output_path.display()
                );
            }

            Ok(AssembleResult {
                output_path: output_path.to_path_buf(),
                needs_linking: false, // WASM doesn't need linking
            })
        }
        _ => Err(LaminaError::ValidationError(format!(
            "Unsupported assembler backend for WASM: {backend:?}"
        ))),
    }
}

/// Assemble native assembly to object file
#[allow(clippy::too_many_arguments)]
fn assemble_native(
    input_path: &Path,
    output_path: &Path,
    target: Target,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
    ras_object_write_options: ObjectWriteOptions,
) -> Result<AssembleResult, LaminaError> {
    let target_arch = target.architecture;
    let target_os = target.operating_system;
    let defaulted = backend.is_none();
    let backend = backend.unwrap_or_else(|| detect_assembler_backend(target_arch, target_os));

    if backend == AssemblerBackend::Ras {
        let reason = match ras_gap(target_arch, target_os) {
            Some(gap) => gap,
            None => match ras_assemble(
                input_path,
                output_path,
                target_arch,
                target_os,
                ras_object_write_options,
                verbose,
            ) {
                Ok(result) => return Ok(result),
                Err(reason) => reason,
            },
        };
        if !defaulted {
            return Err(LaminaError::ValidationError(format!(
                "{reason}; select another toolchain with -c gnu or -c clang"
            )));
        }
        return assemble_with_gnu(
            input_path,
            output_path,
            target,
            additional_flags,
            verbose,
            &reason,
        );
    }

    let backend = validate_assembler_backend_for_target(backend, target_arch, target_os)?;

    let (cmd, args) = match backend {
        AssemblerBackend::Gas => {
            let tool = gnu_tool(target_arch, "as")
                .ok_or_else(|| LaminaError::ValidationError(no_gnu_tool("as", target_arch)))?;
            return run_gnu_as(
                &tool,
                input_path,
                output_path,
                target,
                additional_flags,
                verbose,
            );
        }
        AssemblerBackend::Lld => {
            let mut args = vec!["-c".to_string()];

            let target_triple = match (target_arch, target_os) {
                (TargetArchitecture::X86_64, TargetOperatingSystem::Linux) => {
                    "x86_64-unknown-linux-gnu"
                }
                (TargetArchitecture::X86_64, TargetOperatingSystem::MacOS) => "x86_64-apple-darwin",
                (TargetArchitecture::X86_64, TargetOperatingSystem::Windows) => {
                    "x86_64-pc-windows-msvc"
                }
                (TargetArchitecture::Aarch64, TargetOperatingSystem::Linux) => {
                    "aarch64-unknown-linux-gnu"
                }
                (TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS) => {
                    "aarch64-apple-darwin"
                }
                (TargetArchitecture::Riscv32, TargetOperatingSystem::Linux) => {
                    "riscv32-unknown-linux-gnu"
                }
                (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux) => {
                    "riscv64-unknown-linux-gnu"
                }
                (TargetArchitecture::LoongArch64, TargetOperatingSystem::Linux) => {
                    "loongarch64-unknown-linux-gnu"
                }
                (TargetArchitecture::S390x, TargetOperatingSystem::Linux) => {
                    "s390x-unknown-linux-gnu"
                }
                _ => {
                    return Err(LaminaError::ValidationError(format!(
                        "Unsupported target combination for clang assembler: {target_arch:?} {target_os:?}"
                    )));
                }
            };

            args.push("-target".to_string());
            args.push(target_triple.to_string());
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            ("clang".to_string(), args)
        }
        AssemblerBackend::Custom(name) => {
            let args = vec![
                input_path.to_string_lossy().to_string(),
                "-o".to_string(),
                output_path.to_string_lossy().to_string(),
            ];
            (name.to_string(), args)
        }
        AssemblerBackend::Ras => unreachable!("Ras handled above"),
        AssemblerBackend::Wat2Wasm => {
            return Err(LaminaError::ValidationError(
                "Wat2Wasm is for WASM targets only".to_string(),
            ));
        }
    };

    if verbose {
        println!("[VERBOSE] Assembling with {cmd}: {args:?}");
    }

    let output = Command::new(&cmd).args(&args).output().map_err(|e| {
        LaminaError::ValidationError(format!("Failed to spawn assembler '{cmd}': {e}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(LaminaError::ValidationError(format!(
            "Assembler '{cmd}' failed:\nstdout: {stdout}\nstderr: {stderr}"
        )));
    }

    Ok(AssembleResult {
        output_path: output_path.to_path_buf(),
        needs_linking: true, // Native object files need linking
    })
}

fn ras_assemble(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    ras_object_write_options: ObjectWriteOptions,
    verbose: bool,
) -> Result<AssembleResult, String> {
    if verbose {
        println!(
            "[VERBOSE] Assembling with ras (as/GAS alternative): {} -> {}",
            input_path.display(),
            output_path.display()
        );
    }
    let mut ras = Ras::with_object_write_options(target_arch, target_os, ras_object_write_options)
        .map_err(|e| format!("failed to create ras assembler: {e}"))?;
    ras.assemble_file(input_path, output_path)
        .map_err(|e| format!("ras cannot assemble this input: {e}"))?;
    Ok(AssembleResult {
        output_path: output_path.to_path_buf(),
        needs_linking: true,
    })
}

fn assemble_with_gnu(
    input_path: &Path,
    output_path: &Path,
    target: Target,
    additional_flags: &[String],
    verbose: bool,
    reason: &str,
) -> Result<AssembleResult, LaminaError> {
    let target_arch = target.architecture;
    let target_os = target.operating_system;
    if !targets_elf(target_os) {
        return Err(LaminaError::ValidationError(format!(
            "{reason}; GNU as cannot produce {target_os:?} objects, so select -c clang"
        )));
    }
    let tool = gnu_tool(target_arch, "as").ok_or_else(|| {
        LaminaError::ValidationError(format!("{reason}; {}", no_gnu_tool("as", target_arch)))
    })?;
    eprintln!("[WARN] {reason}; assembling with GNU {tool} instead");
    run_gnu_as(
        &tool,
        input_path,
        output_path,
        target,
        additional_flags,
        verbose,
    )
}

fn run_gnu_as(
    tool: &str,
    input_path: &Path,
    output_path: &Path,
    target: Target,
    additional_flags: &[String],
    verbose: bool,
) -> Result<AssembleResult, LaminaError> {
    let mut args = gas_arch_flags(target.architecture, target.operating_system, target.variant);
    args.push(input_path.to_string_lossy().to_string());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    args.extend(additional_flags.iter().cloned());

    if verbose {
        println!("[VERBOSE] Assembling with {tool}: {args:?}");
    }

    let output = Command::new(tool).args(&args).output().map_err(|e| {
        LaminaError::ValidationError(format!("Failed to spawn assembler '{tool}': {e}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(LaminaError::ValidationError(format!(
            "Assembler '{tool}' failed:\nstdout: {stdout}\nstderr: {stderr}"
        )));
    }

    Ok(AssembleResult {
        output_path: output_path.to_path_buf(),
        needs_linking: true,
    })
}

pub(crate) fn no_gnu_tool(tool: &str, target_arch: TargetArchitecture) -> String {
    match gnu_triple(target_arch) {
        Some(triple) => {
            format!("no GNU {tool} is installed for {target_arch:?} (looked for {triple}-{tool})")
        }
        None => format!("no GNU {tool} is installed for {target_arch:?}"),
    }
}

pub(crate) fn gnu_tool(target_arch: TargetArchitecture, tool: &str) -> Option<String> {
    let name = if host_matches(target_arch) {
        tool.to_string()
    } else {
        format!("{}-{tool}", gnu_triple(target_arch)?)
    };
    Command::new(&name)
        .arg("--version")
        .output()
        .ok()
        .map(|_| name)
}

fn gnu_triple(target_arch: TargetArchitecture) -> Option<&'static str> {
    match target_arch {
        TargetArchitecture::X86_64 => Some("x86_64-linux-gnu"),
        TargetArchitecture::Aarch64 => Some("aarch64-linux-gnu"),
        TargetArchitecture::Riscv32 => Some("riscv32-linux-gnu"),
        TargetArchitecture::Riscv64 => Some("riscv64-linux-gnu"),
        TargetArchitecture::LoongArch64 => Some("loongarch64-linux-gnu"),
        TargetArchitecture::S390x => Some("s390x-linux-gnu"),
        TargetArchitecture::PowerPC64 => Some("powerpc64le-linux-gnu"),
        TargetArchitecture::Armv7 => Some("arm-linux-gnueabihf"),
        TargetArchitecture::I686 => Some("i686-linux-gnu"),
        // The Cortex-M toolchain is `arm-none-eabi`, not a Linux GNU triple.
        TargetArchitecture::Thumbv6m
        | TargetArchitecture::Thumbv7m
        | TargetArchitecture::Arx64
        | TargetArchitecture::Arm32
        | TargetArchitecture::Wasm32
        | TargetArchitecture::Wasm64
        | TargetArchitecture::Nvptx64
        | TargetArchitecture::Amdgcn
        | TargetArchitecture::Unknown => None,
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => None,
    }
}

pub(crate) fn host_matches(target_arch: TargetArchitecture) -> bool {
    match target_arch {
        TargetArchitecture::X86_64 => cfg!(target_arch = "x86_64"),
        TargetArchitecture::Aarch64 => cfg!(target_arch = "aarch64"),
        TargetArchitecture::Riscv64 => cfg!(target_arch = "riscv64"),
        TargetArchitecture::LoongArch64 => cfg!(target_arch = "loongarch64"),
        TargetArchitecture::S390x => cfg!(target_arch = "s390x"),
        TargetArchitecture::PowerPC64 => cfg!(target_arch = "powerpc64"),
        TargetArchitecture::Armv7 => cfg!(target_arch = "arm"),
        // An x86_64 host assembles and links i686 with its own binutils, given
        // `--32` and `-m elf_i386`.
        TargetArchitecture::I686 => cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        TargetArchitecture::Thumbv6m
        | TargetArchitecture::Thumbv7m
        | TargetArchitecture::Arx64
        | TargetArchitecture::Arm32
        | TargetArchitecture::Riscv32
        | TargetArchitecture::Wasm32
        | TargetArchitecture::Wasm64
        | TargetArchitecture::Nvptx64
        | TargetArchitecture::Amdgcn
        | TargetArchitecture::Unknown => false,
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => false,
    }
}

pub(crate) fn targets_elf(target_os: TargetOperatingSystem) -> bool {
    !os_uses_macho(target_os) && target_os != TargetOperatingSystem::Windows
}

fn ras_gap(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Option<String> {
    if matches!(target_arch, TargetArchitecture::PowerPC64) {
        return Some("ras has no PowerPC encoder".to_string());
    }
    if !is_assembly_target_supported(target_arch, target_os)
        || !is_object_file_supported(target_arch, target_os)
    {
        return Some(format!("ras cannot assemble {target_arch:?} {target_os:?}"));
    }
    None
}

/// The assembler used when none is named: ras, linked in-process, so no host tool is needed.
pub fn detect_assembler_backend(
    _target_arch: TargetArchitecture,
    _target_os: TargetOperatingSystem,
) -> AssemblerBackend {
    AssemblerBackend::Ras
}

/// Get the appropriate file extension for assembly output
pub fn get_assembly_output_extension(target_arch: TargetArchitecture) -> &'static str {
    match target_arch {
        TargetArchitecture::Nvptx64 => "ptx",
        TargetArchitecture::Amdgcn => "s",
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => "wasm",
        _ => "o", // Object file for native targets
    }
}

/// Intermediate assembly/WAT file extension for the given target.
fn gas_arch_flags(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    variant: ArchVariant,
) -> Vec<String> {
    let mut args = vec![];
    match target_arch {
        TargetArchitecture::X86_64 => {
            if !os_uses_macho(target_os) {
                args.push("--64".to_string());
            }
        }
        TargetArchitecture::I686 => args.push("--32".to_string()),
        TargetArchitecture::Aarch64 => {
            // Mach-O assemblers take -arch instead of -march
            if os_uses_macho(target_os) {
                args.push("-arch".to_string());
                args.push("arm64".to_string());
            } else {
                let level = variant.aarch64().unwrap_or_default();
                args.push(format!("-march={}", level.march()));
            }
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            let isa = variant.riscv().unwrap_or_else(|| {
                RiscvIsa::general(target_arch.pointer_width().bytes() as u32 * 8)
            });
            // Compressed encodings cost nothing to allow in the assembler, and the ISA
            // string the toolchains expect for a bare riscv target carries them.
            args.push(format!("-march={}c", isa.march().trim_end_matches('c')));
        }
        _ => {}
    }
    args
}

pub fn get_intermediate_extension(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> &'static str {
    match target_arch {
        TargetArchitecture::Nvptx64 => "ptx",
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => "wat",
        _ => {
            if target_os == TargetOperatingSystem::Windows {
                "asm"
            } else {
                "s"
            }
        }
    }
}

fn validate_assembler_backend_for_target(
    backend: AssemblerBackend,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Result<AssemblerBackend, LaminaError> {
    match backend {
        AssemblerBackend::Wat2Wasm => {
            if matches!(
                target_arch,
                TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64
            ) {
                Ok(backend)
            } else {
                Err(LaminaError::ValidationError(
                    "wat2wasm assembler is only valid for WASM targets".to_string(),
                ))
            }
        }
        AssemblerBackend::Ras => match ras_gap(target_arch, target_os) {
            Some(gap) => Err(LaminaError::ValidationError(format!(
                "{gap}; select another toolchain with -c gnu or -c clang"
            ))),
            None => Ok(backend),
        },
        AssemblerBackend::Gas => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(
                    "GNU as/gas is not supported for Windows targets; use --assembler clang"
                        .to_string(),
                ));
            }
            Ok(backend)
        }
        AssemblerBackend::Lld | AssemblerBackend::Custom(_) => Ok(backend),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn intermediate_extension_windows_uses_asm() {
        assert_eq!(
            get_intermediate_extension(TargetArchitecture::X86_64, TargetOperatingSystem::Windows),
            "asm"
        );
    }

    #[test]
    fn intermediate_extension_linux_uses_s() {
        assert_eq!(
            get_intermediate_extension(TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
            "s"
        );
    }

    #[test]
    fn intermediate_extension_macos_uses_s() {
        assert_eq!(
            get_intermediate_extension(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS),
            "s"
        );
    }

    #[test]
    fn intermediate_extension_wasm_uses_wat() {
        assert_eq!(
            get_intermediate_extension(TargetArchitecture::Wasm32, TargetOperatingSystem::Linux),
            "wat"
        );
    }

    #[test]
    fn gas_rejected_for_windows_target() {
        let error = validate_assembler_backend_for_target(
            AssemblerBackend::Gas,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
        )
        .expect_err("gas should be rejected on Windows targets");
        assert!(error.to_string().contains("GNU as/gas"));
    }

    #[test]
    fn ras_gap_covers_powerpc_and_gnu_cannot_write_mach_o_or_coff() {
        assert!(ras_gap(TargetArchitecture::PowerPC64, TargetOperatingSystem::Linux).is_some());
        assert!(ras_gap(TargetArchitecture::X86_64, TargetOperatingSystem::Linux).is_none());
        assert!(targets_elf(TargetOperatingSystem::Linux));
        assert!(!targets_elf(TargetOperatingSystem::MacOS));
        assert!(!targets_elf(TargetOperatingSystem::Windows));
    }

    #[test]
    fn gas_flags_follow_mach_o_predicate() {
        for os in [
            TargetOperatingSystem::MacOS,
            TargetOperatingSystem::IOS,
            TargetOperatingSystem::TvOS,
            TargetOperatingSystem::WatchOS,
            TargetOperatingSystem::VisionOS,
        ] {
            assert!(
                gas_arch_flags(TargetArchitecture::X86_64, os, ArchVariant::None).is_empty(),
                "{os:?}"
            );
            assert_eq!(
                gas_arch_flags(
                    TargetArchitecture::Aarch64,
                    os,
                    TargetArchitecture::Aarch64.default_variant()
                ),
                ["-arch", "arm64"],
                "{os:?}"
            );
        }
        assert_eq!(
            gas_arch_flags(
                TargetArchitecture::X86_64,
                TargetOperatingSystem::Linux,
                ArchVariant::None
            ),
            ["--64"]
        );
        assert_eq!(
            gas_arch_flags(
                TargetArchitecture::Aarch64,
                TargetOperatingSystem::Linux,
                TargetArchitecture::Aarch64.default_variant()
            ),
            ["-march=armv8-a"]
        );
    }

    #[test]
    fn sub_architecture_reaches_the_assembler() {
        let flags = |name: &str| {
            let target: Target = name.parse().expect(name);
            gas_arch_flags(target.architecture, target.operating_system, target.variant)
        };
        assert_eq!(flags("riscv64_linux"), ["-march=rv64gc"]);
        assert_eq!(flags("riscv32_linux"), ["-march=rv32gc"]);
        assert_eq!(flags("rv64imac_linux"), ["-march=rv64imac"]);
        assert_eq!(flags("aarch64_linux"), ["-march=armv8-a"]);
        assert_eq!(flags("aarch64v8.2_linux"), ["-march=armv8.2-a"]);
    }
}
