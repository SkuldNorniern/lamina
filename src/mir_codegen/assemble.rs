//! Assembly/compilation backends for converting intermediate formats to binaries.
//!
//! Supports multiple assemblers:
//! - **Ras**: Drop-in replacement for as/gas; x86_64 and AArch64, ELF (library or `ras` binary).
//! - gas (GNU Assembler), clang's as
//! - WASM: wat2wasm

use crate::error::LaminaError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<AssembleResult, LaminaError> {
    assemble_with_ras_object_options(
        input_path,
        output_path,
        target_arch,
        target_os,
        backend,
        additional_flags,
        verbose,
        ras::ObjectWriteOptions::default(),
    )
}

/// Like [`assemble`], but passes [`ras::ObjectWriteOptions`] when the backend is Ras (ignored for gas/lld/wat2wasm).
#[allow(clippy::too_many_arguments)]
pub fn assemble_with_ras_object_options(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
    ras_object_write_options: ras::ObjectWriteOptions,
) -> Result<AssembleResult, LaminaError> {
    if !input_path.exists() {
        return Err(LaminaError::ValidationError(format!(
            "Input file does not exist: {}",
            input_path.display()
        )));
    }

    match target_arch {
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
            assemble_wasm(input_path, output_path, backend, additional_flags, verbose)
        }
        _ => assemble_native(
            input_path,
            output_path,
            target_arch,
            target_os,
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
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
    ras_object_write_options: ras::ObjectWriteOptions,
) -> Result<AssembleResult, LaminaError> {
    let backend = backend.unwrap_or_else(|| detect_assembler_backend(target_arch, target_os));
    let backend = validate_assembler_backend_for_target(backend, target_arch, target_os)?;

    if backend == AssemblerBackend::Ras {
        if verbose {
            println!(
                "[VERBOSE] Assembling with ras (as/GAS alternative): {} -> {}",
                input_path.display(),
                output_path.display()
            );
        }
        let mut ras =
            ras::Ras::with_object_write_options(target_arch, target_os, ras_object_write_options)
                .map_err(|e| {
                LaminaError::ValidationError(format!("Failed to create ras assembler: {e}"))
            })?;
        ras.assemble_file(input_path, output_path)
            .map_err(|e| LaminaError::ValidationError(format!("ras assembly failed: {e}")))?;
        return Ok(AssembleResult {
            output_path: output_path.to_path_buf(),
            needs_linking: true,
        });
    }

    let (cmd, args) = match backend {
        AssemblerBackend::Gas => {
            let mut args = vec![];
            // Set architecture-specific flags for gas/as
            match target_arch {
                TargetArchitecture::X86_64 => {
                    if target_os != TargetOperatingSystem::MacOS {
                        args.push("--64".to_string());
                    }
                }
                TargetArchitecture::Aarch64 => {
                    // macOS as doesn't support -march, it auto-detects from -arch
                    if target_os == TargetOperatingSystem::MacOS {
                        args.push("-arch".to_string());
                        args.push("arm64".to_string());
                    } else {
                        args.push("-march=aarch64".to_string());
                    }
                }
                TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
                    args.push(format!(
                        "-march=rv{}",
                        if matches!(target_arch, TargetArchitecture::Riscv64) {
                            "64"
                        } else {
                            "32"
                        }
                    ));
                }
                _ => {}
            }
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            args.extend(additional_flags.iter().cloned());
            ("as", args)
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
            ("clang", args)
        }
        AssemblerBackend::Custom(name) => {
            let args = vec![
                input_path.to_string_lossy().to_string(),
                "-o".to_string(),
                output_path.to_string_lossy().to_string(),
            ];
            (name, args)
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

    let output = Command::new(cmd).args(&args).output().map_err(|e| {
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

/// Detect available assembler backend for the given target.
///
/// Prefers a system assembler (gas/clang) when one is installed, because ras
/// cannot yet assemble programs that reference data symbols or external calls
/// (e.g. `print`/printf). ras is used only as a fallback when no system
/// assembler is present; select it explicitly with `--assembler ras` or
/// `-c lamina` for the standalone, no-toolchain path.
pub fn detect_assembler_backend(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> AssemblerBackend {
    if target_os == TargetOperatingSystem::Windows
        && Command::new("clang").arg("--version").output().is_ok()
    {
        return AssemblerBackend::Lld;
    }

    if Command::new("as").arg("--version").output().is_ok() {
        return AssemblerBackend::Gas;
    }

    if Command::new("clang").arg("--version").output().is_ok() {
        return AssemblerBackend::Lld;
    }

    if matches!(
        target_arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64 | TargetArchitecture::Arx64
    ) && ras::is_object_file_supported(target_arch, target_os)
    {
        return AssemblerBackend::Ras;
    }

    AssemblerBackend::Gas
}

/// Get the appropriate file extension for assembly output
pub fn get_assembly_output_extension(target_arch: TargetArchitecture) -> &'static str {
    match target_arch {
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => "wasm",
        _ => "o", // Object file for native targets
    }
}

/// Intermediate assembly/WAT file extension for the given target.
pub fn get_intermediate_extension(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> &'static str {
    match target_arch {
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
        AssemblerBackend::Ras => {
            if !matches!(
                target_arch,
                TargetArchitecture::X86_64
                    | TargetArchitecture::Aarch64
                    | TargetArchitecture::Arx64
            ) {
                return Err(LaminaError::ValidationError(format!(
                    "Ras assembler supports x86_64, AArch64, and ARX64 only, not {target_arch:?}"
                )));
            }
            if !ras::is_object_file_supported(target_arch, target_os) {
                return Err(LaminaError::ValidationError(format!(
                    "Ras assembler does not support object output for {target_arch:?} {target_os:?}"
                )));
            }
            Ok(backend)
        }
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
}
