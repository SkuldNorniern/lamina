//! Assembly/compilation backends for converting intermediate formats to binaries.
//!
//! Supports multiple assemblers:
//! - Native: gas (GNU Assembler), clang's as, custom assemblers
//! - WASM: wat2wasm

use crate::error::LaminaError;
use crate::target::{TargetArchitecture, TargetOperatingSystem};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Assembler backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblerBackend {
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
                    "Failed to spawn wat2wasm: {}. Make sure wat2wasm is installed and in PATH.",
                    e
                ))
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(LaminaError::ValidationError(format!(
                    "wat2wasm failed: {}",
                    stderr
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
            "Unsupported assembler backend for WASM: {:?}",
            backend
        ))),
    }
}

/// Assemble native assembly to object file
fn assemble_native(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    backend: Option<AssemblerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<AssembleResult, LaminaError> {
    let backend = backend.unwrap_or_else(|| detect_assembler_backend());

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
                        "Unsupported target combination for clang assembler: {:?} {:?}",
                        target_arch, target_os
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
        _ => {
            return Err(LaminaError::ValidationError(format!(
                "Unsupported assembler backend for native targets: {:?}",
                backend
            )));
        }
    };

    if verbose {
        println!("[VERBOSE] Assembling with {}: {:?}", cmd, args);
    }

    let output = Command::new(cmd).args(&args).output().map_err(|e| {
        LaminaError::ValidationError(format!("Failed to spawn assembler '{}': {}", cmd, e))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(LaminaError::ValidationError(format!(
            "Assembler '{}' failed:\nstdout: {}\nstderr: {}",
            cmd, stdout, stderr
        )));
    }

    Ok(AssembleResult {
        output_path: output_path.to_path_buf(),
        needs_linking: true, // Native object files need linking
    })
}

/// Detect available assembler backend
fn detect_assembler_backend() -> AssemblerBackend {
    if Command::new("as").arg("--version").output().is_ok() {
        return AssemblerBackend::Gas;
    }

    if Command::new("clang").arg("--version").output().is_ok() {
        return AssemblerBackend::Lld;
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

/// Get the appropriate file extension for intermediate format (assembly/WAT)
pub fn get_intermediate_extension(target_arch: TargetArchitecture) -> &'static str {
    match target_arch {
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => "wat",
        _ => {
            // Assembly file extension
            if cfg!(windows) { "asm" } else { "s" }
        }
    }
}
