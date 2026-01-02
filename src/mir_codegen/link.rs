//! Linking backends for creating final executables or binaries.
//!
//! Supports multiple linkers:
//! - GCC's ld (via gcc)
//! - Clang's ld (via clang)
//! - Mold linker (faster alternative)
//! - MSVC linker (on Windows)
//! - Custom linkers

use crate::error::LaminaError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::path::Path;
use std::process::Command;

fn detect_compiler() -> Option<&'static str> {
    if Command::new("gcc").arg("--version").output().is_ok() {
        Some("gcc")
    } else if Command::new("clang").arg("--version").output().is_ok() {
        Some("clang")
    } else {
        None
    }
}

fn get_crt_file(compiler: &str, crt_name: &str) -> Option<String> {
    Command::new(compiler)
        .arg(format!("--print-file-name={}", crt_name))
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty() && path != crt_name && path.contains('/'))
}

fn get_crt_files(target_os: TargetOperatingSystem) -> Vec<String> {
    if target_os == TargetOperatingSystem::MacOS {
        return Vec::new();
    }

    let Some(compiler) = detect_compiler() else {
        return Vec::new();
    };

    ["crt1.o", "crti.o"]
        .iter()
        .filter_map(|name| get_crt_file(compiler, name))
        .collect()
}

fn get_crtn_files(target_os: TargetOperatingSystem) -> Vec<String> {
    if target_os == TargetOperatingSystem::MacOS {
        return Vec::new();
    }

    let Some(compiler) = detect_compiler() else {
        return Vec::new();
    };

    get_crt_file(compiler, "crtn.o").into_iter().collect()
}

fn build_arch_emulation_flags(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    let mut args = Vec::new();

    match target_arch {
        TargetArchitecture::X86_64 => {
            args.push("-m".to_string());
            args.push("elf_x86_64".to_string());
        }
        TargetArchitecture::Aarch64 => {
            if target_os == TargetOperatingSystem::MacOS {
                args.push("-arch".to_string());
                args.push("arm64".to_string());
            } else {
                args.push("-m".to_string());
                args.push("aarch64linux".to_string());
            }
        }
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            args.push("-m".to_string());
            args.push(format!(
                "elf{}lriscv",
                if matches!(target_arch, TargetArchitecture::Riscv64) {
                    "64"
                } else {
                    "32"
                }
            ));
        }
        _ => {}
    }

    args
}

fn build_crt_args(target_os: TargetOperatingSystem, before_user_object: bool) -> Vec<String> {
    if target_os == TargetOperatingSystem::MacOS {
        return Vec::new();
    }

    if before_user_object {
        get_crt_files(target_os)
    } else {
        get_crtn_files(target_os)
    }
}

fn build_entry_point_arg(target_os: TargetOperatingSystem) -> Vec<String> {
    match target_os {
        TargetOperatingSystem::MacOS => {
            vec!["-e".to_string(), "_main".to_string()]
        }
        _ => {
            vec!["-e".to_string(), "_start".to_string()]
        }
    }
}

fn build_dynamic_linker_arg(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    if target_os != TargetOperatingSystem::Linux {
        return Vec::new();
    }

    let interpreter = match target_arch {
        TargetArchitecture::X86_64 => "/lib64/ld-linux-x86-64.so.2",
        TargetArchitecture::Aarch64 => "/lib/ld-linux-aarch64.so.1",
        TargetArchitecture::Riscv32 => "/lib/ld-linux-riscv32-ilp32d.so.1",
        TargetArchitecture::Riscv64 => "/lib/ld-linux-riscv64-lp64d.so.1",
        _ => return Vec::new(),
    };

    vec!["--dynamic-linker".to_string(), interpreter.to_string()]
}

fn build_library_args(target_os: TargetOperatingSystem) -> Vec<String> {
    let mut args = Vec::new();

    match target_os {
        TargetOperatingSystem::MacOS => {
            if let Ok(output) = Command::new("xcrun").args(["--show-sdk-path"]).output()
                && let Ok(sdk_path) = String::from_utf8(output.stdout)
            {
                let sdk_path = sdk_path.trim();
                if !sdk_path.is_empty() {
                    args.push("-syslibroot".to_string());
                    args.push(sdk_path.to_string());
                }
            }
            args.push("-lSystem".to_string());
        }
        _ => {
            args.push("-lc".to_string());
        }
    }

    args
}

fn build_unix_linker_args(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    additional_flags: &[String],
) -> Vec<String> {
    let mut args = Vec::new();

    args.extend(build_arch_emulation_flags(target_arch, target_os));
    args.extend(build_dynamic_linker_arg(target_arch, target_os));
    args.extend(build_crt_args(target_os, true));
    args.push(input_path.to_string_lossy().to_string());
    args.extend(build_crt_args(target_os, false));
    args.extend(build_entry_point_arg(target_os));
    args.extend(build_library_args(target_os));
    args.extend(additional_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());

    args
}

/// Linker backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkerBackend {
    /// GNU ld (binutils)
    Ld,
    /// LLVM's lld linker
    Lld,
    /// Mold linker (fast alternative)
    Mold,
    /// MSVC linker (Windows)
    Msvc,
    /// Custom linker (specified by name)
    Custom(&'static str),
}

/// Link object files to create final executable
pub fn link(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    backend: Option<LinkerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<(), LaminaError> {
    if matches!(
        target_arch,
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64
    ) {
        return Err(LaminaError::ValidationError(
            "WASM targets do not require linking".to_string(),
        ));
    }

    if !input_path.exists() {
        return Err(LaminaError::ValidationError(format!(
            "Input file does not exist: {}",
            input_path.display()
        )));
    }

    let backend = backend.unwrap_or_else(detect_linker_backend);

    let (cmd, args) = match backend {
        LinkerBackend::Ld => {
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            ("ld", args)
        }
        LinkerBackend::Lld => {
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            ("lld", args)
        }
        LinkerBackend::Mold => {
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            ("mold", args)
        }
        LinkerBackend::Msvc => {
            let mut args = vec!["/nologo".to_string()];
            args.push("/subsystem:console".to_string());
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push(format!("/Fe{}", output_path.display()));
            args.push("/defaultlib:msvcrt".to_string());
            ("link", args)
        }
        LinkerBackend::Custom(name) => {
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            (name, args)
        }
    };

    if verbose {
        println!("[VERBOSE] Linking with {}: {:?}", cmd, args);
    }

    let output = Command::new(cmd).args(&args).output().map_err(|e| {
        LaminaError::ValidationError(format!("Failed to spawn linker '{}': {}", cmd, e))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(LaminaError::ValidationError(format!(
            "Linker '{}' failed:\nstdout: {}\nstderr: {}",
            cmd, stdout, stderr
        )));
    }

    Ok(())
}

/// Detect available linker backend
pub fn detect_linker_backend() -> LinkerBackend {
    if cfg!(windows) && Command::new("link").arg("/?").output().is_ok() {
        return LinkerBackend::Msvc;
    }

    if Command::new("mold").arg("--version").output().is_ok() {
        return LinkerBackend::Mold;
    }

    if Command::new("lld").arg("-v").output().is_ok()
        || Command::new("ld.lld").arg("-v").output().is_ok()
    {
        return LinkerBackend::Lld;
    }

    if Command::new("ld").arg("--version").output().is_ok()
        || Command::new("ld").arg("-v").output().is_ok()
    {
        return LinkerBackend::Ld;
    }

    LinkerBackend::Ld
}

/// Get the appropriate file extension for final output
pub fn get_output_extension(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> &'static str {
    match target_arch {
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => "wasm",
        _ => {
            if target_os == TargetOperatingSystem::Windows {
                "exe"
            } else {
                ""
            }
        }
    }
}
