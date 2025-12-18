//! Linking backends for creating final executables or binaries.
//!
//! Supports multiple linkers:
//! - GCC's ld (via gcc)
//! - Clang's ld (via clang)
//! - Mold linker (faster alternative)
//! - MSVC linker (on Windows)
//! - Custom linkers

use crate::error::LaminaError;
use crate::target::{TargetArchitecture, TargetOperatingSystem};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Linker backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkerBackend {
    /// GCC's linker (via gcc)
    Gcc,
    /// Clang's linker (via clang)
    Clang,
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
    // WASM doesn't need linking (already a binary)
    if matches!(target_arch, TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64) {
        return Err(LaminaError::ValidationError(
            "WASM targets do not require linking".to_string(),
        ));
    }

    let backend = backend.unwrap_or_else(|| detect_linker_backend());

    let (cmd, args) = match backend {
        LinkerBackend::Gcc => {
            let mut args = vec![];
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            ("gcc", args)
        }
        LinkerBackend::Clang => {
            let mut args = vec![];
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            ("clang", args)
        }
        LinkerBackend::Mold => {
            let mut args = vec![];
            args.push("-fuse-ld=mold".to_string());
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            // Use gcc/clang with mold
            let cmd = if Command::new("clang").arg("--version").output().is_ok() {
                "clang"
            } else {
                "gcc"
            };
            (cmd, args)
        }
        LinkerBackend::Msvc => {
            let mut args = vec!["/nologo".to_string()];
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push(format!("/Fe{}", output_path.display()));
            ("cl", args)
        }
        LinkerBackend::Custom(name) => {
            let mut args = vec![];
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push("-o".to_string());
            args.push(output_path.to_string_lossy().to_string());
            (name, args)
        }
    };

    if verbose {
        println!("[VERBOSE] Linking with {}: {:?}", cmd, args);
    }

    let output = Command::new(cmd)
        .args(&args)
        .output()
        .map_err(|e| {
            LaminaError::ValidationError(format!(
                "Failed to spawn linker '{}': {}",
                cmd, e
            ))
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
    if cfg!(windows) {
        // On Windows, try MSVC first
        if Command::new("cl").arg("/?").output().is_ok() {
            return LinkerBackend::Msvc;
        }
    }

    // Try mold (if available and enabled)
    #[cfg(feature = "use_mold")]
    {
        if Command::new("mold").arg("--version").output().is_ok() {
            return LinkerBackend::Mold;
        }
    }

    // Try clang (common on macOS and modern systems)
    if Command::new("clang").arg("--version").output().is_ok() {
        return LinkerBackend::Clang;
    }

    // Fallback to gcc
    if Command::new("gcc").arg("--version").output().is_ok() {
        return LinkerBackend::Gcc;
    }

    // Default to gcc (will fail later if not available)
    LinkerBackend::Gcc
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
                "" // No extension on Unix
            }
        }
    }
}

