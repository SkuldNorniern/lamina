//! Linking backends for creating final executables or binaries.
//!
//! Supports multiple linkers:
//! - **Weld**: Lamina's custom linker; drop-in replacement for ld/lld. Use `-c weld` when available.
//! - GNU ld (binutils)
//! - LLVM lld
//! - Mold (faster alternative)
//! - MSVC linker (on Windows)
//! - Custom linkers

use crate::error::LaminaError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

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

fn windows_sdk_root() -> Option<PathBuf> {
    env::var_os("WindowsSdkDir")
        .or_else(|| env::var_os("UniversalCRTSdkDir"))
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("ProgramFiles(x86)")
                .map(PathBuf::from)
                .map(|program_files_x86| program_files_x86.join("Windows Kits").join("10"))
        })
}

fn windows_sdk_version(sdk_root: &Path) -> Option<String> {
    env::var("WindowsSDKLibVersion")
        .or_else(|_| env::var("UCRTVersion"))
        .ok()
        .map(|version| version.trim_matches(['\\', '/']).to_string())
        .filter(|version| !version.is_empty())
        .or_else(|| {
            let mut versions = fs::read_dir(sdk_root.join("Lib"))
                .ok()?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    if entry.file_type().ok()?.is_dir() {
                        Some(entry.file_name().to_string_lossy().into_owned())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            versions.sort();
            versions.pop()
        })
}

fn msvc_library_arch(target_arch: TargetArchitecture) -> &'static str {
    match target_arch {
        TargetArchitecture::Aarch64 => "arm64",
        _ => "x64",
    }
}

fn build_msvc_library_paths(target_arch: TargetArchitecture) -> Vec<String> {
    let Some(sdk_root) = windows_sdk_root() else {
        return Vec::new();
    };
    let Some(sdk_version) = windows_sdk_version(&sdk_root) else {
        return Vec::new();
    };

    let library_arch = msvc_library_arch(target_arch);
    ["ucrt", "um"]
        .into_iter()
        .map(|library_family| {
            sdk_root
                .join("Lib")
                .join(&sdk_version)
                .join(library_family)
                .join(library_arch)
        })
        .filter(|library_path| library_path.is_dir())
        .map(|library_path| format!("/LIBPATH:{}", library_path.display()))
        .collect()
}

fn latest_child_directory(parent: &Path) -> Option<PathBuf> {
    let mut child_directories = fs::read_dir(parent)
        .ok()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    child_directories.sort();
    child_directories.pop()
}

fn visual_studio_tool_roots() -> Vec<PathBuf> {
    let mut tool_roots = Vec::new();

    if let Some(tool_root) = env::var_os("VCToolsInstallDir").map(PathBuf::from) {
        tool_roots.push(tool_root);
    }

    if let Some(vc_install_dir) = env::var_os("VCINSTALLDIR").map(PathBuf::from) {
        let msvc_root = vc_install_dir.join("Tools").join("MSVC");
        if let Some(tool_root) = latest_child_directory(&msvc_root) {
            tool_roots.push(tool_root);
        }
    }

    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];
    if let Some(program_files) = env::var_os("ProgramFiles").map(PathBuf::from) {
        for edition in editions {
            let msvc_root = program_files
                .join("Microsoft Visual Studio")
                .join("2022")
                .join(edition)
                .join("VC")
                .join("Tools")
                .join("MSVC");
            if let Some(tool_root) = latest_child_directory(&msvc_root) {
                tool_roots.push(tool_root);
            }
        }
    }

    if let Some(program_files_x86) = env::var_os("ProgramFiles(x86)").map(PathBuf::from) {
        for edition in editions {
            let msvc_root = program_files_x86
                .join("Microsoft Visual Studio")
                .join("2019")
                .join(edition)
                .join("VC")
                .join("Tools")
                .join("MSVC");
            if let Some(tool_root) = latest_child_directory(&msvc_root) {
                tool_roots.push(tool_root);
            }
        }
    }

    tool_roots
}

fn build_msvc_toolchain_library_paths(target_arch: TargetArchitecture) -> Vec<String> {
    let library_arch = msvc_library_arch(target_arch);
    visual_studio_tool_roots()
        .into_iter()
        .map(|tool_root| tool_root.join("lib").join(library_arch))
        .filter(|library_path| library_path.is_dir())
        .map(|library_path| format!("/LIBPATH:{}", library_path.display()))
        .collect()
}

fn build_arch_emulation_flags(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    let mut args = Vec::new();

    match target_arch {
        TargetArchitecture::X86_64 => {
            if target_os == TargetOperatingSystem::MacOS {
                args.push("-arch".to_string());
                args.push("x86_64".to_string());
            } else if target_os != TargetOperatingSystem::Windows {
                args.push("-m".to_string());
                args.push("elf_x86_64".to_string());
            }
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

/// Build args for weld as a freestanding, no-libc linker. weld synthesizes its
/// own `_start` and emits a static executable, so no C runtime objects, libc,
/// or dynamic linker are passed. This is the toolchain path that needs no
/// external assembler or linker.
fn build_weld_linker_args(
    input_path: &Path,
    output_path: &Path,
    target_os: TargetOperatingSystem,
    additional_flags: &[String],
) -> Vec<String> {
    let mut args = Vec::new();
    args.push(input_path.to_string_lossy().to_string());
    match target_os {
        TargetOperatingSystem::Windows => {
            // weld detects PE from .exe output suffix; no entry-point arg needed.
        }
        TargetOperatingSystem::MacOS => {
            // macOS: link against libSystem (contains printf, fflush, etc.)
            args.push("-lSystem".to_string());
        }
        _ => {
            // Linux/BSD: link against libc for printf, fflush, etc.
            args.push("-lc".to_string());
        }
    }
    args.extend(additional_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    args
}

fn clang_windows_target_triple(target_arch: TargetArchitecture) -> Option<&'static str> {
    match target_arch {
        TargetArchitecture::X86_64 => Some("x86_64-pc-windows-msvc"),
        TargetArchitecture::Aarch64 => Some("aarch64-pc-windows-msvc"),
        _ => None,
    }
}

fn build_clang_windows_linker_args(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    additional_flags: &[String],
) -> Result<Vec<String>, LaminaError> {
    let Some(target_triple) = clang_windows_target_triple(target_arch) else {
        return Err(LaminaError::ValidationError(format!(
            "Unsupported Windows target architecture for clang linker: {:?}",
            target_arch
        )));
    };

    let mut args = Vec::new();
    args.push("-target".to_string());
    args.push(target_triple.to_string());
    args.push(input_path.to_string_lossy().to_string());
    args.extend(additional_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    Ok(args)
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
    /// Weld (Lamina's custom linker; drop-in replacement for ld/lld)
    Weld,
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

    let backend = backend
        .unwrap_or_else(|| detect_linker_backend(target_arch, target_os));
    let backend = validate_linker_backend_for_target(backend, target_arch, target_os)?;

    let (cmd, args) = match backend {
        LinkerBackend::Ld => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(
                    "GNU ld is not supported for Windows targets; use -c clang or -c msvc"
                        .to_string(),
                ));
            }
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
            if target_os == TargetOperatingSystem::Windows {
                let args = build_clang_windows_linker_args(
                    input_path,
                    output_path,
                    target_arch,
                    additional_flags,
                )?;
                return run_linker("clang", args, verbose);
            }

            let mut args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            if target_os == TargetOperatingSystem::MacOS
                && let Ok(out) = Command::new("xcrun").args(["--show-sdk-version"]).output()
                && let Ok(sdk) = String::from_utf8(out.stdout)
            {
                let sdk_ver = sdk.trim();
                if !sdk_ver.is_empty() {
                    args.insert(0, sdk_ver.to_string());
                    args.insert(0, "10.15".to_string());
                    args.insert(0, "macos".to_string());
                    args.insert(0, "-platform_version".to_string());
                }
            }
            let cmd = if target_os == TargetOperatingSystem::MacOS {
                "ld64.lld"
            } else {
                "lld"
            };
            (cmd, args)
        }
        LinkerBackend::Mold => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(
                    "mold is not supported for Windows targets; use -c clang or -c msvc".to_string(),
                ));
            }
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            ("mold", args)
        }
        LinkerBackend::Weld => {
            let args =
                build_weld_linker_args(input_path, output_path, target_os, additional_flags);
            return run_linker(&resolve_sibling_command("weld"), args, verbose);
        }
        LinkerBackend::Msvc => {
            if target_os != TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(format!(
                    "MSVC link is only supported for Windows targets, not {:?}",
                    target_os
                )));
            }
            let mut args = vec!["/nologo".to_string()];
            args.push("/subsystem:console".to_string());
            args.push("/entry:main".to_string());
            args.extend(build_msvc_library_paths(target_arch));
            args.extend(build_msvc_toolchain_library_paths(target_arch));
            args.extend(additional_flags.iter().cloned());
            args.push(input_path.to_string_lossy().to_string());
            args.push(format!("/OUT:{}", output_path.display()));
            args.push("ucrt.lib".to_string());
            args.push("legacy_stdio_definitions.lib".to_string());
            args.push("kernel32.lib".to_string());
            ("link", args)
        }
        LinkerBackend::Custom(name) => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(format!(
                    "Custom linker '{}' is not supported for Windows targets; use -c clang or -c msvc",
                    name
                )));
            }
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

/// Prefer a binary sitting next to the current executable (the bundled lamina
/// toolchain), falling back to the bare name resolved via PATH.
fn resolve_sibling_command(name: &str) -> String {
    if let Ok(current_exe) = env::current_exe()
        && let Some(dir) = current_exe.parent()
    {
        let suffix = if cfg!(windows) { ".exe" } else { "" };
        let candidate = dir.join(format!("{}{}", name, suffix));
        if candidate.is_file() {
            return candidate.to_string_lossy().to_string();
        }
    }
    name.to_string()
}

fn run_linker(cmd: &str, args: Vec<String>, verbose: bool) -> Result<(), LaminaError> {
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

/// Detect available linker backend for the given target.
/// Tool availability is checked on the host; routing uses target OS/arch.
pub fn detect_linker_backend(
    _target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> LinkerBackend {
    // Check for weld in the same directory as the current executable (bundled toolchain).
    let weld_cmd = resolve_sibling_command("weld");
    let has_weld = std::path::Path::new(&weld_cmd).is_file()
        || Command::new(&weld_cmd).arg("--version").output().is_ok();

    match target_os {
        TargetOperatingSystem::Windows => {
            if Command::new("clang").arg("--version").output().is_ok() {
                return LinkerBackend::Lld;
            }
            if Command::new("link").arg("/?").output().is_ok() {
                return LinkerBackend::Msvc;
            }
            // Weld can link Windows COFF objects to PE executables.
            if has_weld {
                return LinkerBackend::Weld;
            }
            LinkerBackend::Msvc // Last resort — will fail gracefully if not found
        }
        TargetOperatingSystem::MacOS => {
            if Command::new("mold").arg("--version").output().is_ok() {
                return LinkerBackend::Mold;
            }
            if Command::new("ld").arg("-v").output().is_ok()
                || Command::new("ld").arg("--version").output().is_ok()
            {
                return LinkerBackend::Ld;
            }
            if Command::new("ld64.lld").arg("-v").output().is_ok() {
                return LinkerBackend::Lld;
            }
            if has_weld {
                return LinkerBackend::Weld;
            }
            LinkerBackend::Ld
        }
        _ => {
            if Command::new("mold").arg("--version").output().is_ok() {
                return LinkerBackend::Mold;
            }
            if Command::new("ld.lld").arg("-v").output().is_ok()
                || Command::new("lld").arg("-v").output().is_ok()
            {
                return LinkerBackend::Lld;
            }
            if Command::new("ld").arg("--version").output().is_ok()
                || Command::new("ld").arg("-v").output().is_ok()
            {
                return LinkerBackend::Ld;
            }
            if has_weld {
                return LinkerBackend::Weld;
            }
            LinkerBackend::Ld
        }
    }
}

fn validate_linker_backend_for_target(
    backend: LinkerBackend,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Result<LinkerBackend, LaminaError> {
    match (backend, target_os) {
        (LinkerBackend::Msvc, TargetOperatingSystem::Windows) => Ok(backend),
        (LinkerBackend::Msvc, _) => Err(LaminaError::ValidationError(format!(
            "MSVC link is only supported for Windows targets, not {:?}",
            target_os
        ))),
        (LinkerBackend::Weld, TargetOperatingSystem::Windows) => Ok(backend),
        (
            LinkerBackend::Ld | LinkerBackend::Mold,
            TargetOperatingSystem::Windows,
        ) => Err(LaminaError::ValidationError(format!(
            "Linker backend {:?} is not supported for Windows targets; use -c clang or -c msvc",
            backend
        ))),
        (LinkerBackend::Custom(_), TargetOperatingSystem::Windows) => {
            Err(LaminaError::ValidationError(
                "Custom linkers are not supported for Windows targets; use -c clang or -c msvc"
                    .to_string(),
            ))
        }
        (LinkerBackend::Lld, TargetOperatingSystem::Windows) => {
            if clang_windows_target_triple(target_arch).is_some() {
                Ok(backend)
            } else {
                Err(LaminaError::ValidationError(format!(
                    "clang Windows linker does not support {:?}",
                    target_arch
                )))
            }
        }
        _ => Ok(backend),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_x86_64_link_args_use_arch_not_elf() {
        let args = build_arch_emulation_flags(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS,
        );
        assert!(args.contains(&"-arch".to_string()));
        assert!(args.contains(&"x86_64".to_string()));
        assert!(!args.contains(&"elf_x86_64".to_string()));
    }

    #[test]
    fn linux_x86_64_link_args_use_elf_emulation() {
        let args = build_arch_emulation_flags(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
        );
        assert_eq!(args, vec!["-m".to_string(), "elf_x86_64".to_string()]);
    }

    #[test]
    fn windows_x86_64_has_no_unix_emulation_flags() {
        let args = build_arch_emulation_flags(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
        );
        assert!(args.is_empty());
    }

    #[test]
    fn clang_windows_target_triple_x86_64() {
        assert_eq!(
            clang_windows_target_triple(TargetArchitecture::X86_64),
            Some("x86_64-pc-windows-msvc")
        );
    }

    #[test]
    fn build_clang_windows_linker_args_includes_target_and_output() {
        let input = Path::new("module.o");
        let output = Path::new("module.exe");
        let args = build_clang_windows_linker_args(
            input,
            output,
            TargetArchitecture::X86_64,
            &[],
        )
        .expect("x86_64 Windows should be supported");

        assert_eq!(args[0], "-target");
        assert_eq!(args[1], "x86_64-pc-windows-msvc");
        assert!(args.contains(&input.to_string_lossy().to_string()));
        assert!(args.contains(&"-o".to_string()));
        assert!(args.contains(&output.to_string_lossy().to_string()));
    }

    #[test]
    fn msvc_rejected_for_linux_target() {
        let error = validate_linker_backend_for_target(
            LinkerBackend::Msvc,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
        )
        .expect_err("MSVC should be rejected for Linux targets");
        assert!(error.to_string().contains("MSVC link"));
    }

    #[test]
    fn weld_accepted_for_windows_target() {
        // weld now supports Windows PE linking (COFF→PE with import table).
        validate_linker_backend_for_target(
            LinkerBackend::Weld,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
        )
        .expect("weld should be accepted for Windows targets");
    }

    #[test]
    fn weld_args_include_libc_on_linux() {
        let input = Path::new("module.o");
        let output = Path::new("module");
        let args = build_weld_linker_args(input, output, TargetOperatingSystem::Linux, &[]);
        // weld links against libc for printf/fflush access.
        assert!(args.iter().any(|a| a == "-lc"), "expected -lc in args: {args:?}");
        assert!(args.contains(&"-o".to_string()));
        // No CRT objects or dynamic-linker args — weld handles those itself.
        assert!(!args.iter().any(|a| a == "--dynamic-linker"));
        assert!(!args.iter().any(|a| a.contains("crt1.o")));
    }

    #[test]
    fn lld_accepted_for_windows_x86_64() {
        validate_linker_backend_for_target(
            LinkerBackend::Lld,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
        )
        .expect("clang lld should be valid for Windows x86_64");
    }
}
