//! Linking backends for creating final executables or binaries.
//!
//! Supports multiple linkers:
//! - **Weld**: Lamina's own linker, run as a sibling binary. The default.
//! - GNU ld (binutils)
//! - LLVM lld
//! - Mold (faster alternative)
//! - MSVC linker (on Windows)
//! - Custom linkers

use crate::error::LaminaError;
use crate::mir_codegen::assemble::{gnu_tool, host_matches, no_gnu_tool, targets_elf};
use lamina_codegen::os_uses_macho;
use lamina_platform::{OutputArtifact, TargetArchitecture, TargetOperatingSystem};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

fn detect_compiler() -> Option<&'static str> {
    if Command::new("gcc").arg("--version").output().is_ok() {
        Some("gcc")
    } else if Command::new("clang").arg("--version").output().is_ok() {
        Some("clang")
    } else {
        None
    }
}

/// The flag that makes a 64-bit host compiler answer for the 32-bit multilib, so a
/// query for a C runtime object or the compiler runtime finds the i686 copy.
fn multilib_flag(target_arch: TargetArchitecture) -> Option<&'static str> {
    (target_arch == TargetArchitecture::I686).then_some("-m32")
}

fn get_crt_file(compiler: &str, crt_name: &str, multilib: Option<&str>) -> Option<String> {
    Command::new(compiler)
        .args(multilib)
        .arg(format!("--print-file-name={crt_name}"))
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty() && path != crt_name && path.contains('/'))
}

fn get_crt_files(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Vec<String> {
    if os_uses_macho(target_os) {
        return Vec::new();
    }

    let Some(compiler) = detect_compiler() else {
        return Vec::new();
    };

    let multilib = multilib_flag(target_arch);
    ["crt1.o", "crti.o"]
        .iter()
        .filter_map(|name| get_crt_file(compiler, name, multilib))
        .collect()
}

fn get_crtn_files(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    if os_uses_macho(target_os) {
        return Vec::new();
    }

    let Some(compiler) = detect_compiler() else {
        return Vec::new();
    };

    get_crt_file(compiler, "crtn.o", multilib_flag(target_arch))
        .into_iter()
        .collect()
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

/// Return the full path to MSVC `link.exe`, or `None` if unavailable.
///
/// Searches VS tool-root `bin/Host<host_arch>/<target_arch>/link.exe` first so
/// the correct binary is found even when no Developer Command Prompt has been
/// activated (i.e. `link.exe` is not on PATH). All known host-architecture
/// subdirectories (`HostX64`, `HostARM64`, `HostX86`) are tried, making the
/// lookup host-architecture-agnostic.
///
/// Falls back to a bare `link /?` PATH probe only when the output contains the
/// word `"Microsoft"`, ensuring non-MSVC tools also named `link` (e.g. the
/// Unix-style symlink bundled with Git for Windows or MSYS2) are rejected.
fn resolve_msvc_link_cmd(target_arch: TargetArchitecture) -> Option<String> {
    let target_binary_arch = msvc_library_arch(target_arch);
    // The `Host*` subdirectory is the **host** machine architecture, not the
    // target. Try all known host archs so the correct path is found regardless
    // of whether the host is x64, arm64, or x86.
    let host_archs = ["x64", "arm64", "x86"];
    for tool_root in visual_studio_tool_roots() {
        for host_arch in host_archs {
            let candidate = tool_root
                .join("bin")
                .join(format!("Host{}", host_arch.to_uppercase()))
                .join(target_binary_arch)
                .join("link.exe");
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
    }
    // Fallback: probe PATH, but only accept if the binary self-identifies as
    // the Microsoft linker (MSVC `link /?` prints a "Microsoft" header line).
    if let Ok(out) = Command::new("link").arg("/?").output() {
        let header = String::from_utf8_lossy(&out.stdout);
        if header.contains("Microsoft") {
            return Some("link".to_string());
        }
    }
    None
}

fn build_arch_emulation_flags(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    let mut args = Vec::new();

    match target_arch {
        TargetArchitecture::X86_64 => {
            if os_uses_macho(target_os) {
                args.push("-arch".to_string());
                args.push("x86_64".to_string());
            } else if target_os != TargetOperatingSystem::Windows {
                args.push("-m".to_string());
                args.push("elf_x86_64".to_string());
            }
        }
        TargetArchitecture::Aarch64 => {
            if os_uses_macho(target_os) {
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
        TargetArchitecture::PowerPC64 if !os_uses_macho(target_os) => {
            args.push("-m".to_string());
            args.push("elf64lppc".to_string());
        }
        TargetArchitecture::LoongArch64 => {
            args.push("-m".to_string());
            args.push("elf64loongarch".to_string());
        }
        TargetArchitecture::I686 => {
            args.push("-m".to_string());
            args.push("elf_i386".to_string());
        }
        TargetArchitecture::PowerPC64
        | TargetArchitecture::Arx64
        | TargetArchitecture::Arm32
        | TargetArchitecture::Armv7
        | TargetArchitecture::Wasm32
        | TargetArchitecture::Wasm64
        | TargetArchitecture::Nvptx64
        | TargetArchitecture::Amdgcn
        | TargetArchitecture::Unknown => {}
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => {}
    }

    args
}

fn build_crt_args(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    before_user_object: bool,
) -> Vec<String> {
    if os_uses_macho(target_os) || !host_matches(target_arch) {
        return Vec::new();
    }

    if before_user_object {
        get_crt_files(target_arch, target_os)
    } else {
        get_crtn_files(target_arch, target_os)
    }
}

fn build_entry_point_arg(target_os: TargetOperatingSystem) -> Vec<String> {
    match target_os {
        os if os_uses_macho(os) => {
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
        TargetArchitecture::Armv7 => "/lib/ld-linux-armhf.so.3",
        TargetArchitecture::I686 => "/lib/ld-linux.so.2",
        TargetArchitecture::Riscv64 => "/lib/ld-linux-riscv64-lp64d.so.1",
        TargetArchitecture::LoongArch64 => "/lib64/ld-linux-loongarch-lp64d.so.1",
        TargetArchitecture::Arx64
        | TargetArchitecture::Arm32
        | TargetArchitecture::Wasm32
        | TargetArchitecture::Wasm64
        | TargetArchitecture::PowerPC64
        | TargetArchitecture::Nvptx64
        | TargetArchitecture::Amdgcn
        | TargetArchitecture::Unknown => return Vec::new(),
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => return Vec::new(),
    };

    vec!["--dynamic-linker".to_string(), interpreter.to_string()]
}

fn build_library_args(
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Vec<String> {
    let mut args = Vec::new();

    if os_uses_macho(target_os) {
        if target_os == TargetOperatingSystem::MacOS
            && let Ok(output) = Command::new("xcrun").args(["--show-sdk-path"]).output()
            && let Ok(sdk_path) = String::from_utf8(output.stdout)
        {
            let sdk_path = sdk_path.trim();
            if !sdk_path.is_empty() {
                args.push("-syslibroot".to_string());
                args.push(sdk_path.to_string());
            }
        }
        args.push("-lSystem".to_string());
    } else {
        args.push("-lc".to_string());
        if host_matches(target_arch)
            && let Some(libgcc) =
                detect_compiler().and_then(|cc| get_libgcc_file(cc, multilib_flag(target_arch)))
        {
            args.push(libgcc);
        }
    }

    args
}

/// The compiler runtime archive. It supplies the 128-bit integer helpers (`__divti3`
/// and friends) that i128 lowering calls, which `-lc` does not provide.
fn get_libgcc_file(compiler: &str, multilib: Option<&str>) -> Option<String> {
    Command::new(compiler)
        .args(multilib)
        .arg("-print-libgcc-file-name")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|path| path.trim().to_string())
        .filter(|path| path.ends_with(".a") && Path::new(path).exists())
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
    args.extend(build_crt_args(target_arch, target_os, true));
    args.push(input_path.to_string_lossy().to_string());
    args.extend(build_crt_args(target_arch, target_os, false));
    args.extend(build_entry_point_arg(target_os));
    args.extend(build_library_args(target_arch, target_os));
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
        os if os_uses_macho(os) => {
            // Mach-O: link against libSystem (contains printf, fflush, etc.)
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
            "Unsupported Windows target architecture for clang linker: {target_arch:?}"
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
    match target_arch.output_artifact() {
        OutputArtifact::Executable => {}
        OutputArtifact::Module => {
            return Err(LaminaError::ValidationError(
                "WASM targets do not require linking".to_string(),
            ));
        }
        OutputArtifact::Text => {
            return Err(LaminaError::ValidationError(format!(
                "{target_arch} emits assembly text, which is never linked"
            )));
        }
    }

    if !input_path.exists() {
        return Err(LaminaError::ValidationError(format!(
            "Input file does not exist: {}",
            input_path.display()
        )));
    }

    let defaulted = backend.is_none();
    let backend = backend.unwrap_or_else(|| detect_linker_backend(target_arch, target_os));
    if defaulted
        && backend == LinkerBackend::Weld
        && let Some(gap) = weld_gap(target_arch, target_os)
    {
        return link_with_gnu(
            input_path,
            output_path,
            target_arch,
            target_os,
            additional_flags,
            verbose,
            &gap,
        );
    }
    let backend = validate_linker_backend_for_target(backend, target_arch, target_os)?;

    let (cmd, args) = match backend {
        LinkerBackend::Ld => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(
                    "GNU ld is not supported for Windows targets; use -c clang or -c msvc"
                        .to_string(),
                ));
            }
            let cmd = gnu_tool(target_arch, "ld")
                .ok_or_else(|| LaminaError::ValidationError(no_gnu_tool("ld", target_arch)))?;
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            (cmd, args)
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
            let cmd = if os_uses_macho(target_os) {
                "ld64.lld"
            } else {
                "lld"
            };
            (cmd.to_string(), args)
        }
        LinkerBackend::Mold => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(
                    "mold is not supported for Windows targets; use -c clang or -c msvc"
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
            ("mold".to_string(), args)
        }
        LinkerBackend::Weld => {
            let args = build_weld_linker_args(input_path, output_path, target_os, additional_flags);
            return match run_weld(&args, verbose) {
                WeldRun::Linked => Ok(()),
                WeldRun::Failed(reason) if defaulted => link_with_gnu(
                    input_path,
                    output_path,
                    target_arch,
                    target_os,
                    additional_flags,
                    verbose,
                    &reason,
                ),
                WeldRun::Failed(reason) => Err(LaminaError::ValidationError(format!(
                    "{reason}; select another toolchain with -c gnu or -c clang"
                ))),
            };
        }
        LinkerBackend::Msvc => {
            if target_os != TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(format!(
                    "MSVC link is only supported for Windows targets, not {target_os:?}"
                )));
            }
            let link_cmd = resolve_msvc_link_cmd(target_arch).ok_or_else(|| {
                LaminaError::ValidationError(
                    "MSVC link.exe not found; install Visual Studio C++ tools or use -c clang"
                        .to_string(),
                )
            })?;
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
            return run_linker(&link_cmd, args, verbose);
        }
        LinkerBackend::Custom(name) => {
            if target_os == TargetOperatingSystem::Windows {
                return Err(LaminaError::ValidationError(format!(
                    "Custom linker '{name}' is not supported for Windows targets; use -c clang or -c msvc"
                )));
            }
            let args = build_unix_linker_args(
                input_path,
                output_path,
                target_arch,
                target_os,
                additional_flags,
            );
            (name.to_string(), args)
        }
    };

    run_linker(&cmd, args, verbose)
}

fn link_with_gnu(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    additional_flags: &[String],
    verbose: bool,
    reason: &str,
) -> Result<(), LaminaError> {
    if !targets_elf(target_os) {
        return Err(LaminaError::ValidationError(format!(
            "{reason}; GNU ld cannot produce {target_os:?} executables, so select -c clang or -c msvc"
        )));
    }
    let tool = gnu_tool(target_arch, "ld").ok_or_else(|| {
        LaminaError::ValidationError(format!("{reason}; {}", no_gnu_tool("ld", target_arch)))
    })?;
    eprintln!("[WARN] {reason}; linking with GNU {tool} instead");
    let args = build_unix_linker_args(
        input_path,
        output_path,
        target_arch,
        target_os,
        additional_flags,
    );
    run_linker(&tool, args, verbose)
}

/// Outcome of running the weld binary.
enum WeldRun {
    Linked,
    Failed(String),
}

fn run_weld(args: &[String], verbose: bool) -> WeldRun {
    let cmd = resolve_toolchain_command("weld");
    if verbose {
        println!("[VERBOSE] Linking with {cmd}: {args:?}");
    }
    let output = match Command::new(&cmd).args(args).output() {
        Ok(output) => output,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return WeldRun::Failed("weld is not installed beside lamina or on PATH".to_string());
        }
        Err(e) => return WeldRun::Failed(format!("failed to run weld: {e}")),
    };
    if output.status.success() {
        return WeldRun::Linked;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let detail = stderr
        .lines()
        .map(|line| line.trim().trim_start_matches("weld: "))
        .find(|line| !line.is_empty())
        .unwrap_or("no diagnostic");
    WeldRun::Failed(format!("weld cannot link: {detail}"))
}

/// Find a toolchain binary beside the current executable, falling back to PATH.
/// Test binaries live in `deps/`, so that directory's parent is checked too.
fn resolve_toolchain_command(name: &str) -> String {
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    if let Ok(current_exe) = env::current_exe()
        && let Some(dir) = current_exe.parent()
    {
        let mut dirs = vec![dir.to_path_buf()];
        if dir.file_name().is_some_and(|n| n == "deps")
            && let Some(parent) = dir.parent()
        {
            dirs.push(parent.to_path_buf());
        }
        for dir in dirs {
            let candidate = dir.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }
    name.to_string()
}

fn run_linker(cmd: &str, args: Vec<String>, verbose: bool) -> Result<(), LaminaError> {
    if verbose {
        println!("[VERBOSE] Linking with {cmd}: {args:?}");
    }

    let output = Command::new(cmd).args(&args).output().map_err(|e| {
        LaminaError::ValidationError(format!("Failed to spawn linker '{cmd}': {e}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(LaminaError::ValidationError(format!(
            "Linker '{cmd}' failed:\nstdout: {stdout}\nstderr: {stderr}"
        )));
    }

    Ok(())
}

/// The linker used when none is named: weld, linked in-process, so no host tool is needed.
pub fn detect_linker_backend(
    _target_arch: TargetArchitecture,
    _target_os: TargetOperatingSystem,
) -> LinkerBackend {
    LinkerBackend::Weld
}

fn weld_supports(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> bool {
    matches!(
        (target_arch, target_os),
        (
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
                | TargetOperatingSystem::Windows
                | TargetOperatingSystem::MacOS
        ) | (TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS)
            | (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
    )
}

fn weld_gap(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Option<String> {
    (!weld_supports(target_arch, target_os))
        .then(|| format!("weld cannot link {target_arch:?} {target_os:?} yet"))
}

fn validate_linker_backend_for_target(
    backend: LinkerBackend,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Result<LinkerBackend, LaminaError> {
    match (backend, target_os) {
        (LinkerBackend::Weld, _) => {
            if weld_supports(target_arch, target_os) {
                Ok(backend)
            } else {
                Err(LaminaError::ValidationError(format!(
                    "weld cannot link {target_arch:?} {target_os:?} yet; use -c gnu or -c clang"
                )))
            }
        }
        (LinkerBackend::Msvc, TargetOperatingSystem::Windows) => Ok(backend),
        (LinkerBackend::Msvc, _) => Err(LaminaError::ValidationError(format!(
            "MSVC link is only supported for Windows targets, not {target_os:?}"
        ))),
        (LinkerBackend::Ld | LinkerBackend::Mold, TargetOperatingSystem::Windows) => {
            Err(LaminaError::ValidationError(format!(
                "Linker backend {backend:?} is not supported for Windows targets; use -c clang or -c msvc"
            )))
        }
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
                    "clang Windows linker does not support {target_arch:?}"
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
        TargetArchitecture::Nvptx64 => "ptx",
        TargetArchitecture::Amdgcn => "s",
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
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn macos_x86_64_link_args_use_arch_not_elf() {
        let args =
            build_arch_emulation_flags(TargetArchitecture::X86_64, TargetOperatingSystem::MacOS);
        assert!(args.iter().any(|a| a == "-arch"));
        assert!(args.iter().any(|a| a == "x86_64"));
        assert!(!args.iter().any(|a| a == "elf_x86_64"));
    }

    #[test]
    fn linux_x86_64_link_args_use_elf_emulation() {
        let args =
            build_arch_emulation_flags(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
        assert_eq!(args, ["-m", "elf_x86_64"]);
    }

    #[test]
    fn windows_x86_64_has_no_unix_emulation_flags() {
        let args =
            build_arch_emulation_flags(TargetArchitecture::X86_64, TargetOperatingSystem::Windows);
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
        let args = build_clang_windows_linker_args(input, output, TargetArchitecture::X86_64, &[])
            .expect("x86_64 Windows should be supported");

        assert_eq!(args[0], "-target");
        assert_eq!(args[1], "x86_64-pc-windows-msvc");
        let input_str = input.to_string_lossy();
        let output_str = output.to_string_lossy();
        assert!(args.iter().any(|a| a == input_str.as_ref()));
        assert!(args.iter().any(|a| a == "-o"));
        assert!(args.iter().any(|a| a == output_str.as_ref()));
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
    fn weld_refused_for_aarch64_windows() {
        let error = validate_linker_backend_for_target(
            LinkerBackend::Weld,
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Windows,
        )
        .expect_err("weld links only x86-64 PE objects");
        assert!(error.to_string().contains("weld cannot link"));
    }

    #[test]
    fn weld_gate_matches_measured_targets() {
        for (arch, os) in [
            (TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
            (TargetArchitecture::X86_64, TargetOperatingSystem::Windows),
            (TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS),
            (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux),
        ] {
            assert!(weld_gap(arch, os).is_none(), "{arch:?} {os:?}");
        }
        for (arch, os) in [
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Linux),
            (TargetArchitecture::PowerPC64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Windows),
        ] {
            assert!(weld_gap(arch, os).is_some(), "{arch:?} {os:?}");
        }
    }

    #[test]
    fn weld_is_the_default_linker() {
        assert_eq!(
            detect_linker_backend(TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
            LinkerBackend::Weld
        );
    }

    #[test]
    fn weld_args_include_libc_on_linux() {
        let input = Path::new("module.o");
        let output = Path::new("module");
        let args = build_weld_linker_args(input, output, TargetOperatingSystem::Linux, &[]);
        // weld links against libc for printf/fflush access.
        assert!(
            args.iter().any(|a| a == "-lc"),
            "expected -lc in args: {args:?}"
        );
        assert!(args.iter().any(|a| a == "-o"));
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

    #[test]
    fn apple_mobile_link_args_use_mach_o_conventions() {
        for os in [
            TargetOperatingSystem::IOS,
            TargetOperatingSystem::TvOS,
            TargetOperatingSystem::WatchOS,
            TargetOperatingSystem::VisionOS,
        ] {
            assert_eq!(
                build_arch_emulation_flags(TargetArchitecture::Aarch64, os),
                ["-arch", "arm64"],
                "{os:?}"
            );
            assert_eq!(
                build_arch_emulation_flags(TargetArchitecture::X86_64, os),
                ["-arch", "x86_64"],
                "{os:?}"
            );
            let arch = TargetArchitecture::Aarch64;
            assert!(build_crt_args(arch, os, true).is_empty(), "{os:?}");
            assert!(build_crt_args(arch, os, false).is_empty(), "{os:?}");
            assert_eq!(build_entry_point_arg(os), ["-e", "_main"], "{os:?}");
            let libs = build_library_args(arch, os);
            assert!(libs.iter().any(|a| a == "-lSystem"), "{os:?}");
            assert!(!libs.iter().any(|a| a == "-lc"), "{os:?}");
            assert!(!libs.iter().any(|a| a == "-syslibroot"), "{os:?}");
        }
    }
}
