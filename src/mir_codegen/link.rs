//! Linking backends for creating final executables or binaries.
//!
//! Supports multiple linkers:
//! - **Weld**: Lamina's own linker, installed beside `lamina` and run as a sibling
//!   binary. The default.
//! - GNU ld (binutils)
//! - LLVM lld
//! - Mold (faster alternative)
//! - MSVC linker (on Windows)
//! - Custom linkers

use crate::error::LaminaError;
use crate::mir_codegen::assemble::{gnu_tool, host_matches, no_gnu_tool, targets_elf};
use lamina_codegen::os_uses_macho;
use lamina_platform::{
    Libc, OutputArtifact, Target, TargetArchitecture, TargetEnvironment, TargetOperatingSystem,
};
use std::{
    env,
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};
use weld::VERSION as REQUIRED_WELD;

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

/// The driver whose start-up objects belong to the target: the host compiler for a
/// host-architecture link, otherwise the target's cross compiler.
fn target_compiler(
    target_arch: TargetArchitecture,
    environment: TargetEnvironment,
) -> Result<String, String> {
    let Some(libc) = environment.libc() else {
        return Err(format!(
            "the {environment} environment links no C library, so there is no compiler to take C runtime objects from"
        ));
    };
    match libc {
        Libc::Glibc => {
            if host_matches(target_arch) {
                return detect_compiler().map(str::to_string).ok_or_else(|| {
                    "no C compiler is installed on the host (looked for gcc and clang)".to_string()
                });
            }
            gnu_tool(target_arch, "gcc").ok_or_else(|| no_gnu_tool("gcc", target_arch))
        }
        Libc::Musl => musl_tool(target_arch, "gcc"),
        Libc::Lamina => Err(unimplemented_libc_message(Libc::Lamina)),
    }
}

/// Refusal for a C library lamina names but cannot yet link.
///
/// Reached before any start-up object, interpreter path or library is chosen, so a
/// target that asks for an unimplemented libc fails by name instead of quietly taking
/// another C library's.
fn unimplemented_libc_message(libc: Libc) -> String {
    let reason = libc
        .unavailable_reason()
        .unwrap_or("this C library cannot be linked");
    format!("cannot link against the {libc} C library: {reason}")
}

/// Refuse a target whose environment names a C library lamina has no implementation of.
fn check_libc_is_implemented(target: &Target) -> Result<(), LaminaError> {
    let Some(libc) = target.environment.libc() else {
        return Ok(());
    };
    if libc.is_implemented() {
        return Ok(());
    }
    let base = format!("{}_{}", target.arch_token(), target.operating_system);
    Err(LaminaError::ValidationError(format!(
        "cannot link {target}: {}; write `{base}` for the environment this target declares, `{base}_{}` for musl, or `{base}_{}` for a freestanding binary",
        unimplemented_libc_message(libc),
        TargetEnvironment::Musl,
        TargetEnvironment::Freestanding,
    )))
}

/// The musl cross driver for a target, found by the architecture's musl triple.
fn musl_tool(target_arch: TargetArchitecture, tool: &str) -> Result<String, String> {
    let triple = target_arch
        .spec()
        .toolchain
        .musl
        .ok_or_else(|| format!("no musl toolchain triple is declared for {target_arch:?}"))?;
    let name = format!("{triple}-{tool}");
    Command::new(&name)
        .arg("--version")
        .output()
        .ok()
        .map(|_| name.clone())
        .ok_or_else(|| {
            format!("no musl {tool} is installed for {target_arch:?} (looked for {name})")
        })
}

fn crt_compiler(target: &Target) -> Result<String, LaminaError> {
    target_compiler(target.architecture, target.environment)
        .map_err(|reason| missing_crt_error(target, &reason))
}

fn missing_crt_error(target: &Target, reason: &str) -> LaminaError {
    let (target_os, target_arch) = (target.operating_system, target.architecture);
    LaminaError::ValidationError(format!(
        "cannot link {target_os:?} {target_arch:?}: {reason}; crt1.o, crti.o and crtn.o come from that compiler and are required to link"
    ))
}

fn required_crt_file(
    target: &Target,
    compiler: &str,
    crt_name: &str,
) -> Result<String, LaminaError> {
    let (target_os, target_arch) = (target.operating_system, target.architecture);
    get_crt_file(compiler, crt_name, multilib_flag(target_arch)).ok_or_else(|| {
        LaminaError::ValidationError(format!(
            "cannot link {target_os:?} {target_arch:?}: {compiler} cannot find {crt_name}, which is required to link"
        ))
    })
}

fn get_crt_files(target: &Target) -> Result<Vec<String>, LaminaError> {
    if os_uses_macho(target.operating_system) || !target.environment.is_hosted() {
        return Ok(Vec::new());
    }

    let compiler = crt_compiler(target)?;
    ["crt1.o", "crti.o"]
        .iter()
        .map(|name| required_crt_file(target, &compiler, name))
        .collect()
}

fn get_crtn_files(target: &Target) -> Result<Vec<String>, LaminaError> {
    if os_uses_macho(target.operating_system) || !target.environment.is_hosted() {
        return Ok(Vec::new());
    }

    let compiler = crt_compiler(target)?;
    required_crt_file(target, &compiler, "crtn.o").map(|path| vec![path])
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

/// Directory name this architecture's import libraries live under in the Windows SDK.
///
/// Stated per architecture in the arch table. An architecture Windows has no layout for
/// is an error here, not a search of the x86_64 directories.
fn msvc_library_arch(target_arch: TargetArchitecture) -> Result<&'static str, LaminaError> {
    target_arch.spec().msvc_library_arch.ok_or_else(|| {
        LaminaError::ValidationError(format!(
            "cannot link Windows {target_arch:?}: the Windows SDK has no library directory for this architecture, so there is nothing to search"
        ))
    })
}

fn build_msvc_library_paths(target_arch: TargetArchitecture) -> Result<Vec<String>, LaminaError> {
    let library_arch = msvc_library_arch(target_arch)?;
    let Some(sdk_root) = windows_sdk_root() else {
        return Ok(Vec::new());
    };
    let Some(sdk_version) = windows_sdk_version(&sdk_root) else {
        return Ok(Vec::new());
    };

    let paths = ["ucrt", "um"]
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
        .collect::<Vec<_>>();
    Ok(paths)
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

fn build_msvc_toolchain_library_paths(
    target_arch: TargetArchitecture,
) -> Result<Vec<String>, LaminaError> {
    let library_arch = msvc_library_arch(target_arch)?;
    let paths = visual_studio_tool_roots()
        .into_iter()
        .map(|tool_root| tool_root.join("lib").join(library_arch))
        .filter(|library_path| library_path.is_dir())
        .map(|library_path| format!("/LIBPATH:{}", library_path.display()))
        .collect::<Vec<_>>();
    Ok(paths)
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
    let target_binary_arch = msvc_library_arch(target_arch).ok()?;
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
        TargetArchitecture::S390x => {
            args.push("-m".to_string());
            args.push("elf64_s390".to_string());
        }
        TargetArchitecture::PowerPC64Be => {
            args.push("-m".to_string());
            args.push("elf64ppc".to_string());
        }
        TargetArchitecture::Mips64 => {
            args.push("-m".to_string());
            args.push("elf64btsmip".to_string());
        }
        TargetArchitecture::Mips64El => {
            args.push("-m".to_string());
            args.push("elf64ltsmip".to_string());
        }
        TargetArchitecture::I686 => {
            args.push("-m".to_string());
            args.push("elf_i386".to_string());
        }
        TargetArchitecture::PowerPC64
        | TargetArchitecture::Arm32
        | TargetArchitecture::Armv7
        | TargetArchitecture::Thumbv6m
        | TargetArchitecture::Thumbv7m
        | TargetArchitecture::Wasm32
        | TargetArchitecture::Wasm64
        | TargetArchitecture::Nvptx64
        | TargetArchitecture::Amdgcn => {}
        #[cfg(feature = "nightly")]
        TargetArchitecture::Arx64 | TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => {}
    }

    args
}

fn build_crt_args(target: &Target, before_user_object: bool) -> Result<Vec<String>, LaminaError> {
    if os_uses_macho(target.operating_system) || cross_os(target.operating_system) {
        return Ok(Vec::new());
    }

    if before_user_object {
        get_crt_files(target)
    } else {
        get_crtn_files(target)
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

/// FreeBSD's run-time linker lives at one path on every architecture it supports.
const FREEBSD_INTERPRETER: &str = "/libexec/ld-elf.so.1";

/// True when the host toolchain's C runtime objects and libraries belong to a
/// different operating system than the target, so linking them in would be wrong.
fn cross_os(target_os: TargetOperatingSystem) -> bool {
    target_os == TargetOperatingSystem::FreeBSD && !cfg!(target_os = "freebsd")
}

fn build_dynamic_linker_arg(target: &Target) -> Vec<String> {
    let target_os = target.operating_system;
    if target_os == TargetOperatingSystem::FreeBSD {
        return vec![
            "--dynamic-linker".to_string(),
            FREEBSD_INTERPRETER.to_string(),
        ];
    }
    if target_os != TargetOperatingSystem::Linux {
        return Vec::new();
    }

    let Some(interpreter) = target.interpreter() else {
        return Vec::new();
    };
    vec!["--dynamic-linker".to_string(), interpreter.to_string()]
}

fn build_library_args(target: &Target) -> Result<Vec<String>, LaminaError> {
    let target_os = target.operating_system;
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
    } else if !cross_os(target_os) && target.environment.libc().is_some() {
        args.push("-lc".to_string());
        if let Some(libgcc) = libgcc_file(target) {
            args.push(libgcc);
        }
    }

    Ok(args)
}

fn libgcc_file(target: &Target) -> Option<String> {
    let target_arch = target.architecture;
    let compiler = target_compiler(target_arch, target.environment).ok()?;
    get_libgcc_file(&compiler, multilib_flag(target_arch))
}

/// The host compiler runtime archive, linked after our own objects so it supplies
/// only what `builtins/` does not.
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
    target: &Target,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
) -> Result<Vec<String>, LaminaError> {
    let (target_arch, target_os) = (target.architecture, target.operating_system);
    let mut args = Vec::new();

    args.extend(build_arch_emulation_flags(target_arch, target_os));
    let dynamic_linker = build_dynamic_linker_arg(target);
    // gcc and clang ask for PT_GNU_EH_FRAME in every dynamic executable. Without it
    // the unwinder in libgcc finds no FDE, not even for C frames.
    if !dynamic_linker.is_empty() {
        args.push("--eh-frame-hdr".to_string());
    }
    args.extend(dynamic_linker);
    args.extend(build_crt_args(target, true)?);
    args.push(input_path.to_string_lossy().to_string());
    args.extend(object_args(runtime_objects));
    args.extend(build_crt_args(target, false)?);
    args.extend(build_entry_point_arg(target_os));
    args.extend(build_library_args(target)?);
    args.extend(additional_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());

    Ok(args)
}

/// Build args for weld as a freestanding, no-libc linker. weld synthesizes its
/// own `_start` and emits a static executable, so no C runtime objects, libc,
/// or dynamic linker are passed. This is the toolchain path that needs no
/// external assembler or linker.
fn build_weld_linker_args(
    input_path: &Path,
    output_path: &Path,
    target: &Target,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
) -> Vec<String> {
    let target_os = target.operating_system;
    let mut args = Vec::new();
    args.push(input_path.to_string_lossy().to_string());
    args.extend(object_args(runtime_objects));
    if let Some(interpreter) = target.interpreter() {
        args.push("--dynamic-linker".to_string());
        args.push(interpreter.to_string());
    }
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
    runtime_objects: &[PathBuf],
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
    args.extend(object_args(runtime_objects));
    args.extend(additional_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    Ok(args)
}

/// Link the static C runtime as cl does by default, and leave the entry point to
/// mainCRTStartup, which sets up the CRT and TLS before it calls `main`. The CRT also
/// supplies memset, memcpy and memmove, and `_tls_index`.
fn build_msvc_linker_args(
    input_path: &Path,
    output_path: &Path,
    target_arch: TargetArchitecture,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
) -> Result<Vec<String>, LaminaError> {
    let mut args = vec!["/nologo".to_string()];
    args.push("/subsystem:console".to_string());
    args.extend(build_msvc_library_paths(target_arch)?);
    args.extend(build_msvc_toolchain_library_paths(target_arch)?);
    args.extend(additional_flags.iter().cloned());
    args.push(input_path.to_string_lossy().to_string());
    args.extend(object_args(runtime_objects));
    args.push(format!("/OUT:{}", output_path.display()));
    args.push("libcmt.lib".to_string());
    args.push("libvcruntime.lib".to_string());
    args.push("libucrt.lib".to_string());
    args.push("legacy_stdio_definitions.lib".to_string());
    args.push("kernel32.lib".to_string());
    Ok(args)
}

/// Extra objects named on the link line, ahead of any library. Our own compiler
/// runtime must define a symbol before an archive gets the chance to.
fn object_args(objects: &[PathBuf]) -> Vec<String> {
    objects
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect()
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

/// ELF `e_ident[EI_OSABI]` for FreeBSD. FreeBSD's image activator brands a binary by
/// this byte, by an ABI note, or by a recognised interpreter path, and `kern.elf64.
/// fallback_brand` is -1, so an unbranded static executable is refused by the kernel.
const ELFOSABI_FREEBSD: u8 = 9;
const EI_OSABI: u64 = 7;

/// Link object files to create final executable
pub fn link(
    input_path: &Path,
    output_path: &Path,
    target: &Target,
    backend: Option<LinkerBackend>,
    additional_flags: &[String],
    verbose: bool,
) -> Result<(), LaminaError> {
    link_with_runtime(
        input_path,
        output_path,
        target,
        backend,
        &[],
        additional_flags,
        verbose,
    )
}

/// Like [`link`], but names the compiler runtime objects built for this program.
#[allow(clippy::too_many_arguments)]
pub fn link_with_runtime(
    input_path: &Path,
    output_path: &Path,
    target: &Target,
    backend: Option<LinkerBackend>,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
    verbose: bool,
) -> Result<(), LaminaError> {
    link_object(
        input_path,
        output_path,
        target,
        backend,
        runtime_objects,
        additional_flags,
        verbose,
    )?;
    if target.operating_system == TargetOperatingSystem::FreeBSD {
        brand_freebsd(output_path)?;
    }
    Ok(())
}

/// Stamp `ELFOSABI_FREEBSD` into the linked executable, as `brandelf -t FreeBSD` does.
/// Host linkers emit `ELFOSABI_NONE`, which FreeBSD's loader does not recognise.
fn brand_freebsd(output_path: &Path) -> Result<(), LaminaError> {
    use std::io::{Seek, SeekFrom, Write};

    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(output_path)
        .map_err(|e| {
            LaminaError::ValidationError(format!(
                "cannot open {} to brand it for FreeBSD: {e}",
                output_path.display()
            ))
        })?;
    let mut magic = [0u8; 4];
    io::Read::read_exact(&mut file, &mut magic).map_err(|e| {
        LaminaError::ValidationError(format!(
            "cannot read the ELF header of {}: {e}",
            output_path.display()
        ))
    })?;
    if magic != *b"\x7fELF" {
        return Err(LaminaError::ValidationError(format!(
            "{} is not an ELF file, so it cannot be branded for FreeBSD",
            output_path.display()
        )));
    }
    file.seek(SeekFrom::Start(EI_OSABI))
        .and_then(|_| file.write_all(&[ELFOSABI_FREEBSD]))
        .map_err(|e| {
            LaminaError::ValidationError(format!(
                "cannot write the FreeBSD ELF brand into {}: {e}",
                output_path.display()
            ))
        })
}

#[allow(clippy::too_many_arguments)]
fn link_object(
    input_path: &Path,
    output_path: &Path,
    target: &Target,
    backend: Option<LinkerBackend>,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
    verbose: bool,
) -> Result<(), LaminaError> {
    check_libc_is_implemented(target)?;
    let (target_arch, target_os) = (target.architecture, target.operating_system);
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
            target,
            runtime_objects,
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
                target,
                runtime_objects,
                additional_flags,
            )?;
            (cmd, args)
        }
        LinkerBackend::Lld => {
            if target_os == TargetOperatingSystem::Windows {
                let args = build_clang_windows_linker_args(
                    input_path,
                    output_path,
                    target_arch,
                    runtime_objects,
                    additional_flags,
                )?;
                return run_linker("clang", args, verbose);
            }

            let mut args = build_unix_linker_args(
                input_path,
                output_path,
                target,
                runtime_objects,
                additional_flags,
            )?;
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
                target,
                runtime_objects,
                additional_flags,
            )?;
            ("mold".to_string(), args)
        }
        LinkerBackend::Weld => {
            let args = build_weld_linker_args(
                input_path,
                output_path,
                target,
                runtime_objects,
                additional_flags,
            );
            return match run_weld(&args, verbose) {
                WeldRun::Linked => Ok(()),
                WeldRun::Failed(reason) if defaulted => link_with_gnu(
                    input_path,
                    output_path,
                    target,
                    runtime_objects,
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
            let args = build_msvc_linker_args(
                input_path,
                output_path,
                target_arch,
                runtime_objects,
                additional_flags,
            )?;
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
                target,
                runtime_objects,
                additional_flags,
            )?;
            (name.to_string(), args)
        }
    };

    run_linker(&cmd, args, verbose)
}

#[allow(clippy::too_many_arguments)]
fn link_with_gnu(
    input_path: &Path,
    output_path: &Path,
    target: &Target,
    runtime_objects: &[PathBuf],
    additional_flags: &[String],
    verbose: bool,
    reason: &str,
) -> Result<(), LaminaError> {
    let (target_arch, target_os) = (target.architecture, target.operating_system);
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
        target,
        runtime_objects,
        additional_flags,
    )?;
    run_linker(&tool, args, verbose)
}

/// Outcome of running the weld binary.
enum WeldRun {
    Linked,
    Failed(String),
}

/// Numeric release fields of a version string, so `0.2.10` orders after `0.2.9`.
/// Reads the last whitespace-separated field, as `weld --version` prints `weld X.Y.Z`.
fn version_fields(text: &str) -> Option<(u32, u32, u32)> {
    let mut fields = text
        .split_whitespace()
        .next_back()?
        .split(['.', '-', '+'])
        .map_while(|field| field.parse().ok());
    Some((
        fields.next()?,
        fields.next().unwrap_or(0),
        fields.next().unwrap_or(0),
    ))
}

/// What the resolved weld answers `--version` with, or `None` if it does not answer.
fn reported_weld_version(cmd: &str) -> Option<String> {
    let output = Command::new(cmd).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim().to_string();
    (!line.is_empty()).then_some(line)
}

/// Refuse a weld older than the one lamina was built against, naming both versions
/// and the binary that answered. A newer weld is accepted: linking against a weld
/// ahead of the bundled one is the supported way to pick up a linker fix early. A
/// weld that reports no version it understands is left to its own diagnostics.
fn weld_version_skew(cmd: &str) -> Option<String> {
    let reported = reported_weld_version(cmd)?;
    let found = version_fields(&reported)?;
    let required = version_fields(REQUIRED_WELD)?;
    (found < required).then(|| {
        let (major, minor, patch) = found;
        format!(
            "weld {major}.{minor}.{patch} at {cmd} is older than the weld {REQUIRED_WELD} lamina needs, so it rejects objects this compiler emits; run `cargo install lamina` to put a matching weld beside the compiler, or `cargo install lamina-weld` to update the one on PATH"
        )
    })
}

fn checked_weld_version_skew(cmd: &str) -> Option<String> {
    static SKEW: OnceLock<Option<String>> = OnceLock::new();
    SKEW.get_or_init(|| weld_version_skew(cmd)).clone()
}

fn run_weld(args: &[String], verbose: bool) -> WeldRun {
    let cmd = resolve_toolchain_command("weld");
    if let Some(skew) = checked_weld_version_skew(&cmd) {
        return WeldRun::Failed(skew);
    }
    if verbose {
        println!("[VERBOSE] Linking with {cmd}: {args:?}");
    }
    let output = match Command::new(&cmd).args(args).output() {
        Ok(output) => output,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return WeldRun::Failed(format!(
                "weld is not installed beside lamina or on PATH; lamina needs weld {REQUIRED_WELD} or newer, which `cargo install lamina` places beside the compiler"
            ));
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
/// The sibling is preferred because it ships with this lamina and so matches it,
/// so `LAMINA_WELD` is how a deliberately newer one is named. Test binaries live
/// in `deps/`, so that directory's parent is checked too.
fn resolve_toolchain_command(name: &str) -> String {
    toolchain_command(name, env::var_os(format!("LAMINA_{}", name.to_uppercase())))
}

fn toolchain_command(name: &str, named: Option<OsString>) -> String {
    if let Some(named) = named.filter(|named| !named.is_empty()) {
        return named.to_string_lossy().to_string();
    }
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

/// The linker used when none is named: weld, which is installed beside `lamina`, so
/// no host tool is needed.
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
        ) | (
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux
                | TargetOperatingSystem::MacOS
                | TargetOperatingSystem::Windows
        ) | (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
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
    use crate::test_support::TempDir;

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
        let args =
            build_clang_windows_linker_args(input, output, TargetArchitecture::X86_64, &[], &[])
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
    fn build_msvc_linker_args_links_crt_and_names_no_entry() {
        let args = build_msvc_linker_args(
            Path::new("module.o"),
            Path::new("module.exe"),
            TargetArchitecture::X86_64,
            &[],
            &[],
        )
        .expect("x86_64 Windows should be supported");

        assert!(
            !args
                .iter()
                .any(|a| a.to_ascii_lowercase().starts_with("/entry")),
            "{args:?}"
        );
        for lib in ["libcmt.lib", "libvcruntime.lib", "libucrt.lib"] {
            assert!(args.iter().any(|a| a == lib), "missing {lib}: {args:?}");
        }
        assert!(!args.iter().any(|a| a == "ucrt.lib"), "{args:?}");
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
    fn weld_refused_for_powerpc64_linux() {
        let error = validate_linker_backend_for_target(
            LinkerBackend::Weld,
            TargetArchitecture::PowerPC64,
            TargetOperatingSystem::Linux,
        )
        .expect_err("weld writes no powerpc64 image");
        assert!(error.to_string().contains("weld cannot link"));
    }

    #[test]
    fn weld_gate_matches_measured_targets() {
        for (arch, os) in [
            (TargetArchitecture::X86_64, TargetOperatingSystem::Linux),
            (TargetArchitecture::X86_64, TargetOperatingSystem::Windows),
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Linux),
            (TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS),
            (TargetArchitecture::Aarch64, TargetOperatingSystem::Windows),
            (TargetArchitecture::Riscv64, TargetOperatingSystem::Linux),
        ] {
            assert!(weld_gap(arch, os).is_none(), "{arch:?} {os:?}");
        }
        let (arch, os) = (TargetArchitecture::PowerPC64, TargetOperatingSystem::Linux);
        assert!(weld_gap(arch, os).is_some(), "{arch:?} {os:?}");
    }

    #[test]
    fn a_named_toolchain_binary_wins_over_the_shipped_one() {
        assert_eq!(
            toolchain_command("weld", Some(OsString::from("/opt/newer/weld"))),
            "/opt/newer/weld"
        );
        let shipped = toolchain_command("weld", None);
        assert_eq!(toolchain_command("weld", Some(OsString::new())), shipped);
        assert!(
            shipped.ends_with("weld") || shipped.ends_with("weld.exe"),
            "{shipped}"
        );
    }

    #[test]
    fn version_fields_order_by_number_not_by_text() {
        assert_eq!(version_fields("weld 0.2.3"), Some((0, 2, 3)));
        assert_eq!(version_fields("0.2.10"), Some((0, 2, 10)));
        assert_eq!(version_fields("weld 1.0"), Some((1, 0, 0)));
        assert_eq!(version_fields("weld 0.3.0-rc1"), Some((0, 3, 0)));
        assert!(version_fields("0.2.10") > version_fields("0.2.9"));
        assert_eq!(version_fields("weld"), None);
    }

    #[test]
    fn the_shipped_weld_meets_the_version_lamina_needs() {
        let cmd = resolve_toolchain_command("weld");
        assert!(
            weld_version_skew(&cmd).is_none(),
            "the weld beside lamina must not be older than {REQUIRED_WELD}"
        );
        assert!(version_fields(REQUIRED_WELD).is_some(), "{REQUIRED_WELD}");
    }

    #[test]
    fn an_older_weld_is_named_with_both_versions() {
        let dir = TempDir::new("lamina-weld-skew");
        let fake = dir.path().join("weld");
        if cfg!(windows) {
            return;
        }
        fs::write(&fake, "#!/bin/sh\necho 'weld 0.0.1'\n").expect("write stub weld");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).expect("make executable");
        }
        let cmd = fake.to_string_lossy().to_string();
        let message = weld_version_skew(&cmd).expect("0.0.1 is older than anything lamina ships");
        assert!(message.contains("weld 0.0.1"), "{message}");
        assert!(message.contains(REQUIRED_WELD), "{message}");
        assert!(message.contains(&cmd), "{message}");
        assert!(message.contains("cargo install lamina"), "{message}");
    }

    #[test]
    fn a_newer_weld_is_accepted() {
        let dir = TempDir::new("lamina-weld-newer");
        let fake = dir.path().join("weld");
        if cfg!(windows) {
            return;
        }
        fs::write(&fake, "#!/bin/sh\necho 'weld 99.0.0'\n").expect("write stub weld");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).expect("make executable");
        }
        assert!(weld_version_skew(&fake.to_string_lossy()).is_none());
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
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
        let args = build_weld_linker_args(input, output, &target, &[], &[]);
        // weld links against libc for printf/fflush access.
        assert!(
            args.iter().any(|a| a == "-lc"),
            "expected -lc in args: {args:?}"
        );
        assert!(args.iter().any(|a| a == "-o"));
        assert!(!args.iter().any(|a| a.contains("crt1.o")));
    }

    #[test]
    fn weld_args_name_the_interpreter_the_environment_asks_for() {
        let input = Path::new("module.o");
        let output = Path::new("module");
        for (environment, interpreter) in [
            (TargetEnvironment::Gnu, "/lib64/ld-linux-x86-64.so.2"),
            (TargetEnvironment::Musl, "/lib/ld-musl-x86_64.so.1"),
        ] {
            let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .with_environment(environment)
                .expect("x86_64 linux takes gnu and musl");
            let args = build_weld_linker_args(input, output, &target, &[], &[]);
            let named = args
                .windows(2)
                .find(|pair| pair[0] == "--dynamic-linker")
                .map(|pair| pair[1].as_str());
            assert_eq!(named, Some(interpreter), "{environment:?}");
        }
    }

    #[test]
    fn freebsd_uses_the_rtld_elf_interpreter() {
        assert_eq!(FREEBSD_INTERPRETER, "/libexec/ld-elf.so.1");
        for arch in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
            assert_eq!(
                build_dynamic_linker_arg(&Target::new(arch, TargetOperatingSystem::FreeBSD)),
                ["--dynamic-linker", "/libexec/ld-elf.so.1"],
                "{arch:?}"
            );
        }
        assert_eq!(
            build_dynamic_linker_arg(&Target::new(
                TargetArchitecture::X86_64,
                TargetOperatingSystem::Linux
            )),
            ["--dynamic-linker", "/lib64/ld-linux-x86-64.so.2"]
        );
    }

    #[test]
    fn the_environment_picks_the_interpreter_not_the_host_toolchain() {
        let gnu = Target::new(TargetArchitecture::S390x, TargetOperatingSystem::Linux);
        assert_eq!(
            build_dynamic_linker_arg(&gnu),
            ["--dynamic-linker", "/lib/ld64.so.1"]
        );
        let musl = gnu
            .with_environment(TargetEnvironment::Musl)
            .expect("s390x accepts musl");
        assert_eq!(
            build_dynamic_linker_arg(&musl),
            ["--dynamic-linker", "/lib/ld-musl-s390x.so.1"]
        );
    }

    #[test]
    fn a_dynamic_executable_asks_for_an_eh_frame_header() {
        if cfg!(target_os = "freebsd") {
            return;
        }
        let args = |target: &Target| {
            build_unix_linker_args(Path::new("module.o"), Path::new("module"), target, &[], &[])
                .expect("link arguments")
        };
        // Off a FreeBSD host the link takes no host C runtime, so needs no compiler.
        let freebsd = args(&Target::new(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::FreeBSD,
        ));
        assert!(freebsd.iter().any(|a| a == "--eh-frame-hdr"), "{freebsd:?}");
        // gcc leaves it out of a static link, and ld64 has no such option.
        let freestanding = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .with_environment(TargetEnvironment::Freestanding)
            .expect("every architecture accepts a freestanding environment");
        let macos = Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS);
        for target in [freestanding, macos] {
            let args = args(&target);
            assert!(!args.iter().any(|a| a == "--eh-frame-hdr"), "{args:?}");
        }
    }

    #[test]
    fn a_freestanding_target_links_no_libc_and_no_compiler_runtime() {
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .with_environment(TargetEnvironment::Freestanding)
            .expect("every architecture accepts a freestanding environment");
        assert!(build_dynamic_linker_arg(&target).is_empty());
        assert!(
            build_crt_args(&target, true)
                .expect("C runtime objects")
                .is_empty()
        );
        let libs = build_library_args(&target).expect("library arguments");
        assert!(!libs.iter().any(|a| a == "-lc"), "{libs:?}");
        assert!(!libs.iter().any(|a| a.ends_with("libgcc.a")), "{libs:?}");
    }

    #[test]
    fn a_lamina_libc_target_refuses_to_link_and_names_the_library() {
        for arch in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
            let target = Target::new(arch, TargetOperatingSystem::Linux)
                .with_environment(TargetEnvironment::Lamina)
                .expect("x86_64 and aarch64 accept the lamina environment");
            let error = check_libc_is_implemented(&target).expect_err("no Lamina libc exists");
            let message = error.to_string();
            assert!(message.contains("lamina"), "{message}");
            assert!(message.contains("not implemented yet"), "{message}");
            assert!(message.contains("musl"), "{message}");
            let reason = target_compiler(arch, TargetEnvironment::Lamina)
                .expect_err("no compiler supplies a Lamina C runtime");
            assert!(reason.contains("Lamina C library"), "{reason}");
        }
    }

    #[test]
    fn a_lamina_libc_target_names_no_interpreter_so_none_is_guessed() {
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .with_environment(TargetEnvironment::Lamina)
            .expect("x86_64 accepts the lamina environment");
        assert!(build_dynamic_linker_arg(&target).is_empty());
        let gnu = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
        assert_eq!(
            build_dynamic_linker_arg(&gnu),
            ["--dynamic-linker", "/lib64/ld-linux-x86-64.so.2"]
        );
    }

    #[test]
    fn an_implemented_libc_passes_the_link_gate() {
        for environment in [
            TargetEnvironment::Gnu,
            TargetEnvironment::Musl,
            TargetEnvironment::Freestanding,
        ] {
            let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .with_environment(environment)
                .expect("x86_64 linux takes gnu, musl and none");
            check_libc_is_implemented(&target).expect("an implemented libc links");
        }
    }

    #[test]
    fn a_target_with_32bit_registers_links_without_demanding_libgcc() {
        let narrow = Target::new(TargetArchitecture::I686, TargetOperatingSystem::Linux);
        let libs = build_library_args(&narrow).expect("library arguments");
        assert!(libs.iter().any(|a| a == "-lc"), "{libs:?}");
    }

    #[test]
    fn an_architecture_windows_has_no_layout_for_is_refused_by_name() {
        let error = msvc_library_arch(TargetArchitecture::S390x)
            .expect_err("the Windows SDK has no s390x directory");
        assert!(format!("{error}").contains("S390x"), "{error}");
        assert_eq!(
            msvc_library_arch(TargetArchitecture::Aarch64).expect("aarch64 is an SDK arch"),
            "arm64"
        );
    }

    #[test]
    fn freebsd_link_args_leave_out_the_host_c_runtime() {
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::FreeBSD);
        if cfg!(target_os = "freebsd") {
            return;
        }
        assert!(
            build_crt_args(&target, true)
                .expect("C runtime objects")
                .is_empty()
        );
        assert!(
            build_crt_args(&target, false)
                .expect("C runtime objects")
                .is_empty()
        );
        assert!(
            !build_library_args(&target)
                .expect("library arguments")
                .iter()
                .any(|a| a == "-lc")
        );
    }

    #[test]
    fn freebsd_brand_is_written_into_the_elf_header() {
        let dir = TempDir::new("lamina-freebsd-brand");
        let path = dir.path().join("stub.elf");
        let mut header = vec![0u8; 64];
        header[..4].copy_from_slice(b"\x7fELF");
        header[4] = 2;
        header[5] = 1;
        header[6] = 1;
        fs::write(&path, &header).expect("write stub elf");
        brand_freebsd(&path).expect("brand the stub");
        let branded = fs::read(&path).expect("read back");
        assert_eq!(branded[EI_OSABI as usize], ELFOSABI_FREEBSD);
        assert_eq!(branded[..7], header[..7]);
        assert_eq!(branded.len(), header.len());

        fs::write(&path, b"not an elf file at all").expect("write non-elf");
        let error = brand_freebsd(&path).expect_err("a non-ELF file cannot be branded");
        assert!(error.to_string().contains("not an ELF file"));
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
            let target = Target::new(TargetArchitecture::Aarch64, os);
            assert!(
                build_crt_args(&target, true)
                    .expect("C runtime objects")
                    .is_empty(),
                "{os:?}"
            );
            assert!(
                build_crt_args(&target, false)
                    .expect("C runtime objects")
                    .is_empty(),
                "{os:?}"
            );
            assert_eq!(build_entry_point_arg(os), ["-e", "_main"], "{os:?}");
            let libs = build_library_args(&target).expect("library arguments");
            assert!(libs.iter().any(|a| a == "-lSystem"), "{os:?}");
            assert!(!libs.iter().any(|a| a == "-lc"), "{os:?}");
            assert!(!libs.iter().any(|a| a == "-syslibroot"), "{os:?}");
        }
    }
}
