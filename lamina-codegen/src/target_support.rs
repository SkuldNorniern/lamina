//! Cross-platform target support for code generation.
//!
//! This module defines which (architecture, OS) combinations are supported
//! for assembly and binary emission, so callers can cross-compile to a
//! different target than the host.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Returns true if assembly text generation is supported for this (arch, os).
///
/// Cross-compilation: pass the desired target, not the host.
pub fn is_assembly_supported(arch: TargetArchitecture, os: TargetOperatingSystem) -> bool {
    match arch {
        TargetArchitecture::X86_64 => os_supported_for_x86_64(os),
        TargetArchitecture::Aarch64 => os_supported_for_aarch64(os),
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
            os_supported_for_riscv(os)
        }
        TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
            matches!(os, TargetOperatingSystem::Unknown | TargetOperatingSystem::Linux)
        }
        _ => false,
    }
}

/// Returns true if the OS is supported for x86_64 code generation.
fn os_supported_for_x86_64(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::MacOS
            | TargetOperatingSystem::Windows
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::Unknown
    )
}

/// Returns true if the OS is supported for AArch64 code generation.
fn os_supported_for_aarch64(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::MacOS
            | TargetOperatingSystem::Windows
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::Unknown
    )
}

/// Returns true if the OS is supported for RISC-V code generation.
fn os_supported_for_riscv(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Unknown
    )
}

/// Returns a short description of supported targets for error messages.
pub fn supported_assembly_targets_hint() -> &'static str {
    "Supported: x86_64/aarch64/riscv32/riscv64/wasm32 on Linux, macOS, Windows, BSD, Redox. Use lamina_platform::Target for cross-compilation."
}

/// Returns true if the given OS uses ELF object format (Linux, BSD, Redox).
pub fn os_uses_elf(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
    )
}

/// Returns true if the given OS uses Mach-O (macOS).
pub fn os_uses_macho(os: TargetOperatingSystem) -> bool {
    os == TargetOperatingSystem::MacOS
}

/// Returns true if the given OS uses COFF/PE (Windows).
pub fn os_uses_coff(os: TargetOperatingSystem) -> bool {
    os == TargetOperatingSystem::Windows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_64_cross_platform() {
        assert!(is_assembly_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_assembly_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows
        ));
        assert!(is_assembly_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS
        ));
        assert!(is_assembly_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::FreeBSD
        ));
    }

    #[test]
    fn test_aarch64_cross_platform() {
        assert!(is_assembly_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_assembly_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::MacOS
        ));
    }

    #[test]
    fn test_riscv_elf_like_only() {
        assert!(is_assembly_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Linux
        ));
        assert!(!is_assembly_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Windows
        ));
    }

    #[test]
    fn test_wasm_limited_os() {
        assert!(is_assembly_supported(
            TargetArchitecture::Wasm32,
            TargetOperatingSystem::Unknown
        ));
    }

    #[test]
    fn test_unknown_os_fallback() {
        assert!(is_assembly_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Unknown
        ));
        assert!(is_assembly_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Unknown
        ));
    }

    #[test]
    fn test_unsupported_arch() {
        assert!(!is_assembly_supported(
            TargetArchitecture::Unknown,
            TargetOperatingSystem::Linux
        ));
    }

    #[test]
    fn test_os_uses_elf() {
        assert!(os_uses_elf(TargetOperatingSystem::Linux));
        assert!(os_uses_elf(TargetOperatingSystem::FreeBSD));
        assert!(os_uses_elf(TargetOperatingSystem::Redox));
        assert!(!os_uses_elf(TargetOperatingSystem::MacOS));
        assert!(!os_uses_elf(TargetOperatingSystem::Windows));
    }
}
