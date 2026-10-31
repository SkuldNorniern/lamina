//! Cross-platform target support for code generation.
//!
//! This module defines which (architecture, OS) combinations are supported
//! for assembly and binary emission, so callers can cross-compile to a
//! different target than the host.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Whether a backend can *write* assembly text for this (arch, os).
///
/// Distinct from `ras::is_assembly_target_supported`, which answers whether ras can
/// *read* assembly for a target and turn it into an object file. A target can be one
/// and not the other.
///
/// Cross-compilation: pass the desired target, not the host.
pub fn is_assembly_emission_supported(arch: TargetArchitecture, os: TargetOperatingSystem) -> bool {
    match arch {
        TargetArchitecture::Nvptx64 => os == TargetOperatingSystem::Cuda,
        TargetArchitecture::Amdgcn => os == TargetOperatingSystem::AmdHsa,
        TargetArchitecture::X86_64 => os_supported_for_x86_64(os),
        TargetArchitecture::Aarch64 => os_supported_for_aarch64(os),
        #[cfg(feature = "nightly")]
        TargetArchitecture::Arx64 => os_supported_for_arx64(os),
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => os_supported_for_riscv(os),
        TargetArchitecture::LoongArch64 => os == TargetOperatingSystem::Linux,
        TargetArchitecture::S390x => os == TargetOperatingSystem::Linux,
        TargetArchitecture::Mips64 | TargetArchitecture::Mips64El => {
            os == TargetOperatingSystem::Linux
        }
        TargetArchitecture::I686 => os == TargetOperatingSystem::Linux,
        // wasi_snapshot_preview1 passes pointers as i32, so it has no wasm64 form.
        TargetArchitecture::Wasm32 => {
            matches!(
                os,
                TargetOperatingSystem::Unknown
                    | TargetOperatingSystem::Linux
                    | TargetOperatingSystem::Wasi
            )
        }
        TargetArchitecture::Wasm64 => {
            matches!(
                os,
                TargetOperatingSystem::Unknown | TargetOperatingSystem::Linux
            )
        }
        TargetArchitecture::Armv7 => os == TargetOperatingSystem::Linux,
        TargetArchitecture::Thumbv6m | TargetArchitecture::Thumbv7m => {
            matches!(
                os,
                TargetOperatingSystem::Unknown | TargetOperatingSystem::Linux
            )
        }
        TargetArchitecture::PowerPC64 => {
            matches!(
                os,
                TargetOperatingSystem::Linux | TargetOperatingSystem::Unknown
            )
        }
        TargetArchitecture::PowerPC64Be => os == TargetOperatingSystem::Linux,
        TargetArchitecture::Arm32 => false,
        #[cfg(feature = "nightly")]
        TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => false,
    }
}

/// Returns true if the OS is supported for x86_64 code generation.
fn os_supported_for_x86_64(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::Android
            | TargetOperatingSystem::MacOS
            | TargetOperatingSystem::IOS
            | TargetOperatingSystem::TvOS
            | TargetOperatingSystem::WatchOS
            | TargetOperatingSystem::VisionOS
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
            | TargetOperatingSystem::Android
            | TargetOperatingSystem::MacOS
            | TargetOperatingSystem::IOS
            | TargetOperatingSystem::TvOS
            | TargetOperatingSystem::WatchOS
            | TargetOperatingSystem::VisionOS
            | TargetOperatingSystem::Windows
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::Unknown
    )
}

/// Returns true if the OS is supported for ARX64 code generation.
#[cfg(feature = "nightly")]
fn os_supported_for_arx64(os: TargetOperatingSystem) -> bool {
    matches!(os, TargetOperatingSystem::Unknown)
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
    "Supported: x86_64 and aarch64 on Linux, Android, macOS, iOS, Windows, BSD, Redox and bare metal; \n riscv32/riscv64 on Linux, BSD and bare metal; i686, armv7, ppc64le, ppc64, s390x, mips64, mips64el \n and loongarch64 on Linux; thumbv6m/thumbv7m on Linux and bare metal; wasm32 on Linux, WASI and \n bare metal, wasm64 on Linux and bare metal; nvptx64 on CUDA and amdgcn on AMDHSA. \n Use lamina_platform::Target for cross-compilation."
}

/// Returns true if the given OS uses ELF object format (Linux, BSD, Redox, Horizon, Orbis, Prospero).
pub fn os_uses_elf(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::Android
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::Horizon
            | TargetOperatingSystem::Orbis
            | TargetOperatingSystem::Prospero
    )
}

/// Returns true if the given OS uses Mach-O (macOS).
pub fn os_uses_macho(os: TargetOperatingSystem) -> bool {
    matches!(
        os,
        TargetOperatingSystem::MacOS
            | TargetOperatingSystem::IOS
            | TargetOperatingSystem::TvOS
            | TargetOperatingSystem::WatchOS
            | TargetOperatingSystem::VisionOS
    )
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
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::MacOS
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::FreeBSD
        ));
    }

    #[test]
    fn test_aarch64_cross_platform() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::MacOS
        ));
    }

    #[test]
    fn test_riscv_elf_like_only() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Linux
        ));
        assert!(!is_assembly_emission_supported(
            TargetArchitecture::Riscv64,
            TargetOperatingSystem::Windows
        ));
    }

    #[test]
    fn test_wasm_limited_os() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Wasm32,
            TargetOperatingSystem::Unknown
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Wasm32,
            TargetOperatingSystem::Wasi
        ));
        assert!(!is_assembly_emission_supported(
            TargetArchitecture::Wasm64,
            TargetOperatingSystem::Wasi
        ));
        assert!(!is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Wasi
        ));
    }

    #[test]
    fn test_unknown_os_fallback() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Unknown
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Unknown
        ));
    }

    #[test]
    fn test_apple_mobile_assembly_supported() {
        for os in [
            TargetOperatingSystem::IOS,
            TargetOperatingSystem::TvOS,
            TargetOperatingSystem::WatchOS,
            TargetOperatingSystem::VisionOS,
        ] {
            assert!(
                is_assembly_emission_supported(TargetArchitecture::X86_64, os),
                "{os}"
            );
            assert!(
                is_assembly_emission_supported(TargetArchitecture::Aarch64, os),
                "{os}"
            );
        }
    }

    #[test]
    fn test_android_is_elf_and_assembly_supported() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Android
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Android
        ));
        assert!(os_uses_elf(TargetOperatingSystem::Android));
    }

    #[test]
    fn test_apple_mobile_oses_use_macho() {
        for os in [
            TargetOperatingSystem::IOS,
            TargetOperatingSystem::TvOS,
            TargetOperatingSystem::WatchOS,
            TargetOperatingSystem::VisionOS,
        ] {
            assert!(os_uses_macho(os), "{os} should use Mach-O");
            assert!(!os_uses_elf(os), "{os} should not use ELF");
        }
    }

    #[test]
    fn test_unsupported_arch() {
        assert!(!is_assembly_emission_supported(
            TargetArchitecture::Arm32,
            TargetOperatingSystem::Linux
        ));
    }

    #[test]
    fn test_powerpc64_assembly_supported() {
        assert!(is_assembly_emission_supported(
            TargetArchitecture::PowerPC64,
            TargetOperatingSystem::Linux
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::PowerPC64,
            TargetOperatingSystem::Unknown
        ));
        assert!(is_assembly_emission_supported(
            TargetArchitecture::PowerPC64Be,
            TargetOperatingSystem::Linux
        ));
        assert!(!is_assembly_emission_supported(
            TargetArchitecture::PowerPC64Be,
            TargetOperatingSystem::Windows
        ));
    }

    #[test]
    fn test_os_uses_elf() {
        assert!(os_uses_elf(TargetOperatingSystem::Linux));
        assert!(os_uses_elf(TargetOperatingSystem::FreeBSD));
        assert!(os_uses_elf(TargetOperatingSystem::Redox));
        assert!(os_uses_elf(TargetOperatingSystem::Horizon));
        assert!(os_uses_elf(TargetOperatingSystem::Orbis));
        assert!(os_uses_elf(TargetOperatingSystem::Prospero));
        assert!(!os_uses_elf(TargetOperatingSystem::MacOS));
        assert!(!os_uses_elf(TargetOperatingSystem::Windows));
    }
}
