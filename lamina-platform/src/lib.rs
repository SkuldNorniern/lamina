//! lamina-platform - Platform and target detection
//!
//! This crate provides target architecture and operating system detection
//! and definitions. It's separated from the main lamina crate to avoid
//! dependency cycles with ras and other components.
//!
//! ## Modules
//!
//! - [`arch_table`] - the architecture table: every property of every architecture
//! - [`target`] - targets: architecture, environment, operating system and variant
//! - [`environment`] - the C runtime and ABI axis
//! - [`cpu_level`] - x86-64 microarchitecture levels (v1 to v4)
//! - [`arch_variant`] - sub-architecture variants for the other families
//! - [`detection`] - Host system detection functions
//! - [`simd`] - SIMD capabilities detection (nightly feature)

pub mod arch_table;
pub mod arch_variant;
pub mod cpu_level;
pub mod detection;
pub mod environment;
pub mod shared;
pub mod target;
pub mod vector;

#[cfg(feature = "nightly")]
pub mod simd;

// Re-export main types
pub use arch_table::{
    ArchSpec, Endianness, LinuxInterpreters, OutputArtifact, PointerWidth, TargetArchitecture,
    TargetDataLayout, ToolchainTriples, VaListAbi,
};
pub use arch_variant::{
    Aarch64Level, ArchVariant, Armv7Profile, Armv7Variant, PowerLevel, RiscvIsa, VariantAxis,
};
pub use cpu_level::X86CpuLevel;
pub use detection::{cpu_count, detect_host_architecture_only, detect_host_os};
pub use environment::{CompilerRuntime, Libc, TargetEnvironment};
pub use shared::{MemoryOrdering, SourceLoc, TlsModel};
pub use target::{
    Target, TargetOperatingSystem, default_environment, host_arch_list, parse_arch_token,
    parse_architecture, target_alias_list,
};
pub use vector::any_vector_register_bits;

#[cfg(feature = "nightly")]
pub use simd::{ArmSimdExtension, RiscvSimdExtension, SimdCapabilities, X86SimdExtension};

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_from_str_roundtrip() {
        let host = Target::detect_host().expect("host is a supported target");
        let str_repr = host.to_str();
        let parsed_back = Target::from_str(&str_repr).expect("Valid target should parse");

        assert_eq!(
            host, parsed_back,
            "from_str/to_str should be a roundtrip operation"
        );
    }

    #[test]
    fn test_detect_functions_consistency() {
        // Test that detect_host uses the same logic as the separate functions
        let combined = Target::detect_host().expect("host is a supported target");
        let arch_str = detect_host_architecture_only();
        let os_str = detect_host_os();

        assert_eq!(
            combined.architecture,
            TargetArchitecture::from_str(arch_str).expect("host arch parses")
        );
        assert_eq!(
            combined.operating_system,
            TargetOperatingSystem::from_str(os_str).expect("host os parses")
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_target_simd_capabilities() {
        let target = Target::detect_host().expect("host is a supported target");
        let caps = target.simd_capabilities();
        // At minimum, we should know if SIMD is supported or not
        assert!(caps.max_vector_width <= 2048);
    }
}
