//! lamina-platform - Platform and target detection
//!
//! This crate provides target architecture and operating system detection
//! and definitions. It's separated from the main lamina crate to avoid
//! dependency cycles with ras and other components.
//!
//! ## Modules
//!
//! - [`target`] - Target architecture and operating system definitions
//! - [`detection`] - Host system detection functions
//! - [`simd`] - SIMD capabilities detection (nightly feature)

pub mod target;
pub mod detection;

#[cfg(feature = "nightly")]
pub mod simd;

// Re-export main types
pub use target::{Target, TargetArchitecture, TargetOperatingSystem, HOST_ARCH_LIST};
pub use detection::{
    detect_host_architecture, detect_host_architecture_only, detect_host_os,
};

#[cfg(feature = "nightly")]
pub use simd::{
    ArmSimdExtension, RiscvSimdExtension, SimdCapabilities, X86SimdExtension,
};


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_host_consistency() {
        // Test that the deprecated function produces the same result as the new structured approach
        let new_result = Target::detect_host().to_str();
        let old_result = detect_host_architecture();

        assert_eq!(
            new_result, old_result,
            "detect_host().to_str() and detect_host_architecture() should return the same result"
        );
    }

    #[test]
    fn test_from_str_roundtrip() {
        let host = Target::detect_host();
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
        let combined = Target::detect_host();
        let arch_str = detect_host_architecture_only();
        let os_str = detect_host_os();

        assert_eq!(
            combined.architecture,
            TargetArchitecture::from_str(arch_str).unwrap_or(TargetArchitecture::Unknown)
        );
        assert_eq!(
            combined.operating_system,
            TargetOperatingSystem::from_str(os_str).unwrap_or(TargetOperatingSystem::Unknown)
        );
    }

    #[test]
    fn test_deprecated_function_format() {
        // Test that the deprecated function returns the expected combined format
        let result = detect_host_architecture();
        assert!(
            result.contains('_'),
            "Should contain underscore separating arch and os"
        );
        assert!(!result.is_empty(), "Should not be empty");
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_target_simd_capabilities() {
        let target = Target::detect_host();
        let caps = target.simd_capabilities();
        // At minimum, we should know if SIMD is supported or not
        assert!(caps.max_vector_width >= 0);
    }
}
