//! Host system detection functions.

use crate::target::{TargetArchitecture, TargetOperatingSystem};

/// Detect the host system's architecture only.
///
/// Returns a string representing the detected architecture: "x86_64", "aarch64", etc.
///
/// Falls back to "x86_64" if detection fails.
pub fn detect_host_architecture_only() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    return "x86_64";
    #[cfg(target_arch = "aarch64")]
    return "aarch64";
    #[cfg(target_arch = "wasm32")]
    return "wasm32";
    #[cfg(target_arch = "wasm64")]
    return "wasm64";
    #[cfg(target_arch = "riscv32")]
    return "riscv32";
    #[cfg(target_arch = "riscv64")]
    return "riscv64";

    // Default fallback
    #[allow(unreachable_code)]
    "x86_64"
}

/// Detect the host system's operating system only.
///
/// Returns a string representing the detected operating system: "linux", "macos", "windows", etc.
///
/// Falls back to "unknown" if detection fails.
pub fn detect_host_os() -> &'static str {
    #[cfg(target_os = "linux")]
    return "linux";
    #[cfg(target_os = "macos")]
    return "macos";
    #[cfg(target_os = "windows")]
    return "windows";
    #[cfg(target_os = "freebsd")]
    return "freebsd";
    #[cfg(target_os = "openbsd")]
    return "openbsd";
    #[cfg(target_os = "netbsd")]
    return "netbsd";
    #[cfg(target_os = "dragonfly")]
    return "dragonfly";
    #[cfg(target_os = "redox")]
    return "redox";

    // Default fallback
    #[allow(unreachable_code)]
    "unknown"
}


/// Detect the host system's architecture and operating system combination.
///
/// Returns a string representing the detected architecture and host system combination.
///
/// # Supported Targets
/// - x86_64_unknown, x86_64_linux, x86_64_windows, x86_64_macos
/// - aarch64_unknown, aarch64_macos, aarch64_linux, aarch64_windows
/// - wasm32_unknown, wasm64_unknown
/// - riscv32_unknown, riscv64_unknown
/// - riscv128_unknown (nightly feature only)
///
/// Falls back to "x86_64_unknown" for unsupported combinations.
///
/// # Deprecated
/// This function is deprecated. Use `detect_host().to_str()` instead for a more structured approach.
#[deprecated(since = "0.0.8", note = "Use `detect_host().to_str()` instead")]
pub fn detect_host_architecture() -> &'static str {
    let arch = detect_host_architecture_only();
    let os = detect_host_os();
    // For backward compatibility, return the combined format
    // This will be removed once the deprecation period is over
    match (arch, os) {
        ("x86_64", "linux") => "x86_64_linux",
        ("x86_64", "macos") => "x86_64_macos",
        ("x86_64", "windows") => "x86_64_windows",
        ("aarch64", "linux") => "aarch64_linux",
        ("aarch64", "macos") => "aarch64_macos",
        ("aarch64", "windows") => "aarch64_windows",
        ("wasm32", _) => "wasm32_unknown",
        ("wasm64", _) => "wasm64_unknown",
        ("riscv32", _) => "riscv32_unknown",
        ("riscv64", _) => "riscv64_unknown",
        #[cfg(feature = "nightly")]
        ("riscv128", _) => "riscv128_unknown",
        _ => {
            // Fallback for unsupported combinations
            match arch {
                "x86_64" => "x86_64_unknown",
                "aarch64" => "aarch64_unknown",
                _ => "x86_64_unknown",
            }
        }
    }
}

