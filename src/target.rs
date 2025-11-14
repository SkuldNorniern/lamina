use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub architecture: TargetArchitecture,
    pub operating_system: TargetOperatingSystem,
}
impl Target {
    pub fn new(architecture: TargetArchitecture, operating_system: TargetOperatingSystem) -> Self {
        Self { architecture, operating_system }
    }
    pub fn from_str(target: &str) -> Self {
        let parts = target.split('_').collect::<Vec<&str>>();
        if parts.len() != 2 {
            return Self {
                architecture: TargetArchitecture::Unknown,
                operating_system: TargetOperatingSystem::Unknown,
            };
        }
        Self::new(
            TargetArchitecture::from_str(parts[0]),
            TargetOperatingSystem::from_str(parts[1])
        )
    }
    pub fn to_str(&self) -> String {
        format!("{}_{}", self.architecture, self.operating_system)
    }
    pub fn detect_host() -> Self {
        let arch = architecture_name();
        let os = detect_host_os();
        Self::new(
            TargetArchitecture::from_str(arch),
            TargetOperatingSystem::from_str(os)
        )
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.architecture, self.operating_system)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArchitecture {
    X86_64,
    Aarch64,
    Arm32,
    Riscv32,
    Riscv64,
    Riscv128,
    Wasm32,
    Wasm64,
    Lisa,
    Unknown,
}

impl TargetArchitecture {
    pub fn from_str(s: &str) -> Self {
        match s {
            "x86_64" => Self::X86_64,
            "aarch64" => Self::Aarch64,
            "arm32" => Self::Arm32,
            "riscv32" => Self::Riscv32,
            "riscv64" => Self::Riscv64,
            "riscv128" => Self::Riscv128,
            "wasm32" => Self::Wasm32,
            "wasm64" => Self::Wasm64,
            "wasm" => Self::Wasm32,  // fallback for generic "wasm"
            "lisa" => Self::Lisa,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOperatingSystem {
    Linux,
    MacOS,
    Windows,
    BSD,
    Redox,
    Artery,
    Unknown,
}

impl TargetOperatingSystem {
    pub fn from_str(s: &str) -> Self {
        match s {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            "bsd" => Self::BSD,
            "redox" => Self::Redox,
            "artery" => Self::Artery,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for TargetArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Arm32 => "arm32",
            Self::Riscv32 => "riscv32",
            Self::Riscv64 => "riscv64",
            Self::Riscv128 => "riscv128",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::Lisa => "lisa",
            Self::Unknown => "unknown",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for TargetOperatingSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Linux => "linux",
            Self::MacOS => "macos",
            Self::Windows => "windows",
            Self::BSD => "bsd",
            Self::Redox => "redox",
            Self::Artery => "artery",
            Self::Unknown => "unknown",
        };
        write!(f, "{}", s)
    }
}


pub const HOST_ARCH_LIST: &[&str] = &[
    "x86_64_unknown",
    "x86_64_linux",
    "x86_64_windows",
    "x86_64_macos",
    "aarch64_unknown",
    "aarch64_macos",
    "aarch64_linux",
    "aarch64_windows",
    "wasm32_unknown",
    "wasm64_unknown",
    "riscv32_unknown",
    "riscv64_unknown",
    "riscv128_unknown",
];

/// Get the host architecture name.
///
/// Returns the architecture name as a string: "x86_64", "aarch64", etc.
pub fn architecture_name() -> &'static str {
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
    #[cfg(target_arch = "riscv128")]
    return "riscv128";

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
    #[cfg(any(target_os = "freebsd", target_os = "openbsd", target_os = "netbsd", target_os = "dragonfly"))]
    return "bsd";
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
/// - riscv32_unknown, riscv64_unknown, riscv128_unknown
///
/// Falls back to "x86_64_unknown" for unsupported combinations.
///
/// # Deprecated
/// This function is deprecated. Use `detect_host().to_str()` instead for a more structured approach.
#[deprecated(since = "0.0.8", note = "Use `detect_host().to_str()` instead")]
pub fn detect_host_architecture() -> &'static str {
    let arch = architecture_name();
    let os = detect_host_os();
    // For backward compatibility, return the combined format
    // This will be removed once the deprecation period is over
    match (arch, os) {
        ("x86_64", "linux") => "x86_64_linux",
        ("x86_64", "macos") => "x86_64_macos",
        ("x86_64", "windows") => "x86_64_windows",
        ("x86_64", "bsd") => "x86_64_bsd",
        ("aarch64", "linux") => "aarch64_linux",
        ("aarch64", "macos") => "aarch64_macos",
        ("aarch64", "windows") => "aarch64_windows",
        ("aarch64", "bsd") => "aarch64_bsd",
        ("wasm32", _) => "wasm32_unknown",
        ("wasm64", _) => "wasm64_unknown",
        ("riscv32", _) => "riscv32_unknown",
        ("riscv64", _) => "riscv64_unknown",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_host_consistency() {
        // Test that the deprecated function produces the same result as the new structured approach
        let new_result = Target::detect_host().to_str();
        let old_result = detect_host_architecture();

        assert_eq!(new_result, old_result,
            "detect_host().to_str() and detect_host_architecture() should return the same result");
    }

    #[test]
    fn test_from_str_roundtrip() {
        let host = Target::detect_host();
        let str_repr = host.to_str();
        let parsed_back = Target::from_str(&str_repr);

        assert_eq!(host, parsed_back,
            "from_str/to_str should be a roundtrip operation");
    }

    #[test]
    fn test_detect_functions_consistency() {
        // Test that detect_host uses the same logic as the separate functions
        let combined = Target::detect_host();
        let arch_str = architecture_name();
        let os_str = detect_host_os();

        assert_eq!(combined.architecture, TargetArchitecture::from_str(arch_str));
        assert_eq!(combined.operating_system, TargetOperatingSystem::from_str(os_str));
    }

    #[test]
    fn test_deprecated_function_format() {
        // Test that the deprecated function returns the expected combined format
        let result = detect_host_architecture();
        assert!(result.contains('_'), "Should contain underscore separating arch and os");
        assert!(!result.is_empty(), "Should not be empty");
    }

    #[test]
    #[cfg(target_os = "freebsd")]
    fn test_freebsd_detection() {
        let host = Target::detect_host();
        assert_eq!(host.operating_system, TargetOperatingSystem::BSD,
            "FreeBSD should be detected as BSD, not Unknown");
    }

    #[test]
    #[cfg(target_os = "openbsd")]
    fn test_openbsd_detection() {
        let host = Target::detect_host();
        assert_eq!(host.operating_system, TargetOperatingSystem::BSD,
            "OpenBSD should be detected as BSD, not Unknown");
    }

    #[test]
    #[cfg(target_os = "netbsd")]
    fn test_netbsd_detection() {
        let host = Target::detect_host();
        assert_eq!(host.operating_system, TargetOperatingSystem::BSD,
            "NetBSD should be detected as BSD, not Unknown");
    }

    #[test]
    #[cfg(target_os = "dragonfly")]
    fn test_dragonfly_detection() {
        let host = Target::detect_host();
        assert_eq!(host.operating_system, TargetOperatingSystem::BSD,
            "DragonFly BSD should be detected as BSD, not Unknown");
    }
}