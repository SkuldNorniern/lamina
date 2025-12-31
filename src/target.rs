use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub architecture: TargetArchitecture,
    pub operating_system: TargetOperatingSystem,
}
impl Target {
    pub fn new(architecture: TargetArchitecture, operating_system: TargetOperatingSystem) -> Self {
        Self {
            architecture,
            operating_system,
        }
    }

    pub fn to_str(&self) -> String {
        format!("{}_{}", self.architecture, self.operating_system)
    }

    pub fn detect_host() -> Self {
        let arch = detect_host_architecture_only();
        let os = detect_host_os();
        Self::new(
            TargetArchitecture::from_str(arch).unwrap_or(TargetArchitecture::Unknown),
            TargetOperatingSystem::from_str(os).unwrap_or(TargetOperatingSystem::Unknown),
        )
    }
}

impl FromStr for Target {
    type Err = &'static str;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        // Split from the right to handle architectures with underscores (e.g., "x86_64")
        let parts: Vec<&str> = target.rsplitn(2, '_').collect();
        if parts.len() != 2 {
            return Err("Invalid target format: expected 'architecture_os'");
        }
        // parts[0] is the OS (rightmost part), parts[1] is the architecture (everything else)
        Ok(Self::new(
            TargetArchitecture::from_str(parts[1]).map_err(|_| "Invalid architecture")?,
            TargetOperatingSystem::from_str(parts[0]).map_err(|_| "Invalid operating system")?,
        ))
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
    #[cfg(feature = "nightly")]
    Riscv128,
    Wasm32,
    Wasm64,
    #[cfg(feature = "nightly")]
    Lisa,
    Unknown,
}

impl FromStr for TargetArchitecture {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "x86_64" => Ok(Self::X86_64),
            "aarch64" => Ok(Self::Aarch64),
            "arm32" => Ok(Self::Arm32),
            "riscv32" => Ok(Self::Riscv32),
            "riscv64" => Ok(Self::Riscv64),
            #[cfg(feature = "nightly")]
            "riscv128" => Ok(Self::Riscv128),
            "wasm32" => Ok(Self::Wasm32),
            "wasm64" => Ok(Self::Wasm64),
            "wasm" => Ok(Self::Wasm32),
            // default to 32-bit for backward compatibility
            #[cfg(feature = "nightly")]
            "lisa" => Ok(Self::Lisa),
            _ => Err("Unknown architecture"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOperatingSystem {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
    OpenBSD,
    NetBSD,
    DragonFly,
    Redox,
    #[cfg(feature = "nightly")]
    Artery,
    Unknown,
}

impl FromStr for TargetOperatingSystem {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linux" => Ok(Self::Linux),
            "macos" => Ok(Self::MacOS),
            "windows" => Ok(Self::Windows),
            "freebsd" => Ok(Self::FreeBSD),
            "openbsd" => Ok(Self::OpenBSD),
            "netbsd" => Ok(Self::NetBSD),
            "dragonfly" => Ok(Self::DragonFly),
            "redox" => Ok(Self::Redox),
            #[cfg(feature = "nightly")]
            "artery" => Ok(Self::Artery),
            // Keep backward compatibility
            "bsd" => Ok(Self::FreeBSD), // default to FreeBSD for generic "bsd"
            _ => Err("Unknown operating system"),
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
            #[cfg(feature = "nightly")]
            Self::Riscv128 => "riscv128",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            #[cfg(feature = "nightly")]
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
            Self::FreeBSD => "freebsd",
            Self::OpenBSD => "openbsd",
            Self::NetBSD => "netbsd",
            Self::DragonFly => "dragonfly",
            Self::Redox => "redox",
            #[cfg(feature = "nightly")]
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
    #[cfg(feature = "nightly")]
    "riscv128_unknown",
];

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
}
