//! Target architecture and operating system definitions.

use std::fmt;
use std::str::FromStr;

use crate::detection::{detect_host_architecture_only, detect_host_os};

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

    /// Detect the host system's target.
    ///
    /// # Examples
    ///
    /// ```
    /// use lamina_platform::Target;
    /// let host = Target::detect_host();
    /// println!("Host target: {}", host);
    /// ```
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
        let parts: Vec<&str> = target.rsplitn(2, '_').collect();
        if parts.len() != 2 {
            return Err("Invalid target format: expected 'architecture_os'");
        }
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
    Arx64,
    Arm32,
    Riscv32,
    Riscv64,
    #[cfg(feature = "nightly")]
    Riscv128,
    Wasm32,
    Wasm64,
    PowerPC64,
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
            "arx64" => Ok(Self::Arx64),
            "arm32" => Ok(Self::Arm32),
            "riscv32" => Ok(Self::Riscv32),
            "riscv64" => Ok(Self::Riscv64),
            #[cfg(feature = "nightly")]
            "riscv128" => Ok(Self::Riscv128),
            "wasm32" => Ok(Self::Wasm32),
            "wasm64" => Ok(Self::Wasm64),
            "wasm" => Ok(Self::Wasm32),
            "ppc64" | "powerpc64" | "ppc64le" => Ok(Self::PowerPC64),
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
    Android,
    IOS,
    TvOS,
    WatchOS,
    VisionOS,
    FreeBSD,
    OpenBSD,
    NetBSD,
    DragonFly,
    Redox,
    Horizon,
    Psp,
    Psx,
    Vita,
    Orbis,
    Prospero,
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
            "android" => Ok(Self::Android),
            "ios" => Ok(Self::IOS),
            "tvos" => Ok(Self::TvOS),
            "watchos" => Ok(Self::WatchOS),
            "visionos" => Ok(Self::VisionOS),
            "freebsd" => Ok(Self::FreeBSD),
            "openbsd" => Ok(Self::OpenBSD),
            "netbsd" => Ok(Self::NetBSD),
            "dragonfly" => Ok(Self::DragonFly),
            "redox" => Ok(Self::Redox),
            "horizon" => Ok(Self::Horizon),
            "psp" => Ok(Self::Psp),
            "psx" => Ok(Self::Psx),
            "vita" => Ok(Self::Vita),
            "orbis" | "ps4" => Ok(Self::Orbis),
            "prospero" | "ps5" => Ok(Self::Prospero),
            "unknown" => Ok(Self::Unknown),
            #[cfg(feature = "nightly")]
            "artery" => Ok(Self::Artery),
            "bsd" => Ok(Self::FreeBSD),
            _ => Err("Unknown operating system"),
        }
    }
}

impl fmt::Display for TargetArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Arx64 => "arx64",
            Self::Arm32 => "arm32",
            Self::Riscv32 => "riscv32",
            Self::Riscv64 => "riscv64",
            #[cfg(feature = "nightly")]
            Self::Riscv128 => "riscv128",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::PowerPC64 => "ppc64le",
            #[cfg(feature = "nightly")]
            Self::Lisa => "lisa",
            Self::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for TargetOperatingSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Linux => "linux",
            Self::MacOS => "macos",
            Self::Windows => "windows",
            Self::Android => "android",
            Self::IOS => "ios",
            Self::TvOS => "tvos",
            Self::WatchOS => "watchos",
            Self::VisionOS => "visionos",
            Self::FreeBSD => "freebsd",
            Self::OpenBSD => "openbsd",
            Self::NetBSD => "netbsd",
            Self::DragonFly => "dragonfly",
            Self::Redox => "redox",
            Self::Horizon => "horizon",
            Self::Psp => "psp",
            Self::Psx => "psx",
            Self::Vita => "vita",
            Self::Orbis => "orbis",
            Self::Prospero => "prospero",
            #[cfg(feature = "nightly")]
            Self::Artery => "artery",
            Self::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

pub const HOST_ARCH_LIST: &[&str] = &[
    "x86_64_unknown",
    "x86_64_linux",
    "x86_64_windows",
    "x86_64_macos",
    "x86_64_android",
    "x86_64_ios",
    "x86_64_tvos",
    "x86_64_watchos",
    "aarch64_unknown",
    "aarch64_macos",
    "aarch64_linux",
    "aarch64_windows",
    "aarch64_android",
    "aarch64_ios",
    "aarch64_tvos",
    "aarch64_watchos",
    "aarch64_visionos",
    "aarch64_horizon",
    "arm32_android",
    "arm32_ios",
    "arm32_watchos",
    "arm32_vita",
    "x86_64_orbis",
    "x86_64_prospero",
    "arx64_unknown",
    "wasm32_unknown",
    "wasm64_unknown",
    "riscv32_unknown",
    "riscv64_unknown",
    #[cfg(feature = "nightly")]
    "riscv128_unknown",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_mobile_and_console_targets() {
        assert_eq!(
            Target::from_str("aarch64_ios").unwrap(),
            Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::IOS)
        );
        assert_eq!(
            Target::from_str("aarch64_horizon").unwrap(),
            Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Horizon)
        );
        assert_eq!(
            Target::from_str("arm32_vita").unwrap(),
            Target::new(TargetArchitecture::Arm32, TargetOperatingSystem::Vita)
        );
    }

    #[test]
    fn displays_known_mobile_and_console_os_names() {
        assert_eq!(TargetOperatingSystem::VisionOS.to_string(), "visionos");
        assert_eq!(TargetOperatingSystem::Psp.to_string(), "psp");
        assert_eq!(TargetOperatingSystem::Psx.to_string(), "psx");
    }
}
