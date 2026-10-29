//! Target architecture and operating system definitions.

use std::{fmt, str::FromStr};

use crate::detection::{detect_host_architecture_only, detect_host_os};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub architecture: TargetArchitecture,
    pub operating_system: TargetOperatingSystem,
}

/// Width of a data pointer. Pointer size and alignment derive from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerWidth {
    Bits32,
    Bits64,
}

impl PointerWidth {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Bits32 => 4,
            Self::Bits64 => 8,
        }
    }
}

impl Target {
    pub fn new(architecture: TargetArchitecture, operating_system: TargetOperatingSystem) -> Self {
        Self {
            architecture,
            operating_system,
        }
    }

    pub fn pointer_width(&self) -> PointerWidth {
        self.architecture.pointer_width()
    }

    pub fn has_64bit_int_registers(&self) -> bool {
        self.architecture.has_64bit_int_registers()
    }

    pub fn gpu_triple(&self) -> Option<&'static str> {
        match (self.architecture, self.operating_system) {
            (TargetArchitecture::Nvptx64, TargetOperatingSystem::Cuda) => {
                Some("nvptx64-nvidia-cuda")
            }
            (TargetArchitecture::Amdgcn, TargetOperatingSystem::AmdHsa) => {
                Some("amdgcn-amd-amdhsa")
            }
            _ => None,
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
        let target = TARGET_ALIASES
            .iter()
            .find(|(alias, _)| *alias == target)
            .map_or(target, |(_, canonical)| canonical);
        let parts: Vec<&str> = target.rsplitn(2, '_').collect();
        if parts.len() != 2 {
            return Err("Invalid target format: expected 'architecture_os'");
        }
        Ok(Self::new(
            TargetArchitecture::from_str(parts[1])?,
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
    Armv7,
    I686,
    Riscv32,
    Riscv64,
    #[cfg(feature = "nightly")]
    Riscv128,
    Wasm32,
    Wasm64,
    PowerPC64,
    LoongArch64,
    Nvptx64,
    Amdgcn,
    #[cfg(feature = "nightly")]
    Lisa,
    Unknown,
}

impl TargetArchitecture {
    /// Exhaustive on purpose: a new architecture does not compile until its pointer
    /// width is chosen.
    pub const fn pointer_width(self) -> PointerWidth {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Arx64
            | Self::Riscv64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::Wasm64
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => PointerWidth::Bits64,
            Self::Arm32 | Self::Armv7 | Self::I686 | Self::Riscv32 | Self::Wasm32 => {
                PointerWidth::Bits32
            }
            #[cfg(feature = "nightly")]
            Self::Riscv128 => {
                panic!("rv128 needs 16-byte pointers, and PointerWidth has no 128-bit variant yet")
            }
            #[cfg(feature = "nightly")]
            Self::Lisa => PointerWidth::Bits64,
        }
    }

    /// Whether an integer register holds 64 bits. Decides whether an `i64` is one
    /// register or a pair of 32-bit words, which is a different question from
    /// [`TargetArchitecture::pointer_width`]: wasm32 has 32-bit pointers and 64-bit
    /// integer registers. Exhaustive on purpose, like `pointer_width`.
    pub const fn has_64bit_int_registers(self) -> bool {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Arx64
            | Self::Riscv64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::Wasm32
            | Self::Wasm64
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => true,
            Self::Arm32 | Self::Armv7 | Self::I686 | Self::Riscv32 => false,
            #[cfg(feature = "nightly")]
            Self::Riscv128 | Self::Lisa => true,
        }
    }
}

impl FromStr for TargetArchitecture {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nvptx64" => Ok(Self::Nvptx64),
            "amdgcn" => Ok(Self::Amdgcn),
            "x86_64" => Ok(Self::X86_64),
            "aarch64" => Ok(Self::Aarch64),
            "arx64" => Ok(Self::Arx64),
            "arm32" => Ok(Self::Arm32),
            "armv7" | "armv7a" | "armv7hf" => Ok(Self::Armv7),
            "i686" | "i386" | "x86" => Ok(Self::I686),
            "riscv32" => Ok(Self::Riscv32),
            "riscv64" | "riscv64gc" => Ok(Self::Riscv64),
            #[cfg(feature = "nightly")]
            "riscv128" => Ok(Self::Riscv128),
            "wasm32" => Ok(Self::Wasm32),
            "wasm64" => Ok(Self::Wasm64),
            "wasm" => Ok(Self::Wasm32),
            "ppc64le" => Ok(Self::PowerPC64),
            "ppc64" => Err(
                "Architecture 'ppc64' was removed\n  Hint: lamina only supports little-endian POWER. Write 'ppc64le' (e.g. ppc64le_linux)",
            ),
            "loongarch64" | "loong64" => Ok(Self::LoongArch64),
            #[cfg(feature = "nightly")]
            "lisa" => Ok(Self::Lisa),
            _ => Err("Unknown architecture"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOperatingSystem {
    Cuda,
    AmdHsa,
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
    Wasi,
    #[cfg(feature = "nightly")]
    Artery,
    Unknown,
}

impl FromStr for TargetOperatingSystem {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cuda" => Ok(Self::Cuda),
            "amdhsa" => Ok(Self::AmdHsa),
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
            "wasi" => Ok(Self::Wasi),
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
            Self::Nvptx64 => "nvptx64",
            Self::Amdgcn => "amdgcn",
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Arx64 => "arx64",
            Self::Arm32 => "arm32",
            Self::Armv7 => "armv7",
            Self::I686 => "i686",
            Self::Riscv32 => "riscv32",
            Self::Riscv64 => "riscv64",
            #[cfg(feature = "nightly")]
            Self::Riscv128 => "riscv128",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::PowerPC64 => "ppc64le",
            Self::LoongArch64 => "loongarch64",
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
            Self::Cuda => "cuda",
            Self::AmdHsa => "amdhsa",
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
            Self::Wasi => "wasi",
            #[cfg(feature = "nightly")]
            Self::Artery => "artery",
            Self::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

const TARGET_ALIASES: &[(&str, &str)] = &[
    ("riscv64_unknown", "riscv64_linux"),
    ("riscv64gc_linux", "riscv64_linux"),
];

pub const HOST_ARCH_LIST: &[&str] = &[
    "nvptx64_cuda",
    "amdgcn_amdhsa",
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
    "wasm32_wasi",
    "riscv32_unknown",
    "riscv32_linux",
    "riscv64_linux",
    "riscv64_unknown",
    "riscv64gc_linux",
    "ppc64le_linux",
    "ppc64le_unknown",
    "loongarch64_linux",
    "i686_linux",
    "armv7_linux",
    #[cfg(feature = "nightly")]
    "riscv128_unknown",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_targets_round_trip() {
        for (name, triple) in [
            ("nvptx64_cuda", "nvptx64-nvidia-cuda"),
            ("amdgcn_amdhsa", "amdgcn-amd-amdhsa"),
        ] {
            let target: Target = name.parse().unwrap();
            assert_eq!(target.to_string(), name);
            assert_eq!(target.gpu_triple(), Some(triple));
            assert!(HOST_ARCH_LIST.contains(&name));
        }
    }

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
    fn rejects_big_endian_powerpc64_name() {
        assert!(Target::from_str("powerpc64_linux").is_err());
        assert!(
            Target::from_str("ppc64_linux")
                .unwrap_err()
                .contains("ppc64le")
        );
        assert_eq!(
            Target::from_str("ppc64le_linux").unwrap(),
            Target::new(TargetArchitecture::PowerPC64, TargetOperatingSystem::Linux)
        );
        assert!(HOST_ARCH_LIST.contains(&"ppc64le_linux"));
    }

    #[test]
    fn wasm_targets_round_trip() {
        for name in ["wasm32_wasi", "wasm32_unknown", "wasm64_unknown"] {
            let target: Target = name.parse().unwrap();
            assert_eq!(target.to_string(), name);
            assert!(HOST_ARCH_LIST.contains(&name));
        }
    }

    #[test]
    fn displays_known_mobile_and_console_os_names() {
        assert_eq!(TargetOperatingSystem::VisionOS.to_string(), "visionos");
        assert_eq!(TargetOperatingSystem::Psp.to_string(), "psp");
        assert_eq!(TargetOperatingSystem::Psx.to_string(), "psx");
    }

    #[test]
    fn riscv64_aliases_resolve_to_riscv64_linux() {
        let linux = Target::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux);
        for name in ["riscv64_linux", "riscv64_unknown", "riscv64gc_linux"] {
            let target: Target = name.parse().unwrap();
            assert_eq!(target, linux);
            assert!(HOST_ARCH_LIST.contains(&name));
        }
        assert_eq!(linux.to_string(), "riscv64_linux");
    }

    #[test]
    fn pointer_width_follows_the_architecture() {
        let width = |name: &str| name.parse::<Target>().unwrap().pointer_width();
        assert_eq!(width("x86_64_linux"), PointerWidth::Bits64);
        assert_eq!(width("aarch64_linux"), PointerWidth::Bits64);
        assert_eq!(width("riscv64_unknown"), PointerWidth::Bits64);
        assert_eq!(width("riscv32_unknown"), PointerWidth::Bits32);
        assert_eq!(width("arm32_vita"), PointerWidth::Bits32);
        assert_eq!(width("wasm32_unknown"), PointerWidth::Bits32);
        assert_eq!(width("wasm64_unknown"), PointerWidth::Bits64);
        assert_eq!(PointerWidth::Bits32.bytes(), 4);
        assert_eq!(PointerWidth::Bits64.bytes(), 8);
    }

    #[test]
    fn int_registers_are_not_pointers() {
        let regs = |name: &str| name.parse::<Target>().unwrap().has_64bit_int_registers();
        assert!(regs("wasm32_wasi"));
        assert!(regs("wasm64_unknown"));
        assert!(regs("x86_64_linux"));
        assert!(!regs("riscv32_linux"));
        assert!(!regs("i686_linux"));
        assert!(!regs("armv7_linux"));
    }
}
