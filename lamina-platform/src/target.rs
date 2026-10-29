//! Target architecture and operating system definitions.

use std::{fmt, str::FromStr};

use crate::arch_variant::{
    Aarch64Level, ArchVariant, Armv7Profile, Armv7Variant, PowerLevel, RiscvIsa,
};
use crate::detection::{detect_host_architecture_only, detect_host_os};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub architecture: TargetArchitecture,
    pub operating_system: TargetOperatingSystem,
    /// Sub-architecture the target string selected. An added axis: every spelling that
    /// names only the architecture gets [`TargetArchitecture::default_variant`], which is
    /// what that spelling has always meant.
    pub variant: ArchVariant,
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

/// Byte order of multi-byte values in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    Little,
    Big,
}

/// What a finished compilation of this target is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputArtifact {
    /// Object files that a linker turns into an executable.
    Executable,
    /// A self-contained module, assembled but never linked.
    Module,
    /// Assembly text handed on as-is.
    Text,
}

/// The facts a target-independent lowering needs to turn IR into MIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetDataLayout {
    pub pointer_width: PointerWidth,
    pub int_register_bits: u32,
    pub endianness: Endianness,
}

impl TargetDataLayout {
    pub const fn has_64bit_int_registers(self) -> bool {
        self.int_register_bits >= 64
    }
}

impl Target {
    pub fn new(architecture: TargetArchitecture, operating_system: TargetOperatingSystem) -> Self {
        Self {
            architecture,
            operating_system,
            variant: architecture.default_variant(),
        }
    }

    pub fn with_variant(mut self, variant: ArchVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Architecture half of the target string, carrying the variant when it is not the
    /// default, so `Display` and `FromStr` stay inverses.
    pub fn arch_token(&self) -> String {
        if self.variant == self.architecture.default_variant() {
            return self.architecture.to_string();
        }
        match self.variant {
            ArchVariant::Riscv(isa) => isa.march(),
            ArchVariant::None
            | ArchVariant::Armv7(_)
            | ArchVariant::Aarch64(_)
            | ArchVariant::PowerPC64(_) => VARIANT_SPELLINGS
                .iter()
                .find(|(_, arch, variant)| *arch == self.architecture && *variant == self.variant)
                .map_or_else(
                    || self.architecture.to_string(),
                    |(name, _, _)| (*name).to_string(),
                ),
        }
    }

    pub fn pointer_width(&self) -> PointerWidth {
        self.architecture.pointer_width()
    }

    pub fn has_64bit_int_registers(&self) -> bool {
        self.architecture.has_64bit_int_registers()
    }

    pub fn endianness(&self) -> Endianness {
        self.architecture.endianness()
    }

    pub fn stack_alignment_at_call(&self) -> usize {
        self.architecture.stack_alignment_at_call()
    }

    pub fn honors_argument_alignment(&self) -> bool {
        self.architecture.honors_argument_alignment()
    }

    pub fn output_artifact(&self) -> OutputArtifact {
        self.architecture.output_artifact()
    }

    pub fn data_layout(&self) -> TargetDataLayout {
        self.architecture.data_layout()
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
        format!("{}_{}", self.arch_token(), self.operating_system)
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
        let (architecture, variant) = parse_arch_token(parts[1])?;
        let os = TargetOperatingSystem::from_str(parts[0])?;
        Ok(Self::new(architecture, os).with_variant(variant))
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.arch_token(), self.operating_system)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArchitecture {
    X86_64,
    Aarch64,
    Arx64,
    Arm32,
    Armv7,
    /// ARMv6-M, Thumb-1 only: Cortex-M0 and M0+.
    Thumbv6m,
    /// ARMv7-M, Thumb-2: Cortex-M3, M4 and M7.
    Thumbv7m,
    I686,
    Riscv32,
    Riscv64,
    #[cfg(feature = "nightly")]
    Riscv128,
    Wasm32,
    Wasm64,
    PowerPC64,
    LoongArch64,
    S390x,
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
            | Self::S390x
            | Self::Wasm64
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => PointerWidth::Bits64,
            Self::Arm32
            | Self::Armv7
            | Self::Thumbv6m
            | Self::Thumbv7m
            | Self::I686
            | Self::Riscv32
            | Self::Wasm32 => PointerWidth::Bits32,
            #[cfg(feature = "nightly")]
            Self::Riscv128 => {
                panic!("rv128 needs 16-byte pointers, and PointerWidth has no 128-bit variant yet")
            }
            #[cfg(feature = "nightly")]
            Self::Lisa => PointerWidth::Bits64,
        }
    }

    /// Widest integer one register holds. A wider integer is lowered to a tuple of
    /// registers or to a libcall. Not the pointer width: wasm32 has 32-bit pointers and
    /// 64-bit integer registers. Exhaustive on purpose, like `pointer_width`.
    pub const fn max_int_register_bits(self) -> u32 {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Arx64
            | Self::Riscv64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::S390x
            | Self::Wasm32
            | Self::Wasm64
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => 64,
            Self::Arm32
            | Self::Armv7
            | Self::Thumbv6m
            | Self::Thumbv7m
            | Self::I686
            | Self::Riscv32 => 32,
            #[cfg(feature = "nightly")]
            Self::Riscv128 => 128,
            #[cfg(feature = "nightly")]
            Self::Lisa => 64,
        }
    }

    /// Whether an integer register holds 64 bits. Decides whether an `i64` is one
    /// register or a pair of 32-bit words.
    pub const fn has_64bit_int_registers(self) -> bool {
        self.max_int_register_bits() >= 64
    }

    /// Byte order of a multi-byte value in memory. Decides how a literal is laid out in
    /// static data and which half of a split wide value is the low word. Exhaustive on
    /// purpose: `ppc64` and `s390x` are big-endian, and nothing else would catch it.
    pub const fn endianness(self) -> Endianness {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Arx64
            | Self::Arm32
            | Self::Armv7
            | Self::Thumbv6m
            | Self::Thumbv7m
            | Self::I686
            | Self::Riscv32
            | Self::Riscv64
            | Self::Wasm32
            | Self::Wasm64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => Endianness::Little,
            Self::S390x => Endianness::Big,
            #[cfg(feature = "nightly")]
            Self::Riscv128 | Self::Lisa => Endianness::Little,
        }
    }

    /// Alignment the stack pointer holds at a call. `1` means the target has no machine
    /// stack to align. Exhaustive on purpose.
    pub const fn stack_alignment_at_call(self) -> usize {
        match self {
            Self::X86_64
            | Self::I686
            | Self::Aarch64
            | Self::Arx64
            | Self::Armv7
            | Self::Riscv32
            | Self::Riscv64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::Unknown => 16,
            Self::Arm32 | Self::Thumbv6m | Self::Thumbv7m | Self::S390x => 8,
            Self::Wasm32 | Self::Wasm64 | Self::Nvptx64 | Self::Amdgcn => 1,
            #[cfg(feature = "nightly")]
            Self::Riscv128 => 16,
            #[cfg(feature = "nightly")]
            Self::Lisa => 16,
        }
    }

    /// Whether an argument with 16-byte alignment starts at its own alignment rather than
    /// at the next word. AAPCS64 rounds the integer register counter up to even before
    /// such an argument, so an `i128` takes x2:x3 after one `i64`. riscv64 and ppc64
    /// do not round for named arguments: the riscv psABI makes the even-register rule
    /// variadic-only, and clang 18 places a named `i128` at a1:a2 and r4:r5. Keep them
    /// false unless that is settled differently. Exhaustive on purpose, so a new
    /// architecture states its answer.
    pub const fn honors_argument_alignment(self) -> bool {
        match self {
            Self::X86_64 | Self::Aarch64 => true,
            Self::Arx64
            | Self::Arm32
            | Self::Armv7
            | Self::Thumbv6m
            | Self::Thumbv7m
            | Self::I686
            | Self::Riscv32
            | Self::Riscv64
            | Self::Wasm32
            | Self::Wasm64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::S390x
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => false,
            #[cfg(feature = "nightly")]
            Self::Riscv128 | Self::Lisa => false,
        }
    }

    /// What a finished compilation is. A GPU target stops at assembly text and a wasm
    /// target at an assembled module; neither reaches a linker. Exhaustive on purpose.
    pub const fn output_artifact(self) -> OutputArtifact {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Arx64
            | Self::Arm32
            | Self::Armv7
            | Self::Thumbv6m
            | Self::Thumbv7m
            | Self::I686
            | Self::Riscv32
            | Self::Riscv64
            | Self::PowerPC64
            | Self::LoongArch64
            | Self::S390x
            | Self::Unknown => OutputArtifact::Executable,
            Self::Wasm32 | Self::Wasm64 => OutputArtifact::Module,
            Self::Nvptx64 | Self::Amdgcn => OutputArtifact::Text,
            #[cfg(feature = "nightly")]
            Self::Riscv128 | Self::Lisa => OutputArtifact::Executable,
        }
    }

    pub const fn data_layout(self) -> TargetDataLayout {
        TargetDataLayout {
            pointer_width: self.pointer_width(),
            int_register_bits: self.max_int_register_bits(),
            endianness: self.endianness(),
        }
    }

    /// Sub-architecture a bare architecture name selects. Each value is what that name
    /// has always meant to the backend, so adding the variant axis changes no output.
    /// Exhaustive on purpose.
    pub const fn default_variant(self) -> ArchVariant {
        match self {
            Self::Riscv32 => ArchVariant::Riscv(RiscvIsa::general(32)),
            Self::Riscv64 => ArchVariant::Riscv(RiscvIsa::general(64)),
            Self::Armv7 => ArchVariant::Armv7(Armv7Variant::cortex_a7()),
            Self::Thumbv7m => ArchVariant::Armv7(Armv7Variant::cortex_m3()),
            Self::Aarch64 => ArchVariant::Aarch64(Aarch64Level::V8_0),
            Self::PowerPC64 => ArchVariant::PowerPC64(PowerLevel::Power8),
            Self::X86_64
            | Self::Arx64
            | Self::Arm32
            | Self::Thumbv6m
            | Self::I686
            | Self::Wasm32
            | Self::Wasm64
            | Self::LoongArch64
            | Self::S390x
            | Self::Nvptx64
            | Self::Amdgcn
            | Self::Unknown => ArchVariant::None,
            #[cfg(feature = "nightly")]
            Self::Riscv128 => ArchVariant::Riscv(RiscvIsa::general(128)),
            #[cfg(feature = "nightly")]
            Self::Lisa => ArchVariant::None,
        }
    }
}

/// Target strings that name a sub-architecture. Open-ended RISC-V ISA strings are parsed
/// instead, in [`parse_arch_token`].
const VARIANT_SPELLINGS: &[(&str, TargetArchitecture, ArchVariant)] = &[
    (
        "armv7a-neon",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            neon: true,
            fpu: Some("neon-vfpv4"),
            ..Armv7Variant::cortex_a7()
        }),
    ),
    (
        "armv7a-sf",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            hard_float: false,
            fpu: None,
            ..Armv7Variant::cortex_a7()
        }),
    ),
    (
        "armv7r",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            profile: Armv7Profile::R,
            hard_float: false,
            fpu: None,
            ..Armv7Variant::cortex_a7()
        }),
    ),
    (
        "aarch64v8.1",
        TargetArchitecture::Aarch64,
        ArchVariant::Aarch64(Aarch64Level::V8_1),
    ),
    (
        "aarch64v8.2",
        TargetArchitecture::Aarch64,
        ArchVariant::Aarch64(Aarch64Level::V8_2),
    ),
    (
        "ppc64le-power9",
        TargetArchitecture::PowerPC64,
        ArchVariant::PowerPC64(PowerLevel::Power9),
    ),
    (
        "ppc64le-power10",
        TargetArchitecture::PowerPC64,
        ArchVariant::PowerPC64(PowerLevel::Power10),
    ),
];

/// Architecture half of a target string, with the sub-architecture it names. A spelling
/// that names only the architecture takes [`TargetArchitecture::default_variant`].
pub fn parse_arch_token(token: &str) -> Result<(TargetArchitecture, ArchVariant), &'static str> {
    if let Some((_, arch, variant)) = VARIANT_SPELLINGS.iter().find(|(name, _, _)| *name == token) {
        return Ok((*arch, *variant));
    }
    if token.starts_with("rv") {
        let isa = RiscvIsa::from_str(token)?;
        let arch = match isa.xlen {
            32 => TargetArchitecture::Riscv32,
            64 => TargetArchitecture::Riscv64,
            #[cfg(feature = "nightly")]
            128 => TargetArchitecture::Riscv128,
            _ => return Err("no backend for that RISC-V width"),
        };
        return Ok((arch, ArchVariant::Riscv(isa)));
    }
    let arch = TargetArchitecture::from_str(token)?;
    Ok((arch, arch.default_variant()))
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
            "thumbv6m" | "armv6m" | "cortex-m0" => Ok(Self::Thumbv6m),
            "thumbv7m" | "armv7m" | "cortex-m3" => Ok(Self::Thumbv7m),
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
            "s390x" | "systemz" => Ok(Self::S390x),
            "s390" => Err(
                "Architecture 's390' was not added\n  Hint: lamina supports 64-bit z/Architecture only. Write 's390x' (e.g. s390x_linux)",
            ),
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
            "openbsd" => Err(
                "Operating system 'openbsd' is not implemented\n  Hint: lamina has no OpenBSD syscall table, dynamic linker path or ELF brand. FreeBSD is the BSD that is implemented: write 'freebsd' (e.g. x86_64_freebsd)",
            ),
            "netbsd" => Err(
                "Operating system 'netbsd' is not implemented\n  Hint: lamina has no NetBSD syscall table, dynamic linker path or ELF brand. FreeBSD is the BSD that is implemented: write 'freebsd' (e.g. x86_64_freebsd)",
            ),
            "dragonfly" => Err(
                "Operating system 'dragonfly' is not implemented\n  Hint: lamina has no DragonFly syscall table, dynamic linker path or ELF brand. FreeBSD is the BSD that is implemented: write 'freebsd' (e.g. x86_64_freebsd)",
            ),
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
            Self::Thumbv6m => "thumbv6m",
            Self::Thumbv7m => "thumbv7m",
            Self::I686 => "i686",
            Self::Riscv32 => "riscv32",
            Self::Riscv64 => "riscv64",
            #[cfg(feature = "nightly")]
            Self::Riscv128 => "riscv128",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::PowerPC64 => "ppc64le",
            Self::LoongArch64 => "loongarch64",
            Self::S390x => "s390x",
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
    "x86_64_freebsd",
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
    "s390x_linux",
    "i686_linux",
    "armv7_linux",
    "thumbv6m_unknown",
    "thumbv6m_linux",
    "thumbv7m_unknown",
    "thumbv7m_linux",
    #[cfg(feature = "nightly")]
    "riscv128_unknown",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freebsd_is_a_target_the_driver_also_accepts() {
        let target: Target = "x86_64_freebsd".parse().expect("freebsd parses");
        assert_eq!(target.architecture, TargetArchitecture::X86_64);
        assert_eq!(target.operating_system, TargetOperatingSystem::FreeBSD);
        assert!(HOST_ARCH_LIST.contains(&"x86_64_freebsd"));
        assert_eq!(
            "bsd".parse::<TargetOperatingSystem>(),
            Ok(TargetOperatingSystem::FreeBSD)
        );
    }

    #[test]
    fn the_other_bsds_refuse_with_a_reason() {
        for name in ["openbsd", "netbsd", "dragonfly"] {
            let error = name
                .parse::<TargetOperatingSystem>()
                .expect_err("this BSD is not implemented");
            assert!(error.contains("is not implemented"), "{name}: {error}");
            assert!(error.contains("Hint:"), "{name}: {error}");
            assert!(error.contains("freebsd"), "{name}: {error}");
            assert!(
                !HOST_ARCH_LIST.contains(&format!("x86_64_{name}").as_str()),
                "{name}"
            );
        }
    }

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
    fn s390x_is_the_only_big_endian_target() {
        for name in HOST_ARCH_LIST {
            let target: Target = name.parse().expect(name);
            let expected = if target.architecture == TargetArchitecture::S390x {
                Endianness::Big
            } else {
                Endianness::Little
            };
            assert_eq!(target.endianness(), expected, "{name}");
        }
    }

    #[test]
    fn x86_64_and_aarch64_honor_argument_alignment() {
        let honors = |name: &str| name.parse::<Target>().unwrap().honors_argument_alignment();
        assert!(honors("x86_64_linux"));
        assert!(honors("aarch64_linux"));
        for name in [
            "riscv64_linux",
            "riscv32_linux",
            "ppc64le_linux",
            "i686_linux",
        ] {
            assert!(!honors(name), "{name}");
        }
    }

    #[test]
    fn stack_alignment_is_sixteen_where_there_is_a_stack() {
        let align = |name: &str| name.parse::<Target>().unwrap().stack_alignment_at_call();
        for name in [
            "x86_64_linux",
            "i686_linux",
            "aarch64_linux",
            "armv7_linux",
            "riscv64_linux",
            "ppc64le_linux",
        ] {
            assert_eq!(align(name), 16, "{name}");
        }
        assert_eq!(align("wasm32_wasi"), 1);
    }

    #[test]
    fn only_native_targets_reach_a_linker() {
        let artifact = |name: &str| name.parse::<Target>().unwrap().output_artifact();
        assert_eq!(artifact("x86_64_linux"), OutputArtifact::Executable);
        assert_eq!(artifact("wasm32_wasi"), OutputArtifact::Module);
        assert_eq!(artifact("nvptx64_cuda"), OutputArtifact::Text);
        assert_eq!(artifact("amdgcn_amdhsa"), OutputArtifact::Text);
    }

    #[test]
    fn data_layout_separates_pointers_from_registers() {
        let layout = "wasm32_wasi".parse::<Target>().unwrap().data_layout();
        assert_eq!(layout.pointer_width, PointerWidth::Bits32);
        assert_eq!(layout.int_register_bits, 64);
        assert!(layout.has_64bit_int_registers());
        assert_eq!(layout.endianness, Endianness::Little);
    }

    #[test]
    fn bare_spellings_keep_their_default_variant() {
        for name in ["riscv64_linux", "riscv64gc_linux", "riscv64_unknown"] {
            let target: Target = name.parse().expect(name);
            assert_eq!(
                target.variant,
                ArchVariant::Riscv(RiscvIsa::general(64)),
                "{name}"
            );
        }
        for name in [
            "armv7_linux",
            "aarch64_linux",
            "ppc64le_linux",
            "x86_64_linux",
        ] {
            let target: Target = name.parse().expect(name);
            assert_eq!(
                target.variant,
                target.architecture.default_variant(),
                "{name}"
            );
            assert_eq!(target.to_string(), name.replace("gc", ""), "{name}");
        }
    }

    #[test]
    fn variant_spellings_parse_and_print_back() {
        for name in [
            "rv32imac_linux",
            "rv64gc_linux",
            "armv7r_linux",
            "armv7a-neon_linux",
            "aarch64v8.2_linux",
            "ppc64le-power9_linux",
        ] {
            let target: Target = name.parse().expect(name);
            assert_ne!(
                target.variant,
                target.architecture.default_variant(),
                "{name} should not be the default"
            );
            assert_eq!(target.to_string(), name, "{name} must round trip");
        }
        let rv: Target = "rv32imac_linux".parse().unwrap();
        assert_eq!(rv.architecture, TargetArchitecture::Riscv32);
        assert_eq!(rv.variant.riscv().map(|i| i.has_float()), Some(false));
        let a64: Target = "aarch64v8.2_linux".parse().unwrap();
        assert_eq!(a64.architecture, TargetArchitecture::Aarch64);
        assert!(a64.variant.aarch64().unwrap().has_lse_atomics());
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
