//! Targets: an architecture, an environment, an operating system and a variant.
//!
//! Every per-architecture property comes from the table in [`crate::arch_table`]. This
//! module holds the other three axes and the target-string grammar that selects them.

use std::{fmt, str::FromStr};

use crate::arch_table::{
    ArchSpec, Endianness, FloatAggregateAbi, OutputArtifact, PointerWidth, REFUSED_SPELLINGS,
    TargetDataLayout,
};
use crate::arch_variant::{
    Aarch64Level, ArchVariant, Armv7Profile, Armv7Variant, PowerLevel, RiscvIsa,
};
use crate::detection::{detect_host_architecture_only, detect_host_os};
use crate::environment::TargetEnvironment;

pub use crate::arch_table::TargetArchitecture;

/// A target: what to generate code for, and what to link it against.
///
/// The four axes are independent and each is declared, never inferred from the build
/// machine. The target-string grammar is `<architecture>_<operating-system>[_<environment>]`,
/// where the architecture field may name a sub-architecture variant (`rv64gc`,
/// `aarch64v8.2`) and the environment field may be left out, in which case the
/// architecture's declared default for that operating system applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub architecture: TargetArchitecture,
    pub operating_system: TargetOperatingSystem,
    /// The C runtime and ABI. Left out of a target string, it is
    /// [`default_environment`] for the architecture and operating system.
    pub environment: TargetEnvironment,
    /// Sub-architecture the target string selected. Every spelling that names only the
    /// architecture gets the architecture's [`crate::VariantAxis`] default, which is what
    /// that spelling has always meant.
    pub variant: ArchVariant,
}

/// The environment a target means when its target string names none.
///
/// Declared, not discovered: an operating system that supplies its own C runtime, or none
/// at all, declares [`TargetEnvironment::Freestanding`], and every other operating system
/// takes the architecture's own `hosted_environment` from the arch table.
pub const fn default_environment(
    architecture: TargetArchitecture,
    operating_system: TargetOperatingSystem,
) -> TargetEnvironment {
    if operating_system.links_a_c_runtime() {
        architecture.spec().hosted_environment
    } else {
        TargetEnvironment::Freestanding
    }
}

impl Target {
    pub fn new(architecture: TargetArchitecture, operating_system: TargetOperatingSystem) -> Self {
        Self {
            architecture,
            operating_system,
            environment: default_environment(architecture, operating_system),
            variant: architecture.spec().variants.default_variant(),
        }
    }

    /// Select a sub-architecture. A variant of another family is replaced by the
    /// architecture's default, so the target stays consistent.
    pub fn with_variant(mut self, variant: ArchVariant) -> Self {
        self.variant = self.architecture.spec().variants.accept(variant);
        self
    }

    /// Select an environment the architecture accepts.
    pub fn with_environment(mut self, environment: TargetEnvironment) -> Result<Self, String> {
        let spec = self.architecture.spec();
        if !spec.accepts_environment(environment) {
            return Err(format!(
                "Environment '{environment}' is not supported for {}\n  Hint: {} accepts {}",
                spec.token,
                spec.token,
                accepted_environments(&spec)
            ));
        }
        self.environment = environment;
        Ok(self)
    }

    /// Everything the architecture states, as this target's variant refines it.
    pub fn spec(&self) -> ArchSpec {
        self.variant.refine(self.architecture.spec())
    }

    /// Architecture half of the target string, carrying the variant when it is not the
    /// default, so `Display` and `FromStr` stay inverses.
    pub fn arch_token(&self) -> String {
        let spec = self.architecture.spec();
        if self.variant == spec.variants.default_variant() {
            return spec.token.to_string();
        }
        match self.variant {
            ArchVariant::Riscv(isa) => isa.march(),
            ArchVariant::None
            | ArchVariant::Armv7(_)
            | ArchVariant::Aarch64(_)
            | ArchVariant::PowerPC64(_)
            | ArchVariant::X86(_) => self
                .variant_spelling()
                .map_or_else(|| spec.token.to_string(), |(name, _)| name.to_string()),
        }
    }

    fn variant_spelling(&self) -> Option<(&'static str, Option<TargetEnvironment>)> {
        VARIANT_SPELLINGS
            .iter()
            .find(|(_, arch, variant, _)| *arch == self.architecture && *variant == self.variant)
            .map(|(name, _, _, environment)| (*name, *environment))
    }

    /// The environment the rest of the target string already implies, and so the one the
    /// environment field is left out for.
    pub fn implied_environment(&self) -> TargetEnvironment {
        self.variant_spelling()
            .and_then(|(_, environment)| environment)
            .unwrap_or_else(|| default_environment(self.architecture, self.operating_system))
    }

    pub fn pointer_width(&self) -> PointerWidth {
        self.spec().pointer_width
    }

    pub fn has_64bit_int_registers(&self) -> bool {
        self.spec().has_64bit_int_registers()
    }

    pub fn endianness(&self) -> Endianness {
        self.spec().endianness
    }

    pub fn stack_alignment_at_call(&self) -> usize {
        self.spec().stack_alignment_at_call
    }

    pub fn honors_argument_alignment(&self) -> bool {
        self.spec().honors_argument_alignment
    }

    pub fn output_artifact(&self) -> OutputArtifact {
        self.spec().output_artifact
    }

    pub fn data_layout(&self) -> TargetDataLayout {
        let mut layout = self.spec().data_layout();
        layout.float_aggregates = match (layout.float_aggregates, self.operating_system) {
            (FloatAggregateAbi::SysV, TargetOperatingSystem::Windows) => FloatAggregateAbi::Win64,
            (abi, _) => abi,
        };
        layout
    }

    /// The ELF interpreter this target's environment asks for, if it asks for one.
    pub fn interpreter(&self) -> Option<&'static str> {
        self.spec().interpreter(self.environment)
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
        self.to_string()
    }

    /// Detect the host system's target.
    ///
    /// # Examples
    ///
    /// ```
    /// use lamina_platform::Target;
    /// let host = Target::detect_host().expect("host is a supported target");
    /// println!("Host target: {}", host);
    /// ```
    ///
    /// Fails for a host whose architecture or OS lamina has no backend for, rather than
    /// guessing a target for it.
    pub fn detect_host() -> Result<Self, String> {
        let arch = detect_host_architecture_only();
        let os = detect_host_os();
        let architecture = parse_architecture(arch)
            .map_err(|_| format!("host architecture '{arch}' has no lamina backend"))?;
        let operating_system = TargetOperatingSystem::from_str(os)
            .map_err(|_| format!("host operating system '{os}' has no lamina backend"))?;
        Ok(Self::new(architecture, operating_system))
    }
}

fn accepted_environments(spec: &ArchSpec) -> String {
    let mut names: Vec<&str> = spec
        .environments
        .iter()
        .map(|environment| environment.token())
        .collect();
    names.push(TargetEnvironment::Freestanding.token());
    names.join(", ")
}

impl FromStr for Target {
    type Err = String;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let (head, named_environment) = split_environment(target);
        let head = TARGET_ALIASES
            .iter()
            .find(|(alias, _)| *alias == head)
            .map_or(head, |(_, canonical)| *canonical);
        let parts: Vec<&str> = head.rsplitn(2, '_').collect();
        if parts.len() != 2 {
            return Err("Invalid target format: expected 'architecture_os'".to_string());
        }
        let (architecture, variant, spelling_environment) = parse_arch_token(parts[1])?;
        let os = TargetOperatingSystem::from_str(parts[0]).map_err(str::to_string)?;
        let target = Self::new(architecture, os).with_variant(variant);
        match named_environment.or(spelling_environment) {
            Some(environment) => target.with_environment(environment),
            None => Ok(target),
        }
    }
}

/// Split a trailing environment field off a target string. The last underscore field is
/// the environment only when it names one, so `x86_64_linux` is unaffected.
fn split_environment(target: &str) -> (&str, Option<TargetEnvironment>) {
    match target.rsplit_once('_') {
        Some((head, tail)) => match tail.parse::<TargetEnvironment>() {
            Ok(environment) => (head, Some(environment)),
            Err(_) => (target, None),
        },
        None => (target, None),
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.arch_token(), self.operating_system)?;
        if self.environment != self.implied_environment() {
            write!(f, "_{}", self.environment)?;
        }
        Ok(())
    }
}

/// Target strings that name a sub-architecture, with the environment the spelling implies
/// when it is not the architecture's default. Open-ended RISC-V ISA strings are parsed
/// instead, in [`parse_arch_token`].
const VARIANT_SPELLINGS: &[(
    &str,
    TargetArchitecture,
    ArchVariant,
    Option<TargetEnvironment>,
)] = &[
    (
        "armv7a-neon",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            neon: true,
            fpu: Some("neon-vfpv4"),
            ..Armv7Variant::cortex_a7()
        }),
        None,
    ),
    (
        "armv7a-sf",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            fpu: None,
            ..Armv7Variant::cortex_a7()
        }),
        Some(TargetEnvironment::Eabi),
    ),
    (
        "armv7r",
        TargetArchitecture::Armv7,
        ArchVariant::Armv7(Armv7Variant {
            profile: Armv7Profile::R,
            fpu: None,
            ..Armv7Variant::cortex_a7()
        }),
        Some(TargetEnvironment::Eabi),
    ),
    (
        "aarch64v8.1",
        TargetArchitecture::Aarch64,
        ArchVariant::Aarch64(Aarch64Level::V8_1),
        None,
    ),
    (
        "aarch64v8.2",
        TargetArchitecture::Aarch64,
        ArchVariant::Aarch64(Aarch64Level::V8_2),
        None,
    ),
    (
        "ppc64le-power9",
        TargetArchitecture::PowerPC64,
        ArchVariant::PowerPC64(PowerLevel::Power9),
        None,
    ),
    (
        "ppc64le-power10",
        TargetArchitecture::PowerPC64,
        ArchVariant::PowerPC64(PowerLevel::Power10),
        None,
    ),
];

/// Architecture half of a target string, with the sub-architecture it names and the
/// environment that spelling implies. A spelling that names only the architecture takes
/// the architecture's variant-axis default and implies no environment.
pub fn parse_arch_token(
    token: &str,
) -> Result<(TargetArchitecture, ArchVariant, Option<TargetEnvironment>), String> {
    if let Some((_, arch, variant, environment)) = VARIANT_SPELLINGS
        .iter()
        .find(|(name, _, _, _)| *name == token)
    {
        return Ok((*arch, *variant, *environment));
    }
    if token.starts_with("rv") {
        let isa = RiscvIsa::from_str(token).map_err(str::to_string)?;
        let arch = match isa.xlen {
            32 => TargetArchitecture::Riscv32,
            64 => TargetArchitecture::Riscv64,
            #[cfg(feature = "nightly")]
            128 => TargetArchitecture::Riscv128,
            _ => return Err("no backend for that RISC-V width".to_string()),
        };
        return Ok((arch, ArchVariant::Riscv(isa), None));
    }
    let arch = parse_architecture(token).map_err(str::to_string)?;
    Ok((arch, arch.spec().variants.default_variant(), None))
}

/// The architecture a bare spelling names.
///
/// Refusals are consulted first, so a machine lamina has decided not to implement is
/// refused with its reason rather than resolving to a neighbour.
pub fn parse_architecture(spelling: &str) -> Result<TargetArchitecture, &'static str> {
    if let Some((_, reason)) = REFUSED_SPELLINGS.iter().find(|(name, _)| *name == spelling) {
        return Err(reason);
    }
    TargetArchitecture::from_spelling(spelling).ok_or("Unknown architecture")
}

impl FromStr for TargetArchitecture {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_architecture(s)
    }
}

impl fmt::Display for TargetArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.spec().token)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackProbe {
    Chkstk,
    PageLoop,
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

impl TargetOperatingSystem {
    pub const fn stack_probe(self) -> Option<StackProbe> {
        match self {
            Self::Windows => Some(StackProbe::Chkstk),
            Self::Linux | Self::MacOS => Some(StackProbe::PageLoop),
            _ => None,
        }
    }

    /// Whether a target on this operating system links C runtime objects and a libc that
    /// the environment axis chooses.
    ///
    /// False where the operating system supplies its own runtime — Darwin, Windows, WASI
    /// — or where there is nothing to link at all, as on the GPU targets. Those declare
    /// [`TargetEnvironment::Freestanding`] and the axis selects nothing for them.
    pub const fn links_a_c_runtime(self) -> bool {
        match self {
            Self::Linux
            | Self::Android
            | Self::FreeBSD
            | Self::OpenBSD
            | Self::NetBSD
            | Self::DragonFly
            | Self::Redox
            | Self::Horizon
            | Self::Psp
            | Self::Psx
            | Self::Vita
            | Self::Orbis
            | Self::Prospero
            | Self::Unknown => true,
            Self::Cuda
            | Self::AmdHsa
            | Self::MacOS
            | Self::Windows
            | Self::IOS
            | Self::TvOS
            | Self::WatchOS
            | Self::VisionOS
            | Self::Wasi => false,
            #[cfg(feature = "nightly")]
            Self::Artery => false,
        }
    }
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
            "orbis" => Ok(Self::Orbis),
            "prospero" => Ok(Self::Prospero),
            "wasi" => Ok(Self::Wasi),
            "unknown" => Ok(Self::Unknown),
            #[cfg(feature = "nightly")]
            "artery" => Ok(Self::Artery),
            other => OS_ALIASES
                .iter()
                .find(|(alias, _)| *alias == other)
                .map_or(Err("Unknown operating system"), |(_, os)| Ok(*os)),
        }
    }
}

/// Second spellings of an operating system the driver accepts.
///
/// The parser and the list the driver prints both read this table, so an alias cannot
/// be accepted and go unmentioned.
pub const OS_ALIASES: &[(&str, TargetOperatingSystem)] = &[
    ("bsd", TargetOperatingSystem::FreeBSD),
    ("ps4", TargetOperatingSystem::Orbis),
    ("ps5", TargetOperatingSystem::Prospero),
];

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

/// Every target spelling the driver accepts that [`host_arch_list`] does not name.
///
/// Gathered from the tables the parser itself reads: the architecture spellings in the
/// arch table, the sub-architecture spellings in [`VARIANT_SPELLINGS`] and the
/// operating-system aliases in [`OS_ALIASES`]. A spelling cannot be accepted and left
/// out of the list the driver prints.
///
/// Open-ended RISC-V ISA strings such as `rv32imac` are parsed rather than tabulated,
/// so they are not here; the driver names them separately.
pub fn target_alias_list() -> Vec<String> {
    let listed = host_arch_list();
    let mut aliases: Vec<String> = Vec::new();
    for arch in TargetArchitecture::all() {
        let spec = arch.spec();
        for target in spec.targets {
            let Some((_, os)) = target.rsplit_once('_') else {
                continue;
            };
            for spelling in spec.spellings {
                aliases.push(format!("{spelling}_{os}"));
            }
            for (spelling, variant_arch, _, _) in VARIANT_SPELLINGS {
                if *variant_arch == arch {
                    aliases.push(format!("{spelling}_{os}"));
                }
            }
            for (alias, aliased) in OS_ALIASES {
                if os.parse::<TargetOperatingSystem>() == Ok(*aliased) {
                    aliases.push(format!("{}_{alias}", spec.token));
                }
            }
        }
    }
    for (alias, _) in TARGET_ALIASES {
        aliases.push((*alias).to_string());
    }
    aliases.retain(|alias| !listed.contains(&alias.as_str()));
    aliases.sort();
    aliases.dedup();
    aliases
}

/// Every target string the driver advertises, gathered from the arch table.
///
/// Derived from the same blocks the properties come from, so an architecture cannot be
/// added without deciding which targets it offers, and the list cannot fall behind.
pub fn host_arch_list() -> Vec<&'static str> {
    TargetArchitecture::all()
        .into_iter()
        .flat_map(|arch| arch.spec().targets.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::arch_variant::VariantAxis;
    use crate::cpu_level::X86CpuLevel;
    use crate::environment::{CompilerRuntime, Libc};

    fn listed(name: &str) -> bool {
        host_arch_list().contains(&name)
    }

    #[test]
    fn freebsd_is_a_target_the_driver_also_accepts() {
        let target: Target = "x86_64_freebsd".parse().expect("freebsd parses");
        assert_eq!(target.architecture, TargetArchitecture::X86_64);
        assert_eq!(target.operating_system, TargetOperatingSystem::FreeBSD);
        assert!(listed("x86_64_freebsd"));
        assert_eq!(
            "bsd".parse::<TargetOperatingSystem>(),
            Ok(TargetOperatingSystem::FreeBSD)
        );
    }

    #[test]
    fn every_advertised_alias_parses_to_a_listed_target() {
        let aliases = target_alias_list();
        assert!(!aliases.is_empty(), "the alias list is empty");
        for alias in &aliases {
            let target: Target = alias
                .parse()
                .unwrap_or_else(|e| panic!("{alias} does not parse: {e}"));
            let bare = Target::new(target.architecture, target.operating_system);
            assert!(listed(&bare.to_string()), "{alias} resolves to {bare}");
            assert!(!listed(alias), "{alias} is already a listed target");
        }
    }

    #[test]
    fn the_alias_list_names_the_spellings_from_every_table_it_draws_on() {
        let aliases = target_alias_list();
        for witness in [
            "x86_64_bsd",
            "x86_64_ps4",
            "x86_64_ps5",
            "i386_linux",
            "x86_linux",
            "systemz_linux",
            "loong64_linux",
            "mips64le_linux",
            "powerpc64_linux",
            "armv7m_unknown",
            "cortex-m3_unknown",
            "ppc64le-power10_linux",
            "aarch64v8.2_linux",
            "armv7a-neon_linux",
        ] {
            assert!(
                aliases.iter().any(|alias| alias == witness),
                "{witness} is missing from {aliases:?}"
            );
        }
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
            assert!(!listed(&format!("x86_64_{name}")), "{name}");
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
            assert!(listed(name));
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
            Target::from_str("armv7_linux").unwrap(),
            Target::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
        );
    }

    #[test]
    fn big_endian_powerpc64_is_elfv2_and_elfv1_is_refused() {
        for name in ["ppc64_linux", "powerpc64_linux"] {
            assert_eq!(
                Target::from_str(name).unwrap(),
                Target::new(
                    TargetArchitecture::PowerPC64Be,
                    TargetOperatingSystem::Linux
                )
            );
        }
        assert!(
            Target::from_str("ppc64-elfv1_linux")
                .unwrap_err()
                .contains("ELFv1")
        );
        assert_eq!(
            Target::from_str("ppc64le_linux").unwrap(),
            Target::new(TargetArchitecture::PowerPC64, TargetOperatingSystem::Linux)
        );
        assert!(listed("ppc64le_linux"));
        assert!(listed("ppc64_linux"));
    }

    #[test]
    fn arm32_refuses_and_names_armv7() {
        for name in ["arm32_linux", "arm32_android", "arm32_vita"] {
            let error = Target::from_str(name).expect_err("arm32 is not a machine");
            assert!(error.contains("is not implemented"), "{name}: {error}");
            assert!(error.contains("Hint:"), "{name}: {error}");
            assert!(error.contains("armv7"), "{name}: {error}");
            assert!(!listed(name), "{name}");
        }
    }

    #[test]
    fn mips64_spells_both_byte_orders_and_refuses_o32() {
        assert_eq!(
            Target::from_str("mips64_linux").unwrap().endianness(),
            Endianness::Big
        );
        assert_eq!(
            Target::from_str("mips64el_linux").unwrap().endianness(),
            Endianness::Little
        );
        assert!(Target::from_str("mips_linux").unwrap_err().contains("O32"));
    }

    #[test]
    fn wasm_targets_round_trip() {
        for name in ["wasm32_wasi", "wasm32_unknown", "wasm64_unknown"] {
            let target: Target = name.parse().unwrap();
            assert_eq!(target.to_string(), name);
            assert!(listed(name));
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
            assert!(listed(name));
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
        assert_eq!(width("armv7_linux"), PointerWidth::Bits32);
        assert_eq!(width("wasm32_unknown"), PointerWidth::Bits32);
        assert_eq!(width("wasm64_unknown"), PointerWidth::Bits64);
        assert_eq!(PointerWidth::Bits32.bytes(), 4);
        assert_eq!(PointerWidth::Bits64.bytes(), 8);
        assert_eq!(PointerWidth::Bits128.bytes(), 16);
    }

    #[test]
    fn only_the_named_targets_are_big_endian() {
        for name in host_arch_list() {
            let target: Target = name.parse().expect(name);
            let expected = match target.architecture {
                TargetArchitecture::S390x
                | TargetArchitecture::PowerPC64Be
                | TargetArchitecture::Mips64 => Endianness::Big,
                _ => Endianness::Little,
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
                target.architecture.spec().variants.default_variant(),
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
                target.architecture.spec().variants.default_variant(),
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

    #[test]
    fn every_listed_target_prints_back_to_a_target_that_parses_the_same() {
        for name in host_arch_list() {
            let target: Target = name.parse().expect(name);
            let printed = target.to_string();
            assert_eq!(
                printed.parse::<Target>().expect(&printed),
                target,
                "{name} printed as {printed}"
            );
        }
    }

    #[test]
    fn the_variant_axis_rejects_another_familys_variant() {
        let aarch64 = Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .with_variant(ArchVariant::Riscv(RiscvIsa::general(64)));
        assert_eq!(aarch64.variant, ArchVariant::Aarch64(Aarch64Level::V8_0));
        let riscv = Target::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
            .with_variant(ArchVariant::X86(X86CpuLevel::V4));
        assert_eq!(riscv.variant, ArchVariant::Riscv(RiscvIsa::general(64)));
    }

    #[test]
    fn x86_cpu_level_is_a_variant_axis_like_the_others() {
        let spec = TargetArchitecture::X86_64.spec();
        assert_eq!(spec.variants, VariantAxis::X86(X86CpuLevel::V2));
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .with_variant(ArchVariant::X86(X86CpuLevel::V3));
        assert_eq!(target.variant.x86(), Some(X86CpuLevel::V3));
    }

    #[test]
    fn a_bare_linux_target_declares_its_environment() {
        for name in ["s390x_linux", "x86_64_linux", "riscv64_linux"] {
            let target: Target = name.parse().expect(name);
            assert_eq!(target.environment, TargetEnvironment::Gnu, "{name}");
            assert_eq!(target.environment.libc(), Some(Libc::Glibc), "{name}");
            assert_eq!(
                target.environment.compiler_runtime(),
                CompilerRuntime::LaminaBuiltins,
                "{name}"
            );
            assert_eq!(target.to_string(), name, "{name}");
        }
    }

    #[test]
    fn musl_is_a_target_string_and_picks_the_musl_interpreter() {
        let target: Target = "s390x_linux_musl".parse().expect("musl target");
        assert_eq!(target.environment, TargetEnvironment::Musl);
        assert_eq!(target.interpreter(), Some("/lib/ld-musl-s390x.so.1"));
        assert_eq!(target.to_string(), "s390x_linux_musl");
        let gnu: Target = "s390x_linux".parse().unwrap();
        assert_eq!(gnu.interpreter(), Some("/lib/ld64.so.1"));
        assert_eq!(
            gnu.spec().toolchain_triple(TargetEnvironment::Musl),
            Some("s390x-linux-musl")
        );
    }

    #[test]
    fn an_environment_the_architecture_refuses_is_named_in_the_error() {
        let error = Target::from_str("x86_64_linux_eabihf").expect_err("x86_64 has no EABI");
        assert!(error.contains("eabihf"), "{error}");
        assert!(error.contains("Hint:"), "{error}");
        assert!(error.contains("gnu, musl, lamina, none"), "{error}");
    }

    #[test]
    fn lamina_is_a_target_string_that_names_no_interpreter_and_no_toolchain() {
        for name in ["x86_64_linux_lamina", "aarch64_linux_lamina"] {
            let target: Target = name.parse().expect(name);
            assert_eq!(target.environment, TargetEnvironment::Lamina, "{name}");
            assert_eq!(target.environment.libc(), Some(Libc::Lamina), "{name}");
            assert_eq!(
                target.environment.compiler_runtime(),
                CompilerRuntime::LaminaBuiltins,
                "{name}"
            );
            assert_eq!(target.interpreter(), None, "{name}");
            assert_eq!(
                target.spec().toolchain_triple(TargetEnvironment::Lamina),
                None,
                "{name}"
            );
            assert_eq!(target.to_string(), name, "{name}");
        }
    }

    #[test]
    fn an_architecture_that_does_not_accept_lamina_says_so_by_name() {
        for name in ["s390x_linux_lamina", "riscv64_linux_lamina"] {
            let error = Target::from_str(name).expect_err(name);
            assert!(error.contains("lamina"), "{error}");
            assert!(error.contains("not supported"), "{error}");
        }
    }

    #[test]
    fn no_architecture_states_a_lamina_interpreter_or_toolchain_yet() {
        for architecture in TargetArchitecture::all() {
            let spec = architecture.spec();
            assert_eq!(spec.interpreters.lamina, None, "{architecture:?}");
            assert_eq!(spec.toolchain.lamina, None, "{architecture:?}");
            assert_eq!(
                spec.interpreter(TargetEnvironment::Lamina),
                None,
                "{architecture:?}"
            );
            assert_eq!(
                spec.toolchain_triple(TargetEnvironment::Lamina),
                None,
                "{architecture:?}"
            );
        }
    }

    #[test]
    fn no_architecture_defaults_to_the_lamina_environment() {
        for architecture in TargetArchitecture::all() {
            assert_ne!(
                architecture.spec().hosted_environment,
                TargetEnvironment::Lamina,
                "{architecture:?}"
            );
        }
    }

    #[test]
    fn armv7_spellings_keep_their_float_abi() {
        let hard: Target = "armv7_linux".parse().unwrap();
        assert_eq!(hard.environment, TargetEnvironment::Eabihf);
        assert!(hard.environment.hardware_float());
        assert_eq!(hard.interpreter(), Some("/lib/ld-linux-armhf.so.3"));
        for name in ["armv7a-sf_linux", "armv7r_linux"] {
            let soft: Target = name.parse().expect(name);
            assert_eq!(soft.environment, TargetEnvironment::Eabi, "{name}");
            assert!(!soft.environment.hardware_float(), "{name}");
            assert_eq!(soft.interpreter(), Some("/lib/ld-linux.so.3"), "{name}");
            assert_eq!(soft.to_string(), name, "{name}");
        }
        let neon: Target = "armv7a-neon_linux".parse().unwrap();
        assert_eq!(neon.environment, TargetEnvironment::Eabihf);
    }

    #[test]
    fn a_soft_float_riscv_isa_refines_the_interpreter() {
        let soft: Target = "rv64imac_linux".parse().unwrap();
        assert_eq!(soft.interpreter(), Some("/lib/ld-linux-riscv64-lp64.so.1"));
        let hard: Target = "riscv64_linux".parse().unwrap();
        assert_eq!(hard.interpreter(), Some("/lib/ld-linux-riscv64-lp64d.so.1"));
    }

    #[test]
    fn targets_without_a_c_runtime_are_freestanding() {
        for name in ["x86_64_macos", "x86_64_windows", "wasm32_wasi"] {
            let target: Target = name.parse().expect(name);
            assert_eq!(
                target.environment,
                TargetEnvironment::Freestanding,
                "{name}"
            );
            assert!(!target.environment.is_hosted(), "{name}");
            assert_eq!(target.to_string(), name, "{name}");
        }
    }

    #[test]
    fn msvc_library_arch_is_stated_or_absent_never_guessed() {
        assert_eq!(
            TargetArchitecture::X86_64.spec().msvc_library_arch,
            Some("x64")
        );
        assert_eq!(
            TargetArchitecture::Aarch64.spec().msvc_library_arch,
            Some("arm64")
        );
        assert_eq!(
            TargetArchitecture::I686.spec().msvc_library_arch,
            Some("x86")
        );
        for arch in [
            TargetArchitecture::S390x,
            TargetArchitecture::Riscv64,
            TargetArchitecture::PowerPC64,
            TargetArchitecture::Mips64,
            TargetArchitecture::LoongArch64,
        ] {
            assert_eq!(arch.spec().msvc_library_arch, None, "{arch}");
        }
    }

    #[test]
    fn every_architecture_is_in_the_derived_list() {
        let all = TargetArchitecture::all();
        assert!(all.contains(&TargetArchitecture::X86_64));
        assert!(all.contains(&TargetArchitecture::S390x));
        let listed_architectures: Vec<TargetArchitecture> = host_arch_list()
            .into_iter()
            .filter_map(|name| name.parse::<Target>().ok())
            .map(|target| target.architecture)
            .collect();
        for arch in all {
            assert!(!arch.spec().token.is_empty(), "{arch}");
            let has_targets = !arch.spec().targets.is_empty();
            assert_eq!(
                has_targets,
                listed_architectures.contains(&arch),
                "{arch} target list disagrees with the advertised targets"
            );
        }
    }
}
