//! Sub-architecture variants: the part of a target the bare architecture name leaves out.
//!
//! [`crate::X86CpuLevel`] does this for x86-64. The same idea for the other families: a
//! value chosen by the target string, defaulting to what the bare name has always meant.

use std::{fmt, str::FromStr};

use crate::arch_table::{ArchSpec, FloatAggregateAbi};
use crate::cpu_level::X86CpuLevel;

/// RISC-V base width plus the standard extensions the backend decides on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiscvIsa {
    pub xlen: u32,
    /// M: multiply and divide.
    pub m: bool,
    /// A: atomics.
    pub a: bool,
    /// F: single-precision float.
    pub f: bool,
    /// D: double-precision float, implies F.
    pub d: bool,
    /// C: compressed encodings.
    pub c: bool,
    /// V: the vector extension.
    pub v: bool,
}

impl RiscvIsa {
    /// The base integer ISA alone.
    pub const fn base(xlen: u32) -> Self {
        Self {
            xlen,
            m: false,
            a: false,
            f: false,
            d: false,
            c: false,
            v: false,
        }
    }

    /// G, the conventional set: IMAFD. What a bare `riscv32` or `riscv64` means.
    pub const fn general(xlen: u32) -> Self {
        Self {
            xlen,
            m: true,
            a: true,
            f: true,
            d: true,
            c: false,
            v: false,
        }
    }

    pub const fn has_float(self) -> bool {
        self.f || self.d
    }

    /// Canonical ISA string, the spelling the variant parses from.
    pub fn name(self) -> String {
        let mut s = format!("rv{}i", self.xlen);
        for (present, letter) in [
            (self.m, 'm'),
            (self.a, 'a'),
            (self.f, 'f'),
            (self.d, 'd'),
            (self.c, 'c'),
            (self.v, 'v'),
        ] {
            if present {
                s.push(letter);
            }
        }
        s
    }

    /// `-march=` string for an assembler. `g` is the accepted shorthand for IMAFD.
    pub fn march(self) -> String {
        let mut s = format!("rv{}", self.xlen);
        if self.m && self.a && self.f && self.d {
            s.push('g');
        } else {
            s.push('i');
            for (present, letter) in [(self.m, 'm'), (self.a, 'a'), (self.f, 'f'), (self.d, 'd')] {
                if present {
                    s.push(letter);
                }
            }
        }
        if self.c {
            s.push('c');
        }
        if self.v {
            s.push('v');
        }
        s
    }
}

impl FromStr for RiscvIsa {
    type Err = &'static str;

    /// `rv32imac`, `rv64gc`, `rv64g`, `rv32i`. `g` expands to `imafd`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rest = s.strip_prefix("rv").ok_or("expected an rv… ISA string")?;
        let split = rest
            .find(|c: char| !c.is_ascii_digit())
            .ok_or("ISA string has no base after the width")?;
        let (width, letters) = rest.split_at(split);
        let xlen = match width {
            "32" => 32,
            "64" => 64,
            "128" => 128,
            _ => return Err("RISC-V width must be 32, 64 or 128"),
        };
        let mut isa = Self::base(xlen);
        let mut seen_base = false;
        for letter in letters.chars() {
            match letter {
                'i' => seen_base = true,
                'g' => {
                    seen_base = true;
                    isa.m = true;
                    isa.a = true;
                    isa.f = true;
                    isa.d = true;
                }
                'm' => isa.m = true,
                'a' => isa.a = true,
                'f' => isa.f = true,
                'd' => {
                    isa.f = true;
                    isa.d = true;
                }
                'c' => isa.c = true,
                'v' => isa.v = true,
                _ => return Err("unknown RISC-V extension letter"),
            }
        }
        if !seen_base {
            return Err("RISC-V ISA string needs a base, 'i' or 'g'");
        }
        Ok(isa)
    }
}

/// The armv7 profile. A is the application profile with an MMU; R is real-time; M is the
/// microcontroller profile, which is Thumb-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Armv7Profile {
    A,
    R,
    M,
}

impl Armv7Profile {
    /// `.arch` directive name.
    pub const fn arch_name(self) -> &'static str {
        match self {
            Self::A => "armv7-a",
            Self::R => "armv7-r",
            Self::M => "armv7-m",
        }
    }
}

/// Everything the armv7 backend hardcodes today, named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Armv7Variant {
    pub profile: Armv7Profile,
    /// The VFP unit, for the `.fpu` directive. `None` when the profile has no FPU.
    /// Whether floats are *passed* in those registers is the environment's answer,
    /// [`crate::TargetEnvironment::hardware_float`], not the variant's.
    pub fpu: Option<&'static str>,
    pub neon: bool,
    /// Hardware `sdiv`/`udiv`. Cortex-A7 and later in the A profile; always in R and M.
    pub hardware_divide: bool,
}

impl Armv7Variant {
    /// What a bare `thumbv7m` means: a Cortex-M3, which has `sdiv`/`udiv` and no FPU.
    /// An M4F or M7 adds a single-precision VFP, which this backend does not use.
    pub const fn cortex_m3() -> Self {
        Self {
            profile: Armv7Profile::M,
            fpu: None,
            neon: false,
            hardware_divide: true,
        }
    }

    /// What a bare `armv7`, `armv7a` or `armv7hf` has always meant.
    pub const fn cortex_a7() -> Self {
        Self {
            profile: Armv7Profile::A,
            fpu: Some("vfpv3-d16"),
            neon: false,
            hardware_divide: true,
        }
    }
}

impl Default for Armv7Variant {
    fn default() -> Self {
        Self::cortex_a7()
    }
}

/// AArch64 architecture revision. LSE atomics arrive in v8.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Aarch64Level {
    #[default]
    V8_0,
    V8_1,
    V8_2,
}

impl Aarch64Level {
    /// `-march=` string for an assembler.
    pub const fn march(self) -> &'static str {
        match self {
            Self::V8_0 => "armv8-a",
            Self::V8_1 => "armv8.1-a",
            Self::V8_2 => "armv8.2-a",
        }
    }

    /// Large System Extensions: single-instruction atomics instead of a load/store
    /// exclusive loop.
    pub const fn has_lse_atomics(self) -> bool {
        match self {
            Self::V8_0 => false,
            Self::V8_1 | Self::V8_2 => true,
        }
    }
}

/// POWER processor generation for ppc64le.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PowerLevel {
    #[default]
    Power8,
    Power9,
    Power10,
}

impl PowerLevel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Power8 => "power8",
            Self::Power9 => "power9",
            Self::Power10 => "power10",
        }
    }

    /// `modsd` / `modud`, which replace the divide-multiply-subtract remainder sequence.
    pub const fn has_modulo_instructions(self) -> bool {
        match self {
            Self::Power8 => false,
            Self::Power9 | Self::Power10 => true,
        }
    }
}

/// The sub-architecture a target selected, for the families that have one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArchVariant {
    /// The architecture has no sub-architecture axis, or the target named none.
    #[default]
    None,
    Riscv(RiscvIsa),
    Armv7(Armv7Variant),
    Aarch64(Aarch64Level),
    PowerPC64(PowerLevel),
    X86(X86CpuLevel),
}

impl ArchVariant {
    pub fn riscv(self) -> Option<RiscvIsa> {
        match self {
            Self::Riscv(isa) => Some(isa),
            Self::None | Self::Armv7(_) | Self::Aarch64(_) | Self::PowerPC64(_) | Self::X86(_) => {
                None
            }
        }
    }

    pub fn armv7(self) -> Option<Armv7Variant> {
        match self {
            Self::Armv7(v) => Some(v),
            Self::None | Self::Riscv(_) | Self::Aarch64(_) | Self::PowerPC64(_) | Self::X86(_) => {
                None
            }
        }
    }

    pub fn aarch64(self) -> Option<Aarch64Level> {
        match self {
            Self::Aarch64(level) => Some(level),
            Self::None | Self::Riscv(_) | Self::Armv7(_) | Self::PowerPC64(_) | Self::X86(_) => {
                None
            }
        }
    }

    pub fn x86(self) -> Option<X86CpuLevel> {
        match self {
            Self::X86(level) => Some(level),
            Self::None
            | Self::Riscv(_)
            | Self::Armv7(_)
            | Self::Aarch64(_)
            | Self::PowerPC64(_) => None,
        }
    }

    /// The architecture's spec as this variant refines it.
    ///
    /// The single place a sub-architecture changes a property the architecture stated.
    /// A RISC-V ISA without `d` is the soft-float ABI, which takes a different ELF
    /// interpreter than the one the bare architecture names.
    pub fn refine(self, mut spec: ArchSpec) -> ArchSpec {
        match self {
            Self::Riscv(isa) => {
                if !isa.d {
                    spec.interpreters.gnu = spec.interpreters.gnu_soft_float;
                    spec.float_aggregates = FloatAggregateAbi::Refused;
                }
                spec
            }
            Self::None | Self::Armv7(_) | Self::Aarch64(_) | Self::PowerPC64(_) | Self::X86(_) => {
                spec
            }
        }
    }

    pub fn powerpc64(self) -> Option<PowerLevel> {
        match self {
            Self::PowerPC64(level) => Some(level),
            Self::None | Self::Riscv(_) | Self::Armv7(_) | Self::Aarch64(_) | Self::X86(_) => None,
        }
    }
}

impl fmt::Display for ArchVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "default"),
            Self::Riscv(isa) => write!(f, "{}", isa.name()),
            Self::Armv7(v) => write!(
                f,
                "{}{}",
                v.profile.arch_name(),
                if v.neon { "+neon" } else { "" }
            ),
            Self::Aarch64(level) => write!(f, "{}", level.march()),
            Self::PowerPC64(level) => write!(f, "{}", level.name()),
            Self::X86(level) => write!(f, "{level}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_strings_round_trip() {
        let gc: RiscvIsa = "rv64gc".parse().expect("rv64gc");
        assert_eq!(gc.name(), "rv64imafdc");
        assert_eq!(gc.march(), "rv64gc");
        let imac: RiscvIsa = "rv32imac".parse().expect("rv32imac");
        assert!(imac.m && imac.a && imac.c);
        assert!(!imac.has_float());
        assert_eq!(imac.march(), "rv32imac");
        assert_eq!(RiscvIsa::general(64).march(), "rv64g");
        assert!("rv16gc".parse::<RiscvIsa>().is_err());
        assert!("rv64gz".parse::<RiscvIsa>().is_err());
        assert!("rv64".parse::<RiscvIsa>().is_err());
        assert!("rv64mac".parse::<RiscvIsa>().is_err());
        let gcv: RiscvIsa = "rv64gcv".parse().expect("rv64gcv");
        assert!(gcv.v);
        assert_eq!(gcv.name(), "rv64imafdcv");
        assert_eq!(gcv.march(), "rv64gcv");
        assert!(!gc.v);
    }

    #[test]
    fn bare_riscv_keeps_meaning_imafd() {
        let g = RiscvIsa::general(64);
        assert!(g.m && g.a && g.f && g.d && !g.c && !g.v);
        assert_eq!(g.name(), "rv64imafd");
    }

    #[test]
    fn armv7_default_is_what_the_backend_hardcodes() {
        let v = Armv7Variant::default();
        assert_eq!(v.profile.arch_name(), "armv7-a");
        assert_eq!(v.fpu, Some("vfpv3-d16"));
        assert!(v.hardware_divide && !v.neon);
    }

    #[test]
    fn levels_gate_their_instructions() {
        assert!(!Aarch64Level::default().has_lse_atomics());
        assert!(Aarch64Level::V8_1.has_lse_atomics());
        assert_eq!(Aarch64Level::default().march(), "armv8-a");
        assert!(!PowerLevel::default().has_modulo_instructions());
        assert!(PowerLevel::Power9.has_modulo_instructions());
    }

    #[test]
    fn variant_accessors_pick_their_family() {
        let v = ArchVariant::Riscv(RiscvIsa::general(32));
        assert_eq!(v.riscv().map(|i| i.xlen), Some(32));
        assert!(v.armv7().is_none());
        assert!(ArchVariant::None.aarch64().is_none());
        assert_eq!(
            ArchVariant::PowerPC64(PowerLevel::Power10).to_string(),
            "power10"
        );
    }
}

/// The sub-architecture axis an architecture has, carrying the value a bare architecture
/// name selects.
///
/// Stated once per architecture in the arch table. Giving an architecture an axis it does
/// not have yet is one more arm here and one changed line there; nothing else in the tree
/// has to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantAxis {
    /// The architecture has no sub-architecture axis.
    None,
    Riscv(RiscvIsa),
    Armv7(Armv7Variant),
    Aarch64(Aarch64Level),
    Power(PowerLevel),
    X86(X86CpuLevel),
}

impl VariantAxis {
    /// Sub-architecture a bare architecture name selects. Each value is what that name
    /// has always meant to the backend.
    pub const fn default_variant(self) -> ArchVariant {
        match self {
            Self::None => ArchVariant::None,
            Self::Riscv(isa) => ArchVariant::Riscv(isa),
            Self::Armv7(variant) => ArchVariant::Armv7(variant),
            Self::Aarch64(level) => ArchVariant::Aarch64(level),
            Self::Power(level) => ArchVariant::PowerPC64(level),
            Self::X86(level) => ArchVariant::X86(level),
        }
    }

    /// `chosen` when it belongs to this axis, the axis default otherwise.
    ///
    /// Every route into a [`crate::Target`] goes through this, so a target cannot hold a
    /// variant of a family its architecture does not have.
    pub const fn accept(self, chosen: ArchVariant) -> ArchVariant {
        let belongs = match (self, chosen) {
            (Self::Riscv(_), ArchVariant::Riscv(_))
            | (Self::Armv7(_), ArchVariant::Armv7(_))
            | (Self::Aarch64(_), ArchVariant::Aarch64(_))
            | (Self::Power(_), ArchVariant::PowerPC64(_))
            | (Self::X86(_), ArchVariant::X86(_))
            | (Self::None, ArchVariant::None) => true,
            (Self::None, _)
            | (Self::Riscv(_), _)
            | (Self::Armv7(_), _)
            | (Self::Aarch64(_), _)
            | (Self::Power(_), _)
            | (Self::X86(_), _) => false,
        };
        if belongs {
            chosen
        } else {
            self.default_variant()
        }
    }
}
