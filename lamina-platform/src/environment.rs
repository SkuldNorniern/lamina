//! The C runtime and ABI a target means.
//!
//! The fourth axis of a target, beside the architecture, the operating system and the
//! sub-architecture variant. It is what decides the ELF interpreter, the C runtime
//! objects, the libc and the compiler runtime, so that `s390x_linux` *declares* glibc
//! rather than inheriting it from whichever cross compiler the build machine has.

use std::{fmt, str::FromStr};

use crate::arch_table::ArchSpec;

/// The C library an environment links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Libc {
    Glibc,
    Musl,
}

impl Libc {
    /// Component of a cross-toolchain triple that names this C library.
    pub const fn triple_component(self) -> &'static str {
        match self {
            Self::Glibc => "gnu",
            Self::Musl => "musl",
        }
    }
}

impl fmt::Display for Libc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Glibc => "glibc",
            Self::Musl => "musl",
        };
        write!(f, "{s}")
    }
}

/// Where the compiler-runtime helpers (`__divti3` and friends) come from.
///
/// A separate choice from the [`Libc`]: a musl toolchain links the host compiler runtime
/// too. Another runtime arrives as one more value here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompilerRuntime {
    /// Lamina's own `builtins/`, compiled per program for the target. It has the 128-bit
    /// helpers and no 64-bit ones, so on some architectures the host runtime is still
    /// needed beside it — see [`CompilerRuntime::requires_host_runtime`].
    LaminaBuiltins,
    /// No compiler runtime is linked; a call to a helper is a link error.
    None,
}

impl CompilerRuntime {
    /// Whether the host toolchain's `libgcc.a` is still required beside this runtime for
    /// an architecture.
    ///
    /// `builtins/` has no 64-bit routines, so an architecture whose integer registers
    /// hold fewer than 64 bits splits every `i64` and needs the `__divdi3` family, which
    /// is not ours. The runtime says what is selected and the architecture says what it
    /// has to cover; neither answers alone.
    pub const fn requires_host_runtime(self, spec: &ArchSpec) -> bool {
        match self {
            Self::LaminaBuiltins => !spec.has_64bit_int_registers(),
            Self::None => false,
        }
    }
}

/// The C runtime and ABI a target means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetEnvironment {
    /// glibc, with the architecture's own floating-point ABI.
    Gnu,
    /// musl libc.
    Musl,
    /// ARM EABI with the soft-float calling convention.
    Eabi,
    /// ARM EABI with the hard-float calling convention: floats in VFP registers.
    Eabihf,
    /// No libc and no compiler runtime: bare metal, or an operating system that supplies
    /// its own runtime and leaves nothing for this axis to choose.
    Freestanding,
}

impl TargetEnvironment {
    /// Every environment a target string may name, in the order the diagnostics list them.
    pub const ALL: [Self; 5] = [
        Self::Gnu,
        Self::Musl,
        Self::Eabi,
        Self::Eabihf,
        Self::Freestanding,
    ];

    /// Spelling in a target string.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Gnu => "gnu",
            Self::Musl => "musl",
            Self::Eabi => "eabi",
            Self::Eabihf => "eabihf",
            Self::Freestanding => "none",
        }
    }

    /// The C library this environment links, or `None` when it links none.
    pub const fn libc(self) -> Option<Libc> {
        match self {
            Self::Gnu | Self::Eabi | Self::Eabihf => Some(Libc::Glibc),
            Self::Musl => Some(Libc::Musl),
            Self::Freestanding => None,
        }
    }

    /// Where the 128-bit integer helpers and the other compiler-runtime symbols come from.
    pub const fn compiler_runtime(self) -> CompilerRuntime {
        match self {
            Self::Gnu | Self::Musl | Self::Eabi | Self::Eabihf => CompilerRuntime::LaminaBuiltins,
            Self::Freestanding => CompilerRuntime::None,
        }
    }

    /// Whether the calling convention passes floating-point values in floating-point
    /// registers. This is the eabi/eabihf distinction on 32-bit ARM.
    pub const fn hardware_float(self) -> bool {
        match self {
            Self::Gnu | Self::Musl | Self::Eabihf => true,
            Self::Eabi | Self::Freestanding => false,
        }
    }

    /// Whether C runtime objects and a libc are linked at all.
    pub const fn is_hosted(self) -> bool {
        self.libc().is_some()
    }
}

impl fmt::Display for TargetEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl FromStr for TargetEnvironment {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gnu" => Ok(Self::Gnu),
            "musl" => Ok(Self::Musl),
            "eabi" => Ok(Self::Eabi),
            "eabihf" => Ok(Self::Eabihf),
            "none" => Ok(Self::Freestanding),
            _ => Err("Unknown target environment"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_round_trip() {
        for env in TargetEnvironment::ALL {
            assert_eq!(env.token().parse(), Ok(env));
            assert_eq!(env.to_string(), env.token());
        }
        assert!("glibc".parse::<TargetEnvironment>().is_err());
    }

    #[test]
    fn freestanding_links_neither_libc_nor_a_compiler_runtime() {
        assert_eq!(TargetEnvironment::Freestanding.libc(), None);
        assert_eq!(
            TargetEnvironment::Freestanding.compiler_runtime(),
            CompilerRuntime::None
        );
        assert!(!TargetEnvironment::Freestanding.is_hosted());
    }

    #[test]
    fn musl_takes_the_same_compiler_runtime_as_gnu() {
        assert_eq!(TargetEnvironment::Musl.libc(), Some(Libc::Musl));
        assert_eq!(
            TargetEnvironment::Musl.compiler_runtime(),
            CompilerRuntime::LaminaBuiltins
        );
        assert_eq!(
            TargetEnvironment::Gnu.compiler_runtime(),
            CompilerRuntime::LaminaBuiltins
        );
        assert_eq!(Libc::Musl.triple_component(), "musl");
        assert_eq!(Libc::Glibc.triple_component(), "gnu");
    }

    #[test]
    fn the_host_runtime_is_needed_only_where_our_builtins_fall_short() {
        use crate::TargetArchitecture;
        let runtime = TargetEnvironment::Gnu.compiler_runtime();
        for arch in [
            TargetArchitecture::I686,
            TargetArchitecture::Armv7,
            TargetArchitecture::Riscv32,
            TargetArchitecture::Thumbv6m,
            TargetArchitecture::Thumbv7m,
        ] {
            assert!(runtime.requires_host_runtime(&arch.spec()), "{arch}");
        }
        for arch in [
            TargetArchitecture::X86_64,
            TargetArchitecture::Aarch64,
            TargetArchitecture::Riscv64,
            TargetArchitecture::S390x,
            TargetArchitecture::PowerPC64,
        ] {
            assert!(!runtime.requires_host_runtime(&arch.spec()), "{arch}");
        }
        assert!(!CompilerRuntime::None.requires_host_runtime(&TargetArchitecture::I686.spec()));
    }

    #[test]
    fn only_eabi_and_freestanding_are_soft_float() {
        assert!(!TargetEnvironment::Eabi.hardware_float());
        assert!(!TargetEnvironment::Freestanding.hardware_float());
        assert!(TargetEnvironment::Eabihf.hardware_float());
        assert!(TargetEnvironment::Gnu.hardware_float());
        assert!(TargetEnvironment::Musl.hardware_float());
    }
}
