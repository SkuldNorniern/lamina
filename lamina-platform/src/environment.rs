//! The C runtime and ABI a target means.
//!
//! The fourth axis of a target, beside the architecture, the operating system and the
//! sub-architecture variant. It is what decides the ELF interpreter, the C runtime
//! objects, the libc and the compiler runtime, so that `s390x_linux` *declares* glibc
//! rather than inheriting it from whichever cross compiler the build machine has.

use std::{fmt, str::FromStr};

/// The C library an environment links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Libc {
    Glibc,
    Musl,
    /// Lamina's own C library. The name is reserved and the plumbing is in place; no
    /// implementation exists yet, so [`Libc::is_implemented`] answers `false` for it.
    Lamina,
}

impl Libc {
    /// Component of a cross-toolchain triple that names this C library.
    pub const fn triple_component(self) -> &'static str {
        match self {
            Self::Glibc => "gnu",
            Self::Musl => "musl",
            Self::Lamina => "lamina",
        }
    }

    /// Whether lamina has an implementation of this C library to link a program against.
    ///
    /// `false` is a promise that the link refuses and names the library, rather than
    /// taking another C library's interpreter, start-up objects and symbols instead.
    pub const fn is_implemented(self) -> bool {
        match self {
            Self::Glibc | Self::Musl => true,
            Self::Lamina => false,
        }
    }

    /// Why this C library cannot be linked, for a diagnostic, or `None` when it can.
    pub const fn unavailable_reason(self) -> Option<&'static str> {
        match self {
            Self::Glibc | Self::Musl => None,
            Self::Lamina => Some(
                "the Lamina C library is not implemented yet: there is no Lamina libc to link, no start-up objects and no dynamic loader for it",
            ),
        }
    }
}

impl fmt::Display for Libc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Glibc => "glibc",
            Self::Musl => "musl",
            Self::Lamina => "lamina",
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
    /// Lamina's own `builtins/`, compiled per program for the target. It has both the
    /// 128-bit helpers and the 64-bit ones a target with 32-bit integer registers needs,
    /// so no host compiler runtime is required beside it.
    LaminaBuiltins,
    /// No compiler runtime is linked; a call to a helper is a link error.
    None,
}

/// The C runtime and ABI a target means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetEnvironment {
    /// glibc, with the architecture's own floating-point ABI.
    Gnu,
    /// musl libc.
    Musl,
    /// Lamina's own C library, beside musl rather than on top of it.
    ///
    /// Reserved: the axis carries the choice so the target grammar is settled, but no
    /// Lamina libc has been written. Every environment-derived property that would name
    /// a file answers `None` for it, and a link refuses by name.
    Lamina,
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
    pub const ALL: [Self; 6] = [
        Self::Gnu,
        Self::Musl,
        Self::Lamina,
        Self::Eabi,
        Self::Eabihf,
        Self::Freestanding,
    ];

    /// Spelling in a target string.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Gnu => "gnu",
            Self::Musl => "musl",
            Self::Lamina => "lamina",
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
            Self::Lamina => Some(Libc::Lamina),
            Self::Freestanding => None,
        }
    }

    /// Where the 128-bit integer helpers and the other compiler-runtime symbols come from.
    pub const fn compiler_runtime(self) -> CompilerRuntime {
        match self {
            Self::Gnu | Self::Musl | Self::Lamina | Self::Eabi | Self::Eabihf => {
                CompilerRuntime::LaminaBuiltins
            }
            Self::Freestanding => CompilerRuntime::None,
        }
    }

    /// Whether the calling convention passes floating-point values in floating-point
    /// registers. This is the eabi/eabihf distinction on 32-bit ARM.
    pub const fn hardware_float(self) -> bool {
        match self {
            Self::Gnu | Self::Musl | Self::Lamina | Self::Eabihf => true,
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
            "lamina" => Ok(Self::Lamina),
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
    fn every_hosted_environment_selects_our_own_compiler_runtime() {
        for env in TargetEnvironment::ALL {
            let expected = if env.is_hosted() {
                CompilerRuntime::LaminaBuiltins
            } else {
                CompilerRuntime::None
            };
            assert_eq!(env.compiler_runtime(), expected, "{env}");
        }
    }

    #[test]
    fn lamina_is_a_libc_of_its_own_taking_our_own_compiler_runtime() {
        let env = TargetEnvironment::Lamina;
        assert_eq!(env.token(), "lamina");
        assert_eq!(env.libc(), Some(Libc::Lamina));
        assert_eq!(env.compiler_runtime(), CompilerRuntime::LaminaBuiltins);
        assert!(env.is_hosted());
        assert!(env.hardware_float());
        assert_eq!(Libc::Lamina.triple_component(), "lamina");
        assert_eq!(Libc::Lamina.to_string(), "lamina");
    }

    #[test]
    fn the_lamina_libc_is_the_one_that_is_not_implemented_yet() {
        assert!(Libc::Glibc.is_implemented());
        assert!(Libc::Musl.is_implemented());
        assert!(!Libc::Lamina.is_implemented());
        assert_eq!(Libc::Glibc.unavailable_reason(), None);
        assert_eq!(Libc::Musl.unavailable_reason(), None);
        let reason = Libc::Lamina
            .unavailable_reason()
            .expect("an unimplemented libc states why");
        assert!(reason.contains("Lamina C library"), "{reason}");
        assert!(reason.contains("not implemented"), "{reason}");
    }

    #[test]
    fn an_unimplemented_libc_is_still_a_hosted_environment() {
        for env in TargetEnvironment::ALL {
            let hosted = env.libc().is_some();
            assert_eq!(env.is_hosted(), hosted, "{env}");
        }
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
