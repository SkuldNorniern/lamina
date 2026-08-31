//! Which RISC-V a target actually is: register width and extension set.
//!
//! `TargetArchitecture` only distinguishes `Riscv32` from `Riscv64`, so the backend had
//! no way to say whether multiply and divide were available. It emitted `mul` and `div`
//! unconditionally, which are M-extension instructions an rv64i core cannot execute.

use lamina_platform::TargetArchitecture;

/// Register width in bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Xlen {
    Rv32,
    Rv64,
}

impl Xlen {
    pub fn bits(self) -> u32 {
        match self {
            Xlen::Rv32 => 32,
            Xlen::Rv64 => 64,
        }
    }
}

/// Base integer ISA plus the standard extensions the backend cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiscVTarget {
    pub xlen: Xlen,
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
}

impl RiscVTarget {
    /// The base integer ISA on its own, rv32i or rv64i.
    pub fn base(xlen: Xlen) -> Self {
        Self {
            xlen,
            m: false,
            a: false,
            f: false,
            d: false,
            c: false,
        }
    }

    /// G, the conventional general-purpose set: IMAFD.
    pub fn general(xlen: Xlen) -> Self {
        Self {
            xlen,
            m: true,
            a: true,
            f: true,
            d: true,
            c: false,
        }
    }

    /// Default for a bare `riscv32`/`riscv64` target.
    ///
    /// G, matching what most toolchains assume, and what this backend already emitted
    /// before extensions were modelled at all.
    pub fn from_arch(arch: TargetArchitecture) -> Option<Self> {
        match arch {
            TargetArchitecture::Riscv32 => Some(Self::general(Xlen::Rv32)),
            TargetArchitecture::Riscv64 => Some(Self::general(Xlen::Rv64)),
            _ => None,
        }
    }

    /// Bytes in a general-purpose register.
    pub fn word_bytes(self) -> i32 {
        match self.xlen {
            Xlen::Rv32 => 4,
            Xlen::Rv64 => 8,
        }
    }

    /// Mnemonic for storing a whole register. `sd` has no rv32 encoding.
    pub fn store_word(self) -> &'static str {
        match self.xlen {
            Xlen::Rv32 => "sw",
            Xlen::Rv64 => "sd",
        }
    }

    /// Mnemonic for loading a whole register.
    pub fn load_word(self) -> &'static str {
        match self.xlen {
            Xlen::Rv32 => "lw",
            Xlen::Rv64 => "ld",
        }
    }

    /// Check the float extension a width needs is present.
    ///
    /// F covers f32, D covers f64. Emitting `fadd.d` on a target without D produces an
    /// instruction the core traps on.
    pub fn require_float(self, is_f32: bool) -> Result<(), String> {
        let (have, needs) = if is_f32 { (self.f, 'F') } else { (self.d, 'D') };
        if have {
            return Ok(());
        }
        Err(format!(
            "needs the {needs} extension, but the target is {}. Select an ISA with \
             {needs} or lower the operation to soft float before codegen.",
            self.isa_name()
        ))
    }

    /// ISA string, for diagnostics.
    pub fn isa_name(&self) -> String {
        let mut s = format!("rv{}i", self.xlen.bits());
        for (present, letter) in [
            (self.m, 'm'),
            (self.a, 'a'),
            (self.f, 'f'),
            (self.d, 'd'),
            (self.c, 'c'),
        ] {
            if present {
                s.push(letter);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_has_no_multiply() {
        assert!(!RiscVTarget::base(Xlen::Rv64).m);
        assert!(RiscVTarget::general(Xlen::Rv64).m);
    }

    #[test]
    fn float_widths_need_their_extension() {
        let base = RiscVTarget::base(Xlen::Rv64);
        assert!(base.require_float(true).is_err(), "rv64i has no F");
        assert!(base.require_float(false).is_err(), "rv64i has no D");

        let g = RiscVTarget::general(Xlen::Rv64);
        assert!(g.require_float(true).is_ok());
        assert!(g.require_float(false).is_ok());

        let f_only = RiscVTarget {
            f: true,
            ..RiscVTarget::base(Xlen::Rv32)
        };
        assert!(f_only.require_float(true).is_ok(), "F covers f32");
        let err = f_only
            .require_float(false)
            .expect_err("F does not cover f64");
        assert!(err.contains('D'), "message should name D: {err}");
        assert!(err.contains("rv32if"), "message should name the ISA: {err}");
    }

    #[test]
    fn word_size_follows_xlen() {
        let rv32 = RiscVTarget::general(Xlen::Rv32);
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        assert_eq!(
            (rv32.word_bytes(), rv32.store_word(), rv32.load_word()),
            (4, "sw", "lw")
        );
        assert_eq!(
            (rv64.word_bytes(), rv64.store_word(), rv64.load_word()),
            (8, "sd", "ld")
        );
    }

    #[test]
    fn isa_names_read_the_usual_way() {
        assert_eq!(RiscVTarget::base(Xlen::Rv32).isa_name(), "rv32i");
        assert_eq!(RiscVTarget::general(Xlen::Rv64).isa_name(), "rv64imafd");
    }

    #[test]
    fn bare_targets_default_to_general() {
        let t = RiscVTarget::from_arch(TargetArchitecture::Riscv64).expect("riscv64");
        assert_eq!(t.xlen, Xlen::Rv64);
        assert!(
            t.m,
            "default must keep multiply, the backend already emitted it"
        );
        assert_eq!(RiscVTarget::from_arch(TargetArchitecture::X86_64), None);
    }
}
