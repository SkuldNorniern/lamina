//! x86-64 microarchitecture levels: the instruction-set baseline a generated binary may rely on.

use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum X86CpuLevel {
    V1,
    #[default]
    V2,
    V3,
    V4,
}

impl X86CpuLevel {
    pub const ALL: [X86CpuLevel; 4] = [Self::V1, Self::V2, Self::V3, Self::V4];

    pub fn features(self) -> &'static str {
        match self {
            Self::V1 => "SSE2",
            Self::V2 => "SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B",
            Self::V3 => "v2 + AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE",
            Self::V4 => "v3 + AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL",
        }
    }
}

impl fmt::Display for X86CpuLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
            Self::V4 => 4,
        };
        write!(f, "x86-64-v{n}")
    }
}

impl FromStr for X86CpuLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let digit = lower
            .strip_prefix("x86-64-v")
            .or_else(|| lower.strip_prefix('v'))
            .unwrap_or("");
        match digit {
            "1" => Ok(Self::V1),
            "2" => Ok(Self::V2),
            "3" => Ok(Self::V3),
            "4" => Ok(Self::V4),
            _ => Err(format!(
                "unknown CPU level '{s}' (expected v1, v2, v3 or v4)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_short_and_psabi_names() {
        assert_eq!("v1".parse(), Ok(X86CpuLevel::V1));
        assert_eq!("V2".parse(), Ok(X86CpuLevel::V2));
        assert_eq!("x86-64-v3".parse(), Ok(X86CpuLevel::V3));
        assert_eq!("X86-64-V4".parse(), Ok(X86CpuLevel::V4));
        assert!("v5".parse::<X86CpuLevel>().is_err());
        assert!("native".parse::<X86CpuLevel>().is_err());
    }

    #[test]
    fn levels_are_ordered_and_default_to_v2() {
        assert!(X86CpuLevel::V1 < X86CpuLevel::V2);
        assert!(X86CpuLevel::V4 > X86CpuLevel::V3);
        assert_eq!(X86CpuLevel::default(), X86CpuLevel::V2);
        assert_eq!(X86CpuLevel::V2.to_string(), "x86-64-v2");
    }
}
