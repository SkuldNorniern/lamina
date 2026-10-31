//! Machine vector register widths, as a property of the target rather than a constant.

use crate::arch_table::TargetArchitecture;

impl TargetArchitecture {
    /// Bit widths a vector register of this architecture comes in, ascending.
    ///
    /// Empty for an architecture whose backend emits no vector registers, and for a
    /// scalable vector machine whose width is not a compile-time constant. A `<N x T>`
    /// whose lanes do not add up to one of these has no register to live in.
    pub fn vector_register_bits(self) -> &'static [u32] {
        match self {
            TargetArchitecture::X86_64 => &[128, 256],
            TargetArchitecture::Aarch64
            | TargetArchitecture::Armv7
            | TargetArchitecture::Wasm32
            | TargetArchitecture::Wasm64 => &[128],
            TargetArchitecture::I686
            | TargetArchitecture::Arm32
            | TargetArchitecture::Thumbv6m
            | TargetArchitecture::Thumbv7m
            | TargetArchitecture::Riscv32
            | TargetArchitecture::Riscv64
            | TargetArchitecture::PowerPC64
            | TargetArchitecture::PowerPC64Be
            | TargetArchitecture::Mips64
            | TargetArchitecture::Mips64El
            | TargetArchitecture::LoongArch64
            | TargetArchitecture::S390x
            | TargetArchitecture::Nvptx64
            | TargetArchitecture::Amdgcn => &[],
            #[cfg(feature = "nightly")]
            TargetArchitecture::Arx64 | TargetArchitecture::Riscv128 | TargetArchitecture::Lisa => {
                &[]
            }
        }
    }
}

/// Every vector register width some target has.
///
/// What a check with no target in hand can say: a width absent here fits no machine at
/// all, while a width present here still has to be one the chosen target offers, which
/// [`TargetArchitecture::vector_register_bits`] answers.
pub fn any_vector_register_bits() -> &'static [u32] {
    &[128, 256]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_width_is_one_a_target_independent_check_allows() {
        for architecture in TargetArchitecture::all() {
            for bits in architecture.vector_register_bits() {
                assert!(
                    any_vector_register_bits().contains(bits),
                    "{architecture:?} has {bits}-bit vectors that no target-independent check allows"
                );
            }
        }
    }

    #[test]
    fn widths_are_ascending_and_powers_of_two() {
        for architecture in TargetArchitecture::all() {
            let bits = architecture.vector_register_bits();
            assert!(bits.windows(2).all(|w| w[0] < w[1]), "{architecture:?}");
            assert!(bits.iter().all(|b| b.is_power_of_two()), "{architecture:?}");
        }
    }
}
