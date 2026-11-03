//! Immediate operand forms for AArch64 data processing instructions.
//!
//! A lowering asks whether a constant can ride along inside the instruction that consumes
//! it. When the answer is no the caller falls back to materialising the constant into a
//! register, so every test here refuses rather than approximates.

/// The `#imm` text for the 12-bit unsigned immediate of `add`, `sub`, `cmp` and `cmn`,
/// optionally shifted left by twelve, or `None` when `value` needs a register.
pub fn arith_immediate(value: u64) -> Option<String> {
    if value <= 0xFFF {
        return Some(format!("#{value}"));
    }
    if value & 0xFFF == 0 && (value >> 12) <= 0xFFF {
        return Some(format!("#{}, lsl #12", value >> 12));
    }
    None
}

/// The same form for the negation of `value`, used to turn `add` into `sub`, `sub` into
/// `add` and `cmp` into `cmn` when the constant is negative.
pub fn negated_arith_immediate(value: u64) -> Option<String> {
    arith_immediate((value as i64).checked_neg()? as u64)
}

/// The `#imm` text for the bitmask immediate of `and`, `orr` and `eor` on an x register, or
/// `None` when `value` is not a repeated rotated run of ones.
pub fn logical_immediate(value: u64) -> Option<String> {
    is_logical_immediate(value).then(|| format!("#0x{value:x}"))
}

/// Whether `value` is encodable as a 64-bit logical immediate, that is as `N:immr:imms`.
///
/// The encoding holds an element of 2, 4, 8, 16, 32 or 64 bits repeated to fill the
/// register, and that element must be a rotation of a run of ones. All-zeros and all-ones
/// have no encoding.
fn is_logical_immediate(value: u64) -> bool {
    if value == 0 || value == u64::MAX {
        return false;
    }
    let mut element = 64u32;
    while element > 2 {
        let half = element / 2;
        let low = (1u64 << half) - 1;
        if value & low != (value >> half) & low {
            break;
        }
        element = half;
    }
    let element_mask = if element == 64 {
        u64::MAX
    } else {
        (1u64 << element) - 1
    };
    let pattern = value & element_mask;
    if pattern == 0 || pattern == element_mask {
        return false;
    }
    let ones = pattern.count_ones();
    let run = (1u64 << ones) - 1;
    (0..element).any(|r| {
        let rotated = if r == 0 {
            run
        } else {
            ((run >> r) | (run << (element - r))) & element_mask
        };
        rotated == pattern
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn small_constant_is_an_arithmetic_immediate() {
        assert_eq!(arith_immediate(0).as_deref(), Some("#0"));
        assert_eq!(arith_immediate(4095).as_deref(), Some("#4095"));
    }

    #[test]
    fn multiple_of_four_thousand_ninety_six_uses_the_shifted_form() {
        assert_eq!(arith_immediate(4096).as_deref(), Some("#1, lsl #12"));
        assert_eq!(arith_immediate(16773120).as_deref(), Some("#4095, lsl #12"));
    }

    #[test]
    fn constant_between_the_two_forms_needs_a_register() {
        assert_eq!(arith_immediate(4096 + 1), None);
        assert_eq!(arith_immediate(12345), None);
        assert_eq!(arith_immediate(16773120 + 4096), None);
    }

    #[test]
    fn negative_constant_folds_into_the_opposite_mnemonic() {
        assert_eq!(negated_arith_immediate(-1i64 as u64).as_deref(), Some("#1"));
        assert_eq!(negated_arith_immediate(-12345i64 as u64), None);
        assert_eq!(negated_arith_immediate(i64::MIN as u64), None);
    }

    #[test]
    fn run_of_ones_is_a_logical_immediate() {
        assert_eq!(logical_immediate(0xFFFF).as_deref(), Some("#0xffff"));
        assert_eq!(
            logical_immediate(0x7FFF_FFFF).as_deref(),
            Some("#0x7fffffff")
        );
        assert_eq!(logical_immediate(1).as_deref(), Some("#0x1"));
        assert!(logical_immediate(0xFFFF_FFFF_FFFF_FF00).is_some());
    }

    #[test]
    fn repeated_element_is_a_logical_immediate() {
        assert!(logical_immediate(0x5555_5555_5555_5555).is_some());
        assert!(logical_immediate(0xAAAA_AAAA_AAAA_AAAA).is_some());
        assert!(logical_immediate(0x00F0_00F0_00F0_00F0).is_some());
    }

    #[test]
    fn all_zeros_and_all_ones_have_no_logical_encoding() {
        assert_eq!(logical_immediate(0), None);
        assert_eq!(logical_immediate(u64::MAX), None);
    }

    #[test]
    fn broken_pattern_has_no_logical_encoding() {
        assert_eq!(logical_immediate(12345), None);
        assert_eq!(logical_immediate(0x4000_0001), None);
        assert_eq!(logical_immediate(0x1234_5678_9ABC_DEF0), None);
    }

    /// The predicate must agree with the definition it stands in for: a bitmask immediate is
    /// an element of 2, 4, 8, 16, 32 or 64 bits, holding a rotation of a run of ones shorter
    /// than the element, repeated to fill the register.
    #[test]
    fn predicate_agrees_with_the_enumerated_set() {
        let mut encodable = HashSet::new();
        for element in [2u32, 4, 8, 16, 32, 64] {
            let element_mask = if element == 64 {
                u64::MAX
            } else {
                (1u64 << element) - 1
            };
            for ones in 1..element {
                let run = (1u64 << ones) - 1;
                for r in 0..element {
                    let rotated = if r == 0 {
                        run
                    } else {
                        ((run >> r) | (run << (element - r))) & element_mask
                    };
                    let mut value = 0u64;
                    let mut shift = 0;
                    while shift < 64 {
                        value |= rotated << shift;
                        shift += element;
                    }
                    encodable.insert(value);
                }
            }
        }
        assert_eq!(encodable.len(), 5334);
        for v in &encodable {
            assert!(is_logical_immediate(*v), "{v:#x} should be encodable");
        }
        for v in 0..=0xFFFFu64 {
            assert_eq!(is_logical_immediate(v), encodable.contains(&v), "{v:#x}");
        }
        for v in [
            0u64,
            u64::MAX,
            12345,
            0x4000_0001,
            0x1234_5678_9ABC_DEF0,
            0xDEAD_BEEF,
            0x7FFF_FFFF,
            0xFFFF_FFFF_FFFF_FF00,
        ] {
            assert_eq!(is_logical_immediate(v), encodable.contains(&v), "{v:#x}");
        }
    }
}
