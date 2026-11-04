//! Whole-word division by a constant as a multiply by the reciprocal.
//!
//! `sdiv` and `udiv` take several times longer than `smulh`. The constants are from
//! Hacker's Delight, chapter 10.

/// The multiplier and right shift that divide a signed word by `divisor`, which must not
/// be -1, 0, 1 or `i64::MIN`.
pub(super) fn signed_reciprocal(divisor: i64) -> (i64, u32) {
    const TWO_63: u64 = 1 << 63;
    let magnitude = divisor.unsigned_abs();
    let bound = TWO_63 + ((divisor as u64) >> 63);
    let limit = bound - 1 - bound % magnitude;
    let mut power = 63;
    let (mut q1, mut q2) = (TWO_63 / limit, TWO_63 / magnitude);
    let (mut r1, mut r2) = (TWO_63 - q1 * limit, TWO_63 - q2 * magnitude);
    loop {
        power += 1;
        q1 = q1.wrapping_mul(2);
        r1 = r1.wrapping_mul(2);
        if r1 >= limit {
            q1 = q1.wrapping_add(1);
            r1 -= limit;
        }
        q2 = q2.wrapping_mul(2);
        r2 = r2.wrapping_mul(2);
        if r2 >= magnitude {
            q2 = q2.wrapping_add(1);
            r2 -= magnitude;
        }
        let delta = magnitude - r2;
        if !(q1 < delta || (q1 == delta && r1 == 0)) {
            break;
        }
    }
    let multiplier = q2.wrapping_add(1) as i64;
    (
        if divisor < 0 {
            multiplier.wrapping_neg()
        } else {
            multiplier
        },
        power - 64,
    )
}

/// The multiplier, whether the product overflowed 64 bits, and the right shift that
/// divide an unsigned word by `divisor`, which must be at least 2.
pub(super) fn unsigned_reciprocal(divisor: u64) -> (u64, bool, u32) {
    const TWO_63: u64 = 1 << 63;
    const BELOW_TWO_63: u64 = TWO_63 - 1;
    let limit = u64::MAX - 0u64.wrapping_sub(divisor) % divisor;
    let mut power = 63;
    let mut add = false;
    let (mut q1, mut q2) = (TWO_63 / limit, BELOW_TWO_63 / divisor);
    let (mut r1, mut r2) = (TWO_63 - q1 * limit, BELOW_TWO_63 - q2 * divisor);
    loop {
        power += 1;
        if r1 >= limit - r1 {
            q1 = q1.wrapping_mul(2).wrapping_add(1);
            r1 = r1.wrapping_mul(2).wrapping_sub(limit);
        } else {
            q1 = q1.wrapping_mul(2);
            r1 = r1.wrapping_mul(2);
        }
        if r2 + 1 >= divisor - r2 {
            if q2 >= BELOW_TWO_63 {
                add = true;
            }
            q2 = q2.wrapping_mul(2).wrapping_add(1);
            r2 = r2.wrapping_mul(2).wrapping_add(1).wrapping_sub(divisor);
        } else {
            if q2 >= TWO_63 {
                add = true;
            }
            q2 = q2.wrapping_mul(2);
            r2 = r2.wrapping_mul(2).wrapping_add(1);
        }
        let delta = divisor - 1 - r2;
        if !(power < 128 && (q1 < delta || (q1 == delta && r1 == 0))) {
            break;
        }
    }
    (q2.wrapping_add(1), add, power - 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIVISORS: [i64; 24] = [
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        25,
        90,
        100,
        125,
        625,
        641,
        1000,
        4096,
        65537,
        1_000_000_007,
        (1 << 32) + 1,
        (1 << 40) - 87,
        (1 << 62) + 1,
        i64::MAX - 1,
        i64::MAX,
        i64::MIN + 1,
    ];

    fn numerators() -> Vec<i64> {
        let mut values = vec![
            0,
            1,
            -1,
            2,
            -2,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            i64::MAX - 1,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for shift in 0..64 {
            values.push((1u64 << shift) as i64);
            values.push(((1u64 << shift) - 1) as i64);
            values.push((1u64 << shift).wrapping_neg() as i64);
        }
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(state as i64);
        }
        values
    }

    /// The instruction sequence `emit_signed` writes, on plain integers.
    fn signed_by_reciprocal(n: i64, d: i64) -> i64 {
        let (m, s) = signed_reciprocal(d);
        let mut q = ((i128::from(n) * i128::from(m)) >> 64) as i64;
        if d > 0 && m < 0 {
            q = q.wrapping_add(n);
        }
        if d < 0 && m > 0 {
            q = q.wrapping_sub(n);
        }
        q >>= s;
        q.wrapping_add(((q as u64) >> 63) as i64)
    }

    fn unsigned_by_reciprocal(n: u64, d: u64) -> u64 {
        let (m, add, s) = unsigned_reciprocal(d);
        let q = ((u128::from(n) * u128::from(m)) >> 64) as u64;
        if add {
            (((n - q) >> 1) + q) >> (s - 1)
        } else {
            q >> s
        }
    }

    #[test]
    fn signed_reciprocal_matches_division() {
        let numerators = numerators();
        for d in DIVISORS {
            for d in [d, d.wrapping_neg()] {
                for &n in &numerators {
                    assert_eq!(signed_by_reciprocal(n, d), n.wrapping_div(d), "{n} / {d}");
                }
            }
        }
    }

    #[test]
    fn unsigned_reciprocal_matches_division() {
        let numerators = numerators();
        for d in DIVISORS {
            for d in [d as u64, u64::MAX - d as u64 / 2, 1 << (d as u64 % 63 + 1)] {
                if d < 2 {
                    continue;
                }
                for &n in &numerators {
                    let n = n as u64;
                    assert_eq!(unsigned_by_reciprocal(n, d), n / d, "{n} / {d}");
                }
            }
        }
    }
}
