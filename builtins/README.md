# lamina-builtins

Layer 1 of the runtime: the compiler runtime routines that lamina's own code
generation calls for integer arithmetic no register holds. The user never
writes these calls; the wide lowering in `src/mir/codegen/convert/wide.rs`
emits them, for `i128` everywhere and for `i64` on a target whose integer
registers are 32 bits wide.

This is not libc. Nothing here is called by a program that did not ask for
wide division, and nothing here provides printf, malloc or any other
library function. Layer 2 (libc) is a separate, optional project. See
`Plan/runtime-layering.md`.

## Files

| file | symbol | status |
| --- | --- | --- |
| `udivti3.lamina` | `__udivti3` | done |
| `umodti3.lamina` | `__umodti3` | done |
| `divti3.lamina` | `__divti3` | done, calls `__udivti3` |
| `modti3.lamina` | `__modti3` | done, calls `__umodti3` |
| `floatuntidf.lamina` | `__floatuntidf` | done |
| `floattidf.lamina` | `__floattidf` | done, calls `__floatuntidf` |
| `floatuntisf.lamina` | `__floatuntisf` | done |
| `floattisf.lamina` | `__floattisf` | done, calls `__floatuntisf` |
| `fixunsdfti.lamina` | `__fixunsdfti` | done |
| `fixdfti.lamina` | `__fixdfti` | done, calls `__fixunsdfti` |
| `fixunssfti.lamina` | `__fixunssfti` | done |
| `fixsfti.lamina` | `__fixsfti` | done, calls `__fixunssfti` |
| `udivdi3.lamina` | `__udivdi3` | done |
| `umoddi3.lamina` | `__umoddi3` | done |
| `divdi3.lamina` | `__divdi3` | done, calls `__udivdi3` |
| `moddi3.lamina` | `__moddi3` | done, calls `__umoddi3` |
| `floatundidf.lamina` | `__floatundidf` | done |
| `floatdidf.lamina` | `__floatdidf` | done, calls `__floatundidf` |
| `floatundisf.lamina` | `__floatundisf` | done |
| `floatdisf.lamina` | `__floatdisf` | done, calls `__floatundisf` |
| `fixunsdfdi.lamina` | `__fixunsdfdi` | done |
| `fixdfdi.lamina` | `__fixdfdi` | done, calls `__fixunsdfdi` |
| `fixunssfdi.lamina` | `__fixunssfdi` | done |
| `fixsfdi.lamina` | `__fixsfdi` | done, calls `__fixunssfdi` |

The `ti` half is reached by `i128` arithmetic on every target. The `di` half is
reached by plain `i64` arithmetic on i686, armv7, arm32, thumbv6m, thumbv7m and
riscv32, which split every 64-bit integer into a register pair. With both
halves present no target needs the host `libgcc.a`; the driver still puts it on
the link line when the host compiler has one, but the link no longer fails
without it.


## Linking

The compiler embeds these sources and links them itself. When a module calls
one of the twenty-four symbols, leaves it undefined, and has not redirected it with
`LAMINA_BUILTINS`, the driver compiles that routine and everything it calls for
the requested target and names the objects on the link line ahead of `-lc` and
any host archive. A program that never reaches one of these symbols builds
exactly as it did before and gains nothing. A routine the target's backend
cannot compile yet is left out and resolved from the host compiler runtime
instead.

## Conventions

- Each routine is `@export`, so it has global linkage and can be linked
  against a program that declares it `@extern`.
- A 128-bit argument or result uses the same two-word convention the compiler
  uses for every `i128` and `u128` function boundary. The routines are plain
  IR functions with no special ABI.
- `__divti3`, `__modti3`, `__divdi3` and `__moddi3` call their unsigned
  counterparts, so link each signed routine with the unsigned one it calls.
- Division by zero traps. The routines divide one by the folded-together halves
  of the divisor, so the trap is a hardware fault on x86 and is not guaranteed
  on targets whose integer divide does not trap.
- `__divti3(MIN, -1)`, `__modti3(MIN, -1)`, `__divdi3(MIN, -1)` and
  `__moddi3(MIN, -1)` trap, matching the compiler's inline path. They do not
  return the wrapped value that libgcc returns.

## Algorithm

`__udivti3` and `__umodti3` use a fast path when both operands fit in 64 bits
(one `div.u64` or `rem.u64`). Otherwise they run restoring shift-and-subtract
division over 128 bit positions, using only 128-bit shifts, `or`, `sub` and
comparisons. The IR cannot use 128-bit `div` or `rem` here, because lowering
them calls back into these routines.

`__udivdi3` and `__umoddi3` are the same algorithm one width down: a fast path
of one `div.u32` or `rem.u32` when both operands fit in 32 bits, and otherwise
64 rounds of restoring division built from 64-bit shifts, `or`, `sub` and
comparisons. They cannot use 64-bit `div` or `rem` either, for the same reason:
on the targets that call them, lowering a 64-bit divide emits a call to the
routine being written.

## Float conversions

Int to float rounds to nearest, ties to even, in a single rounding.

The 128-bit routines normalise the magnitude with `ctlz`, keep the top 64 bits,
and OR the bits below them into bit 0 as a sticky bit. One hardware
`inttofp.u64` then rounds that 64-bit value to 53 or 24 bits, so the sticky bit
always sits below the rounding position, and the result is scaled by an exact
power of two. Rounding once from 64 bits avoids the double rounding that
`f32(f64(x))` would do, which gets some 128-bit values wrong.

The 64-bit routines cannot borrow the hardware that way. `inttofp.u64` is the
very conversion they implement, and `inttofp.u32` rounds nothing, since 32 bits
land exactly in both formats. So `__floatundidf` and `__floatundisf` assemble
the result bit pattern themselves and round once by hand: shift the magnitude
left until bit 63 is set, take the top 53 or 24 bits as the significand, and
add the single increment `(rest + half - 1 + (significand & 1)) >> width`,
which is one exactly when the discarded remainder is above half, or is exactly
half and the significand is odd. A significand that carries out of its width
adds one to the exponent on its own, because the exponent field is added rather
than OR'd. The shift count comes from a six-step binary normalisation rather
than `ctlz`, which the riscv32 backend does not implement.

Nothing in the 64-bit conversions does floating-point arithmetic: they are
integer code and a `bitcast`. A soft-float target therefore reaches none of
`__adddf3` and its relatives through them.

- `__floatuntidf`, `__floatuntisf`: zero gives +0. A value that rounds past
  the top of the range gives infinity (`u128::MAX` as f32 is `+inf`, as IEEE
  requires; as f64 it is 2^128).
- `__floattidf`, `__floattisf`: round the magnitude symmetrically and negate.
  `i128::MIN` converts exactly to -2^127.
- `__floatundidf`, `__floatundisf`: zero gives +0. `u64::MAX` rounds up to
  2^64 in both formats, which both represent exactly.
- `__floatdidf`, `__floatdisf`: round the magnitude and set the sign bit.
  `i64::MIN` converts exactly to -2^63.

Float to int truncates toward zero. Out-of-range and NaN inputs are defined
here. libgcc leaves them undefined: on the host, NaN and +inf give garbage or
zero, so there is nothing to match, and this divergence is deliberate.

- In range: exact truncation.
- NaN: 0 for every target.
- Signed (`__fixdfti`, `__fixsfti`): saturates. Values at or above 2^127
  give `i128::MAX`, values at or below -2^127 give `i128::MIN`, and -inf gives
  `i128::MIN`.
- Unsigned (`__fixunsdfti`, `__fixunssfti`): any negative value, including
  -0.0, gives 0. Values at or above 2^128 give `u128::MAX`, and +inf gives
  `u128::MAX`.
- Signed (`__fixdfdi`, `__fixsfdi`): saturates. Values at or above 2^63 give
  `i64::MAX`, values at or below -2^63 give `i64::MIN`, and -inf gives
  `i64::MIN`.
- Unsigned (`__fixunsdfdi`, `__fixunssfdi`): any negative value, including
  -0.0, gives 0. Values at or above 2^64 give `u64::MAX`, and +inf gives
  `u64::MAX`.

Signed routines call the unsigned ones on `|x|`. Link each signed routine with
the unsigned one it calls.

## Verification

The 64-bit routines were checked against the same kind of independent Python
oracle: exact half-even rounding to a 53- or 24-bit significand and exact
truncation with the saturation defined above, all in arbitrary-precision
integers. 18,947 inputs produce 37,894 checked results, and every one of the
twelve routines matches the oracle with zero mismatches, both when the routines
are compiled for x86_64 and when they are compiled for i686 by lamina's own
32-bit backend and run under `-m32`. The inputs include every power of two and
its neighbours through 2^64, 300 exact f64 ties and 300 exact f32 ties
constructed so the discarded remainder is exactly half an ulp, the significand
boundaries at 2^24, 2^53 and 2^54, `i64::MIN`, `u64::MAX`, quiet and signalling
NaN payloads, both infinities, negative zero, subnormals, and random values
across the whole range. Division covers the 32-bit fast path, the 64-round slow
path and the mixed case, with `n/1`, `u64::MAX/n` and `n/n` over every boundary
value.

The 128-bit routines were verified the same way: over 5,260 128-bit inputs,
10,871 f64 inputs and 3,043 f32 inputs, all eight routines match the oracle
with zero mismatches. The set includes 116 exact f64 ties and 65 exact f32
ties, plus powers of two around 2^24, 2^25, 2^53 and 2^54, 2^128 minus
2^103 for the f32 overflow boundary, NaN payloads, infinities and negative
zero. Against the host libgcc, the int-to-float routines agree on every input.
The float-to-int routines agree on every in-range input and differ only where
libgcc is undefined.

