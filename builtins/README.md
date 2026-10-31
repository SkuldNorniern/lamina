# lamina-builtins

Layer 1 of the runtime: the compiler runtime routines that lamina's own code
generation calls for 128-bit integer arithmetic. The user never writes these
calls; i128 lowering in `src/mir/codegen/convert/wide.rs` emits them.

This is not libc. Nothing here is called by a program that did not ask for
128-bit division, and nothing here provides printf, malloc or any other
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


## Linking

The compiler embeds these sources and links them itself. When a module calls
one of the twelve symbols, leaves it undefined, and has not redirected it with
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
- `__divti3` and `__modti3` call their unsigned counterparts, so link all four
  objects together when using the signed routines.
- Division by zero traps. The routines use the same `div.u64 1, x` idiom as
  the compiler's inline i128 division, so the trap is a hardware fault on x86
  and is not guaranteed on targets whose integer divide does not trap.
- `__divti3(MIN, -1)` and `__modti3(MIN, -1)` trap, matching the compiler's
  inline path. They do not return the wrapped value that libgcc returns.

## Algorithm

`__udivti3` and `__umodti3` use a fast path when both operands fit in 64 bits
(one `div.u64` or `rem.u64`). Otherwise they run restoring shift-and-subtract
division over 128 bit positions, using only 128-bit shifts, `or`, `sub` and
comparisons. The IR cannot use 128-bit `div` or `rem` here, because lowering
them calls back into these routines.

## Float conversions

Int to float rounds to nearest, ties to even, in a single rounding. The
magnitude is normalised with `ctlz`, the top 64 bits are kept, and the bits
below them are OR'd into bit 0 as a sticky bit. One hardware `inttofp.u64`
then rounds that 64-bit value to 53 or 24 bits, so the sticky bit always sits
below the rounding position. The result is scaled by an exact power of two.
Rounding once from 64 bits avoids the double rounding that
`f32(f64(x))` would do, which gets some 128-bit values wrong.

- `__floatuntidf`, `__floatuntisf`: zero gives +0. A value that rounds past
  the top of the range gives infinity (`u128::MAX` as f32 is `+inf`, as IEEE
  requires; as f64 it is 2^128).
- `__floattidf`, `__floattisf`: round the magnitude symmetrically and negate.
  `i128::MIN` converts exactly to -2^127.

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

Signed routines call the unsigned ones on `|x|`. Link all eight objects
together when using the signed routines.

Verified against an independent Python oracle (exact half-even integer
rounding and exact truncation with saturation). Over 5,260 128-bit inputs,
10,871 f64 inputs and 3,043 f32 inputs, all eight routines match the oracle
with zero mismatches. The set includes 116 exact f64 ties and 65 exact f32
ties, plus powers of two around 2^24, 2^25, 2^53 and 2^54, 2^128 minus
2^103 for the f32 overflow boundary, NaN payloads, infinities and negative
zero. Against the host libgcc, the int-to-float routines agree on every input.
The float-to-int routines agree on every in-range input and differ only where
libgcc is undefined.

