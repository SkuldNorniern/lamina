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

Not yet written: `__floattidf`, `__floatuntidf`, `__fixdfti`, `__fixunsdfti`,
`__floattisf`, `__floatuntisf`, `__fixsfti`, `__fixunssfti`.

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
