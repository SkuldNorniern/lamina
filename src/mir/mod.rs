/// LUMIR — Lamina Unified Machine Intermediate Representation
///
/// LUMIR is a **low-level, machine-friendly layer** produced after IR Processing
/// (and used before/after register allocation). It is assembly-like
/// easy to apply optimizations, and straightforward to lower into target assembly.
///
/// Types
///   Scalars : i8 | i16 | i32 | i64 | f32 | f64 | ptr | i1
///   Vectors : v128<lane> | v256<lane>  where lane is one of { i8, i16, i32, i64, f32, f64 }
///
/// Registers & classes
///   Virtual regs: v0, v1, …
///   Classes     : gpr (int/ptr), fpr (scalar FP), vec (SIMD)
///   Note        : physical regs appear only post-RA
///
/// Addressing
///   [base + imm12]  |  [base + idx<<scale + imm4]  (where scale is 1, 2, 4, or 8)
///   'lea' computes addresses; backends may fold into ld/st if legal
///
/// Core ops
///   int   : add/sub/mul/udiv/sdiv/urem/srem/and/or/xor/not/shl/lshr/ashr/…
///   fp    : fadd/fsub/fmul/fdiv/fneg/fsqrt/(fma?)
///   cmp   : cmp.{eq,ne,ult,ule,ugt,uge,slt,sle,sgt,sge}.i* → i1
///           fcmp.{eq,ne,lt,le,gt,ge}.f*                   → i1
///   select: select <ty> dst, i1 cond, r_true, r_false      (scalar or vector)
///   mem   : ld.<ty> dst, [addr] {align=A, volatile?}
///           st.<ty> src, [addr] {align=A, volatile?}
///           ld.v128<lane> / st.v128<lane> (and v256<lane>)
///   vec   : vadd/vsub/vmul/vand/vor/vxor/vshl/vlshr/vashr
///           vsplat/vextractlane/vinsertlane/vshuffle(mask)
///   ctrl  : jmp, br, switch, call, ret
///   meta  : unrechable, safepoint, stackmap <id>, patchpoint <id> (for GC and profiling)
///
/// Conventions
///   Arguments: v0..v7  •  Return: v0
///   The abstract calling convention is mapped to the real ABI during emission.
///
/// Example (tiny)
///   .func add_store
///   bb0:
///     v2:i64 = add.i64 v0, v1
///     lea     vA, v3, 0
///     st.i64  v2, [vA] {align=8}
///     ret
///   .endfunc