pub mod block;
pub mod codegen;
pub mod function;
pub mod instruction;
pub mod module;
pub mod register;
/// LUMIR — Lamina Unified Machine Intermediate Representation
///
/// LUMIR is a **low-level, machine-friendly layer** produced after IR Processing
/// (and used before/after register allocation). It is assembly-like,
/// easy to apply optimizations, and straightforward to lower into target assembly.
///
/// # Architecture Overview
///
/// ```text
/// Parser → IR → LUMIR → [Optimizations] → Code Generator → Assembly
/// ```
///
/// ## Types
///
/// - **Scalars**: `i8 | i16 | i32 | i64 | f32 | f64 | ptr | i1`
/// - **Vectors**: `v128<lane> | v256<lane>`
///   - where `lane` ∈ `{ i8, i16, i32, i64, f32, f64 }`
///
/// ## Registers & Classes
///
/// - **Virtual regs**: `v0, v1, v2, ...` (unlimited)
/// - **Register classes**:
///   - `gpr` — General Purpose (integers, pointers)
///   - `fpr` — Floating Point (scalar float ops)
///   - `vec` — SIMD Vector registers
/// - **Note**: Physical registers (like `%rax`, `x0`) appear only **post-RA** (after register allocation)
///
/// ## Addressing Modes
///
/// - **Simple**: `[base + imm12]` — base register + 12-bit immediate
/// - **Indexed**: `[base + idx<<scale + imm4]` — with scale ∈ `{1, 2, 4, 8}`
/// - **LEA**: `lea dst, base, offset` — computes addresses; backends may fold into `ld`/`st`
///
/// ## Core Operations
///
/// ### Integer Arithmetic
/// `add, sub, mul, udiv, sdiv, urem, srem, and, or, xor, not, shl, lshr, ashr`
///
/// ### Floating Point
/// `fadd, fsub, fmul, fdiv, fneg, fsqrt, (fma?)`
///
/// ### Comparisons
/// - **Integer**: `cmp.{eq,ne,ult,ule,ugt,uge,slt,sle,sgt,sge}.i* → i1`
/// - **Float**: `fcmp.{eq,ne,lt,le,gt,ge}.f* → i1`
///
/// ### Select (Conditional Move)
/// `select <ty> dst, i1 cond, r_true, r_false` — works on scalars & vectors
///
/// ### Memory Operations
/// - **Load**: `ld.<ty> dst, [addr] {align=A, volatile?}`
/// - **Store**: `st.<ty> src, [addr] {align=A, volatile?}`
/// - **Vector**: `ld.v128<lane>`, `st.v128<lane>` (and `v256<lane>`)
///
/// ### Vector Operations
/// `vadd, vsub, vmul, vand, vor, vxor, vshl, vlshr, vashr`  
/// `vsplat, vextractlane, vinsertlane, vshuffle(mask)`
///
/// ### Control Flow
/// `jmp, br, switch, call, ret`
///
/// ### Meta Operations
/// `unreachable, safepoint, stackmap <id>, patchpoint <id>` (for GC and profiling)
///
/// ## Calling Convention
///
/// - **Arguments**: `v0..v7` (abstract; 8 argument registers)
/// - **Return**: `v0` (abstract return register)
/// - The abstract calling convention is mapped to the real ABI during code emission:
///   - **x86_64**: System V ABI (`rdi, rsi, rdx, rcx, r8, r9`)
///   - **AArch64**: AAPCS (`x0-x7`)
///   - **WASM**: WASM ABI (`i32, i64, f32, f64`)
///
/// ## Core ops
/// ```text
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
/// ## Example (Minimal Function)
/// ```asm
/// .func add_store
/// bb0:
///   v2:i64 = add.i64 v0, v1    ; v2 = v0 + v1
///   lea     vA, v3, 0          ; vA = v3 + 0 (address calculation)
///   st.i64  v2, [vA] {align=8} ; *vA = v2
///   ret                        ; return (implicitly v0)
/// .endfunc
/// ```
///
/// ## Transform Passes
///
/// Optimizations operate on LUMIR before final code generation. See `transform` module.
// Core modules
pub mod types;

// Transform system
pub mod transform;

// Re-exports for convenience
pub use block::Block;
pub use function::{Function, FunctionBuilder, Parameter, Signature};
pub use instruction::{
    AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction, IntBinOp, IntCmpOp,
    MemoryAttrs, Operand, VectorOp,
};
#[cfg(feature = "nightly")]
pub use instruction::{AtomicBinOp, MemoryOrdering, SimdOp};
pub use module::{Global, Module, ModuleBuilder};
pub use register::{PhysicalReg, Register, RegisterClass, VirtualReg, VirtualRegAllocator};
pub use transform::{
    DeadCodeElimination, FunctionInlining, LoopInvariantCodeMotion, LoopUnrolling, ModuleInlining,
    Peephole, Transform, TransformPipeline, TransformStats,
};
pub use types::{MirType, ScalarType, VectorLane, VectorType};
