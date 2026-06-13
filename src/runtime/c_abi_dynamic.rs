//! Dynamic `extern "C"` calls with a runtime-known `i64` argument count.
//!
//! ## C limits
//!
//! ISO C does **not** specify a maximum parameter count. Informative minimums (e.g. 127
//! parameters in a prototype) appear in older annexes; **GCC** and **Clang** accept very
//! large lists in practice. Lamina uses [`MAX_JIT_ARGS`] as a **host policy cap** aligned
//! with “what real toolchains allow,” not a standard ceiling.
//!
//! ## ABI
//!
//! - **AArch64**: AAPCS64 (first eight integer/pointer args in `x0`–`x7`, then stack).
//! - **x86_64** (non-Windows): System V AMD64 (`rdi`…`r9`, then stack).
//! - **Other hosts** (Windows x86_64, non-x86_64, etc.): fixed transmute table for up to **15**
//!   `i64` parameters (same as the previous executor path). [`MAX_JIT_ARGS`] is **15** there.
//!
//! Prefer packing parameters for hot paths; see [`JIT_ARG_SOFT_WARN_THRESHOLD`] for hints.

use std::mem;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_os = "windows")),
))]
use std::arch::asm;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_os = "windows")),
))]
use crate::mir_codegen::MAX_MIR_CALL_PARAMETERS;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_os = "windows")),
))]
/// Upper bound for [`call_function_dynamic`] on AArch64 and SysV x86_64 (stack shim).
pub const MAX_JIT_ARGS: usize = MAX_MIR_CALL_PARAMETERS;

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_os = "windows")),
)))]
/// Upper bound where only the transmute table is available (no generic stack path).
pub const MAX_JIT_ARGS: usize = 15;

/// Past this count, tooling may warn: extra arguments use the stack under the C ABI.
pub const JIT_ARG_SOFT_WARN_THRESHOLD: usize = 8;

fn call_args_valid(function_ptr: *const u8, args: &[i64]) -> bool {
    !function_ptr.is_null() && args.len() <= MAX_JIT_ARGS
}

#[cfg(target_arch = "aarch64")]
/// Call an arbitrary C-ABI function at `function_ptr` with up to [`MAX_JIT_ARGS`] `i64` args.
///
/// # Safety
///
/// - `function_ptr` must point to valid, executable code that follows AAPCS64.
/// - `args` must be the exact set of arguments expected by the callee.
/// - The callee must be safe to call from the current thread context.
pub unsafe fn call_function_dynamic(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    if !call_args_valid(function_ptr, args) {
        return None;
    }
    if !(function_ptr as usize).is_multiple_of(4) {
        return None;
    }

    let n = args.len();
    if n == 0 {
        let mut out: i64 = 0;
        unsafe {
            asm!(
                "mov x16, {fp}",
                "mov x8, xzr",
                "blr x16",
                fp = in(reg) function_ptr,
                lateout("x0") out,
                clobber_abi("C"),
            );
        }
        return if returns_value { Some(out) } else { None };
    }

    let mut regbuf = [0i64; 8];
    let reg_fill = n.min(8);
    regbuf[..reg_fill].copy_from_slice(&args[..reg_fill]);

    let stack_n = n.saturating_sub(8);
    let rp = regbuf.as_ptr();
    let mut out: i64 = 0;
    if stack_n == 0 {
        unsafe {
            asm!(
                "mov x16, {fp}",
                "ldr x0, [{rp}]",
                "ldr x1, [{rp}, #8]",
                "ldr x2, [{rp}, #16]",
                "ldr x3, [{rp}, #24]",
                "ldr x4, [{rp}, #32]",
                "ldr x5, [{rp}, #40]",
                "ldr x6, [{rp}, #48]",
                "ldr x7, [{rp}, #56]",
                "mov x8, xzr",
                "blr x16",
                fp = in(reg) function_ptr,
                rp = in(reg) rp,
                lateout("x0") out,
                clobber_abi("C"),
            );
        }
    } else {
        // Outgoing stack args must not be placed by subtracting sp in asm only: LLVM may use
        // stack slots below the pre-asm sp, so we copy the tail into a heap buffer and point
        // sp at it for the call, then restore the host sp.
        //
        // Keep the saved sp in x20 and declare `lateout("x20") _`: (1) addresses passed via
        // `in(reg)` are in caller-saved registers and the JIT callee may clobber them before we
        // reload; (2) AArch64 Rust reserves x19 for LLVM, so it cannot be an asm operand.
        let byte_len = stack_n * core::mem::size_of::<i64>() + 16;
        let backing = vec![0u8; byte_len];
        let base = backing.as_ptr() as usize;
        let call_sp = (base + 15) & !15;
        let dst = call_sp as *mut i64;
        unsafe {
            core::ptr::copy_nonoverlapping(args.as_ptr().add(8), dst, stack_n);
        }
        unsafe {
            asm!(
                "mov x20, sp",
                "mov sp, {csp}",
                "mov x16, {fp}",
                "ldr x0, [{rp}]",
                "ldr x1, [{rp}, #8]",
                "ldr x2, [{rp}, #16]",
                "ldr x3, [{rp}, #24]",
                "ldr x4, [{rp}, #32]",
                "ldr x5, [{rp}, #40]",
                "ldr x6, [{rp}, #48]",
                "ldr x7, [{rp}, #56]",
                "mov x8, xzr",
                "blr x16",
                "mov sp, x20",
                fp = in(reg) function_ptr,
                rp = in(reg) rp,
                csp = in(reg) call_sp,
                lateout("x0") out,
                lateout("x20") _,
                clobber_abi("C"),
            );
        }
    }

    if returns_value { Some(out) } else { None }
}

#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
/// Call an arbitrary C-ABI function at `function_ptr` with up to [`MAX_JIT_ARGS`] `i64` args.
///
/// # Safety
///
/// - `function_ptr` must point to valid, executable code that follows the System V AMD64 ABI.
/// - `args` must be the exact set of arguments expected by the callee.
/// - The callee must be safe to call from the current thread context.
pub unsafe fn call_function_dynamic(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    if !call_args_valid(function_ptr, args) {
        return None;
    }
    if args.len() <= 15 {
        return unsafe { transmute_dynamic_call(function_ptr, args, returns_value) };
    }

    let n = args.len();
    if n == 0 {
        let out: i64;
        unsafe {
            asm!(
                "mov r11, {fp}",
                "call r11",
                fp = in(reg) function_ptr,
                lateout("rax") out,
                clobber_abi("C"),
            );
        }
        return if returns_value { Some(out) } else { None };
    }

    let mut regbuf = [0i64; 6];
    let reg_fill = n.min(6);
    regbuf[..reg_fill].copy_from_slice(&args[..reg_fill]);

    let stack_n = n.saturating_sub(6);
    let stack_src: *const i64 = if stack_n > 0 {
        args.as_ptr().wrapping_add(6)
    } else {
        args.as_ptr()
    };

    let rp = regbuf.as_ptr();
    let mut out: i64;

    if stack_n == 0 {
        unsafe {
            asm!(
                "mov r11, {fp}",
                "mov r10, {rp}",
                "mov rdi, [r10]",
                "mov rsi, [r10 + 8]",
                "mov rdx, [r10 + 16]",
                "mov rcx, [r10 + 24]",
                "mov r8, [r10 + 32]",
                "mov r9, [r10 + 40]",
                "call r11",
                fp = in(reg) function_ptr,
                rp = in(reg) rp,
                lateout("rax") out,
                clobber_abi("C"),
            );
        }
    } else {
        // Copy outgoing stack args into a heap buffer and point rsp at it for the
        // call, then restore the host sp. Same rationale as the AArch64 path:
        // adjusting rsp in asm alone can clobber Rust stack slots below the frame.
        let byte_len = stack_n * core::mem::size_of::<i64>() + 16;
        let backing = vec![0u8; byte_len];
        let call_sp = (backing.as_ptr() as usize + 15) & !15;
        let dst = call_sp as *mut i64;
        unsafe {
            core::ptr::copy_nonoverlapping(stack_src, dst, stack_n);
        }
        let mut saved_sp = 0usize;
        let saved_sp_ptr = &mut saved_sp as *mut usize;
        unsafe {
            asm!(
                "mov [{sp}], rsp",
                "mov rsp, {csp}",
                "mov r11, {fp}",
                "mov r10, {rp}",
                "mov rdi, [r10]",
                "mov rsi, [r10 + 8]",
                "mov rdx, [r10 + 16]",
                "mov rcx, [r10 + 24]",
                "mov r8, [r10 + 32]",
                "mov r9, [r10 + 40]",
                "call r11",
                "mov rsp, [{sp}]",
                fp = in(reg) function_ptr,
                rp = in(reg) rp,
                csp = in(reg) call_sp,
                sp = in(reg) saved_sp_ptr,
                lateout("rax") out,
                clobber_abi("C"),
            );
        }
    }

    if returns_value { Some(out) } else { None }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_os = "windows")),
)))]
pub unsafe fn call_function_dynamic(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    if !call_args_valid(function_ptr, args) {
        return None;
    }
    unsafe { transmute_dynamic_call(function_ptr, args, returns_value) }
}

unsafe fn transmute_dynamic_call(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    unsafe {
        match args.len() {
            0 => {
                if returns_value {
                    let callee: unsafe extern "C" fn() -> i64 = mem::transmute(function_ptr);
                    Some(callee())
                } else {
                    let callee: unsafe extern "C" fn() = mem::transmute(function_ptr);
                    callee();
                    None
                }
            }
            1 => {
                let a0 = args[0];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0))
                } else {
                    let callee: unsafe extern "C" fn(i64) = mem::transmute(function_ptr);
                    callee(a0);
                    None
                }
            }
            2 => {
                let a0 = args[0];
                let a1 = args[1];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64) = mem::transmute(function_ptr);
                    callee(a0, a1);
                    None
                }
            }
            3 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64) = mem::transmute(function_ptr);
                    callee(a0, a1, a2);
                    None
                }
            }
            4 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3);
                    None
                }
            }
            5 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4);
                    None
                }
            }
            6 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5);
                    None
                }
            }
            7 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                if returns_value {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                        mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6);
                    None
                }
            }
            8 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6, a7))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7);
                    None
                }
            }
            9 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6, a7, a8))
                } else {
                    let callee: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                        mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8);
                    None
                }
            }
            10 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
                    None
                }
            }
            11 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                let a10 = args[10];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
                    None
                }
            }
            12 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                let a10 = args[10];
                let a11 = args[11];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
                    None
                }
            }
            13 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                let a10 = args[10];
                let a11 = args[11];
                let a12 = args[12];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(
                        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                    ))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
                    None
                }
            }
            14 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                let a10 = args[10];
                let a11 = args[11];
                let a12 = args[12];
                let a13 = args[13];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(
                        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
                    ))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                    None
                }
            }
            15 => {
                let a0 = args[0];
                let a1 = args[1];
                let a2 = args[2];
                let a3 = args[3];
                let a4 = args[4];
                let a5 = args[5];
                let a6 = args[6];
                let a7 = args[7];
                let a8 = args[8];
                let a9 = args[9];
                let a10 = args[10];
                let a11 = args[11];
                let a12 = args[12];
                let a13 = args[13];
                let a14 = args[14];
                if returns_value {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) -> i64 = mem::transmute(function_ptr);
                    Some(callee(
                        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                    ))
                } else {
                    let callee: unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                    ) = mem::transmute(function_ptr);
                    callee(
                        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                    );
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod call_dynamic_tests {
    #[cfg(target_arch = "aarch64")]
    extern "C" fn c_abi_dyn_test_ninth(
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        mark: i64,
    ) -> i64 {
        mark
    }

    #[cfg(target_arch = "aarch64")]
    extern "C" fn c_abi_dyn_test_tenth(
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        mark: i64,
    ) -> i64 {
        mark
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_call_dynamic_reads_stack_arg8() {
        let args = [0i64, 0, 0, 0, 0, 0, 0, 0, 77];
        let got =
            unsafe { super::call_function_dynamic(c_abi_dyn_test_ninth as *const u8, &args, true) };
        assert_eq!(got, Some(77));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_call_dynamic_reads_stack_arg9_heap_sp() {
        let args = [0i64, 0, 0, 0, 0, 0, 0, 0, 0, 88];
        let got =
            unsafe { super::call_function_dynamic(c_abi_dyn_test_tenth as *const u8, &args, true) };
        assert_eq!(got, Some(88));
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
    extern "C" fn c_abi_dyn_test_seven_sum(
        a0: i64,
        a1: i64,
        a2: i64,
        a3: i64,
        a4: i64,
        a5: i64,
        a6: i64,
    ) -> i64 {
        a0 + a1 + a2 + a3 + a4 + a5 + a6
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
    #[test]
    fn x86_64_call_dynamic_seventh_arg_on_stack() {
        let args = [1i64, 2, 3, 4, 5, 6, 7];
        let got = unsafe {
            super::call_function_dynamic(c_abi_dyn_test_seven_sum as *const u8, &args, true)
        };
        assert_eq!(got, Some(28));
    }
}
