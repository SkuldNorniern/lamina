//! JIT function execution
//!
//! Provides functions to execute JIT-compiled functions with dynamic argument counts
//! using platform-specific calling conventions.

use crate::mir::Signature;

/// Calls a function pointer with a dynamic number of i64 arguments.
///
/// Supports any number of arguments using platform-specific calling conventions:
/// - AArch64: First 8 arguments in x0-x7, all remaining arguments on stack (unlimited)
/// - x86_64: First 6 arguments in rdi, rsi, rdx, rcx, r8, r9, all remaining arguments on stack (unlimited)
///
/// # Safety
///
/// This function is unsafe because it:
/// - Transmutes a raw pointer to a function pointer
/// - Uses inline assembly to manipulate the stack and registers
/// - Assumes the function signature matches the provided arguments
///
/// # Arguments
///
/// * `function_ptr` - Raw pointer to the function to call
/// * `args` - Slice of i64 arguments to pass to the function
/// * `returns_value` - Whether the function returns an i64 value
///
/// # Returns
///
/// Returns `Some(i64)` if `returns_value` is true, otherwise `None`.
pub unsafe fn call_function_dynamic(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    match args.len() {
        0 => {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::asm;
                let mut result: i64 = 0;
                let func_ptr_val = function_ptr as usize;
                if returns_value {
                    unsafe {
                        asm!("blr {}", in(reg) func_ptr_val, lateout("x0") result, options(nostack));
                    }
                    Some(result)
                } else {
                    unsafe {
                        asm!("blr {}", in(reg) func_ptr_val, options(nostack));
                    }
                    None
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::asm;
                let mut result: i64 = 0;
                if returns_value {
                    unsafe {
                        asm!("call {}", in(reg) function_ptr, lateout("rax") result, options(nostack));
                    }
                    Some(result)
                } else {
                    unsafe {
                        asm!("call {}", in(reg) function_ptr, options(nostack));
                    }
                    None
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            None
        }
        _ => unsafe {
            call_dynamic_helper(function_ptr, args, returns_value)
        }
    }
}

/// Helper function implementing platform-specific calling conventions for dynamic argument counts.
///
/// Supports any number of arguments by using the platform's calling convention:
///
/// # AArch64 Calling Convention (AAPCS64)
/// - Arguments 0-7: x0-x7 registers
/// - Arguments 8+: stack (16-byte aligned, unlimited)
/// - Return value: x0 register
///
/// # x86_64 Calling Convention (System V ABI)
/// - Arguments 0-5: rdi, rsi, rdx, rcx, r8, r9 registers
/// - Arguments 6+: stack (right-to-left order, unlimited)
/// - Return value: rax register
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
unsafe fn call_dynamic_helper(
    function_ptr: *const u8,
    args: &[i64],
    returns_value: bool,
) -> Option<i64> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::asm;
        let mut result: i64 = 0;
        let func_ptr_val = function_ptr as usize;
        
        let reg_args: Vec<i64> = args.iter().take(8).copied().collect();
        let stack_args: Vec<i64> = args.iter().skip(8).copied().collect();
        
        let stack_size = if stack_args.is_empty() { 0 } else { (stack_args.len() * 8 + 15) & !15 };
        
        if stack_size > 0 {
            unsafe {
                asm!("sub sp, sp, {}", in(reg) stack_size, options(nostack));
            }
        }
        
        for (i, &arg) in stack_args.iter().enumerate() {
            let offset = i * 8;
            unsafe {
                let mut addr = 0usize;
                asm!(
                    "add {}, sp, {}",
                    out(reg) addr,
                    in(reg) offset,
                    options(nostack)
                );
                asm!("str {}, [{}]", in(reg) arg, in(reg) addr, options(nostack));
            }
        }
        
        let x0 = reg_args.get(0).copied().unwrap_or(0);
        let x1 = reg_args.get(1).copied().unwrap_or(0);
        let x2 = reg_args.get(2).copied().unwrap_or(0);
        let x3 = reg_args.get(3).copied().unwrap_or(0);
        let x4 = reg_args.get(4).copied().unwrap_or(0);
        let x5 = reg_args.get(5).copied().unwrap_or(0);
        let x6 = reg_args.get(6).copied().unwrap_or(0);
        let x7 = reg_args.get(7).copied().unwrap_or(0);
        
        if returns_value {
            unsafe {
                asm!(
                    "mov x0, {}; mov x1, {}; mov x2, {}; mov x3, {}; mov x4, {}; mov x5, {}; mov x6, {}; mov x7, {}; blr {}",
                    in(reg) x0, in(reg) x1, in(reg) x2, in(reg) x3,
                    in(reg) x4, in(reg) x5, in(reg) x6, in(reg) x7,
                    in(reg) func_ptr_val,
                    lateout("x0") result,
                    options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                    "mov x0, {}; mov x1, {}; mov x2, {}; mov x3, {}; mov x4, {}; mov x5, {}; mov x6, {}; mov x7, {}; blr {}",
                    in(reg) x0, in(reg) x1, in(reg) x2, in(reg) x3,
                    in(reg) x4, in(reg) x5, in(reg) x6, in(reg) x7,
                    in(reg) func_ptr_val,
                    options(nostack)
                );
            }
        }
        
        if stack_size > 0 {
            unsafe {
                asm!("add sp, sp, {}", in(reg) stack_size, options(nostack));
            }
        }
        
        if returns_value { Some(result) } else { None }
    }
    
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::asm;
        let mut result: i64 = 0;
        
        let reg_args: Vec<i64> = args.iter().take(6).copied().collect();
        let stack_args: Vec<i64> = args.iter().skip(6).copied().collect();
        
        for &arg in stack_args.iter().rev() {
            unsafe {
                asm!("push {}", in(reg) arg, options(nostack));
            }
        }
        
        let rdi = reg_args.get(0).copied().unwrap_or(0);
        let rsi = reg_args.get(1).copied().unwrap_or(0);
        let rdx = reg_args.get(2).copied().unwrap_or(0);
        let rcx = reg_args.get(3).copied().unwrap_or(0);
        let r8 = reg_args.get(4).copied().unwrap_or(0);
        let r9 = reg_args.get(5).copied().unwrap_or(0);
        
        if returns_value {
            unsafe {
                asm!(
                    "mov rdi, {}; mov rsi, {}; mov rdx, {}; mov rcx, {}; mov r8, {}; mov r9, {}; call {}",
                    in(reg) rdi, in(reg) rsi, in(reg) rdx, in(reg) rcx,
                    in(reg) r8, in(reg) r9, in(reg) function_ptr,
                    lateout("rax") result,
                    options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                    "mov rdi, {}; mov rsi, {}; mov rdx, {}; mov rcx, {}; mov r8, {}; mov r9, {}; call {}",
                    in(reg) rdi, in(reg) rsi, in(reg) rdx, in(reg) rcx,
                    in(reg) r8, in(reg) r9, in(reg) function_ptr,
                    options(nostack)
                );
            }
        }
        
        if !stack_args.is_empty() {
            let adjust = stack_args.len() * 8;
            unsafe {
                asm!("add rsp, {}", in(reg) adjust, options(nostack));
            }
        }
        
        if returns_value { Some(result) } else { None }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unsafe fn call_dynamic_helper(
    _function_ptr: *const u8,
    _args: &[i64],
    _returns_value: bool,
) -> Option<i64> {
    None
}

/// Executes a JIT-compiled function using its signature to determine calling convention.
///
/// Validates the function signature and calls it with the provided arguments.
/// Currently supports only i64 parameters and i64 or void return types.
///
/// # Arguments
///
/// * `sig` - Function signature from MIR
/// * `function_ptr` - Raw pointer to the compiled function
/// * `args` - Arguments to pass to the function (defaults to zeros if not provided)
/// * `verbose` - Whether to print execution details
///
/// # Errors
///
/// Returns an error if:
/// - Function has non-i64 parameters
/// - Function has unsupported return type
/// - Function should return a value but didn't
pub fn execute_jit_function(
    sig: &Signature,
    function_ptr: *const u8,
    args: Option<&[i64]>,
    verbose: bool,
) -> Result<Option<i64>, Box<dyn std::error::Error>> {
    let param_count = sig.params.len();
    
    let all_i64 = sig.params.iter()
        .all(|p| matches!(p.ty, crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)));
    
    let returns_i64 = matches!(
        sig.ret_ty.as_ref(),
        Some(crate::mir::MirType::Scalar(crate::mir::ScalarType::I64))
    );
    let returns_void = sig.ret_ty.is_none();
    
    if !all_i64 {
        return Err(format!(
            "JIT execution: Function has non-i64 parameters, not yet supported. Parameter types: {:?}",
            sig.params.iter().map(|p| &p.ty).collect::<Vec<_>>()
        ).into());
    }
    
    if !returns_i64 && !returns_void {
        return Err(format!(
            "JIT execution: Unsupported return type: {:?}",
            sig.ret_ty
        ).into());
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::asm;
        let sp: usize;
        unsafe {
            asm!("mov {}, sp", out(reg) sp);
        }
        if !sp.is_multiple_of(16) && verbose {
            eprintln!("[WARNING] Stack pointer is not 16-byte aligned: SP={:p}, alignment={} bytes", 
                     sp as *const u8, sp % 16);
            eprintln!("[WARNING] This may cause the STP instruction to fault!");
        }
    }
    
    let default_args: Vec<i64> = vec![0; param_count];
    let args = args.unwrap_or(&default_args);
    
    if args.len() != param_count {
        return Err(format!(
            "JIT execution: Argument count mismatch. Expected {}, got {}",
            param_count, args.len()
        ).into());
    }
    
    if verbose {
        println!("[JIT] Calling function with {} i64 parameter(s), {} return...", 
                 param_count,
                 if returns_i64 { "i64" } else { "void" });
    }
    
    unsafe {
        let result = call_function_dynamic(function_ptr, args, returns_i64);
        
        if returns_i64 {
            if let Some(value) = result {
                if verbose {
                    println!("[JIT] Function returned: {}", value);
                }
                Ok(Some(value))
            } else {
                Err("JIT execution: Function should have returned a value but didn't".into())
            }
        } else {
            if verbose {
                println!("[JIT] Function executed successfully (void return)");
            }
            Ok(None)
        }
    }
}

