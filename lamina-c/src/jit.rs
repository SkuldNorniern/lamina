// JIT compilation API — nightly feature only.
// Gated at the module level in lib.rs: #[cfg(feature = "nightly")].

use std::ffi::c_char;
use std::panic::AssertUnwindSafe;

use crate::error::{clear_error, set_error};
use crate::types::{LaminaJit, LaminaModule};
use crate::{LaminaStatus, catch, cstr_to_str};

use lamina::runtime::compile_to_runtime;
use lamina_platform::Target;

// ---------------------------------------------------------------------------
// JIT lifecycle
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_compile_ir(
    ir: *const c_char,
    function_name: *const c_char,
    jit_out: *mut *mut LaminaJit,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let ir_str = match cstr_to_str(ir) {
            Some(s) => s,
            None => { set_error("ir is null or invalid UTF-8"); return LaminaStatus::ErrorInvalidArgument; }
        };
        let func_name = match cstr_to_str(function_name) {
            Some(s) => s,
            None => { set_error("function_name is null or invalid UTF-8"); return LaminaStatus::ErrorInvalidArgument; }
        };
        if jit_out.is_null() {
            set_error("jit_out is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        match compile_jit(ir_str, func_name) {
            Ok(h) => { *jit_out = Box::into_raw(Box::new(h)); clear_error(); LaminaStatus::Ok }
            Err(msg) => { set_error(msg); LaminaStatus::ErrorCodegen }
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_module_compile_jit(
    module: *const LaminaModule,
    function_name: *const c_char,
    jit_out: *mut *mut LaminaJit,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if module.is_null() {
            set_error("module is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let func_name = match cstr_to_str(function_name) {
            Some(s) => s,
            None => { set_error("function_name is null or invalid UTF-8"); return LaminaStatus::ErrorInvalidArgument; }
        };
        if jit_out.is_null() {
            set_error("jit_out is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let ir_str = &(*module).0;
        match compile_jit(ir_str, func_name) {
            Ok(h) => { *jit_out = Box::into_raw(Box::new(h)); clear_error(); LaminaStatus::Ok }
            Err(msg) => { set_error(msg); LaminaStatus::ErrorCodegen }
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_get_function(
    jit: *const LaminaJit,
    function_out: *mut *const std::ffi::c_void,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if jit.is_null() || function_out.is_null() {
            set_error("jit or function_out is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        *function_out = (*jit).function_ptr as *const std::ffi::c_void;
        clear_error();
        LaminaStatus::Ok
    }))
}

/// Frees JIT handle and its executable memory. Safe to call with NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_free(jit: *mut LaminaJit) {
    if !jit.is_null() {
        unsafe { drop(Box::from_raw(jit)) };
    }
}

// ---------------------------------------------------------------------------
// Typed call helpers
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_call_i64_0(jit: *const LaminaJit, result: *mut i64) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if jit.is_null() || result.is_null() {
            set_error("jit or result is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let f: extern "C" fn() -> i64 = std::mem::transmute((*jit).function_ptr);
        *result = f();
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_call_i64_1(jit: *const LaminaJit, a: i64, result: *mut i64) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if jit.is_null() || result.is_null() {
            set_error("jit or result is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let f: extern "C" fn(i64) -> i64 = std::mem::transmute((*jit).function_ptr);
        *result = f(a);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_jit_call_i64_2(jit: *const LaminaJit, a: i64, b: i64, result: *mut i64) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if jit.is_null() || result.is_null() {
            set_error("jit or result is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute((*jit).function_ptr);
        *result = f(a, b);
        clear_error();
        LaminaStatus::Ok
    }))
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn compile_jit(ir_str: &str, func_name: &str) -> Result<LaminaJit, String> {
    let host = Target::detect_host();
    let module = lamina::parser::parse_module(ir_str)
        .map_err(|e| format!("parse error: {}", e))?;
    let mir_module = lamina::mir::codegen::from_ir(&module, "jit_c")
        .map_err(|e| format!("IR lowering error: {}", e))?;
    let lookup = func_name.strip_prefix('@').unwrap_or(func_name);
    let result = compile_to_runtime(&mir_module, host.architecture, host.operating_system, Some(lookup))
        .map_err(|e| format!("JIT compilation failed: {}", e))?;
    Ok(LaminaJit { memory: result.memory, function_ptr: result.function_ptr })
}
