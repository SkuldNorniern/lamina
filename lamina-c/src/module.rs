// C API for modules: IR serialisation and AOT compilation.

use std::ffi::c_char;
use std::panic::AssertUnwindSafe;

use crate::error::{clear_error, set_error};
use crate::types::{LaminaBuffer, LaminaModule};
use crate::{LaminaStatus, catch, cstr_to_str};

// ---------------------------------------------------------------------------
// Compile options
// ---------------------------------------------------------------------------

/// Options for AOT assembly generation.
/// `target` may be NULL to select the host target.
/// `opt_level` is reserved and must be 0.
#[repr(C)]
pub struct LaminaCompileOptions {
    pub target: *const c_char,
    pub codegen_units: usize,
    pub opt_level: u8,
}

// ---------------------------------------------------------------------------
// Module lifecycle
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_module_free(module: *mut LaminaModule) {
    if !module.is_null() {
        unsafe { drop(Box::from_raw(module)) };
    }
}

// ---------------------------------------------------------------------------
// IR serialisation
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_module_emit_ir(
    module: *const LaminaModule,
    output: *mut LaminaBuffer,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if module.is_null() {
            set_error("module is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        if output.is_null() {
            set_error("output is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let ir = &(*module).0;
        *output = LaminaBuffer::from_vec(ir.as_bytes().to_vec());
        clear_error();
        LaminaStatus::Ok
    }))
}

// ---------------------------------------------------------------------------
// AOT compilation
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_compile_ir_to_assembly(
    ir: *const c_char,
    options: *const LaminaCompileOptions,
    output: *mut LaminaBuffer,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let ir_str = match cstr_to_str(ir) {
            Some(s) => s,
            None => {
                set_error("ir is null or invalid UTF-8");
                return LaminaStatus::ErrorInvalidArgument;
            }
        };
        if output.is_null() {
            set_error("output is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        if let Some(level) = unsafe { opt_level(options) } {
            if level != 0 {
                set_error("opt_level != 0 not yet supported; pass 0");
                return LaminaStatus::ErrorInvalidArgument;
            }
        }
        unsafe { compile_to_buf(ir_str, options, output) }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_module_compile_to_assembly(
    module: *const LaminaModule,
    options: *const LaminaCompileOptions,
    output: *mut LaminaBuffer,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        if module.is_null() {
            set_error("module is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        if output.is_null() {
            set_error("output is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        if let Some(level) = unsafe { opt_level(options) } {
            if level != 0 {
                set_error("opt_level != 0 not yet supported; pass 0");
                return LaminaStatus::ErrorInvalidArgument;
            }
        }
        let ir_str = &(*module).0;
        unsafe { compile_to_buf(ir_str, options, output) }
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn compile_to_buf(
    ir_str: &str,
    options: *const LaminaCompileOptions,
    output: *mut LaminaBuffer,
) -> LaminaStatus {
    // SAFETY: all pointer validity checked by callers before reaching here.
    let tgt = unsafe { target_str(options) };
    let units = unsafe { codegen_units(options) };
    let mut asm: Vec<u8> = Vec::new();
    let result = if let Some(ref tgt_str) = tgt {
        lamina::compile_lamina_ir_to_target_assembly(ir_str, &mut asm, tgt_str, units)
    } else {
        lamina::compile_lamina_ir_to_assembly(ir_str, &mut asm, units)
    };
    match result {
        Ok(()) => {
            // SAFETY: output validity checked by all callers before reaching here.
            unsafe { *output = LaminaBuffer::from_vec(asm) };
            clear_error();
            LaminaStatus::Ok
        }
        Err(e) => {
            set_error(e.to_string());
            // SAFETY: same as above.
            unsafe { *output = LaminaBuffer::null() };
            LaminaStatus::ErrorCodegen
        }
    }
}

// SAFETY: callers pass valid (non-null, aligned, initialised) pointers or NULL.
unsafe fn target_str(options: *const LaminaCompileOptions) -> Option<String> {
    if options.is_null() {
        return None;
    }
    // SAFETY: non-null, caller upholds validity.
    let opts = unsafe { &*options };
    if opts.target.is_null() {
        None
    } else {
        cstr_to_str(opts.target).map(|s| s.to_string())
    }
}

unsafe fn codegen_units(options: *const LaminaCompileOptions) -> usize {
    if options.is_null() {
        return 1;
    }
    // SAFETY: non-null, caller upholds validity.
    let n = unsafe { (*options).codegen_units };
    if n == 0 { 1 } else { n }
}

unsafe fn opt_level(options: *const LaminaCompileOptions) -> Option<u8> {
    if options.is_null() {
        None
    } else {
        // SAFETY: non-null, caller upholds validity.
        Some(unsafe { (*options).opt_level })
    }
}
