// lamina-c — stable C API for the Lamina compiler.
//
// ABI rules:
//   - Every exported fn wrapped with catch_unwind; panics → LAMINA_ERROR_INTERNAL.
//   - Null pointer arguments → LAMINA_ERROR_INVALID_ARGUMENT.
//   - Incoming C strings copied on entry; no caller-owned pointer retained.
//   - Failing call stores a message; succeeding call clears stale error state.

#![allow(clippy::missing_safety_doc)]
// Nested `unsafe {}` inside `|| unsafe { ... }` closures are intentional —
// each inner block marks the specific unsafe operation being performed.
#![allow(unused_unsafe)]

mod error;
pub mod types;

pub mod builder;
pub mod module;

#[cfg(feature = "nightly")]
pub mod jit;

use std::ffi::{CStr, c_char};

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Return status for every fallible Lamina C API call.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaminaStatus {
    /// Operation succeeded.
    Ok = 0,
    /// A required pointer argument was NULL or an enum value was out of range.
    ErrorInvalidArgument = 1,
    /// IR text could not be parsed.
    ErrorParse = 2,
    /// IR failed structural validation.
    ErrorValidation = 3,
    /// Assembly or object code generation failed.
    ErrorCodegen = 4,
    /// A file I/O operation failed.
    ErrorIo = 5,
    /// An unexpected internal error occurred (e.g. Rust panic).
    ErrorInternal = 6,
}

// ---------------------------------------------------------------------------
// Error API
// ---------------------------------------------------------------------------

/// Returns a pointer to the last error message set on this thread.
///
/// Valid until the next Lamina C API call on this thread. Returns NULL if no
/// error has been set.
#[unsafe(no_mangle)]
pub extern "C" fn lia_last_error() -> *const c_char {
    error::last_error_ptr()
}

/// Clears the last error state on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn lia_clear_error() {
    error::clear_error();
}

// ---------------------------------------------------------------------------
// Version / info
// ---------------------------------------------------------------------------

// ABI version — bumped independently from the Rust crate version.
static ABI_VERSION: &str = "0.1.0\0";

static HOST_TARGET: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();

/// Returns the lamina-c ABI version string (null-terminated, static lifetime).
///
/// Versioned independently from the Rust crate version.
#[unsafe(no_mangle)]
pub extern "C" fn lia_version() -> *const c_char {
    ABI_VERSION.as_ptr() as *const c_char
}

/// Returns the host target identifier (e.g. `"x86_64_linux"`).
///
/// Null-terminated, static lifetime.
#[unsafe(no_mangle)]
pub extern "C" fn lia_host_target() -> *const c_char {
    HOST_TARGET
        .get_or_init(|| {
            let t = lamina_platform::Target::detect_host();
            std::ffi::CString::new(t.to_str()).unwrap_or_default()
        })
        .as_ptr()
}

// ---------------------------------------------------------------------------
// Buffer free
// ---------------------------------------------------------------------------

/// Frees a `lamina_buffer_t` returned by the API.
///
/// Safe to call when `buf->data` is NULL. Do not call twice on the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_buffer_free(buf: *mut types::LaminaBuffer) {
    if buf.is_null() {
        return;
    }
    unsafe {
        let buf = &mut *buf;
        if !buf.data.is_null() && buf.len > 0 {
            drop(Box::from_raw(std::slice::from_raw_parts_mut(buf.data, buf.len)));
        }
        buf.data = std::ptr::null_mut();
        buf.len = 0;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

pub(crate) fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok() }
}

pub(crate) fn catch<F: FnOnce() -> LaminaStatus + std::panic::UnwindSafe>(f: F) -> LaminaStatus {
    match std::panic::catch_unwind(f) {
        Ok(status) => status,
        Err(_) => {
            error::set_error("internal panic in Lamina C API");
            LaminaStatus::ErrorInternal
        }
    }
}
