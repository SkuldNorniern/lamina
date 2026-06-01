// lamina-c — stable C API for the Lamina compiler.
//
// ABI rules enforced here:
//   - Every exported fn wrapped with catch_unwind; panics → LAMINA_ERROR_INTERNAL.
//   - Null pointer arguments → LAMINA_ERROR_INVALID_ARGUMENT.
//   - Incoming C strings copied on entry; no caller-owned pointer retained.
//   - Failing call stores a message; succeeding call clears stale error state.

#![allow(clippy::missing_safety_doc)]

mod error;
mod types;

pub mod builder;
pub mod module;

#[cfg(feature = "nightly")]
pub mod jit;

use std::ffi::{CStr, c_char};

// ---------------------------------------------------------------------------
// Status codes (stable — no JIT variant; JIT is nightly only)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaminaStatus {
    Ok = 0,
    ErrorInvalidArgument = 1,
    ErrorParse = 2,
    ErrorValidation = 3,
    ErrorCodegen = 4,
    ErrorIo = 5,
    ErrorInternal = 6,
}

// ---------------------------------------------------------------------------
// Error API
// ---------------------------------------------------------------------------

/// Returns pointer to last error message on this thread.
/// Valid until next Lamina C API call on this thread. NULL if no error.
#[unsafe(no_mangle)]
pub extern "C" fn lamina_last_error() -> *const c_char {
    error::last_error_ptr()
}

/// Clears last error state on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn lamina_clear_error() {
    error::clear_error();
}

// ---------------------------------------------------------------------------
// Version / info
// ---------------------------------------------------------------------------

// ABI version — bumped independently from the Rust crate version.
// Increment when the public C ABI changes in a backwards-incompatible way.
// The crate version (CARGO_PKG_VERSION) may advance for internal reasons
// without changing this string.
static ABI_VERSION: &str = "0.1.0\0";

static HOST_TARGET: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();

/// Returns the lamina-c ABI version string (null-terminated, static lifetime).
/// This is versioned independently from the Rust crate version.
#[unsafe(no_mangle)]
pub extern "C" fn lamina_version() -> *const c_char {
    ABI_VERSION.as_ptr() as *const c_char
}

/// Returns host target identifier (null-terminated, static lifetime).
#[unsafe(no_mangle)]
pub extern "C" fn lamina_host_target() -> *const c_char {
    HOST_TARGET
        .get_or_init(|| {
            let t = lamina_platform::Target::detect_host();
            std::ffi::CString::new(t.to_str()).unwrap_or_default()
        })
        .as_ptr()
}

// ---------------------------------------------------------------------------
// Buffer API
// ---------------------------------------------------------------------------

use types::LaminaBuffer;

/// Frees a `lamina_buffer_t` returned by the API. Safe to call with null data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lamina_buffer_free(buf: *mut LaminaBuffer) {
    if buf.is_null() {
        return;
    }
    unsafe {
        let buf = &mut *buf;
        if !buf.data.is_null() && buf.len > 0 {
            // Buffers are allocated via Box<[u8]>, so reconstruct a slice and drop it.
            drop(Box::from_raw(std::slice::from_raw_parts_mut(buf.data, buf.len)));
        }
        buf.data = std::ptr::null_mut();
        buf.len = 0;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Safely convert `*const c_char` to `&str`. Returns None for null or non-UTF-8.
pub(crate) fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null; C callers must pass valid null-terminated strings.
    unsafe { CStr::from_ptr(ptr).to_str().ok() }
}

/// Run `f` catching panics; map panic → LAMINA_ERROR_INTERNAL.
pub(crate) fn catch<F: FnOnce() -> LaminaStatus + std::panic::UnwindSafe>(f: F) -> LaminaStatus {
    match std::panic::catch_unwind(f) {
        Ok(status) => status,
        Err(_) => {
            error::set_error("internal panic in Lamina C API");
            LaminaStatus::ErrorInternal
        }
    }
}
