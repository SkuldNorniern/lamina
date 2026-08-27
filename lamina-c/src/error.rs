// Thread-local error state for the C API.
//
// Stores CString so last_error_ptr() always returns a NUL-terminated pointer.

use std::cell::RefCell;
use std::ffi::CString;
use std::ffi::c_char;
use std::ptr::null;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub fn set_error(msg: impl Into<String>) {
    let s = msg.into();
    // Replace embedded NULs so CString::new never fails on malformed messages.
    let safe = if s.contains('\0') {
        s.replace('\0', "\\0")
    } else {
        s
    };
    let cs = CString::new(safe).unwrap_or_else(|_| c"(error message encoding failed)".to_owned());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(cs));
}

pub fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Returns a NUL-terminated pointer valid until the next Lamina API call on
/// this thread. Returns NULL if no error is set.
pub fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(cs) => cs.as_ptr(),
        None => null(),
    })
}
