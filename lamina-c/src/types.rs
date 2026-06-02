// Opaque handle types for the C API.

use lamina_ir::owned::{OwnedIRBuilder, OwnedType, OwnedValue};

pub struct LaminaBuilder(pub OwnedIRBuilder);
pub struct LaminaModule(pub String); // owns serialised IR text
pub struct LaminaType(pub OwnedType);
pub struct LaminaValue(pub OwnedValue);

/// Byte buffer returned to C callers. Free with `lamina_buffer_free`.
#[repr(C)]
pub struct LaminaBuffer {
    pub data: *mut u8,
    pub len: usize,
}

impl LaminaBuffer {
    /// Convert a `Vec<u8>` into a C-owned buffer.
    /// Uses `Box<[u8]>` so capacity always equals length — freeing with
    /// the exact allocation size is safe without storing a separate cap field.
    pub fn from_vec(v: Vec<u8>) -> Self {
        let boxed = v.into_boxed_slice();
        let len = boxed.len();
        let data = Box::into_raw(boxed) as *mut u8;
        LaminaBuffer { data, len }
    }

    pub fn null() -> Self {
        LaminaBuffer {
            data: std::ptr::null_mut(),
            len: 0,
        }
    }
}

/// JIT compilation result — nightly only.
/// Owns executable memory; invalidates all function pointers on free.
#[cfg(feature = "nightly")]
pub struct LaminaJit {
    pub memory: lamina::runtime::ExecutableMemory,
    pub function_ptr: *const u8,
}
