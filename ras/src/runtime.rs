//! Runtime compilation support
//!
//! This module provides runtime code generation (JIT compilation) using ras.
//! It can compile MIR directly to executable memory.
//!
//! Requires the `mir` feature to be enabled.

#[cfg(feature = "mir")]
use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
#[cfg(feature = "mir")]
use lamina::mir::Module as MirModule;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Executable memory for runtime-compiled code
pub struct ExecutableMemory {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableMemory {
    /// Allocate executable memory
    pub fn allocate(size: usize) -> Result<Self, RasError> {
        #[cfg(unix)]
        {
            use libc::{mmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, PROT_EXEC};
            use std::ptr;

            let aligned_size = (size + 4095) & !4095; // Page align

            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_ANONYMOUS | MAP_PRIVATE,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(RasError::IoError("mmap failed".to_string()));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size: aligned_size,
            })
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualAlloc;
            use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE};

            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                )
            };

            if ptr.is_null() {
                return Err(RasError::IoError("VirtualAlloc failed".to_string()));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Executable memory allocation not supported on this platform".to_string(),
            ))
        }
    }

    /// Write code to executable memory
    pub fn write_code(&mut self, code: &[u8]) -> Result<(), RasError> {
        if code.len() > self.size {
            return Err(RasError::IoError("Code too large".to_string()));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
        }

        Ok(())
    }

    /// Get function pointer (unsafe - caller must ensure signature matches)
    pub unsafe fn as_function_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use libc::munmap;
            unsafe {
                munmap(self.ptr as *mut libc::c_void, self.size);
            }
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualFree;
            use winapi::um::winnt::MEM_RELEASE;
            unsafe {
                VirtualFree(self.ptr as *mut winapi::um::winnt::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

/// Runtime compiler - compiles MIR to executable memory
pub struct RasRuntime {
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
}

impl RasRuntime {
    /// Create a new runtime compiler
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Self {
        Self {
            target_arch,
            target_os,
        }
    }

    /// Compile MIR module to executable memory
    ///
    /// Uses ras to compile MIR directly to binary, then allocates
    /// executable memory and writes the code.
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "mir")]
    pub fn compile_to_memory(
        &mut self,
        module: &MirModule,
    ) -> Result<ExecutableMemory, RasError> {
        // 1. Use ras to compile MIR to binary
        use crate::assembler::RasAssembler;
        let mut assembler = RasAssembler::new(self.target_arch, self.target_os)?;
        let code = assembler.compile_mir_to_binary(module)?;

        // 2. Allocate executable memory
        let mut mem = ExecutableMemory::allocate(code.len())?;

        // 3. Write code to memory
        mem.write_code(&code)?;

        Ok(mem)
    }

    /// Compile a specific function from MIR module to executable memory
    ///
    /// Returns a function pointer that can be called directly.
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "mir")]
    pub fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, RasError> {
        // For now, compile entire module
        // TODO: Support compiling individual functions
        let mem = self.compile_to_memory(module)?;
        
        // Get function pointer (unsafe - caller must ensure signature matches)
        unsafe {
            let ptr: *const unsafe extern "C" fn() -> T = mem.as_function_ptr();
            Ok(*ptr)
        }
    }
}

