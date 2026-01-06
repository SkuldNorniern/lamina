//! Runtime compilation support
//!
//! This module provides runtime code generation (JIT compilation) using ras.
//! It can compile MIR directly to executable memory.
//!
//! Requires the `mir` feature to be enabled.

#[cfg(feature = "encoder")]
use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
#[cfg(feature = "encoder")]
use lamina_mir::Module as MirModule;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

// Note: errno access varies by platform
// On macOS, we can use libc::errno directly in some cases, but for better compatibility
// we'll use a simpler error message approach

/// Executable memory for runtime-compiled code
pub struct ExecutableMemory {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableMemory {
    /// Allocate writable memory (not executable yet)
    /// This allows writing code before making it executable
    pub fn allocate_writable(size: usize) -> Result<Self, RasError> {
        #[cfg(unix)]
        {
            use libc::{mmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE};
            use std::ptr;

            let aligned_size = (size + 4095) & !4095; // Page align

            // On macOS Apple Silicon, we need MAP_JIT for JIT memory
            #[cfg(target_os = "macos")]
            #[cfg(target_arch = "aarch64")]
            const MAP_JIT: libc::c_int = 0x0800; // From sys/mman.h on macOS

            #[cfg(target_os = "macos")]
            #[cfg(target_arch = "aarch64")]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_JIT;

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE;

            // Allocate with PROT_READ | PROT_WRITE (not executable yet)
            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE,
                    map_flags,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(RasError::IoError(format!(
                    "mmap failed (size: {} bytes, aligned: {}). This may be due to system security restrictions or insufficient permissions.",
                    size, aligned_size
                )));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size: aligned_size,
            })
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualAlloc;
            use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};

            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
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

    /// Make the memory executable
    /// Must be called after writing code
    pub fn make_executable(&mut self) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use libc::{mprotect, PROT_READ, PROT_EXEC};

            // Change permissions to executable using mprotect
            let mprotect_result = unsafe {
                mprotect(self.ptr as *mut libc::c_void, self.size, PROT_READ | PROT_EXEC)
            };

            if mprotect_result != 0 {
                return Err(RasError::IoError(format!(
                    "mprotect failed (size: {} bytes). Cannot make memory executable.",
                    self.size
                )));
            }

            // Flush instruction cache on ARM architectures
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    // Data Memory Barrier (DMB) - ensure all memory writes are visible
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    
                    // Use GCC/Clang builtin to flush instruction cache
                    // This performs IC IALLU (Invalidate all instruction caches to PoU)
                    // and ISB (Instruction Synchronization Barrier)
                    unsafe extern "C" {
                        fn __clear_cache(start: *const u8, end: *const u8);
                    }
                    // Flush the entire code region
                    let start = self.ptr;
                    let end = self.ptr.add(self.size);
                    __clear_cache(start, end);
                }
            }

            Ok(())
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualProtect;
            use winapi::um::winnt::PAGE_EXECUTE_READ;
            use std::ptr;

            let mut old_protect: winapi::um::winnt::DWORD = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr as *mut winapi::um::winnt::c_void,
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
            }

            Ok(())
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Making memory executable not supported on this platform".to_string(),
            ))
        }
    }

    /// Allocate executable memory (legacy method - kept for compatibility)
    /// For new code, use allocate_writable + make_executable
    pub fn allocate(size: usize) -> Result<Self, RasError> {
        let mut mem = Self::allocate_writable(size)?;
        mem.make_executable()?;
        Ok(mem)
    }

    /// Write code to executable memory
    pub fn write_code(&mut self, code: &[u8]) -> Result<(), RasError> {
        if code.len() > self.size {
            return Err(RasError::IoError("Code too large".to_string()));
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_arch = "aarch64")]
        {
            // On macOS with AArch64, we need to disable write protection before writing
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(0); // Disable write protection
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_arch = "aarch64")]
        {
            // Re-enable write protection after writing
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(1); // Re-enable write protection
            }
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
    #[cfg(feature = "encoder")]
    pub fn compile_to_memory(
        &mut self,
        module: &MirModule,
    ) -> Result<ExecutableMemory, RasError> {
        // 1. Use ras to compile MIR to binary
        use crate::assembler::RasAssembler;
        let mut assembler = RasAssembler::new(self.target_arch, self.target_os)?;
        let (code, _) = assembler.compile_mir_to_binary_function(module, None)?;

        // 2. Allocate writable memory (not executable yet)
        let mut mem = ExecutableMemory::allocate_writable(code.len())?;

        // 3. Write code to memory (while it's still writable)
        mem.write_code(&code)?;

        // 4. Make memory executable
        mem.make_executable()?;

        Ok(mem)
    }

    /// Compile a specific function from MIR module to executable memory
    ///
    /// Returns a function pointer that can be called directly.
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "encoder")]
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

