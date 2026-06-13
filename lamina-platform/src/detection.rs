//! Host system detection functions.

#[cfg(target_os = "macos")]
use std::ffi::CString;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::mem;
use std::ffi::c_void;

/// Returns the host architecture name: "x86_64", "aarch64", "riscv64", etc.
pub fn detect_host_architecture_only() -> &'static str {
    std::env::consts::ARCH
}

/// Returns the host OS name: "linux", "macos", "windows", "freebsd", etc.
pub fn detect_host_os() -> &'static str {
    std::env::consts::OS
}

/// Get the number of available CPU cores.
///
/// Returns the number of logical CPU cores available on the system.
/// Falls back to 1 if detection fails.
///
/// # Examples
///
/// ```
/// use lamina_platform::detection::cpu_count;
/// let cores = cpu_count();
/// println!("System has {} CPU cores", cores);
/// ```
pub fn cpu_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Try reading from /proc/cpuinfo first (most reliable)
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            let count = content
                .lines()
                .filter(|line| line.starts_with("processor"))
                .count();
            if count > 0 {
                return count;
            }
        }

        // Fallback to sysconf
        unsafe {
            unsafe extern "C" {
                fn sysconf(name: i32) -> i64;
            }
            const _SC_NPROCESSORS_ONLN: i32 = 84;
            let count = sysconf(_SC_NPROCESSORS_ONLN);
            if count > 0 {
                return count as usize;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        unsafe {
            unsafe extern "C" {
                fn sysctlbyname(
                    name: *const i8,
                    oldp: *mut std::ffi::c_void,
                    oldlenp: *mut usize,
                    newp: *const std::ffi::c_void,
                    newlen: usize,
                ) -> i32;
            }
            // SAFETY: "hw.ncpu" contains no interior nul bytes, so CString::new never fails.
            #[allow(clippy::unwrap_used)]
            let name = CString::new("hw.ncpu").unwrap();
            let mut count: u32 = 0;
            let mut size = mem::size_of::<u32>();
            if sysctlbyname(
                name.as_ptr(),
                &mut count as *mut _ as *mut std::ffi::c_void,
                &mut size,
                std::ptr::null(),
                0,
            ) == 0
            {
                return count as usize;
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        unsafe {
            unsafe extern "system" {
                fn GetSystemInfo(lp_system_info: *mut SystemInfo);
            }
            // Field order matches the Win32 SYSTEM_INFO layout (repr(C)); names
            // are snake_case since only the layout is ABI-significant.
            #[repr(C)]
            struct SystemInfo {
                processor_architecture: u16,
                reserved: u16,
                page_size: u32,
                minimum_application_address: *mut c_void,
                maximum_application_address: *mut c_void,
                active_processor_mask: *mut u32,
                number_of_processors: u32,
                processor_type: u32,
                allocation_granularity: u32,
                processor_level: u16,
                processor_revision: u16,
            }
            let mut info = mem::zeroed::<SystemInfo>();
            GetSystemInfo(&mut info);
            info.number_of_processors as usize
        }
    }

    // Fallback for other platforms
    #[cfg(not(target_os = "windows"))]
    {
        1
    }
}
