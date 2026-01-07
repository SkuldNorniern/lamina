//! Host system detection functions.

/// Detect the host system's architecture only.
///
/// Returns a string representing the detected architecture: "x86_64", "aarch64", etc.
///
/// Falls back to "x86_64" if detection fails.
pub fn detect_host_architecture_only() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    return "x86_64";
    #[cfg(target_arch = "aarch64")]
    return "aarch64";
    #[cfg(target_arch = "wasm32")]
    return "wasm32";
    #[cfg(target_arch = "wasm64")]
    return "wasm64";
    #[cfg(target_arch = "riscv32")]
    return "riscv32";
    #[cfg(target_arch = "riscv64")]
    return "riscv64";

    // Default fallback
    #[allow(unreachable_code)]
    "x86_64"
}

/// Detect the host system's operating system only.
///
/// Returns a string representing the detected operating system: "linux", "macos", "windows", etc.
///
/// Falls back to "unknown" if detection fails.
pub fn detect_host_os() -> &'static str {
    #[cfg(target_os = "linux")]
    return "linux";
    #[cfg(target_os = "macos")]
    return "macos";
    #[cfg(target_os = "windows")]
    return "windows";
    #[cfg(target_os = "freebsd")]
    return "freebsd";
    #[cfg(target_os = "openbsd")]
    return "openbsd";
    #[cfg(target_os = "netbsd")]
    return "netbsd";
    #[cfg(target_os = "dragonfly")]
    return "dragonfly";
    #[cfg(target_os = "redox")]
    return "redox";

    // Default fallback
    #[allow(unreachable_code)]
    "unknown"
}


/// Detect the host system's architecture and operating system combination.
///
/// Returns a string representing the detected architecture and host system combination.
///
/// # Supported Targets
/// - x86_64_unknown, x86_64_linux, x86_64_windows, x86_64_macos
/// - aarch64_unknown, aarch64_macos, aarch64_linux, aarch64_windows
/// - wasm32_unknown, wasm64_unknown
/// - riscv32_unknown, riscv64_unknown
/// - riscv128_unknown (nightly feature only)
///
/// Falls back to "x86_64_unknown" for unsupported combinations.
///
/// # Deprecated
/// This function is deprecated. Use `detect_host().to_str()` instead for a more structured approach.
#[deprecated(since = "0.0.8", note = "Use `detect_host().to_str()` instead")]
pub fn detect_host_architecture() -> &'static str {
    let arch = detect_host_architecture_only();
    let os = detect_host_os();
    // For backward compatibility, return the combined format
    // This will be removed once the deprecation period is over
    match (arch, os) {
        ("x86_64", "linux") => "x86_64_linux",
        ("x86_64", "macos") => "x86_64_macos",
        ("x86_64", "windows") => "x86_64_windows",
        ("aarch64", "linux") => "aarch64_linux",
        ("aarch64", "macos") => "aarch64_macos",
        ("aarch64", "windows") => "aarch64_windows",
        ("wasm32", _) => "wasm32_unknown",
        ("wasm64", _) => "wasm64_unknown",
        ("riscv32", _) => "riscv32_unknown",
        ("riscv64", _) => "riscv64_unknown",
        #[cfg(feature = "nightly")]
        ("riscv128", _) => "riscv128_unknown",
        _ => {
            // Fallback for unsupported combinations
            match arch {
                "x86_64" => "x86_64_unknown",
                "aarch64" => "aarch64_unknown",
                _ => "x86_64_unknown",
            }
        }
    }
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
        use std::ffi::CString;
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
            let name = CString::new("hw.ncpu").unwrap();
            let mut count: u32 = 0;
            let mut size = std::mem::size_of::<u32>();
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
                fn GetSystemInfo(lpSystemInfo: *mut SystemInfo);
            }
            #[repr(C)]
            struct SystemInfo {
                wProcessorArchitecture: u16,
                wReserved: u16,
                dwPageSize: u32,
                lpMinimumApplicationAddress: *mut std::ffi::c_void,
                lpMaximumApplicationAddress: *mut std::ffi::c_void,
                dwActiveProcessorMask: *mut u32,
                dwNumberOfProcessors: u32,
                dwProcessorType: u32,
                dwAllocationGranularity: u32,
                wProcessorLevel: u16,
                wProcessorRevision: u16,
            }
            let mut info = std::mem::zeroed::<SystemInfo>();
            GetSystemInfo(&mut info);
            return info.dwNumberOfProcessors as usize;
        }
    }
    
    // Fallback for other platforms
    1
}

