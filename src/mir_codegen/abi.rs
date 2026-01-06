//! Common ABI trait and utilities for code generation backends.
//!
//! This module defines a trait that all ABI implementations must follow,
//! reducing code duplication and providing a consistent interface.

use lamina_platform::TargetOperatingSystem;

/// Trait defining the common interface for all ABI implementations.
///
/// This trait provides a standardized way to interact with platform-specific
/// ABI details across different architectures. Each backend implements this
/// trait to provide architecture and OS-specific behavior.
pub trait Abi {
    /// Returns the target operating system for this ABI instance.
    fn target_os(&self) -> TargetOperatingSystem;

    /// Returns the mangled function name with platform-specific prefix.
    ///
    /// This method handles platform-specific symbol naming conventions,
    /// such as the underscore prefix on macOS.
    fn mangle_function_name(&self, name: &str) -> String;

    /// Maps well-known intrinsic/runtime names to platform symbol stubs.
    ///
    /// Returns `None` if the name is not a known intrinsic that needs
    /// special handling. This allows backends to map internal names
    /// (like "print") to platform-specific symbols (like "_printf" on macOS).
    ///
    /// Default implementation returns `None` for all names, meaning
    /// no special mapping is needed.
    fn call_stub(&self, name: &str) -> Option<String> {
        let _ = name;
        None
    }
}

/// Helper function to mangle function names for macOS (adds underscore prefix).
///
/// This is a common pattern used by multiple backends, so we provide
/// it as a shared utility function.
pub fn mangle_macos_name(name: &str) -> String {
    if name == "main" {
        "_main".to_string()
    } else {
        format!("_{}", name)
    }
}

/// Helper function to get the platform-specific printf symbol name.
///
/// Returns "_printf" on macOS, "printf" on other platforms.
pub fn get_printf_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_printf",
        _ => "printf",
    }
}

/// Helper function to get the platform-specific malloc symbol name.
///
/// Returns "_malloc" on macOS, "malloc" on other platforms.
pub fn get_malloc_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_malloc",
        _ => "malloc",
    }
}

/// Helper function to get the platform-specific free symbol name.
///
/// Returns "_free" on macOS, "free" on other platforms.
pub fn get_free_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_free",
        _ => "free",
    }
}

/// Helper function to create a common call_stub implementation.
///
/// This handles the common pattern of mapping "print", "malloc", and "dealloc"
/// to their platform-specific symbols. Backends can use this to reduce
/// duplication in their `call_stub` implementations.
pub fn common_call_stub(name: &str, target_os: TargetOperatingSystem) -> Option<String> {
    match name {
        "print" => Some(get_printf_symbol(target_os).to_string()),
        "malloc" => Some(get_malloc_symbol(target_os).to_string()),
        "dealloc" => Some(get_free_symbol(target_os).to_string()),
        _ => None,
    }
}







