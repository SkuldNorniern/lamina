//! Target detection and platform information
//!
//! Re-exports target types from `lamina-platform` for backward compatibility.
//! New code should use `lamina_platform` directly.

pub use lamina_platform::{
    Target, TargetArchitecture, TargetOperatingSystem, detect_host_architecture_only,
    detect_host_os, host_arch_list,
};
