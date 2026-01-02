//! Target detection and platform information
//!
//! This module re-exports target types from `lamina-platform` for backward compatibility.
//! New code should use `lamina_platform` directly.

pub use lamina_platform::{
    detect_host_architecture, detect_host_architecture_only, detect_host_os, HOST_ARCH_LIST,
    Target, TargetArchitecture, TargetOperatingSystem,
};
