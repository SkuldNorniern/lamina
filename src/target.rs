//! Target detection and platform information
//!
//! Re-exports target types from `lamina-platform` for backward compatibility.
//! New code should use `lamina_platform` directly.

pub use lamina_platform::{
    HOST_ARCH_LIST, Target, TargetArchitecture, TargetOperatingSystem,
    detect_host_architecture_only, detect_host_os,
};
