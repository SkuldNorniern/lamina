use crate::target::TargetOperatingSystem;

/// x86_64 ABI utilities for different platforms
pub struct X86ABI {
    target_os: TargetOperatingSystem,
}

impl X86ABI {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Get the appropriate function name with platform-specific prefix
    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOperatingSystem::MacOS => {
                if name == "main" {
                    "_main".to_string()
                } else {
                    format!("_{}", name)
                }
            }
            _ => name.to_string(),
        }
    }

    /// Get the appropriate global declaration for main
    pub fn get_main_global(&self) -> &'static str {
        ".globl main"
    }
}
