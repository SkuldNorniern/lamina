use crate::mir_codegen::TargetOs;

/// RISC-V ABI utilities
pub struct RiscVAbi {
    target_os: TargetOs,
}

impl RiscVAbi {
    pub fn new(target_os: TargetOs) -> Self {
        Self { target_os }
    }

    /// Get the appropriate function name with platform-specific prefix
    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOs::MacOs => format!("_{}", name),
            _ => name.to_string(),
        }
    }

    /// Get the appropriate global declaration for main
    pub fn get_main_global(&self) -> &'static str {
        ".globl main"
    }

    /// Get the data section directive
    pub fn get_data_section(&self) -> &'static str {
        ".data"
    }

    /// Get the text section directive
    pub fn get_text_section(&self) -> &'static str {
        ".text"
    }

    /// Get the format string for printing integers
    pub fn get_print_format(&self) -> &'static str {
        match self.target_os {
            TargetOs::MacOs => "__mir_fmt_int: .asciz \"%lld\\n\"",
            _ => ".L_mir_fmt_int: .string \"%lld\\n\"",
        }
    }
}
