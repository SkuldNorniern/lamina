/// x86_64 register information and calling convention
/// x86_64 argument registers (System V ABI)
pub const ARG_REGISTERS: &[&str] = &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];

/// Return register for integers/pointers
pub const RETURN_REGISTER: &str = "%rax";

/// Callee-saved registers
pub const CALLEE_SAVED_REGISTERS: &[&str] = &["%rbx", "%rbp", "%r12", "%r13", "%r14", "%r15"];

/// Caller-saved registers  
pub const CALLER_SAVED_REGISTERS: &[&str] = &[
    "%rax", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11",
];

/// Stack pointer register
pub const STACK_POINTER: &str = "%rsp";

/// Frame pointer register
pub const FRAME_POINTER: &str = "%rbp";

/// Get the appropriate register suffix for a given type size
pub fn get_register_suffix_for_size(size_bytes: u64) -> &'static str {
    match size_bytes {
        1 => "b",
        2 => "w",
        4 => "l",
        8 => "q",
        _ => "q", // Default to 64-bit
    }
}

/// Get the appropriate x86_64 register name for a given size
pub fn get_sized_register(base_reg: &str, size_bytes: u64) -> String {
    let suffix = get_register_suffix_for_size(size_bytes);

    // Handle special register mappings for different sizes
    match (base_reg, size_bytes) {
        // RAX family
        ("%rax", 1) => "%al".to_string(),
        ("%rax", 2) => "%ax".to_string(),
        ("%rax", 4) => "%eax".to_string(),
        ("%rax", 8) => "%rax".to_string(),

        // RBX family
        ("%rbx", 1) => "%bl".to_string(),
        ("%rbx", 2) => "%bx".to_string(),
        ("%rbx", 4) => "%ebx".to_string(),
        ("%rbx", 8) => "%rbx".to_string(),

        // RCX family
        ("%rcx", 1) => "%cl".to_string(),
        ("%rcx", 2) => "%cx".to_string(),
        ("%rcx", 4) => "%ecx".to_string(),
        ("%rcx", 8) => "%rcx".to_string(),

        // RDX family
        ("%rdx", 1) => "%dl".to_string(),
        ("%rdx", 2) => "%dx".to_string(),
        ("%rdx", 4) => "%edx".to_string(),
        ("%rdx", 8) => "%rdx".to_string(),

        // Extended registers (r8-r15) - different naming convention
        (reg, 1) if reg.starts_with("%r") && reg.len() >= 3 => {
            format!("{}b", reg)
        }
        (reg, 2) if reg.starts_with("%r") && reg.len() >= 3 => {
            format!("{}w", reg)
        }
        (reg, 4) if reg.starts_with("%r") && reg.len() >= 3 => {
            format!("{}d", reg)
        }
        (reg, 8) if reg.starts_with("%r") => reg.to_string(),

        // Default case - just append suffix
        (reg, _) => {
            if let Some(stripped) = reg.strip_prefix('%') {
                format!("%{}{}", stripped, suffix)
            } else {
                format!("%{}{}", reg, suffix)
            }
        }
    }
}

/// Check if a register is a general-purpose register
pub fn is_gp_register(reg: &str) -> bool {
    matches!(
        reg,
        "%rax"
            | "%rbx"
            | "%rcx"
            | "%rdx"
            | "%rsi"
            | "%rdi"
            | "%rbp"
            | "%rsp"
            | "%r8"
            | "%r9"
            | "%r10"
            | "%r11"
            | "%r12"
            | "%r13"
            | "%r14"
            | "%r15"
    )
}

/// Check if a register needs to be saved by the callee
pub fn is_callee_saved(reg: &str) -> bool {
    CALLEE_SAVED_REGISTERS.contains(&reg)
}

/// Check if a register is volatile (caller-saved)
pub fn is_caller_saved(reg: &str) -> bool {
    CALLER_SAVED_REGISTERS.contains(&reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_constants() {
        assert_eq!(ARG_REGISTERS.len(), 6);
        assert_eq!(RETURN_REGISTER, "%rax");
        assert_eq!(STACK_POINTER, "%rsp");
        assert_eq!(FRAME_POINTER, "%rbp");
    }

    #[test]
    fn test_get_register_suffix_for_size() {
        assert_eq!(get_register_suffix_for_size(1), "b");
        assert_eq!(get_register_suffix_for_size(2), "w");
        assert_eq!(get_register_suffix_for_size(4), "l");
        assert_eq!(get_register_suffix_for_size(8), "q");
    }

    #[test]
    fn test_get_sized_register() {
        // Test RAX family
        assert_eq!(get_sized_register("%rax", 1), "%al");
        assert_eq!(get_sized_register("%rax", 2), "%ax");
        assert_eq!(get_sized_register("%rax", 4), "%eax");
        assert_eq!(get_sized_register("%rax", 8), "%rax");

        // Test extended registers
        assert_eq!(get_sized_register("%r8", 1), "%r8b");
        assert_eq!(get_sized_register("%r8", 2), "%r8w");
        assert_eq!(get_sized_register("%r8", 4), "%r8d");
        assert_eq!(get_sized_register("%r8", 8), "%r8");
    }

    #[test]
    fn test_register_classification() {
        assert!(is_gp_register("%rax"));
        assert!(is_gp_register("%r15"));
        assert!(!is_gp_register("%xmm0"));

        assert!(is_callee_saved("%rbx"));
        assert!(!is_callee_saved("%rax"));

        assert!(is_caller_saved("%rax"));
        assert!(!is_caller_saved("%rbx"));
    }
}
