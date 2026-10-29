//! Register names accepted by inline assembly.

const X86_64_NAMES: &[(&str, &str, u8)] = &[
    ("rax", "rax", 8),
    ("eax", "rax", 4),
    ("rbx", "rbx", 8),
    ("ebx", "rbx", 4),
    ("rcx", "rcx", 8),
    ("ecx", "rcx", 4),
    ("rdx", "rdx", 8),
    ("edx", "rdx", 4),
    ("rsi", "rsi", 8),
    ("esi", "rsi", 4),
    ("rdi", "rdi", 8),
    ("edi", "rdi", 4),
    ("r8", "r8", 8),
    ("r8d", "r8", 4),
    ("r9", "r9", 8),
    ("r9d", "r9", 4),
    ("r10", "r10", 8),
    ("r10d", "r10", 4),
    ("r11", "r11", 8),
    ("r11d", "r11", 4),
    ("r12", "r12", 8),
    ("r12d", "r12", 4),
    ("r13", "r13", 8),
    ("r13d", "r13", 4),
    ("r14", "r14", 8),
    ("r14d", "r14", 4),
    ("r15", "r15", 8),
    ("r15d", "r15", 4),
];

/// Names of the frame and stack pointers. They are rejected as operands and clobbers on
/// every target.
pub const FRAME_REGISTERS: &[&str] = &["rsp", "esp", "rbp", "ebp", "rip", "eip", "sp", "fp"];

/// The register class accepted in place of a named register.
pub const GPR_CLASS: &str = "reg";

/// Named x86_64 register: the canonical 64-bit register and the access width in bytes.
pub fn x86_64_register(name: &str) -> Option<(&'static str, u8)> {
    X86_64_NAMES
        .iter()
        .find(|(spelling, _, _)| *spelling == name)
        .map(|&(_, canonical, width)| (canonical, width))
}

/// Every x86_64 register name accepted in an operand or clobber.
pub fn x86_64_register_names() -> Vec<&'static str> {
    X86_64_NAMES.iter().map(|(name, _, _)| *name).collect()
}

/// Whether `name` is a frame or stack pointer name.
pub fn is_frame_register(name: &str) -> bool {
    FRAME_REGISTERS.contains(&name)
}
