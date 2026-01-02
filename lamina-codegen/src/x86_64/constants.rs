//! x86_64 platform-specific constants.

/// macOS syscall numbers (with 0x2000000 offset)
pub mod macos {
    /// write syscall on macOS
    pub const SYS_WRITE: i64 = 0x2000004;
}

/// Linux syscall numbers
pub mod linux {
    /// write syscall on Linux
    pub const SYS_WRITE: i64 = 1;
}

/// Standard file descriptors
pub mod fd {
    /// Standard output
    pub const STDOUT: i64 = 1;
    /// Standard input
    pub const STDIN: i64 = 0;
    /// Standard error
    pub const STDERR: i64 = 2;
}

/// Windows x64 calling convention constants
pub mod windows {
    /// Shadow space size (32 bytes) required before function calls
    pub const SHADOW_SPACE_SIZE: i32 = 32;
}

/// Stack alignment constants
pub mod stack {
    /// Minimum stack alignment (16 bytes)
    pub const ALIGNMENT: usize = 16;
    /// Size of a stack slot (8 bytes)
    pub const SLOT_SIZE: usize = 8;
}
