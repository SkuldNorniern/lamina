//! x86_64 codegen utilities
//!
//! This module provides register allocation, ABI handling, and frame management
//! utilities for x86_64 code generation.

pub mod abi;
pub mod constants;
pub mod frame;
pub mod regalloc;

pub use abi::X86ABI;
pub use frame::X86Frame;
pub use regalloc::X64RegAlloc;

use std::io::{self, Write};

/// Extend low integer bits to the full register width.
pub fn extend_integer(
    writer: &mut impl Write,
    register: &str,
    size: usize,
    signed: bool,
) -> io::Result<()> {
    if size >= 8 {
        return Ok(());
    }
    let source = register_for_size(register, size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid integer register"))?;
    let (instruction, destination) = if signed {
        (
            match size {
                1 => "movsbq",
                2 => "movswq",
                _ => "movslq",
            },
            register.to_string(),
        )
    } else {
        (
            match size {
                1 => "movzbl",
                2 => "movzwl",
                _ => "movl",
            },
            register_for_size(register, 4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "invalid integer register")
            })?,
        )
    };
    writeln!(writer, "    {instruction} %{source}, %{destination}")
}

pub fn register_for_size(register: &str, size: usize) -> Option<String> {
    let alias = match (register, size) {
        ("rax", 1) => "al".to_string(),
        ("rax", 2) => "ax".to_string(),
        ("rax", 4) => "eax".to_string(),
        ("rbx", 1) => "bl".to_string(),
        ("rbx", 2) => "bx".to_string(),
        ("rbx", 4) => "ebx".to_string(),
        ("rcx", 1) => "cl".to_string(),
        ("rcx", 2) => "cx".to_string(),
        ("rcx", 4) => "ecx".to_string(),
        ("rdx", 1) => "dl".to_string(),
        ("rdx", 2) => "dx".to_string(),
        ("rdx", 4) => "edx".to_string(),
        ("rsi", 1) => "sil".to_string(),
        ("rsi", 2) => "si".to_string(),
        ("rsi", 4) => "esi".to_string(),
        ("rdi", 1) => "dil".to_string(),
        ("rdi", 2) => "di".to_string(),
        ("rdi", 4) => "edi".to_string(),
        ("rbp", 1) => "bpl".to_string(),
        ("rbp", 2) => "bp".to_string(),
        ("rbp", 4) => "ebp".to_string(),
        ("rsp", 1) => "spl".to_string(),
        ("rsp", 2) => "sp".to_string(),
        ("rsp", 4) => "esp".to_string(),
        (name, 1)
            if matches!(
                name,
                "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15"
            ) =>
        {
            format!("{name}b")
        }
        (name, 2)
            if matches!(
                name,
                "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15"
            ) =>
        {
            format!("{name}w")
        }
        (name, 4)
            if matches!(
                name,
                "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15"
            ) =>
        {
            format!("{name}d")
        }
        (name, 8) => name.to_string(),
        _ => return None,
    };
    Some(alias)
}
