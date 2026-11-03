//! x86_64 stack frame management utilities.

use std::io::{Error, ErrorKind, Write};

use lamina_platform::StackProbe;

/// Stack frame utilities for x86_64 code generation.
pub struct X86Frame;

impl X86Frame {
    /// Generates the function prologue: saves frame pointer and allocates stack space.
    pub fn generate_prologue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        unwind: bool,
        probe: Option<StackProbe>,
        page_size: usize,
    ) -> Result<(), Error> {
        writeln!(writer, "    pushq %rbp")?;
        if unwind {
            writeln!(writer, "    .cfi_def_cfa_offset 16")?;
            writeln!(writer, "    .cfi_offset %rbp, -16")?;
        }
        writeln!(writer, "    movq %rsp, %rbp")?;
        if unwind {
            writeln!(writer, "    .cfi_def_cfa_register %rbp")?;
        }
        let stack_size = Self::aligned_frame_size(stack_size);
        match probe {
            Some(probe) if stack_size > page_size => {
                Self::emit_probed_allocation(writer, stack_size, page_size, probe)?
            }
            _ if stack_size > 0 => writeln!(writer, "    subq ${stack_size}, %rsp")?,
            _ => {}
        }
        Ok(())
    }

    fn emit_probed_allocation<W: Write>(
        writer: &mut W,
        stack_size: usize,
        page_size: usize,
        probe: StackProbe,
    ) -> Result<(), Error> {
        if stack_size > i32::MAX as usize {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("frame of {stack_size} bytes is too large to probe"),
            ));
        }
        match probe {
            StackProbe::Chkstk => {
                writeln!(writer, "    movl ${stack_size}, %eax")?;
                writeln!(writer, "    call __chkstk")?;
                writeln!(writer, "    subq %rax, %rsp")?;
            }
            StackProbe::PageLoop => {
                writeln!(writer, "    leaq -{stack_size}(%rsp), %r11")?;
                let remainder = stack_size % page_size;
                if remainder != 0 {
                    writeln!(writer, "    subq ${remainder}, %rsp")?;
                    writeln!(writer, "    orq $0, (%rsp)")?;
                }
                writeln!(writer, "1:")?;
                writeln!(writer, "    subq ${page_size}, %rsp")?;
                writeln!(writer, "    orq $0, (%rsp)")?;
                writeln!(writer, "    cmpq %r11, %rsp")?;
                writeln!(writer, "    ja 1b")?;
            }
        }
        Ok(())
    }

    /// Frames are a multiple of 16 so rsp stays aligned at every call, which
    /// SSE code in the C library relies on.
    pub fn aligned_frame_size(stack_size: usize) -> usize {
        stack_size.div_ceil(16) * 16
    }

    /// Generates the function epilogue: restores stack and frame pointer, then returns.
    /// The unwind state is restored after the `ret`, so later code keeps the frame's description.
    pub fn generate_epilogue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        unwind: bool,
    ) -> Result<(), Error> {
        let stack_size = Self::aligned_frame_size(stack_size);
        if unwind {
            writeln!(writer, "    .cfi_remember_state")?;
        }
        if stack_size > 0 {
            writeln!(writer, "    addq ${stack_size}, %rsp")?;
        }
        writeln!(writer, "    popq %rbp")?;
        if unwind {
            writeln!(writer, "    .cfi_def_cfa %rsp, 8")?;
        }
        writeln!(writer, "    ret")?;
        if unwind {
            writeln!(writer, "    .cfi_restore_state")?;
        }
        Ok(())
    }

    /// Calculates the stack slot offset from RBP for a given slot index.
    pub fn calculate_stack_offset(slot_index: usize) -> i32 {
        -((slot_index as i32 + 1) * 8)
    }
}
