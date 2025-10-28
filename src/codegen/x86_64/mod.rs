pub mod functions;
pub mod globals;
pub mod instructions;
pub mod load_store_opt;
pub mod optimization;
pub mod register_allocator;
pub mod register_info;
pub mod state;
pub mod util;

use crate::{Module, LaminaError};
use std::io::Write;
use std::result::Result;

/// Generate x86_64 assembly for a module
pub fn generate_x86_64_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<(), LaminaError> {
    let mut state = state::CodegenState::new();

    // Add note for non-executable stack
    writeln!(writer, ".section .note.GNU-stack,\"\",@progbits")?;

    // --- 1. Process Globals and emit .data/.bss ---
    globals::generate_global_data_section(module, writer, &mut state)?;

    // --- 2. Emit .text section ---
    writeln!(writer, "\n.section .text")?;

    // --- 2.5. Add extern declarations for heap functions ---
    writeln!(writer, "    .extern malloc")?;
    writeln!(writer, "    .extern free")?;

    // --- 3. Process functions ---
    functions::generate_functions(module, writer, &mut state)?;

    // Generate global variable sections (.rodata, .data, .bss)
    globals::generate_globals(&state, writer)?;

    Ok(())
}
