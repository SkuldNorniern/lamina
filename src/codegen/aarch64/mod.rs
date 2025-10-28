pub mod functions;
pub mod globals;
pub mod instructions;
pub mod state;
pub mod util;

use crate::{Module, LaminaError};
use std::io::Write;
use std::result::Result;

/// Generate aarch64 assembly for a module
pub fn generate_aarch64_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<(), LaminaError> {
    let mut state = state::CodegenState::new();

    // --- 1. Process Globals and emit data/BSS ---
    globals::generate_global_data_section(module, writer, &mut state)?;

    // --- 2. Emit text section ---
    // Use generic .text which is accepted by clang on macOS as well
    writeln!(writer, "\n.text")?;

    // --- 2.5. Add extern declarations for heap functions ---
    writeln!(writer, "    .extern _malloc")?;
    writeln!(writer, "    .extern _free")?;

    // --- 3. Process functions ---
    functions::generate_functions(module, writer, &mut state)?;

    // --- 4. Emit any read-only data we've collected (e.g., format strings) ---
    globals::generate_globals(&state, writer)?;

    Ok(())
}
