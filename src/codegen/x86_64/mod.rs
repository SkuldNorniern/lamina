use crate::{Module, Result};
use std::io::Write;

// Declare internal modules
mod state;
mod util;
mod globals;
mod functions;
mod instructions;

// Re-export the main entry point
pub use state::CodegenState;
// Re-export needed state components for sibling module tests
pub use state::{FunctionContext, ValueLocation};
pub use functions::generate_functions;
pub use globals::generate_global_data_section;
pub use globals::generate_globals;

// Declare the tests module

/// Generates x86-64 assembly text (AT&T syntax) from a Lamina IR Module.
///
/// Writes the output assembly to the provided `writer`.
pub fn generate_x86_64_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<()> {
    let mut state = CodegenState::new();

    // Add note for non-executable stack
    writeln!(writer, ".section .note.GNU-stack,\"\",@progbits")?;

    // --- 1. Process Globals and emit .data/.bss --- 
    generate_global_data_section(module, writer, &mut state)?;

    // --- 2. Emit .text section --- 
    writeln!(writer, "\n.section .text")?;

    // --- 3. Process functions --- 
    generate_functions(module, writer, &mut state)?;

    // Generate global variable sections (.rodata, .data, .bss)
    generate_globals(&state, writer)?;

    Ok(())
}
