pub mod functions;
pub mod globals;
pub mod instructions;
pub mod state;
pub mod util;

use crate::{Module, LaminaError};
use std::io::Write;
use Result;

/// Generate RISC-V RV32I assembly for a module
pub fn generate_riscv32_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<(), LaminaError> {
    let mut state = state::CodegenState::new(IsaWidth::Rv32);
    globals::generate_global_data_section(module, writer, &mut state)?;
    writeln!(writer, "\n.text")?;
    functions::generate_functions(module, writer, &mut state)
}

/// Generate RISC-V RV64I assembly for a module
pub fn generate_riscv64_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<(), LaminaError> {
    let mut state = state::CodegenState::new(IsaWidth::Rv64);
    globals::generate_global_data_section(module, writer, &mut state)?;
    writeln!(writer, "\n.text")?;
    functions::generate_functions(module, writer, &mut state)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaWidth {
    Rv32,
    Rv64,
    Rv128,
}

/// Generate RISC-V RV128I assembly for a module (experimental)
pub fn generate_riscv128_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
) -> Result<(), LaminaError> {
    let mut state = state::CodegenState::new(IsaWidth::Rv128);
    globals::generate_global_data_section(module, writer, &mut state)?;
    writeln!(writer, "\n.text")?;
    functions::generate_functions(module, writer, &mut state)
}
