//! x86_64 code generation for MIR (Mid-level IR).
//!
//! Code generation from MIR to x86_64 assembly,
//! supporting System V AMD64 and Microsoft x64 calling conventions.

pub mod constants;
mod function;
mod instruction;
mod moves;
mod selection;
mod simd;
#[cfg(test)]
mod tests;

use function::*;
use instruction::*;
pub use moves::RaxTrackingWriter;
use moves::*;
use selection::*;
pub mod util;

use constants::{fd, linux, macos, stack, windows};
use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame, extend_integer, register_for_size};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    io::{self, Error, Write},
};
use util::*;

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Instruction as MirInst,
        IntBinOp, IntCmpOp, MirType, Module as MirModule, Operand, Register, Signature, VirtualReg,
        instruction::Immediate, register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::CapabilitySet, validate_module_call_parameters,
    },
};

use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator, LiveInterval,
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::{collections::hash_map::Entry, sync::Arc};

use crate::mir_codegen::common::{
    CodegenBase, CopyLocation, assign_stack_slots, assign_stack_slots_where, coalesce_dying_inputs,
    compile_functions_parallel, emit_print_format_section, parallel_codegen_error,
    parallel_copy_schedule, self_tail_entry_label, use_counts,
};

/// MIR to x86_64 code generator implementing the `Codegen` trait.
pub struct X86Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> X86Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
        }
    }

    /// Attach the MIR module that should be emitted in the next codegen pass.
    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    /// Drain the internal assembly buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    /// Emit assembly for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), LaminaError> {
        generate_mir_x86_64_with_units(module, writer, self.base.target_os, codegen_units)
    }
}

impl<'a> Codegen for X86Codegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn capabilities() -> CapabilitySet {
        CapabilitySet::standard_native()
    }

    fn prepare(
        &mut self,
        types: &HashMap<String, MirType>,
        globals: &HashMap<String, Global>,
        funcs: &HashMap<String, Signature>,
        codegen_units: usize,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError> {
        self.base.prepare_base(
            types,
            globals,
            funcs,
            codegen_units,
            verbose,
            options,
            input_name,
        )
    }

    fn compile(&mut self) -> Result<(), CodegenError> {
        self.base.compile_base()
    }

    fn finalize(&mut self) -> Result<(), CodegenError> {
        self.base.finalize_base()
    }

    fn emit_asm(&mut self) -> Result<(), CodegenError> {
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_x86_64_with_units(module, writer, target_os, codegen_units)
            },
            "x86_64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not supported".to_string(),
        ))
    }
}

pub fn generate_mir_x86_64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_x86_64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_x86_64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_x86_64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_x86_64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::X86_64)?;
    let abi = X86ABI::new(target_os);

    emit_print_format_section(writer, target_os)?;

    // Emit external function declarations first
    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    writeln!(writer, ".text")?;
    writeln!(writer, "{}", abi.get_main_global())?;

    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_x86_64(name, func, os, settings_arc.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    Ok(())
}
