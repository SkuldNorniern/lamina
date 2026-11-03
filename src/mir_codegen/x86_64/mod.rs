//! x86_64 code generation for MIR (Mid-level IR).
//!
//! Code generation from MIR to x86_64 assembly,
//! supporting System V AMD64 and Microsoft x64 calling conventions.

mod atomic;
pub mod constants;
#[cfg(test)]
mod cpu_level_tests;
mod function;
mod inline_asm;
#[cfg(test)]
mod inline_asm_tests;
mod instruction;
mod moves;
mod selection;
mod simd;
#[cfg(test)]
mod stack_probe_tests;
#[cfg(test)]
mod tests;
mod variadic;
#[cfg(test)]
mod vector_surface_tests;

use function::*;
use instruction::*;
pub use moves::RaxTrackingWriter;
use moves::*;
use selection::*;
pub mod util;

use constants::{fd, freebsd, linux, macos, stack, windows};
use lamina_codegen::LocalRegisterAllocator;
use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame, extend_integer, register_for_size};
use lamina_codegen::{os_uses_coff, os_uses_elf, os_uses_macho};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    io::{self, Error, Write},
};
use util::*;

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, Aggregate, AggregateClass, FloatBinOp, FloatCmpOp, FloatUnOp, Function,
        Global, Instruction as MirInst, IntBinOp, IntCmpOp, IntRotOp, IntUnOp, MirType,
        Module as MirModule, Operand, Parameter, Register, ScalarType, Signature, Signedness,
        VirtualReg, instruction::Immediate, register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::{CapabilitySet, CodegenCapability},
        validate_module_call_parameters,
    },
};

use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator, LiveInterval,
};
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem, X86CpuLevel};
use std::{collections::hash_map::Entry, sync::Arc};

use crate::mir_codegen::common::external_names;
use crate::mir_codegen::common::{
    CodegenBase, CopyLocation, assign_stack_slots, assign_stack_slots_where, coalesce_dying_inputs,
    compile_functions_parallel, emit_global_directives, emit_symbol_visibility,
    emit_x86_64_global_data, parallel_codegen_error, parallel_copy_schedule,
    reject_x86_64_thread_local_globals, self_tail_entry_label, use_counts,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note};

/// MIR to x86_64 code generator implementing the `Codegen` trait.
pub struct X86Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> X86Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem, pointer_width: PointerWidth) -> Self {
        Self {
            base: CodegenBase::new(target_os, pointer_width),
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
            .with(CodegenCapability::AggregateArguments)
            .with(CodegenCapability::IndirectCalls)
            .with(CodegenCapability::GlobalRegisterAllocation)
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

fn reject_misclassified_windows_aggregates(
    module: &MirModule,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Windows {
        return Ok(());
    }
    let mut names: Vec<&String> = module
        .functions
        .iter()
        .filter(|(_, func)| {
            func.sig
                .ret_aggregate
                .is_some_and(|aggregate| aggregate.class != AggregateClass::Integer)
                || func.sig.params.iter().any(|p| {
                    p.aggregate
                        .is_some_and(|aggregate| aggregate.class != AggregateClass::Integer)
                })
        })
        .map(|(name, _)| name)
        .collect();
    names.sort();
    match names.first() {
        Some(name) => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "x86_64 Windows cannot place an aggregate classified for another ABI, as @{name} has"
            ),
        ))),
        None => Ok(()),
    }
}

pub fn generate_mir_x86_64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::X86_64)?;
    reject_x86_64_thread_local_globals(module, target_os)?;
    reject_misclassified_windows_aggregates(module, target_os)?;
    let abi = X86ABI::new(target_os);

    // Emit external function declarations first
    for func_name in external_names(module) {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
        if os_uses_coff(target_os) {
            writeln!(writer, ".def {label}")?;
            writeln!(writer, ".scl 2")?;
            writeln!(writer, ".type 32")?;
            writeln!(writer, ".endef")?;
        }
    }

    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings, module)?;
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;

    let settings_arc = Arc::new(settings.clone());
    let abi_extensions = Arc::new(AbiExtensions::from_module(module, target_os));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_x86_64(
                name,
                func,
                os,
                settings_arc.as_ref(),
                abi_extensions.as_ref(),
            )
        })
        .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    emit_x86_64_global_data(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;
    emit_gnu_stack_note(writer, target_os)?;

    Ok(())
}
