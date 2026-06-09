//! PowerPC64 ELFv2 code generator for MIR.
//!
//! Targets little-endian PowerPC64 (ppc64le) running Linux, using the
//! ELFv2 ABI.  Assembly output is suitable for GNU `as` or LLVM `as`.
//!
//! ## Register conventions used internally
//!
//! - `r3`   — primary integer scratch / return value
//! - `r4`   — secondary integer scratch
//! - `r3–r10` — integer arguments (in order)
//! - `f1`   — primary float scratch / return value
//! - `f2`   — secondary float scratch
//! - Stack pointer: `r1`
//! - TOC: `r2` (never clobbered)
//! - Link register saved/restored in prologue/epilogue

mod util;

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::result::Result;
use std::sync::Arc;
use util::{
    load_operand_to_register, load_register_to_r3, load_register_to_register, store_r3_to_register,
};

use crate::mir::register::RegisterClass;
use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::common::{CodegenBase, compile_functions_parallel, parallel_codegen_error};
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
    capability::CapabilitySet,
};

use lamina_codegen::powerpc::{Ppc64Abi, Ppc64Frame, Ppc64RegAlloc};
use lamina_codegen::{Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

pub struct Ppc64Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> Ppc64Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
        }
    }

    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_ppc64_with_units_and_settings(
            module,
            writer,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for Ppc64Codegen<'a> {
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
        types: &HashMap<String, crate::mir::MirType>,
        globals: &HashMap<String, crate::mir::Global>,
        funcs: &HashMap<String, crate::mir::Signature>,
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
                generate_mir_ppc64_with_units_and_settings(
                    module,
                    writer,
                    target_os,
                    codegen_units,
                    &MirCodegenSettings::default(),
                )
            },
            "PowerPC64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not yet supported for PowerPC64".to_string(),
        ))
    }
}

fn ppc64_stack_offset_for_linear_spill(off: i32) -> i32 {
    let k = ((-off) / 8) as usize;
    let slot_ix = k.saturating_sub(1);
    Ppc64Frame::calculate_stack_offset(slot_ix)
}

fn compile_single_function_ppc64(
    func_name: &str,
    func: &crate::mir::Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let abi = Ppc64Abi::new(target_os);

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{}:", label)
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("I/O error: {}", e)))?;

    if settings.emit_asm_debug_lines {
        let tag = settings.debug_file_tag.replace('\"', "'");
        writeln!(output, "    .file 1 \"{}\"", tag)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("I/O error: {}", e)))?;
    }

    let mut reg_alloc = Ppc64RegAlloc::new(target_os);
    let mut stack_slots: HashMap<crate::mir::VirtualReg, i32> = HashMap::new();

    if settings.regalloc != RegallocStrategy::Incremental {
        let mut def_regs: HashSet<crate::mir::VirtualReg> = HashSet::new();
        let mut used_regs: HashSet<crate::mir::VirtualReg> = HashSet::new();
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                {
                    def_regs.insert(*vreg);
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        used_regs.insert(*vreg);
                    }
                }
            }
        }
        for vreg in &def_regs {
            if !stack_slots.contains_key(vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(*vreg, Ppc64Frame::calculate_stack_offset(slot_index));
            }
        }
        for vreg in used_regs {
            if !def_regs.contains(&vreg) && !stack_slots.contains_key(&vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(vreg, Ppc64Frame::calculate_stack_offset(slot_index));
            }
        }
        let pool = Ppc64RegAlloc::gpr_pool_for_global_allocation();
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| i.vreg.class == RegisterClass::Gpr)
            .collect();
        let plan = match settings.regalloc {
            RegallocStrategy::LinearScanGlobal => {
                LinearScanAllocator::allocate(&intervals, pool.as_slice())
            }
            RegallocStrategy::GraphColorGlobal => {
                GraphColorAllocator::allocate(&intervals, pool.as_slice())
            }
            RegallocStrategy::Incremental => {
                return Err(CodegenError::InvalidCodegenOptions(
                    "internal: incremental in global branch".to_string(),
                ));
            }
        };
        reg_alloc = Ppc64RegAlloc::from_global_plan(target_os, &plan);
        for (v, a) in &plan {
            if let MirAllocation::Spill(off) = a {
                stack_slots.insert(*v, ppc64_stack_offset_for_linear_spill(*off));
            }
        }
    } else {
        let mut next_slot = 0usize;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                    && !stack_slots.contains_key(vreg)
                {
                    stack_slots.insert(*vreg, Ppc64Frame::calculate_stack_offset(next_slot));
                    next_slot += 1;
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg)
                    {
                        stack_slots.insert(*vreg, Ppc64Frame::calculate_stack_offset(next_slot));
                        next_slot += 1;
                    }
                }
            }
        }
    }

    let local_bytes = stack_slots.len() * 8;
    Ppc64Frame::generate_prologue(&mut output, local_bytes)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    let mut debug_line: u32 = 0;
    for block in &func.blocks {
        writeln!(output, ".L_{}:", block.label)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("I/O error: {}", e)))?;

        for inst in &block.instructions {
            emit_instruction_ppc64(
                inst,
                &mut output,
                &mut reg_alloc,
                &stack_slots,
                target_os,
                settings,
                &mut debug_line,
            )
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
    }

    Ok(output)
}

pub fn generate_mir_ppc64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    generate_mir_ppc64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_ppc64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), crate::error::LaminaError> {
    generate_mir_ppc64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_ppc64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), crate::error::LaminaError> {
    crate::mir_codegen::validate_module_call_parameters(module, TargetArchitecture::PowerPC64)?;
    let abi = Ppc64Abi::new(target_os);

    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_print_format())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    writeln!(writer, "{}", abi.get_main_global())?;

    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {}", label)?;
    }

    let settings_arc = Arc::new(settings.clone());
    let results = compile_functions_parallel(module, target_os, codegen_units, {
        let settings_arc = settings_arc.clone();
        move |name, func, os| compile_single_function_ppc64(name, func, os, settings_arc.as_ref())
    })
    .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn emit_instruction_ppc64<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &mut Ppc64RegAlloc,
    stack_slots: &HashMap<crate::mir::VirtualReg, i32>,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    debug_line: &mut u32,
) -> Result<(), crate::error::LaminaError> {
    if settings.emit_asm_debug_lines {
        *debug_line = debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", *debug_line)?;
    }
    match inst {
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
            match op {
                crate::mir::IntBinOp::Add => writeln!(writer, "    add 3, 3, 4")?,
                crate::mir::IntBinOp::Sub => writeln!(writer, "    sub 3, 3, 4")?,
                crate::mir::IntBinOp::Mul => writeln!(writer, "    mulld 3, 3, 4")?,
                crate::mir::IntBinOp::SDiv => writeln!(writer, "    divd 3, 3, 4")?,
                crate::mir::IntBinOp::UDiv => writeln!(writer, "    divdu 3, 3, 4")?,
                crate::mir::IntBinOp::SRem => {
                    // rem = lhs - (lhs/rhs)*rhs
                    writeln!(writer, "    divd 5, 3, 4")?;
                    writeln!(writer, "    mulld 5, 5, 4")?;
                    writeln!(writer, "    sub 3, 3, 5")?;
                }
                crate::mir::IntBinOp::URem => {
                    writeln!(writer, "    divdu 5, 3, 4")?;
                    writeln!(writer, "    mulld 5, 5, 4")?;
                    writeln!(writer, "    sub 3, 3, 5")?;
                }
                crate::mir::IntBinOp::And => writeln!(writer, "    and 3, 3, 4")?,
                crate::mir::IntBinOp::Or => writeln!(writer, "    or 3, 3, 4")?,
                crate::mir::IntBinOp::Xor => writeln!(writer, "    xor 3, 3, 4")?,
                crate::mir::IntBinOp::Shl => writeln!(writer, "    sld 3, 3, 4")?,
                crate::mir::IntBinOp::AShr => writeln!(writer, "    srad 3, 3, 4")?,
                crate::mir::IntBinOp::LShr => writeln!(writer, "    srd 3, 3, 4")?,
            }
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
            // cmpd sets CR0; use branch + li to materialise 0/1 into r3.
            writeln!(writer, "    cmpd 3, 4")?;
            let (set_true, set_false) = match op {
                crate::mir::IntCmpOp::Eq => ("beq", "bne"),
                crate::mir::IntCmpOp::Ne => ("bne", "beq"),
                crate::mir::IntCmpOp::SLt => ("blt", "bge"),
                crate::mir::IntCmpOp::SLe => ("ble", "bgt"),
                crate::mir::IntCmpOp::SGt => ("bgt", "ble"),
                crate::mir::IntCmpOp::SGe => ("bge", "blt"),
                // Unsigned: use cmplw/cmpld
                crate::mir::IntCmpOp::ULt
                | crate::mir::IntCmpOp::ULe
                | crate::mir::IntCmpOp::UGt
                | crate::mir::IntCmpOp::UGe => {
                    // Redo with unsigned compare
                    writeln!(writer, "    cmpld 3, 4")?;
                    match op {
                        crate::mir::IntCmpOp::ULt => ("blt", "bge"),
                        crate::mir::IntCmpOp::ULe => ("ble", "bgt"),
                        crate::mir::IntCmpOp::UGt => ("bgt", "ble"),
                        _ => ("bge", "blt"),
                    }
                }
            };
            writeln!(writer, "    {} .L_ppc_cmp_true_{:p}", set_true, lhs)?;
            writeln!(writer, "    li 3, 0")?;
            writeln!(writer, "    b .L_ppc_cmp_end_{:p}", lhs)?;
            writeln!(writer, ".L_ppc_cmp_true_{:p}:", lhs)?;
            writeln!(writer, "    li 3, 1")?;
            writeln!(writer, ".L_ppc_cmp_end_{:p}:", lhs)?;
            let _ = set_false; // used in branch selection above
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;
            let suffix = if is_f32 { "s" } else { "d" };
            // Load float operands from integer stack slots into FPRs via memory.
            emit_load_fp_operand(lhs, writer, reg_alloc, stack_slots, "1", is_f32)?;
            emit_load_fp_operand(rhs, writer, reg_alloc, stack_slots, "2", is_f32)?;
            match op {
                crate::mir::FloatBinOp::FAdd => writeln!(writer, "    fadd{} 1, 1, 2", suffix)?,
                crate::mir::FloatBinOp::FSub => writeln!(writer, "    fsub{} 1, 1, 2", suffix)?,
                crate::mir::FloatBinOp::FMul => writeln!(writer, "    fmul{} 1, 1, 2", suffix)?,
                crate::mir::FloatBinOp::FDiv => writeln!(writer, "    fdiv{} 1, 1, 2", suffix)?,
            }
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let is_f32 = ty.size_bytes() == 4;
            let suffix = if is_f32 { "s" } else { "d" };
            emit_load_fp_operand(src, writer, reg_alloc, stack_slots, "1", is_f32)?;
            match op {
                crate::mir::FloatUnOp::FNeg => writeln!(writer, "    fneg{} 1, 1", suffix)?,
                crate::mir::FloatUnOp::FSqrt => writeln!(writer, "    fsqrt{} 1, 1", suffix)?,
            }
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;
            emit_load_fp_operand(lhs, writer, reg_alloc, stack_slots, "1", is_f32)?;
            emit_load_fp_operand(rhs, writer, reg_alloc, stack_slots, "2", is_f32)?;
            writeln!(writer, "    fcmpu 0, 1, 2")?;
            let branch = match op {
                crate::mir::FloatCmpOp::Eq => "beq",
                crate::mir::FloatCmpOp::Ne => "bne",
                crate::mir::FloatCmpOp::Lt => "blt",
                crate::mir::FloatCmpOp::Le => "ble",
                crate::mir::FloatCmpOp::Gt => "bgt",
                crate::mir::FloatCmpOp::Ge => "bge",
            };
            writeln!(writer, "    {} .L_ppc_fcmp_true_{:p}", branch, lhs)?;
            writeln!(writer, "    li 3, 0")?;
            writeln!(writer, "    b .L_ppc_fcmp_end_{:p}", lhs)?;
            writeln!(writer, ".L_ppc_fcmp_true_{:p}:", lhs)?;
            writeln!(writer, "    li 3, 1")?;
            writeln!(writer, ".L_ppc_fcmp_end_{:p}:", lhs)?;
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // Materialise both values, then pick using isel or a branch.
            load_operand_to_register(false_val, writer, reg_alloc, stack_slots, "4")?;
            load_operand_to_register(true_val, writer, reg_alloc, stack_slots, "3")?;
            // Test condition in r5.
            match cond {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
            }
            writeln!(writer, "    cmpdi 5, 0")?;
            // isel: if CR0[EQ] is set (cond == 0), select false (r4); else true (r3)
            writeln!(writer, "    isel 3, 4, 3, 2")?; // CR0 bit 2 = EQ
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            match value {
                Register::Virtual(v) => {
                    load_register_to_r3(v, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            for (case_val, case_label) in cases {
                writeln!(writer, "    cmpdi 3, {}", case_val)?;
                writeln!(writer, "    beq .L_{}", case_label)?;
            }
            writeln!(writer, "    b .L_{}", default)?;
        }
        MirInst::Call { name, args, ret } => {
            let abi = Ppc64Abi::new(target_os);
            let arg_regs = Ppc64Abi::ARG_REGISTERS;

            if name == "print" {
                if let Some(arg) = args.first() {
                    writeln!(writer, "    addis 3, 2, .L_mir_fmt_int@toc@ha")?;
                    writeln!(writer, "    addi 3, 3, .L_mir_fmt_int@toc@l")?;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "4")?;
                    writeln!(writer, "    bl printf")?;
                    writeln!(writer, "    nop")?;
                }
            } else {
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                for (i, arg) in args.iter().take(num_reg_args).enumerate() {
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[i])?;
                }

                let stack_space = ((num_stack_args * 8) + 15) & !15;
                if stack_space > 0 {
                    writeln!(writer, "    addi 1, 1, -{}", stack_space)?;
                    for (i, arg) in args.iter().skip(num_reg_args).enumerate() {
                        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "11")?;
                        writeln!(writer, "    std 11, {}(1)", i * 8)?;
                    }
                }

                let target_sym = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                writeln!(writer, "    bl {}", target_sym)?;
                writeln!(writer, "    nop")?;

                if stack_space > 0 {
                    writeln!(writer, "    addi 1, 1, {}", stack_space)?;
                }
            }

            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::TailCall { name, args } => {
            let abi = Ppc64Abi::new(target_os);
            let arg_regs = Ppc64Abi::ARG_REGISTERS;
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());
            for (i, arg) in args.iter().take(num_reg_args).enumerate() {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[i])?;
            }
            let local_bytes = stack_slots.len() * 8;
            let frame_size = Ppc64Frame::aligned_frame_size(local_bytes);
            if num_stack_args > 0 {
                for (j, arg) in args.iter().skip(num_reg_args).enumerate() {
                    let disp = frame_size as i32 + (j as i32) * 8;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "11")?;
                    writeln!(writer, "    std 11, {}(1)", disp)?;
                }
            }
            Ppc64Frame::generate_tail_epilogue(writer, local_bytes).map_err(|e| {
                crate::error::LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(
                    e.to_string(),
                ))
            })?;
            let target_sym = abi
                .call_stub(name)
                .unwrap_or_else(|| abi.mangle_function_name(name));
            writeln!(writer, "    b {}", target_sym)?;
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            let load_op = match ty {
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I1)
                | crate::mir::MirType::Scalar(crate::mir::ScalarType::I8) => "lbz",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I16) => "lhz",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I32) => "lwz",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)
                | crate::mir::MirType::Scalar(crate::mir::ScalarType::Ptr) => "ld",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F32) => "lfs",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F64) => "lfd",
                other => {
                    return Err(crate::error::LaminaError::CodegenError(
                        CodegenError::UnsupportedFeature(format!(
                            "PowerPC64 load: unsupported type {:?}",
                            other
                        )),
                    ));
                }
            };
            match addr {
                crate::mir::instruction::AddressMode::BaseOffset { base, offset } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                        }
                        Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
                    }
                    writeln!(writer, "    {} 3, {}(5)", load_op, offset)?;
                    if let Register::Virtual(vreg) = dst {
                        store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
                    }
                }
                _ => {
                    return Err(crate::error::LaminaError::CodegenError(
                        CodegenError::UnsupportedFeature(
                            "PowerPC64 load: only base+offset addressing supported".to_string(),
                        ),
                    ));
                }
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs: _,
        } => {
            let store_op = match ty {
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I1)
                | crate::mir::MirType::Scalar(crate::mir::ScalarType::I8) => "stb",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I16) => "sth",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I32) => "stw",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)
                | crate::mir::MirType::Scalar(crate::mir::ScalarType::Ptr) => "std",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F32) => "stfs",
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F64) => "stfd",
                other => {
                    return Err(crate::error::LaminaError::CodegenError(
                        CodegenError::UnsupportedFeature(format!(
                            "PowerPC64 store: unsupported type {:?}",
                            other
                        )),
                    ));
                }
            };
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "3")?;
            match addr {
                crate::mir::instruction::AddressMode::BaseOffset { base, offset } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                        }
                        Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
                    }
                    writeln!(writer, "    {} 3, {}(5)", store_op, offset)?;
                }
                _ => {
                    return Err(crate::error::LaminaError::CodegenError(
                        CodegenError::UnsupportedFeature(
                            "PowerPC64 store: only base+offset addressing supported".to_string(),
                        ),
                    ));
                }
            }
        }
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "3")?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            if *offset != 0 {
                writeln!(writer, "    addi 3, 3, {}", offset)?;
            }
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    b .L_{}", target)?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            match cond {
                Register::Virtual(v) => {
                    load_register_to_r3(v, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            writeln!(writer, "    cmpdi 3, 0")?;
            writeln!(writer, "    bne .L_{}", true_target)?;
            writeln!(writer, "    b .L_{}", false_target)?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                load_operand_to_register(val, writer, reg_alloc, stack_slots, "3")?;
            }
            let local_bytes = stack_slots.len() * 8;
            Ppc64Frame::generate_epilogue(writer, local_bytes).map_err(|e| {
                crate::error::LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(
                    e.to_string(),
                ))
            })?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {}", text)?;
        }
        other => {
            return Err(crate::error::LaminaError::CodegenError(
                CodegenError::UnsupportedFeature(format!(
                    "PowerPC64 backend: instruction not yet supported: {:?}",
                    other
                )),
            ));
        }
    }
    Ok(())
}

/// Load a float operand (stored as integer bits in a stack slot) into FPR `fN`.
///
/// Strategy: store the integer bits to a temp stack slot and then `lfs`/`lfd` them
/// into the FPR, which reinterprets the bit pattern as a float.
fn emit_load_fp_operand<W: Write>(
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &mut Ppc64RegAlloc,
    stack_slots: &HashMap<crate::mir::VirtualReg, i32>,
    fpr: &str,
    is_f32: bool,
) -> Result<(), crate::error::LaminaError> {
    let load_fp = if is_f32 { "lfs" } else { "lfd" };
    match operand {
        crate::mir::Operand::Register(reg) => {
            match reg {
                Register::Virtual(v) => {
                    if let Some(offset) = stack_slots.get(v) {
                        writeln!(writer, "    {} {}, {}(1)", load_fp, fpr, offset)?;
                    } else if let Some(phys) = reg_alloc.get_mapping_for(v) {
                        // Store to temp stack location then load as float
                        writeln!(writer, "    std {}, -8(1)", phys)?;
                        writeln!(writer, "    {} {}, -8(1)", load_fp, fpr)?;
                    }
                }
                Register::Physical(p) => {
                    writeln!(writer, "    std {}, -8(1)", p.name)?;
                    writeln!(writer, "    {} {}, -8(1)", load_fp, fpr)?;
                }
            }
        }
        crate::mir::Operand::Immediate(imm) => {
            let bits: i64 = match imm {
                crate::mir::instruction::Immediate::F32(v) => v.to_bits() as i64,
                crate::mir::instruction::Immediate::F64(v) => v.to_bits() as i64,
                crate::mir::instruction::Immediate::I32(v) => *v as i64,
                crate::mir::instruction::Immediate::I64(v) => *v,
                other => {
                    return Err(crate::error::LaminaError::CodegenError(
                        CodegenError::UnsupportedFeature(format!(
                            "PowerPC64: float immediate {:?} not supported",
                            other
                        )),
                    ));
                }
            };
            writeln!(writer, "    li 11, {}", bits)?;
            writeln!(writer, "    std 11, -8(1)")?;
            writeln!(writer, "    {} {}, -8(1)", load_fp, fpr)?;
        }
    }
    Ok(())
}

/// Store FPR result back to a virtual register stack slot as integer bits.
fn emit_store_fp_result<W: Write>(
    fpr: &str,
    vreg: &crate::mir::VirtualReg,
    writer: &mut W,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<crate::mir::VirtualReg, i32>,
    is_f32: bool,
) -> Result<(), crate::error::LaminaError> {
    let store_fp = if is_f32 { "stfs" } else { "stfd" };
    if let Some(offset) = stack_slots.get(vreg) {
        writeln!(writer, "    {} {}, {}(1)", store_fp, fpr, offset)?;
    } else if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
        writeln!(writer, "    {} {}, -8(1)", store_fp, fpr)?;
        writeln!(writer, "    ld {}, -8(1)", phys)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, MirType, ScalarType, VirtualReg};
    use crate::mir::instruction::{Immediate, Instruction, IntBinOp, Operand};

    fn make_module_with_empty_func() -> MirModule {
        let mut m = MirModule::new("test_ppc64");
        let f = FunctionBuilder::new("test_fn")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();
        m.add_function(f);
        m
    }

    #[test]
    fn test_generate_empty_function() {
        let module = make_module_with_empty_func();
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("test_fn:"), "Expected function label: {s}");
        assert!(s.contains("blr"), "Expected blr in output: {s}");
    }

    #[test]
    fn test_generate_integer_add() {
        let mut m = MirModule::new("test_ppc64_add");
        let v0 = VirtualReg::gpr(0);
        let v2 = VirtualReg::gpr(2);
        let f = FunctionBuilder::new("add_fn")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(v2),
                lhs: Operand::Register(Register::Virtual(v0)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(v2))),
            })
            .build();
        m.add_function(f);
        let mut output = Vec::new();
        generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("add 3, 3, 4"), "Expected add instruction: {s}");
    }

    #[test]
    fn test_generate_jmp_br() {
        let mut m = MirModule::new("test_ppc64_cf");
        let cond = VirtualReg::gpr(0);
        let f = FunctionBuilder::new("cf_fn")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Br {
                cond: Register::Virtual(cond),
                true_target: "then".to_string(),
                false_target: "else".to_string(),
            })
            .block("then")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(
                    1,
                ))),
            })
            .block("else")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(
                    0,
                ))),
            })
            .build();
        m.add_function(f);
        let mut output = Vec::new();
        generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("bne"), "Expected bne in output: {s}");
        assert!(s.contains(".L_then"), "Expected .L_then in output: {s}");
    }
}
