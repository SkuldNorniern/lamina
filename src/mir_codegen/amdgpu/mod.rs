use super::gpu::{error, kernels};
use crate::{
    error::LaminaError,
    mir::{
        AddressSpace, Function, Instruction, MirType, Module, Register, RegisterClass, ScalarType,
        gpu::{GpuAxis, GpuIntrinsic},
    },
};
use std::{collections::HashMap, io::Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmdGpuTarget {
    Gfx900,
    #[default]
    Gfx1200,
}

impl AmdGpuTarget {
    pub fn cpu(self) -> &'static str {
        match self {
            Self::Gfx900 => "gfx900",
            Self::Gfx1200 => "gfx1200",
        }
    }

    fn wave_size(self) -> u32 {
        match self {
            Self::Gfx900 => 64,
            Self::Gfx1200 => 32,
        }
    }
}

#[derive(Clone, Copy)]
struct Value {
    start: u32,
    ty: MirType,
}

struct Argument {
    offset: usize,
    ty: MirType,
}

struct Kernel {
    body: Vec<u8>,
    values: HashMap<Register, Value>,
    arguments: Vec<Argument>,
    kernarg_size: usize,
    kernarg_align: usize,
    vgprs: u32,
    sgprs: u32,
    target: AmdGpuTarget,
    group_ids: bool,
}

impl Kernel {
    fn new(function: &Function, target: AmdGpuTarget) -> Result<Self, LaminaError> {
        let mut kernel = Self {
            body: Vec::new(),
            values: HashMap::new(),
            arguments: Vec::new(),
            kernarg_size: 0,
            kernarg_align: 4,
            vgprs: if target == AmdGpuTarget::Gfx900 { 3 } else { 1 },
            sgprs: 0,
            target,
            group_ids: false,
        };
        for parameter in &function.sig.params {
            let ty = parameter.ty;
            if !matches!(
                ty,
                MirType::Scalar(
                    ScalarType::I32
                        | ScalarType::I64
                        | ScalarType::Ptr
                        | ScalarType::PtrIn(AddressSpace::Global)
                )
            ) {
                return Err(error(
                    "AMDGPU arguments require i32, i64, or global pointers",
                ));
            }
            let size = ty.size_bytes();
            let offset = kernel.kernarg_size.next_multiple_of(size);
            kernel.kernarg_align = kernel.kernarg_align.max(size);
            kernel.kernarg_size = offset + size;
            if kernel.kernarg_size > 4096 {
                return Err(error("AMDGPU kernarg segment exceeds 4096 bytes"));
            }
            kernel.arguments.push(Argument { offset, ty });
            let value = kernel.allocate(&parameter.reg, ty)?;
            kernel.sgprs = 10;
            if value.ty.size_bytes() == 8 {
                writeln!(kernel.body, "    s_load_dwordx2 s[8:9], s[0:1], {offset}")?;
            } else {
                writeln!(kernel.body, "    s_load_dword s8, s[0:1], {offset}")?;
            }
            writeln!(
                kernel.body,
                "    s_waitcnt lgkmcnt(0)\n    v_mov_b32 v{}, s8",
                value.start
            )?;
            if size == 8 {
                writeln!(kernel.body, "    v_mov_b32 v{}, s9", value.start + 1)?;
            }
        }
        kernel.kernarg_size = kernel.kernarg_size.next_multiple_of(kernel.kernarg_align);
        Ok(kernel)
    }

    fn allocate(&mut self, register: &Register, ty: MirType) -> Result<Value, LaminaError> {
        if !register.is_virtual()
            || register.class() != RegisterClass::Gpr
            || self.values.contains_key(register)
        {
            return Err(error(
                "AMDGPU values require unique virtual GPR definitions",
            ));
        }
        let words = (ty.size_bytes() / 4) as u32;
        self.vgprs = self.vgprs.next_multiple_of(words);
        let value = Value {
            start: self.vgprs,
            ty,
        };
        self.vgprs += words;
        if self.vgprs > 128 {
            return Err(error(
                "AMDGPU kernel exceeds 128 VGPRs; spilling is unsupported",
            ));
        }
        self.values.insert(register.clone(), value);
        Ok(value)
    }

    fn instruction(&mut self, instruction: &Instruction) -> Result<(), LaminaError> {
        match instruction {
            Instruction::Ret { value: None } => writeln!(self.body, "    s_endpgm")?,
            Instruction::Call { name, args, ret }
                if args.is_empty()
                    && ret.is_none()
                    && GpuIntrinsic::from_name(name) == Some(GpuIntrinsic::WorkgroupBarrier) =>
            {
                match self.target {
                    AmdGpuTarget::Gfx900 => writeln!(
                        self.body,
                        "    s_waitcnt vmcnt(0) lgkmcnt(0)\n    s_barrier"
                    )?,
                    AmdGpuTarget::Gfx1200 => writeln!(
                        self.body,
                        "    s_wait_loadcnt 0\n    s_wait_storecnt 0\n    s_barrier_signal -1\n    s_barrier_wait -1"
                    )?,
                }
            }
            Instruction::Call {
                name,
                args,
                ret: Some(dst),
            } if args.is_empty() => {
                let intrinsic = GpuIntrinsic::from_name(name)
                    .ok_or_else(|| error("unsupported AMDGPU call"))?;
                let value = self.allocate(dst, MirType::i32())?;
                let axis_index = |axis| match axis {
                    GpuAxis::X => 0,
                    GpuAxis::Y => 1,
                    GpuAxis::Z => 2,
                };
                match intrinsic {
                    GpuIntrinsic::ThreadId(axis) => {
                        let axis = axis_index(axis);
                        if self.target == AmdGpuTarget::Gfx900 {
                            writeln!(self.body, "    v_mov_b32 v{}, v{axis}", value.start)?;
                        } else {
                            writeln!(
                                self.body,
                                "    v_bfe_u32 v{}, v0, {}, 10",
                                value.start,
                                axis * 10
                            )?;
                        }
                    }
                    GpuIntrinsic::BlockId(axis) => {
                        self.group_ids = true;
                        let base = if self.arguments.is_empty() { 0 } else { 2 };
                        self.sgprs = self.sgprs.max(base + 3);
                        writeln!(
                            self.body,
                            "    v_mov_b32 v{}, s{}",
                            value.start,
                            base + axis_index(axis)
                        )?;
                    }
                    _ => {
                        return Err(error(
                            "AMDGPU supports work-item and workgroup ID intrinsics only",
                        ));
                    }
                }
            }
            _ => return Err(error("unsupported AMDGPU instruction")),
        }
        Ok(())
    }
}

pub fn emit<W: Write>(module: &Module, writer: &mut W) -> Result<(), LaminaError> {
    emit_with_target(module, writer, AmdGpuTarget::default())
}

pub fn emit_with_target<W: Write>(
    module: &Module,
    writer: &mut W,
    target: AmdGpuTarget,
) -> Result<(), LaminaError> {
    let mut output = Vec::new();
    let kernels = kernels(module, true)?;
    let mut compiled = Vec::new();
    writeln!(
        output,
        ".amdgcn_target \"amdgcn-amd-amdhsa--{}\"",
        target.cpu()
    )?;
    for (function, name, _) in &kernels {
        let mut kernel = Kernel::new(function, target)?;
        for instruction in &function.blocks[0].instructions {
            kernel.instruction(instruction)?;
        }
        writeln!(
            output,
            ".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:"
        )?;
        output.extend(&kernel.body);
        writeln!(output, ".L{name}_end:\n.size {name}, .L{name}_end-{name}")?;
        writeln!(
            output,
            ".section .rodata\n.p2align 6\n.amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 0\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size {}",
            kernel.kernarg_size
        )?;
        if target == AmdGpuTarget::Gfx900 {
            writeln!(output, "    .amdhsa_user_sgpr_private_segment_buffer 0")?;
        }
        if target == AmdGpuTarget::Gfx1200 {
            writeln!(output, "    .amdhsa_wavefront_size32 1")?;
        }
        let group_ids = u8::from(kernel.group_ids);
        writeln!(
            output,
            "    .amdhsa_user_sgpr_kernarg_segment_ptr {}\n    .amdhsa_system_sgpr_workgroup_id_x {group_ids}\n    .amdhsa_system_sgpr_workgroup_id_y {group_ids}\n    .amdhsa_system_sgpr_workgroup_id_z {group_ids}\n    .amdhsa_system_vgpr_workitem_id {}\n    .amdhsa_next_free_vgpr {}\n    .amdhsa_next_free_sgpr {}\n    .amdhsa_reserve_vcc 0\n.end_amdhsa_kernel",
            u8::from(!kernel.arguments.is_empty()),
            2,
            kernel.vgprs,
            kernel.sgprs
        )?;
        compiled.push(kernel);
    }
    writeln!(
        output,
        ".amdgpu_metadata\n---\namdhsa.version: [1, 0]\namdhsa.kernels:"
    )?;
    for ((_, name, metadata), kernel) in kernels.iter().zip(&compiled) {
        writeln!(
            output,
            "  - .name: {name}\n    .symbol: {name}.kd\n    .kernarg_segment_size: {}\n    .kernarg_segment_align: {}\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: {}\n    .sgpr_count: {}\n    .vgpr_count: {}\n    .max_flat_workgroup_size: {}",
            kernel.kernarg_size,
            kernel.kernarg_align,
            target.wave_size(),
            kernel.sgprs,
            kernel.vgprs,
            metadata.max_threads.unwrap_or(1024)
        )?;
        if let Some([x, y, z]) = metadata.required_workgroup_size {
            writeln!(output, "    .reqd_workgroup_size: [{x}, {y}, {z}]")?;
        }
        if !kernel.arguments.is_empty() {
            writeln!(output, "    .args:")?;
            for (index, argument) in kernel.arguments.iter().enumerate() {
                let pointer = matches!(
                    argument.ty,
                    MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_))
                );
                writeln!(
                    output,
                    "      - .name: arg{index}\n        .size: {}\n        .offset: {}\n        .value_kind: {}",
                    argument.ty.size_bytes(),
                    argument.offset,
                    if pointer { "global_buffer" } else { "by_value" }
                )?;
                if pointer {
                    writeln!(output, "        .address_space: global")?;
                }
            }
        }
    }
    writeln!(output, "...\n.end_amdgpu_metadata")?;
    writer.write_all(&output)?;
    Ok(())
}
