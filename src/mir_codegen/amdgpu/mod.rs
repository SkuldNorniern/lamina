use super::gpu::{error, kernels};
use crate::{
    error::LaminaError,
    mir::{
        AddressMode, AddressSpace, Function, Immediate, Instruction, IntBinOp, MemoryAttrs,
        MirType, Module, Operand, Register, RegisterClass, ScalarType,
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

impl Value {
    fn assembly(self) -> String {
        if self.ty.size_bytes() == 8 {
            format!("v[{}:{}]", self.start, self.start + 1)
        } else {
            format!("v{}", self.start)
        }
    }
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
    dispatch_ptr: bool,
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
            dispatch_ptr: function.blocks.iter().flat_map(|block| &block.instructions).any(|inst|
                matches!(inst, Instruction::Call { name, .. }
                    if matches!(GpuIntrinsic::from_name(name), Some(GpuIntrinsic::BlockDim(_) | GpuIntrinsic::GridDim(_))))),
        };
        if kernel.dispatch_ptr {
            kernel.sgprs = 2;
        }
        let kernarg = if kernel.dispatch_ptr {
            "s[2:3]"
        } else {
            "s[0:1]"
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
                writeln!(
                    kernel.body,
                    "    s_load_dwordx2 s[8:9], {kernarg}, {offset}"
                )?;
            } else {
                writeln!(kernel.body, "    s_load_dword s8, {kernarg}, {offset}")?;
            }
            let wait = if target == AmdGpuTarget::Gfx1200 {
                "s_wait_kmcnt 0"
            } else {
                "s_waitcnt lgkmcnt(0)"
            };
            writeln!(
                kernel.body,
                "    {wait}\n    v_mov_b32 v{}, s8",
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
        let value = self.temporary(ty)?;
        self.values.insert(register.clone(), value);
        Ok(value)
    }

    fn temporary(&mut self, ty: MirType) -> Result<Value, LaminaError> {
        if !matches!(
            ty,
            MirType::Scalar(
                ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::Ptr
                    | ScalarType::PtrIn(AddressSpace::Global)
            )
        ) {
            return Err(error("AMDGPU values require i32, i64, or global pointers"));
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
        Ok(value)
    }

    fn operand(&mut self, operand: &Operand, ty: MirType) -> Result<Value, LaminaError> {
        match operand {
            Operand::Register(register) => self
                .values
                .get(register)
                .copied()
                .filter(|value| value.ty == ty)
                .ok_or_else(|| error("AMDGPU operand has an undefined or mismatched type")),
            Operand::Immediate(immediate) => {
                let bits = match (immediate, ty) {
                    (Immediate::I32(value), MirType::Scalar(ScalarType::I32)) => {
                        u64::from(*value as u32)
                    }
                    (Immediate::I64(value), MirType::Scalar(ScalarType::I64)) => *value as u64,
                    _ => return Err(error("unsupported AMDGPU immediate")),
                };
                let value = self.temporary(ty)?;
                writeln!(
                    self.body,
                    "    v_mov_b32 v{}, 0x{:08x}",
                    value.start, bits as u32
                )?;
                if ty.size_bytes() == 8 {
                    writeln!(
                        self.body,
                        "    v_mov_b32 v{}, 0x{:08x}",
                        value.start + 1,
                        (bits >> 32) as u32
                    )?;
                }
                Ok(value)
            }
        }
    }

    fn copy(&mut self, dst: Value, src: Value) -> Result<(), LaminaError> {
        for word in 0..(dst.ty.size_bytes() / 4) as u32 {
            writeln!(
                self.body,
                "    v_mov_b32 v{}, v{}",
                dst.start + word,
                src.start + word
            )?;
        }
        Ok(())
    }

    fn binary(
        &mut self,
        op: IntBinOp,
        dst: Value,
        lhs: Value,
        rhs: Value,
    ) -> Result<(), LaminaError> {
        let (d, l, r) = (dst.start, lhs.start, rhs.start);
        let mnemonic = match op {
            IntBinOp::Add => "v_add_u32",
            IntBinOp::Sub => "v_sub_u32",
            IntBinOp::Mul => "v_mul_lo_u32",
            IntBinOp::And => "v_and_b32",
            IntBinOp::Or => "v_or_b32",
            IntBinOp::Xor => "v_xor_b32",
            _ => return Err(error("unsupported AMDGPU integer operation")),
        };
        if dst.ty.size_bytes() == 4 {
            writeln!(self.body, "    {mnemonic} v{d}, v{l}, v{r}")?;
        } else {
            match op {
                IntBinOp::Add | IntBinOp::Sub => {
                    let (low, high) = match (op, self.target) {
                        (IntBinOp::Add, AmdGpuTarget::Gfx900) => ("v_add_co_u32", "v_addc_co_u32"),
                        (IntBinOp::Sub, AmdGpuTarget::Gfx900) => ("v_sub_co_u32", "v_subb_co_u32"),
                        (IntBinOp::Add, AmdGpuTarget::Gfx1200) => {
                            ("v_add_co_u32", "v_add_co_ci_u32")
                        }
                        _ => ("v_sub_co_u32", "v_sub_co_ci_u32"),
                    };
                    let carry = if self.target == AmdGpuTarget::Gfx900 {
                        "s[16:17]"
                    } else {
                        "s16"
                    };
                    self.sgprs = self.sgprs.max(if self.target == AmdGpuTarget::Gfx900 {
                        18
                    } else {
                        17
                    });
                    writeln!(
                        self.body,
                        "    {low} v{d}, {carry}, v{l}, v{r}\n    {high} v{}, {carry}, v{}, v{}, {carry}",
                        d + 1,
                        l + 1,
                        r + 1
                    )?;
                }
                IntBinOp::Mul => {
                    let t = self.temporary(MirType::i32())?.start;
                    writeln!(
                        self.body,
                        "    v_mul_hi_u32 v{}, v{l}, v{r}\n    v_mul_lo_u32 v{t}, v{l}, v{}\n    v_add_u32 v{}, v{}, v{t}\n    v_mul_lo_u32 v{t}, v{}, v{r}\n    v_add_u32 v{}, v{}, v{t}\n    v_mul_lo_u32 v{d}, v{l}, v{r}",
                        d + 1,
                        r + 1,
                        d + 1,
                        d + 1,
                        l + 1,
                        d + 1,
                        d + 1
                    )?;
                }
                _ => {
                    for word in 0..2 {
                        writeln!(
                            self.body,
                            "    {mnemonic} v{}, v{}, v{}",
                            d + word,
                            l + word,
                            r + word
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn memory_address(
        &self,
        address: &AddressMode,
        attrs: &MemoryAttrs,
        ty: MirType,
    ) -> Result<Value, LaminaError> {
        if !matches!(ty, MirType::Scalar(ScalarType::I32 | ScalarType::I64))
            || attrs.volatile
            || usize::from(attrs.align) < ty.size_bytes()
            || !attrs.align.is_power_of_two()
        {
            return Err(error(
                "AMDGPU memory requires aligned, non-volatile i32 or i64 accesses",
            ));
        }
        let AddressMode::BaseOffset {
            base,
            offset: 0,
            address_space: AddressSpace::Generic | AddressSpace::Global,
        } = address
        else {
            return Err(error(
                "AMDGPU memory requires a global pointer with zero instruction offset",
            ));
        };
        self.values
            .get(base)
            .copied()
            .filter(|value| {
                matches!(
                    value.ty,
                    MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(AddressSpace::Global))
                )
            })
            .ok_or_else(|| error("AMDGPU memory base must be a global pointer"))
    }

    fn wait_memory(&mut self, store: bool) -> Result<(), LaminaError> {
        let instruction = match (self.target, store) {
            (AmdGpuTarget::Gfx900, _) => "s_waitcnt vmcnt(0)",
            (AmdGpuTarget::Gfx1200, false) => "s_wait_loadcnt 0",
            (AmdGpuTarget::Gfx1200, true) => "s_wait_storecnt 0",
        };
        writeln!(self.body, "    {instruction}")?;
        Ok(())
    }

    fn instruction(&mut self, instruction: &Instruction) -> Result<(), LaminaError> {
        match instruction {
            Instruction::Lea { dst, base, offset } => {
                let base = self
                    .values
                    .get(base)
                    .copied()
                    .filter(|value| {
                        matches!(
                            value.ty,
                            MirType::Scalar(
                                ScalarType::Ptr | ScalarType::PtrIn(AddressSpace::Global)
                            )
                        )
                    })
                    .ok_or_else(|| error("AMDGPU address calculation requires a global pointer"))?;
                let dst = self.allocate(dst, MirType::ptr())?;
                if *offset == 0 {
                    self.copy(dst, base)?;
                } else {
                    let offset = self.operand(
                        &Operand::Immediate(Immediate::I64(i64::from(*offset))),
                        MirType::i64(),
                    )?;
                    self.binary(IntBinOp::Add, dst, base, offset)?;
                }
            }
            Instruction::Load {
                ty,
                dst,
                addr,
                attrs,
            } => {
                let address = self.memory_address(addr, attrs, *ty)?;
                let dst = self.allocate(dst, *ty)?;
                let mnemonic = if ty.size_bytes() == 8 {
                    "global_load_dwordx2"
                } else {
                    "global_load_dword"
                };
                writeln!(
                    self.body,
                    "    {mnemonic} {}, {}, off",
                    dst.assembly(),
                    address.assembly()
                )?;
                self.wait_memory(false)?;
            }
            Instruction::Store {
                ty,
                src,
                addr,
                attrs,
            } => {
                let address = self.memory_address(addr, attrs, *ty)?;
                let src = self.operand(src, *ty)?;
                let mnemonic = if ty.size_bytes() == 8 {
                    "global_store_dwordx2"
                } else {
                    "global_store_dword"
                };
                writeln!(
                    self.body,
                    "    {mnemonic} {}, {}, off",
                    address.assembly(),
                    src.assembly()
                )?;
                self.wait_memory(true)?;
            }
            Instruction::Bitcast { from, to, dst, src } => {
                let source = self.operand(src, *from)?;
                let destination = self.allocate(dst, *to)?;
                self.copy(destination, source)?;
            }
            Instruction::Copy { ty, dst, src } => {
                let src = self.operand(src, *ty)?;
                let dst = self.allocate(dst, *ty)?;
                self.copy(dst, src)?;
            }
            Instruction::IntConvert {
                from,
                to,
                signed,
                dst,
                src,
            } => {
                let source = self.operand(src, *from)?;
                let destination = self.allocate(dst, *to)?;
                writeln!(
                    self.body,
                    "    v_mov_b32 v{}, v{}",
                    destination.start, source.start
                )?;
                if to.size_bytes() == 8 {
                    if from.size_bytes() == 8 {
                        writeln!(
                            self.body,
                            "    v_mov_b32 v{}, v{}",
                            destination.start + 1,
                            source.start + 1
                        )?;
                    } else if *signed {
                        writeln!(
                            self.body,
                            "    v_ashrrev_i32 v{}, 31, v{}",
                            destination.start + 1,
                            source.start
                        )?;
                    } else {
                        writeln!(self.body, "    v_mov_b32 v{}, 0", destination.start + 1)?;
                    }
                }
            }
            Instruction::IntBinary {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                if !matches!(ty, MirType::Scalar(ScalarType::I32 | ScalarType::I64)) {
                    return Err(error("AMDGPU arithmetic requires i32 or i64"));
                }
                let lhs = self.operand(lhs, *ty)?;
                let dst = self.allocate(dst, *ty)?;
                if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) {
                    let shift = match (rhs, *ty) {
                        (
                            Operand::Immediate(Immediate::I32(value)),
                            MirType::Scalar(ScalarType::I32),
                        ) => i64::from(*value),
                        (
                            Operand::Immediate(Immediate::I64(value)),
                            MirType::Scalar(ScalarType::I64),
                        ) => *value,
                        _ => return Err(error("AMDGPU shifts require a matching immediate count")),
                    };
                    let shift = shift & (ty.size_bytes() as i64 * 8 - 1);
                    let mnemonic = match (op, ty.size_bytes()) {
                        (IntBinOp::Shl, 4) => "v_lshlrev_b32",
                        (IntBinOp::LShr, 4) => "v_lshrrev_b32",
                        (IntBinOp::AShr, 4) => "v_ashrrev_i32",
                        (IntBinOp::Shl, _) => "v_lshlrev_b64",
                        (IntBinOp::LShr, _) => "v_lshrrev_b64",
                        _ => "v_ashrrev_i64",
                    };
                    writeln!(
                        self.body,
                        "    {mnemonic} {}, {shift}, {}",
                        dst.assembly(),
                        lhs.assembly()
                    )?;
                } else {
                    let rhs = self.operand(rhs, *ty)?;
                    self.binary(*op, dst, lhs, rhs)?;
                }
            }
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
                        if self.target == AmdGpuTarget::Gfx900 {
                            let base = u32::from(self.dispatch_ptr) * 2
                                + if self.arguments.is_empty() { 0 } else { 2 };
                            self.sgprs = self.sgprs.max(base + 3);
                            writeln!(
                                self.body,
                                "    v_mov_b32 v{}, s{}",
                                value.start,
                                base + axis_index(axis)
                            )?;
                        } else {
                            match axis {
                                GpuAxis::X => {
                                    writeln!(self.body, "    v_mov_b32 v{}, ttmp9", value.start)?
                                }
                                GpuAxis::Y | GpuAxis::Z => {
                                    self.sgprs = self.sgprs.max(6);
                                    if axis == GpuAxis::Y {
                                        writeln!(self.body, "    s_and_b32 s5, ttmp7, 0xffff")?;
                                    } else {
                                        writeln!(self.body, "    s_lshr_b32 s5, ttmp7, 16")?;
                                    }
                                    writeln!(self.body, "    v_mov_b32 v{}, s5", value.start)?;
                                }
                            }
                        }
                    }
                    GpuIntrinsic::BlockDim(axis) | GpuIntrinsic::GridDim(axis) => {
                        self.sgprs = self.sgprs.max(13);
                        let axis = axis_index(axis);
                        let size_offset = if axis == 2 { 8 } else { 4 };
                        writeln!(self.body, "    s_load_dword s9, s[0:1], {size_offset}")?;
                        let grid = matches!(intrinsic, GpuIntrinsic::GridDim(_));
                        if grid {
                            writeln!(self.body, "    s_load_dword s8, s[0:1], {}", 12 + axis * 4)?;
                        }
                        let wait = if self.target == AmdGpuTarget::Gfx1200 {
                            "s_wait_kmcnt 0"
                        } else {
                            "s_waitcnt lgkmcnt(0)"
                        };
                        writeln!(
                            self.body,
                            "    {wait}\n    s_bfe_u32 s9, s9, {}",
                            (16 << 16) | if axis == 1 { 16 } else { 0 }
                        )?;
                        if grid {
                            writeln!(self.body, "    s_mov_b32 s10, 0\n    s_mov_b32 s11, 0")?;
                            for bit in (0..32).rev() {
                                writeln!(
                                    self.body,
                                    "    s_lshl_b32 s11, s11, 1\n    s_bfe_u32 s12, s8, {}\n    s_or_b32 s11, s11, s12\n    s_sub_u32 s12, s11, s9\n    s_cmp_ge_u32 s11, s9\n    s_cselect_b32 s11, s12, s11\n    s_cselect_b32 s12, 1, 0\n    s_lshl_b32 s10, s10, 1\n    s_or_b32 s10, s10, s12",
                                    (1 << 16) | bit
                                )?;
                            }
                            writeln!(
                                self.body,
                                "    s_cmp_lg_u32 s11, 0\n    s_cselect_b32 s12, 1, 0\n    s_add_u32 s9, s10, s12"
                            )?;
                        }
                        writeln!(self.body, "    v_mov_b32 v{}, s9", value.start)?;
                    }
                    _ => {
                        return Err(error("unsupported AMDGPU intrinsic"));
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
        writeln!(
            output,
            "    .amdhsa_user_sgpr_dispatch_ptr {}",
            u8::from(kernel.dispatch_ptr)
        )?;
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

#[cfg(test)]
mod emission_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{
        env,
        error::Error,
        fs,
        process::{self, Command},
    };

    #[test]
    fn gpu_cli_emits_text_without_native_tools() -> Result<(), Box<dyn Error>> {
        let directory = env::temp_dir().join(format!("lamina-amdgpu-cli-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("kernel.lamina");
        fs::write(
            &source,
            "@kernel fn @empty() -> void {\nentry:\ncall @__lamina_gpu_workgroup_barrier()\nret.void\n}\n",
        )?;
        for (target, extension, expected) in [("amdgcn_amdhsa", "s", ".amdhsa_kernel empty")] {
            for mode in [None, Some("--emit-asm"), Some("--emit-mir-asm")] {
                let output_path = directory.join(target);
                let mut flags = vec!["--target", target];
                if let Some(mode) = mode {
                    flags.push(mode);
                }
                compile(&source, &output_path, &flags)?;
                let artifact = output_path.with_extension(extension);
                assert!(fs::read_to_string(&artifact)?.contains(expected));
                fs::remove_file(artifact)?;
            }
        }
        fs::remove_file(source)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn amdgpu_kernels_assemble_for_both_processors() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["clang"]) {
            return Ok(());
        }
        let targets = Command::new("clang").arg("--print-targets").output()?;
        if !String::from_utf8_lossy(&targets.stdout).contains("amdgcn") {
            eprintln!("skipping AMDGPU assembly: clang lacks amdgcn target");
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-amdgpu-asm-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("barrier.lamina");
        fs::write(&source, include_str!("amdgpu.lamina"))?;
        for cpu in ["gfx1200", "gfx900"] {
            for level in 0..=3 {
                let level = level.to_string();
                let path = directory.join(format!("{cpu}-o{level}"));
                compile(
                    &source,
                    &path,
                    &[
                        "--emit-asm",
                        "--opt-level",
                        &level,
                        "--target",
                        "amdgcn_amdhsa",
                        "--mcpu",
                        cpu,
                    ],
                )?;
                let assembly = path.with_extension("s");
                let object = path.with_extension("o");
                let text = fs::read_to_string(&assembly)?;
                assert!(text.contains(".amdhsa_user_sgpr_kernarg_segment_ptr 1"));
                assert!(text.contains(".kernarg_segment_size: 24"));
                assert!(text.contains(".offset: 8"));
                assert!(text.contains(".offset: 16"));
                assert!(text.contains(".value_kind: global_buffer"));
                assert!(text.contains("global_load_dwordx2"));
                assert!(text.contains("global_store_dwordx2"));
                if cpu == "gfx1200" {
                    assert!(text.contains("ttmp9"));
                    assert!(text.contains("s_and_b32 s5, ttmp7, 0xffff"));
                    assert!(text.contains("s_lshr_b32 s5, ttmp7, 16"));
                    assert!(text.contains("s_wait_kmcnt 0"));
                }
                let dimensions = text
                    .split("dimensions:\n")
                    .nth(1)
                    .ok_or("missing dimensions kernel")?
                    .split(".Ldimensions_end:")
                    .next()
                    .ok_or("missing dimensions body")?;
                assert!(dimensions.contains("s_load_dwordx2 s[8:9], s[2:3], 0"));
                assert!(dimensions.contains("s_load_dwordx2 s[8:9], s[2:3], 8"));
                for offset in [4, 8] {
                    assert!(dimensions.contains(&format!("s_load_dword s9, s[0:1], {offset}")));
                }
                for offset in [12, 16, 20] {
                    assert!(dimensions.contains(&format!("s_load_dword s8, s[0:1], {offset}")));
                }
                assert!(text.contains(".amdhsa_user_sgpr_dispatch_ptr 1"));
                if cpu == "gfx900" {
                    for register in [4, 5, 6] {
                        assert!(dimensions.lines().any(|line| line.contains("v_mov_b32")
                            && line.ends_with(&format!(", s{register}"))));
                    }
                    assert!(dimensions.contains("v_add_co_u32") && dimensions.contains("s[16:17]"));
                }
                assert!(text.contains("v_mul_hi_u32"));
                assert!(text.contains("v_ashrrev_i32"));
                assert!(text.contains(if cpu == "gfx1200" {
                    "v_add_co_ci_u32"
                } else {
                    "v_addc_co_u32"
                }));
                let output = Command::new("clang")
                    .args(["--target=amdgcn-amd-amdhsa", &format!("-mcpu={cpu}"), "-c"])
                    .arg(&assembly)
                    .arg("-o")
                    .arg(&object)
                    .output()?;
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                fs::remove_file(assembly)?;
                fs::remove_file(object)?;
            }
        }
        fs::remove_file(source)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn amdgpu_rejects_unsupported_kernels_without_partial_output() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, mir_codegen::amdgpu, parser::parse_module};

        for source in [
            "@kernel fn @bad(i32 %a) -> void {\nentry:\n%x = div.i32 %a, 2\nret.void\n}",
            "@kernel fn @bad(i32 %a) -> void {\nentry:\n%x = shl.i32 %a, %a\nret.void\n}",
            "@kernel fn @bad(f32 %a) -> void {\nentry:\nret.void\n}",
            "@kernel fn @bad() -> void {\nentry:\n%p = alloc.stack i64\nret.void\n}",
            "@kernel fn @bad() -> void {\nentry:\n%x = call @__lamina_gpu_lane_id()\nret.void\n}",
        ] {
            let ir = parse_module(source)?;
            let mir = from_ir(&ir, "unsupported").map_err(|error| format!("{error:?}"))?;
            for target in [amdgpu::AmdGpuTarget::Gfx900, amdgpu::AmdGpuTarget::Gfx1200] {
                let mut output = Vec::new();
                assert!(amdgpu::emit_with_target(&mir, &mut output, target).is_err());
                assert!(output.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn gpu_backends_reject_local_unaligned_and_volatile_memory() {
        use crate::{
            mir::{
                AddressMode, AddressSpace, FunctionBuilder, Instruction, KernelMetadata,
                MemoryAttrs, MirType, Module, Register, VirtualReg,
            },
            mir_codegen::amdgpu,
        };
        for (space, align, volatile) in [
            (AddressSpace::Local, 4, false),
            (AddressSpace::Private, 4, false),
            (AddressSpace::Global, 1, false),
            (AddressSpace::Global, 4, true),
        ] {
            let base = Register::Virtual(VirtualReg::gpr(0));
            let mut function = FunctionBuilder::new("bad")
                .param(base.clone(), MirType::ptr_in(AddressSpace::Global))
                .block("entry")
                .instr(Instruction::Load {
                    ty: MirType::i32(),
                    dst: Register::Virtual(VirtualReg::gpr(1)),
                    addr: AddressMode::BaseOffset {
                        base,
                        offset: 0,
                        address_space: space,
                    },
                    attrs: MemoryAttrs { align, volatile },
                })
                .instr(Instruction::Ret { value: None })
                .build();
            function.sig.kernel = Some(KernelMetadata::default());
            let mut module = Module::new("unsupported");
            module.add_function(function);
            let mut output = Vec::new();
            assert!(amdgpu::emit(&module, &mut output).is_err());
            assert!(output.is_empty());
        }
    }
}
