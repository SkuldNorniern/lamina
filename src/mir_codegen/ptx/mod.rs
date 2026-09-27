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

fn width(ty: MirType) -> Result<usize, LaminaError> {
    match ty {
        MirType::Scalar(ScalarType::I32) => Ok(32),
        MirType::Scalar(
            ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(AddressSpace::Global),
        ) => Ok(64),
        _ => Err(error("PTX values require i32, i64, or global pointers")),
    }
}

fn is_pointer(ty: MirType) -> bool {
    matches!(
        ty,
        MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(AddressSpace::Global))
    )
}

struct Kernel {
    body: Vec<u8>,
    declarations: Vec<(String, MirType)>,
    values: HashMap<Register, (String, MirType)>,
}

impl Kernel {
    fn new(function: &Function) -> Result<Self, LaminaError> {
        let mut kernel = Self {
            body: Vec::new(),
            declarations: Vec::new(),
            values: HashMap::new(),
        };
        for (index, parameter) in function.sig.params.iter().enumerate() {
            let dst = kernel.allocate(&parameter.reg, parameter.ty)?;
            writeln!(
                kernel.body,
                "    ld.param.u{} {dst}, [arg{index}];",
                width(parameter.ty)?
            )?;
        }
        Ok(kernel)
    }

    fn allocate(&mut self, register: &Register, ty: MirType) -> Result<String, LaminaError> {
        width(ty)?;
        let Register::Virtual(reg) = register else {
            return Err(error("PTX values require virtual GPRs"));
        };
        if reg.class != RegisterClass::Gpr || self.values.contains_key(register) {
            return Err(error("PTX values require unique virtual GPR definitions"));
        }
        let name = format!("%r{}", reg.id);
        self.values.insert(register.clone(), (name.clone(), ty));
        self.declarations.push((name.clone(), ty));
        Ok(name)
    }

    fn value(&self, register: &Register) -> Result<(String, MirType), LaminaError> {
        self.values
            .get(register)
            .cloned()
            .ok_or_else(|| error("undefined PTX register"))
    }

    fn operand(&self, operand: &Operand, ty: MirType) -> Result<String, LaminaError> {
        match operand {
            Operand::Register(register) => {
                let (name, actual) = self.value(register)?;
                if actual != ty {
                    return Err(error("PTX operand type mismatch"));
                }
                Ok(name)
            }
            Operand::Immediate(immediate) => match (immediate, ty) {
                (Immediate::I32(value), MirType::Scalar(ScalarType::I32)) => {
                    Ok(format!("0x{:08x}", *value as u32))
                }
                (Immediate::I64(value), MirType::Scalar(ScalarType::I64)) => {
                    Ok(format!("0x{:016x}", *value as u64))
                }
                _ => Err(error("unsupported PTX immediate")),
            },
        }
    }

    fn address(&self, address: &AddressMode) -> Result<String, LaminaError> {
        let AddressMode::BaseOffset {
            base,
            offset: 0,
            address_space: AddressSpace::Generic | AddressSpace::Global,
        } = address
        else {
            return Err(error(
                "PTX memory requires global base pointers with zero offset",
            ));
        };
        let (name, ty) = self.value(base)?;
        if !is_pointer(ty) {
            return Err(error("PTX memory base must be a global pointer"));
        }
        Ok(name)
    }

    fn instruction(&mut self, instruction: &Instruction) -> Result<(), LaminaError> {
        match instruction {
            Instruction::Ret { value: None } => writeln!(self.body, "    ret;")?,
            Instruction::Copy { ty, dst, src } => {
                let source = self.operand(src, *ty)?;
                let dst = self.allocate(dst, *ty)?;
                writeln!(self.body, "    mov.b{} {dst}, {source};", width(*ty)?)?;
            }
            Instruction::Lea { dst, base, offset } => {
                let (base, ty) = self.value(base)?;
                if !is_pointer(ty) {
                    return Err(error("PTX address calculation requires a global pointer"));
                }
                let dst = self.allocate(dst, MirType::ptr())?;
                writeln!(
                    self.body,
                    "    add.u64 {dst}, {base}, 0x{:016x};",
                    i64::from(*offset) as u64
                )?;
            }
            Instruction::IntConvert {
                from,
                to,
                signed,
                dst,
                src,
            } => {
                let source = self.operand(src, *from)?;
                let from_width = width(*from)?;
                let to_width = width(*to)?;
                let dst = self.allocate(dst, *to)?;
                if from_width == to_width {
                    writeln!(self.body, "    mov.b{to_width} {dst}, {source};")?;
                } else {
                    let sign = if *signed { "s" } else { "u" };
                    writeln!(
                        self.body,
                        "    cvt.{sign}{to_width}.{sign}{from_width} {dst}, {source};"
                    )?;
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
                    return Err(error("PTX arithmetic requires i32 or i64"));
                }
                let bits = width(*ty)?;
                let lhs = self.operand(lhs, *ty)?;
                let (mnemonic, rhs) =
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
                            _ => {
                                return Err(error("PTX shifts require a matching immediate count"));
                            }
                        };
                        if !(0..bits as i64).contains(&shift) {
                            return Err(error("PTX shift count must be below the value width"));
                        }
                        let mnemonic = match op {
                            IntBinOp::Shl => format!("shl.b{bits}"),
                            IntBinOp::LShr => format!("shr.u{bits}"),
                            _ => format!("shr.s{bits}"),
                        };
                        (mnemonic, shift.to_string())
                    } else {
                        let mnemonic = match op {
                            IntBinOp::Add => format!("add.u{bits}"),
                            IntBinOp::Sub => format!("sub.u{bits}"),
                            IntBinOp::Mul => format!("mul.lo.u{bits}"),
                            IntBinOp::And => format!("and.b{bits}"),
                            IntBinOp::Or => format!("or.b{bits}"),
                            IntBinOp::Xor => format!("xor.b{bits}"),
                            _ => return Err(error("unsupported PTX integer operation")),
                        };
                        (mnemonic, self.operand(rhs, *ty)?)
                    };
                let dst = self.allocate(dst, *ty)?;
                writeln!(self.body, "    {mnemonic} {dst}, {lhs}, {rhs};")?;
            }
            Instruction::Load {
                ty,
                dst,
                addr,
                attrs,
            } => {
                Self::memory_type(*ty, attrs)?;
                let address = self.address(addr)?;
                let dst = self.allocate(dst, *ty)?;
                writeln!(
                    self.body,
                    "    ld.global.u{} {dst}, [{address}];",
                    width(*ty)?
                )?;
            }
            Instruction::Store {
                ty,
                src,
                addr,
                attrs,
            } => {
                Self::memory_type(*ty, attrs)?;
                let address = self.address(addr)?;
                let source = self.operand(src, *ty)?;
                writeln!(
                    self.body,
                    "    st.global.u{} [{address}], {source};",
                    width(*ty)?
                )?;
            }
            Instruction::Call { name, args, ret } if args.is_empty() => {
                let intrinsic =
                    GpuIntrinsic::from_name(name).ok_or_else(|| error("unsupported PTX call"))?;
                if intrinsic == GpuIntrinsic::WorkgroupBarrier {
                    if ret.is_some() {
                        return Err(error("barrier cannot return a value"));
                    }
                    writeln!(self.body, "    bar.sync 0;")?;
                } else {
                    let dst = ret
                        .as_ref()
                        .ok_or_else(|| error("GPU ID calls require a result"))?;
                    let dst = self.allocate(dst, MirType::i32())?;
                    let (special, axis) = match intrinsic {
                        GpuIntrinsic::ThreadId(axis) => ("tid", Some(axis)),
                        GpuIntrinsic::BlockId(axis) => ("ctaid", Some(axis)),
                        GpuIntrinsic::BlockDim(axis) => ("ntid", Some(axis)),
                        GpuIntrinsic::GridDim(axis) => ("nctaid", Some(axis)),
                        GpuIntrinsic::LaneId => ("laneid", None),
                        GpuIntrinsic::WorkgroupBarrier => unreachable!(),
                    };
                    let suffix = match axis {
                        Some(GpuAxis::X) => ".x",
                        Some(GpuAxis::Y) => ".y",
                        Some(GpuAxis::Z) => ".z",
                        None => "",
                    };
                    writeln!(self.body, "    mov.u32 {dst}, %{special}{suffix};")?;
                }
            }
            _ => return Err(error("unsupported PTX instruction")),
        }
        Ok(())
    }

    fn memory_type(ty: MirType, attrs: &MemoryAttrs) -> Result<(), LaminaError> {
        if !matches!(ty, MirType::Scalar(ScalarType::I32 | ScalarType::I64))
            || attrs.volatile
            || usize::from(attrs.align) < ty.size_bytes()
            || !attrs.align.is_power_of_two()
        {
            return Err(error(
                "PTX memory requires aligned, non-volatile i32 or i64 accesses",
            ));
        }
        Ok(())
    }
}

pub fn emit<W: Write>(module: &Module, writer: &mut W) -> Result<(), LaminaError> {
    let mut output = Vec::new();
    writeln!(output, ".version 6.0\n.target sm_60\n.address_size 64")?;
    for (function, name, metadata) in kernels(module, true)? {
        if metadata.max_threads.is_some() {
            return Err(error(
                "PTX max_threads lowering is not implemented; use required_workgroup_size",
            ));
        }
        let mut kernel = Kernel::new(function)?;
        for instruction in &function.blocks[0].instructions {
            kernel.instruction(instruction)?;
        }
        let mut parameters = Vec::new();
        for (index, parameter) in function.sig.params.iter().enumerate() {
            let pointer = if is_pointer(parameter.ty) {
                " .ptr .global .align 1"
            } else {
                ""
            };
            parameters.push(format!(
                ".param .u{}{pointer} arg{index}",
                width(parameter.ty)?
            ));
        }
        writeln!(
            output,
            "\n.visible .entry {name}({})",
            parameters.join(", ")
        )?;
        if let Some([x, y, z]) = metadata.required_workgroup_size {
            writeln!(output, ".reqntid {x}, {y}, {z}")?;
        }
        writeln!(output, "{{")?;
        if kernel.declarations.is_empty() {
            writeln!(output, "    .reg .b32 %r_unused;")?;
        }
        for (register, ty) in &kernel.declarations {
            writeln!(output, "    .reg .b{} {register};", width(*ty)?)?;
        }
        output.extend(kernel.body);
        writeln!(output, "}}")?;
    }
    writer.write_all(&output)?;
    Ok(())
}
