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

#[cfg(test)]
mod emission_tests {
    use crate::mir_codegen::test_support::compile;
    use std::{env, error::Error, fs, process};

    #[test]
    fn gpu_cli_emits_text_without_native_tools() -> Result<(), Box<dyn Error>> {
        let directory = env::temp_dir().join(format!("lamina-ptx-cli-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("kernel.lamina");
        fs::write(
            &source,
            "@kernel fn @empty() -> void {\nentry:\ncall @__lamina_gpu_workgroup_barrier()\nret.void\n}\n",
        )?;
        for (target, extension, expected) in [("nvptx64_cuda", "ptx", ".visible .entry empty()")] {
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
    fn ptx_scaffolding_declares_typed_registers_for_supported_kernels() -> Result<(), Box<dyn Error>>
    {
        use crate::{
            mir::{TransformPipeline, codegen::from_ir},
            mir_codegen::ptx,
            parser::parse_module,
        };
        use std::collections::HashMap;

        let ir = parse_module(include_str!("../amdgpu/amdgpu.lamina"))?;
        for level in 0..=3 {
            let mut mir = from_ir(&ir, "ptx").map_err(|error| format!("{error:?}"))?;
            TransformPipeline::default_for_opt_level(level)
                .apply_to_module(&mut mir)
                .map_err(|error| format!("{error:?}"))?;
            let mut output = Vec::new();
            ptx::emit(&mir, &mut output)?;
            let text = String::from_utf8(output)?;
            assert!(text.starts_with(".version 6.0\n.target sm_60\n.address_size 64"));
            for expected in [
                ".param .u32 arg0",
                ".param .u64 arg1",
                ".param .u64 .ptr .global .align 1 arg0",
                "ld.param.u32",
                "ld.param.u64",
                "bar.sync 0;",
                "cvt.u64.u32",
                "cvt.s64.s32",
                "cvt.u32.u64",
            ] {
                assert!(text.contains(expected), "missing {expected}");
            }
            for bits in [32, 64] {
                for opcode in [
                    "add.u",
                    "sub.u",
                    "mul.lo.u",
                    "and.b",
                    "or.b",
                    "xor.b",
                    "shl.b",
                    "shr.u",
                    "shr.s",
                    "ld.global.u",
                    "st.global.u",
                ] {
                    assert!(
                        text.contains(&format!("{opcode}{bits}")),
                        "missing {opcode}{bits}"
                    );
                }
            }
            for special in ["tid", "ctaid", "ntid", "nctaid"] {
                for axis in ["x", "y", "z"] {
                    assert!(text.contains(&format!("%{special}.{axis};")));
                }
            }
            for kernel in text.split(".visible .entry").skip(1) {
                let mut registers = HashMap::new();
                for line in kernel
                    .lines()
                    .map(str::trim)
                    .filter(|line| line.starts_with(".reg "))
                {
                    let fields: Vec<_> = line.trim_end_matches(';').split_whitespace().collect();
                    assert!(registers.insert(fields[2], fields[1]).is_none());
                }
                for line in kernel
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.starts_with(".reg "))
                {
                    for register in line
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '%' && c != '_')
                        .filter(|token| token.starts_with("%r"))
                    {
                        assert!(
                            registers.contains_key(register),
                            "undeclared {register}: {line}"
                        );
                    }
                    let fields: Vec<_> = line.split_whitespace().collect();
                    if fields.first().is_some_and(|op| {
                        op.starts_with("ld.") || op.starts_with("cvt.") || op.starts_with("mov.")
                    }) {
                        let destination = fields[1].trim_end_matches(',');
                        let width = if fields[0].starts_with("cvt.") {
                            fields[0]
                                .split('.')
                                .nth(1)
                                .ok_or("missing destination type")?
                        } else {
                            fields[0].rsplit('.').next().ok_or("missing load type")?
                        };
                        assert_eq!(registers[destination], format!(".b{}", &width[1..]));
                    }
                }
            }
            assert!(!text.contains(".local"));
            assert!(!text.contains("%rsp"));
        }
        Ok(())
    }

    #[test]
    fn ptx_scaffolding_rejects_unsupported_operations_without_partial_output()
    -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, mir_codegen::ptx, parser::parse_module};
        for source in [
            "@kernel fn @bad(i32 %a) -> void {\nentry:\n%x = div.i32 %a, 2\nret.void\n}",
            "@kernel fn @bad(i64 %a) -> void {\nentry:\n%x = shl.i64 %a, %a\nret.void\n}",
            "@kernel fn @bad(i64 %a) -> void {\nentry:\n%x = shl.i64 %a, 64\nret.void\n}",
            "@kernel fn @bad(f32 %a) -> void {\nentry:\nret.void\n}",
            "@kernel fn @bad() -> void {\nentry:\n%p = alloc.stack i64\nret.void\n}",
        ] {
            let ir = parse_module(source)?;
            let mir = from_ir(&ir, "unsupported").map_err(|error| format!("{error:?}"))?;
            let mut output = Vec::new();
            assert!(ptx::emit(&mir, &mut output).is_err());
            assert!(output.is_empty());
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
            mir_codegen::ptx,
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
            assert!(ptx::emit(&module, &mut output).is_err());
            assert!(output.is_empty());
        }
    }
}
