use super::gpu::{error, kernels};
use crate::{
    error::LaminaError,
    mir::{
        Instruction, Module, Register, RegisterClass,
        gpu::{GpuAxis, GpuIntrinsic},
    },
};
use std::{collections::BTreeSet, io::Write};

pub fn emit<W: Write>(module: &Module, writer: &mut W) -> Result<(), LaminaError> {
    let mut output = Vec::new();
    writeln!(output, ".version 6.0\n.target sm_60\n.address_size 64")?;
    for (function, name, metadata) in kernels(module)? {
        if metadata.max_threads.is_some() {
            return Err(error(
                "PTX max_threads lowering is not implemented; use required_workgroup_size",
            ));
        }
        let mut registers = BTreeSet::new();
        let mut body = Vec::new();
        for instruction in &function.blocks[0].instructions {
            match instruction {
                Instruction::Ret { value: None } => writeln!(body, "    ret;")?,
                Instruction::Call { name, args, ret } if args.is_empty() => {
                    let intrinsic = GpuIntrinsic::from_name(name)
                        .ok_or_else(|| error("unsupported call (including Print and host I/O)"))?;
                    if intrinsic == GpuIntrinsic::WorkgroupBarrier {
                        if ret.is_some() {
                            return Err(error("barrier cannot return a value"));
                        }
                        writeln!(body, "    bar.sync 0;")?;
                    } else {
                        let Some(Register::Virtual(reg)) = ret else {
                            return Err(error("GPU ID calls require a virtual i32 result"));
                        };
                        if reg.class != RegisterClass::Gpr {
                            return Err(error("GPU ID result must be a GPR"));
                        }
                        registers.insert(reg.id);
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
                        writeln!(body, "    mov.u32 %r{}, %{special}{suffix};", reg.id)?;
                    }
                }
                _ => return Err(error("unsupported PTX instruction")),
            }
        }
        writeln!(output, "\n.visible .entry {name}()")?;
        if let Some([x, y, z]) = metadata.required_workgroup_size {
            writeln!(output, ".reqntid {x}, {y}, {z}")?;
        }
        writeln!(output, "{{")?;
        if registers.is_empty() {
            writeln!(output, "    .reg .b32 %r_unused;")?;
        }
        for register in registers {
            writeln!(output, "    .reg .b32 %r{register};")?;
        }
        output.extend(body);
        writeln!(output, "}}")?;
    }
    writer.write_all(&output)?;
    Ok(())
}
