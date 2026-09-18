use super::gpu::{error, kernels};
use crate::{
    error::LaminaError,
    mir::{Instruction, Module, gpu::GpuIntrinsic},
};
use std::io::Write;

pub fn emit<W: Write>(module: &Module, writer: &mut W) -> Result<(), LaminaError> {
    let mut output = Vec::new();
    let kernels = kernels(module)?;
    writeln!(output, ".amdgcn_target \"amdgcn-amd-amdhsa--gfx900\"")?;
    for (function, name, _) in &kernels {
        writeln!(
            output,
            ".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:"
        )?;
        for instruction in &function.blocks[0].instructions {
            match instruction {
                Instruction::Ret { value: None } => writeln!(output, "    s_endpgm")?,
                Instruction::Call { name, args, ret }
                    if args.is_empty()
                        && ret.is_none()
                        && GpuIntrinsic::from_name(name)
                            == Some(GpuIntrinsic::WorkgroupBarrier) =>
                {
                    writeln!(output, "    s_waitcnt lgkmcnt(0)\n    s_barrier")?;
                }
                _ => {
                    return Err(error(
                        "AMDGPU skeleton supports only void return and workgroup barrier",
                    ));
                }
            }
        }
        writeln!(output, ".L{name}_end:\n.size {name}, .L{name}_end-{name}")?;
        writeln!(
            output,
            ".section .rodata\n.p2align 6\n.amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 0\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size 0\n    .amdhsa_user_sgpr_private_segment_buffer 0\n    .amdhsa_next_free_vgpr 1\n    .amdhsa_next_free_sgpr 0\n.end_amdhsa_kernel"
        )?;
    }
    writeln!(
        output,
        ".amdgpu_metadata\n---\namdhsa.version: [1, 0]\namdhsa.kernels:"
    )?;
    for (_, name, metadata) in &kernels {
        writeln!(
            output,
            "  - .name: {name}\n    .symbol: {name}.kd\n    .kernarg_segment_size: 0\n    .kernarg_segment_align: 4\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\n    .sgpr_count: 0\n    .vgpr_count: 1\n    .max_flat_workgroup_size: {}",
            metadata.max_threads.unwrap_or(1024)
        )?;
        if let Some([x, y, z]) = metadata.required_workgroup_size {
            writeln!(output, "    .reqd_workgroup_size: [{x}, {y}, {z}]")?;
        }
    }
    writeln!(output, "...\n.end_amdgpu_metadata")?;
    writer.write_all(&output)?;
    Ok(())
}
