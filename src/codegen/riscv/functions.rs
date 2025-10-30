use super::instructions::generate_instruction;
use super::state::{ARG_REGISTERS, CodegenState, FunctionContext, ValueLocation};
use crate::{
    BasicBlock, Function, FunctionAnnotation, Identifier, Instruction, LaminaError, PrimitiveType,
    Type,
};
use std::io::Write;
use std::result::Result;

pub fn generate_functions<'a, W: Write>(
    module: &'a crate::Module<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<(), LaminaError> {
    for (func_name, func) in &module.functions {
        generate_function(func_name, func, writer, state)?;
    }
    Ok(())
}

pub fn generate_function<'a, W: Write>(
    func_name: Identifier<'a>,
    func: &'a Function<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<(), LaminaError> {
    let is_exported = func.annotations.contains(&FunctionAnnotation::Export);
    let is_main = func_name == "main";
    let asm_label = if is_main {
        "main".to_string()
    } else {
        format!("func_{}", func_name)
    };

    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name);

    // Minimal prologue: create frame pointer and save return address
    writeln!(writer, "\n# Function: @{}", func_name)?;
    if is_exported || is_main {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, "{}:", asm_label)?;

    // Prologue: save ra, s0; set up frame pointer
    match state.width() {
        super::IsaWidth::Rv32 => {
            writeln!(writer, "    addi sp, sp, -8")?;
            writeln!(writer, "    sw ra, 4(sp)")?;
            writeln!(writer, "    sw s0, 0(sp)")?;
            writeln!(writer, "    addi s0, sp, 8")?;
        }
        _ => {
            writeln!(writer, "    addi sp, sp, -16")?;
            writeln!(writer, "    sd ra, 8(sp)")?;
            writeln!(writer, "    sd s0, 0(sp)")?;
            writeln!(writer, "    addi s0, sp, 16")?; // s0 points above saved area
        }
    }

    precompute_function_layout(func, &mut func_ctx, state)?;

    // Allocate stack frame for locals/spills computed in layout
    let frame_size = func_ctx.total_stack_size as i64;
    if frame_size > 0 {
        writeln!(writer, "    addi sp, sp, -{}", frame_size)?;
    }

    // Spill arguments to assigned stack slots if necessary
    for (i, param) in func.signature.params.iter().enumerate() {
        if let Some(ValueLocation::StackOffset(off)) = func_ctx.value_locations.get(param.name) {
            if i < ARG_REGISTERS.len() {
                writeln!(writer, "    addi t0, s0, {}", off)?;
                match state.width() {
                    super::IsaWidth::Rv32 => {
                        writeln!(writer, "    sw {}, 0(t0)", ARG_REGISTERS[i])?
                    }
                    _ => writeln!(writer, "    sd {}, 0(t0)", ARG_REGISTERS[i])?,
                }
            } else {
                // For simplicity, skip stack-passed args for now
            }
        }
    }

    if let Some(entry) = func.basic_blocks.get(&func.entry_block) {
        let label = func_ctx.get_block_label(&func.entry_block)?;
        writeln!(writer, "{}:", label)?;
        generate_basic_block(entry, writer, state, &mut func_ctx, func_name)?;
    }
    for (ir_label, block) in &func.basic_blocks {
        if *ir_label != func.entry_block {
            let label = func_ctx.get_block_label(ir_label)?;
            writeln!(writer, "{}:", label)?;
            generate_basic_block(block, writer, state, &mut func_ctx, func_name)?;
        }
    }

    // Epilogue
    writeln!(writer, "{}:", func_ctx.epilogue_label)?;
    // Restore local frame
    if frame_size > 0 {
        writeln!(writer, "    addi sp, sp, {}", frame_size)?;
    }
    match state.width() {
        super::IsaWidth::Rv32 => {
            writeln!(writer, "    lw ra, 4(sp)")?;
            writeln!(writer, "    lw s0, 0(sp)")?;
            writeln!(writer, "    addi sp, sp, 8")?;
        }
        _ => {
            writeln!(writer, "    ld ra, 8(sp)")?;
            writeln!(writer, "    ld s0, 0(sp)")?;
            writeln!(writer, "    addi sp, sp, 16")?;
        }
    }
    writeln!(writer, "    ret")?;
    Ok(())
}

fn precompute_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
) -> Result<(), LaminaError> {
    // Assign simple block labels
    for ir_label in func.basic_blocks.keys() {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    // Place params and locals at negative offsets from s0
    let saved_area: i64 = match state.width() {
        super::IsaWidth::Rv32 => 8,
        _ => 16,
    };
    let word: i64 = match state.width() {
        super::IsaWidth::Rv32 => 4,
        super::IsaWidth::Rv64 => 8,
        super::IsaWidth::Rv128 => 16,
    };
    let mut current: i64 = 0; // locals start at -word, grow downwards

    // Param spill slots
    for param in &func.signature.params {
        current -= word;
        func_ctx
            .value_locations
            .insert(param.name, ValueLocation::StackOffset(current));
    }

    // Collect SSA results and allocate stack slots
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for block in func.basic_blocks.values() {
        for instr in &block.instructions {
            let (maybe_res, size_bytes) = match instr {
                Instruction::Alloc {
                    result,
                    allocated_ty,
                    ..
                } => {
                    // Allocate space for value or pointer
                    let sz = match allocated_ty {
                        Type::Primitive(pt) => prim_size(*pt, word as u64)?,
                        Type::Array { .. } | Type::Struct(_) => word as u64, // store pointer
                        _ => word as u64,
                    };
                    (Some(*result), sz)
                }
                Instruction::Binary { result, ty, .. } | Instruction::Cmp { result, ty, .. } => {
                    (Some(*result), prim_size(*ty, word as u64)?)
                }
                Instruction::Load { result, ty, .. } => {
                    let (_, sz) = super::util::get_type_size_directive_and_bytes(ty)?;
                    (Some(*result), align_to(sz, word as u64))
                }
                Instruction::GetFieldPtr { result, .. }
                | Instruction::GetElemPtr { result, .. }
                | Instruction::Tuple { result, .. }
                | Instruction::ExtractTuple { result, .. }
                | Instruction::IntToPtr { result, .. }
                | Instruction::PtrToInt { result, .. } => (Some(*result), word as u64),
                Instruction::Phi { result, ty, .. } => {
                    let (_, sz) = super::util::get_type_size_directive_and_bytes(ty)?;
                    (Some(*result), align_to(sz, word as u64))
                }
                Instruction::Call {
                    result: Some(res), ..
                } => (Some(*res), word as u64),
                Instruction::ZeroExtend {
                    result,
                    target_type,
                    ..
                } => (Some(*result), prim_size(*target_type, word as u64)?),
                Instruction::Write { result, .. }
                | Instruction::Read { result, .. }
                | Instruction::WriteByte { result, .. }
                | Instruction::ReadByte { result, .. }
                | Instruction::WritePtr { result, .. } => (Some(*result), word as u64),
                _ => (None, 0),
            };
            if let Some(res) = maybe_res
                && !seen.contains(res)
            {
                seen.insert(res);
                current -= align_to(size_bytes, word as u64) as i64;
                func_ctx
                    .value_locations
                    .insert(res, ValueLocation::StackOffset(current));
            }
        }
    }

    // Total frame size (locals + param spills), aligned to 16, excluding saved_area handled in prologue
    let total = (-current) as u64;
    func_ctx.total_stack_size = align_to(total, 16);
    Ok(())
}

fn generate_basic_block<'a, W: Write>(
    block: &BasicBlock<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    func_name: Identifier<'a>,
) -> Result<(), LaminaError> {
    for instr in &block.instructions {
        generate_instruction(instr, writer, state, func_ctx, func_name)?;
    }
    Ok(())
}

fn prim_size(pt: PrimitiveType, word: u64) -> Result<u64, LaminaError> {
    Ok(match pt {
        PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => 1,
        PrimitiveType::I16 | PrimitiveType::U16 => 2,
        PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
        PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr | PrimitiveType::F64 => word,
    })
}

fn align_to(size: u64, align: u64) -> u64 {
    (size + align - 1) & !(align - 1)
}
