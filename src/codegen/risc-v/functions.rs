use super::instructions::generate_instruction;
use super::state::{CodegenState, FunctionContext, ValueLocation, ARG_REGISTERS, RETURN_REGISTER};
use crate::{BasicBlock, Function, FunctionAnnotation, Identifier, Result};
use std::io::Write;

pub fn generate_functions<'a, W: Write>(
    module: &'a crate::Module<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
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
) -> Result<()> {
    let is_exported = func.annotations.contains(&FunctionAnnotation::Export);
    let is_main = func_name == "main";
    let asm_label = if is_main { "main".to_string() } else { format!("func_{}", func_name) };

    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name);

    // Minimal prologue: create frame pointer and save return address
    writeln!(writer, "\n# Function: @{}", func_name)?;
    if is_exported || is_main {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, "{}:", asm_label)?;

    // Prologue: save ra, s0; set up frame pointer
    writeln!(writer, "    addi sp, sp, -16")?;
    writeln!(writer, "    sd ra, 8(sp)")?;
    writeln!(writer, "    sd s0, 0(sp)")?;
    writeln!(writer, "    addi s0, sp, 16")?; // s0 points above saved area

    precompute_function_layout(func, &mut func_ctx, state)?;

    // Spill arguments to assigned stack slots if necessary
    for (i, param) in func.signature.params.iter().enumerate() {
        if let Some(ValueLocation::StackOffset(off)) = func_ctx.value_locations.get(param.name) {
            if i < ARG_REGISTERS.len() {
                writeln!(writer, "    addi t0, s0, {}", off)?;
                writeln!(writer, "    sd {}, 0(t0)", ARG_REGISTERS[i])?;
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
    writeln!(writer, "    ld ra, 8(sp)")?;
    writeln!(writer, "    ld s0, 0(sp)")?;
    writeln!(writer, "    addi sp, sp, 16")?;
    writeln!(writer, "    ret")?;
    Ok(())
}

fn precompute_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    // Assign simple block labels
    for ir_label in func.basic_blocks.keys() {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    // Very basic: place all params on stack sequentially at negative offsets
    let mut current: i64 = -16; // below saved area
    for param in &func.signature.params {
        current -= 8; // 8-byte slots
        func_ctx
            .value_locations
            .insert(param.name, ValueLocation::StackOffset(current));
    }

    // No locals scan for now; keep frame size minimal
    func_ctx.total_stack_size = (-current) as u64;
    Ok(())
}

fn generate_basic_block<'a, W: Write>(
    block: &BasicBlock<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    func_name: Identifier<'a>,
) -> Result<()> {
    for instr in &block.instructions {
        generate_instruction(instr, writer, state, func_ctx, func_name)?;
    }
    Ok(())
}


