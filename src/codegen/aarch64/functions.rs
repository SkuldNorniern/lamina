use super::instructions::generate_instruction;
use super::state::{ARG_REGISTERS, CodegenState, FunctionContext, ValueLocation};
use super::util::get_type_size_directive_and_bytes;
use crate::{BasicBlock, Function, FunctionAnnotation, Identifier, Instruction, LaminaError, PrimitiveType, Result};
use std::collections::HashSet;
use std::io::Write;

/// Generates AArch64 assembly for all functions in the module
///
/// This function iterates through all functions in the provided module and generates
/// their corresponding AArch64 assembly code following Apple's ARM64 ABI.
///
/// # Arguments
/// * `module` - The Lamina IR module containing functions to compile
/// * `writer` - Output writer for the generated assembly
/// * `state` - Code generation state shared across functions
///
/// # Returns
/// * `Result<()>` - Ok if all functions compile successfully, Err with error details otherwise
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
    let asm_label = if is_main {
        "_main".to_string() // Mach-O entry symbol uses underscore
    } else {
        format!("func_{}", func_name)
    };

    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name);

    let (required_regs, _inline_hint) = analyze_register_requirements(func);

    // Prologue (AArch64): save FP/LR and set FP
    writeln!(writer, "\n// Function: @{}", func_name)?;
    if is_exported || is_main {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, "{}:", asm_label)?;
    writeln!(writer, "    stp x29, x30, [sp, #-16]! // BUG: Hardcoded 16 bytes, may need more for frame_size")?;
    writeln!(writer, "    mov x29, sp")?;

    // Precompute layout
    precompute_function_layout(func, &mut func_ctx, state, &required_regs)?;

    // Allocate stack frame
    let frame_size = func_ctx.total_stack_size as i64;
    if frame_size > 0 {
        writeln!(writer, "    sub sp, sp, #{}", frame_size)?;
    }

    // Spill arg registers
    writeln!(writer, "    // Spill argument registers to stack slots")?;
    for (i, arg) in func.signature.params.iter().enumerate() {
        if let Some(loc) = func_ctx.value_locations.get(arg.name) {
            if let ValueLocation::StackOffset(offset) = loc {
                if i < ARG_REGISTERS.len() {
                    writeln!(writer, "        add x10, x29, #{}", offset)?;
                    // FIXED: ARG_REGISTERS[i] is already the register name (string), use it directly
                    writeln!(writer, "        str {}, [x10] // Spill arg {}", ARG_REGISTERS[i], arg.name)?;
                            } else {
                // FIXED: Stack arguments in AAPCS64 start at [sp, #0], calculate correct offset
                // x11 needs to point to the incoming stack argument location
                let stack_arg_offset = ((i - ARG_REGISTERS.len()) * 8) as i64;
                writeln!(writer, "        add x11, x29, #{}", stack_arg_offset)?; // AAPCS64 stack arg offset
                writeln!(writer, "        ldr x10, [x11] // Load stack arg {}", arg.name)?;
                writeln!(writer, "        add x11, x29, #{}", offset)?;
                writeln!(writer, "        str x10, [x11]")?;
            }
            }
        }
    }

    // Entry block
    if let Some(entry_block) = func.basic_blocks.get(&func.entry_block) {
        let asm_label = func_ctx.get_block_label(&func.entry_block)?;
        writeln!(writer, "{}:", asm_label)?;
        generate_basic_block(entry_block, writer, state, &func_ctx, func_name)?;
    }
    for (ir_label, block) in &func.basic_blocks {
        if *ir_label != func.entry_block {
            let asm_label = func_ctx.get_block_label(ir_label)?;
            writeln!(writer, "{}:", asm_label)?;
            generate_basic_block(block, writer, state, &func_ctx, func_name)?;
        }
    }

    // Epilogue
    writeln!(writer, "{}:", func_ctx.epilogue_label)?;
    if frame_size > 0 {
        writeln!(writer, "    add sp, sp, #{}", frame_size)?;
    }
    writeln!(writer, "    ldp x29, x30, [sp], #16")?;
    writeln!(writer, "    ret")?;

    Ok(())
}

fn analyze_register_requirements<'a>(func: &'a Function<'a>) -> (HashSet<&'static str>, bool) {
    let mut required = HashSet::new();
    let total_instructions: usize = func
        .basic_blocks
        .values()
        .map(|b| b.instructions.len())
        .sum();
    if total_instructions > 40 {
        required.insert("x19");
        required.insert("x20");
    }
    (required, false)
}

fn precompute_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
    _required_regs: &HashSet<&'static str>,
) -> Result<()> {
    // FIXED: Use local AArch64 function instead of x86_64 import
    use super::util::get_type_size_directive_and_bytes as aarch64_size;

    // FIXED: Stack arguments in AAPCS64 start at [x29, #0], not [x29, #16]
    let mut tmp_param_locs = Vec::new();
    let mut stack_arg_offset = 0i64; // AAPCS64: First stack arg at [sp, #0]
    for (i, param) in func.signature.params.iter().enumerate() {
        let loc = if i < ARG_REGISTERS.len() {
            ValueLocation::Register(ARG_REGISTERS[i].to_string())
        } else {
            let loc = ValueLocation::StackOffset(stack_arg_offset);
            let (_, sz) = aarch64_size(&param.ty)?;
            stack_arg_offset += ((sz + 7) & !7) as i64;
            loc
        };
        tmp_param_locs.push((param.name, loc));
    }

    for ir_label in func.basic_blocks.keys() {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    let mut local_size = 0u64;
    let mut local_allocs = Vec::new();
    for block in func.basic_blocks.values() {
        for instr in &block.instructions {
            let result_info: Option<(&Identifier<'a>, u64)> = match instr {
                Instruction::Alloc { result, allocated_ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(allocated_ty)?;
                    Some((result, s))
                }
                Instruction::Binary { result, ty, .. } | Instruction::Cmp { result, ty, .. } => {
                    let s = match ty {
                        PrimitiveType::I32 => 4,
                        PrimitiveType::I64 | PrimitiveType::Ptr => 8,
                        PrimitiveType::Bool | PrimitiveType::I8 => 1,
                        PrimitiveType::F32 => 4,
                        _ => return Err(LaminaError::CodegenError(format!("Unsupported type for stack allocation: {:?}", ty))),
                    };
                    Some((result, s))
                }
                Instruction::ZeroExtend { result, target_type, .. } => {
                    let s = match target_type {
                        PrimitiveType::I32 => 4,
                        PrimitiveType::I64 | PrimitiveType::Ptr => 8,
                        PrimitiveType::Bool | PrimitiveType::I8 => 1,
                        PrimitiveType::F32 => 4,
                        _ => return Err(LaminaError::CodegenError(format!("Unsupported target type for zero extension: {:?}", target_type))),
                    };
                    Some((result, s))
                }
                Instruction::Load { result, ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(ty)?;
                    Some((result, s))
                }
                Instruction::GetFieldPtr { result, .. }
                | Instruction::GetElemPtr { result, .. }
                | Instruction::Tuple { result, .. }
                | Instruction::ExtractTuple { result, .. } => Some((result, 8)),
                Instruction::Phi { result, ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(ty)?;
                    Some((result, s))
                }
                Instruction::Call { result: Some(res), .. } => Some((res, 8)),
                _ => None,
            };
            if let Some((res, size)) = result_info {
                let aligned = (size + 7) & !7;
                local_size += aligned;
                local_allocs.push((res, aligned));
            }
        }
    }

    let aligned_local = (local_size + 15) & !15;
    let mut current = -(16 + aligned_local as i64);
    let locals_start = current;
    for (res, sz) in local_allocs {
        func_ctx.value_locations.insert(res, ValueLocation::StackOffset(current));
        current += sz as i64;
    }

    current = (locals_start / 16) * 16;
    for (param_name, init_loc) in tmp_param_locs {
        if let ValueLocation::Register(reg) = init_loc {
            current -= 8;
            func_ctx.arg_register_spills.insert(reg.clone(), current);
            func_ctx
                .value_locations
                .insert(param_name, ValueLocation::StackOffset(current));
        } else {
            func_ctx.value_locations.insert(param_name, init_loc);
        }
    }

    let unaligned = -current as u64;
    func_ctx.total_stack_size = (unaligned + 15) & !15;
    Ok(())
}

fn generate_basic_block<'a, W: Write>(
    block: &BasicBlock<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
    func_name: Identifier<'a>,
) -> Result<()> {
    for instr in &block.instructions {
        generate_instruction(instr, writer, state, func_ctx, func_name)?;
    }
    Ok(())
}


