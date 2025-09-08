use super::instructions::generate_instruction;
use super::register_allocator::{AllocationResult, GraphColoringAllocator};
use super::state::{ARG_REGISTERS, CodegenState, FunctionContext, ValueLocation};
use super::util::get_type_size_directive_and_bytes;
use crate::codegen::CodegenError;
use crate::{
    BasicBlock, Function, FunctionAnnotation, Identifier, Instruction, LaminaError, PrimitiveType,
    Result, Type, Value,
};
use std::collections::HashSet;
use std::io::Write;

/// Generates x86_64 assembly for all functions in the module
///
/// This function iterates through all functions in the provided module and generates
/// their corresponding x86_64 assembly code. Each function is processed individually
/// using the `generate_function` helper.
///
/// # Arguments
/// * `module` - The Lamina IR module containing functions to compile
/// * `writer` - Output writer for the generated assembly
/// * `state` - Code generation state shared across functions
///
/// # Returns
/// * `Result<()>` - Ok if all functions compile successfully, Err with error details otherwise
pub fn generate_functions<'a, W: Write>(
    module: &'a crate::Module<'a>, // Use full path to avoid ambiguity
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    for (func_name, func) in &module.functions {
        generate_function(func_name, func, writer, state)?;
    }
    Ok(())
}

// Generate the assembly for all functions with graph coloring optimization
pub fn generate_functions_with_optimization<'a, W: Write>(
    module: &'a crate::Module<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    for (func_name, func) in &module.functions {
        generate_function_with_register_allocation(func_name, func, writer, state)?;
    }
    Ok(())
}

// Generate assembly for a single function with graph coloring register allocation
pub fn generate_function_with_register_allocation<'a, W: Write>(
    func_name: Identifier<'a>,
    func: &'a Function<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    // Apply graph coloring register allocation
    let mut allocator = GraphColoringAllocator::new();
    let allocation_result = allocator.allocate_registers(func)?;

    // Generate function using optimized register assignments
    generate_function_with_allocation(func_name, func, writer, state, &allocation_result)
}

// Core function generation with optional register allocation results
fn generate_function_with_allocation<'a, W: Write>(
    func_name: Identifier<'a>,
    func: &'a Function<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    // Determine assembly label
    let is_exported = func.annotations.contains(&FunctionAnnotation::Export);
    let is_main = func_name == "main";
    let asm_label = if is_main {
        "main".to_string() // Special case for C runtime entry point
    } else {
        format!("func_{}", func_name)
    };

    // Create context and determine epilogue label *before* generating blocks
    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name); // Set epilogue label early

    // Use allocation results to optimize register usage
    let callee_saved_needed: Vec<&str> = allocation_result
        .callee_saved_used
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Function Prologue - optimized based on register allocation
    writeln!(
        writer,
        "\n# Function: @{} (Graph Coloring Optimized)",
        func_name
    )?;
    if is_exported || is_main {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, ".type {}, @function", asm_label)?;
    writeln!(writer, ".align 16")?; // Ensure function entry is aligned
    writeln!(writer, "{}:", asm_label)?;
    writeln!(writer, "    pushq %rbp")?;
    writeln!(writer, "    movq %rsp, %rbp")?;

    // Only save callee-saved registers that are actually used by register allocation
    let mut saved_regs = Vec::new();
    for reg in &callee_saved_needed {
        if *reg != "%rbp" {
            // %rbp is handled separately
            writeln!(writer, "    pushq {}", reg)?;
            saved_regs.push(*reg);
        }
    }

    // Calculate stack layout using optimized allocation
    precompute_optimized_function_layout(func, &mut func_ctx, state, allocation_result)?;

    // Calculate required stack size for spilled variables
    let spill_stack_size = allocation_result.spilled_vars.len() * 8; // 8 bytes per spilled var
    let frame_size = ((spill_stack_size + 15) & !15) as u64; // Round up to 16-byte alignment

    if frame_size > 0 {
        writeln!(writer, "    subq ${}, %rsp", frame_size)?;
    }

    // Handle register-allocated vs spilled variables
    writeln!(writer, "    # Optimized register assignments:")?;
    for (var, reg) in &allocation_result.assignments {
        writeln!(writer, "    # {} -> {}", var, reg)?;
    }
    if !allocation_result.spilled_vars.is_empty() {
        writeln!(
            writer,
            "    # Spilled variables: {:?}",
            allocation_result.spilled_vars
        )?;
    }

    // Layout function stack and assign value locations based on register allocation
    precompute_optimized_function_layout(func, &mut func_ctx, state, allocation_result)?;

    // Process the entry block first
    if let Some(entry_block) = func.basic_blocks.get(&func.entry_block) {
        let asm_label = func_ctx.get_block_label(&func.entry_block)?;
        writeln!(writer, "{}:", asm_label)?;
        generate_optimized_basic_block(
            entry_block,
            writer,
            state,
            &mut func_ctx,
            func_name,
            allocation_result,
        )?;
    }

    // Process the remaining blocks (sorted for deterministic order)
    let mut sorted_blocks: Vec<_> = func.basic_blocks.keys().collect();
    sorted_blocks.sort();
    for ir_label in sorted_blocks {
        if *ir_label != func.entry_block {
            let block = &func.basic_blocks[ir_label];
            let asm_label = func_ctx.get_block_label(ir_label)?;
            writeln!(writer, "{}:", asm_label)?;
            generate_optimized_basic_block(
                block,
                writer,
                state,
                &mut func_ctx,
                func_name,
                allocation_result,
            )?;
        }
    }

    // Function Epilogue
    writeln!(writer, "{}:", func_ctx.epilogue_label)?;

    // Restore stack pointer
    if frame_size > 0 {
        writeln!(writer, "    addq ${}, %rsp", frame_size)?;
    }

    // Restore callee-saved registers in reverse order
    for reg in saved_regs.iter().rev() {
        writeln!(writer, "    popq {}", reg)?;
    }

    writeln!(writer, "    popq %rbp")?;
    writeln!(writer, "    ret")?;
    writeln!(writer, ".size {}, .-{}\n", asm_label, asm_label)?;

    Ok(())
}

/// Generates x86_64 assembly for a single function
///
/// This function converts a Lamina IR function into x86_64 assembly code, handling:
/// - Function prologue (stack setup, register preservation)
/// - Basic block generation
/// - Instruction translation
/// - Function epilogue (stack cleanup, register restoration)
///
/// # Arguments
/// * `func_name` - Name of the function to compile
/// * `func` - The Lamina IR function definition
/// * `writer` - Output writer for the generated assembly
/// * `state` - Code generation state
///
/// # Returns
/// * `Result<()>` - Ok if function compiles successfully, Err with error details otherwise
pub fn generate_function<'a, W: Write>(
    func_name: Identifier<'a>,
    func: &'a Function<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    // Determine assembly label
    let is_exported = func.annotations.contains(&FunctionAnnotation::Export);
    let is_main = func_name == "main";
    let asm_label = if is_main {
        "main".to_string() // Special case for C runtime entry point
    } else {
        format!("func_{}", func_name)
    };

    // Create context and determine epilogue label *before* generating blocks
    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name); // Set epilogue label early

    // Analyze which registers this function actually requires and if it should be inlined
    let (required_regs, should_inline) = analyze_register_requirements(func);

    // Store inlining preference in function state
    if should_inline {
        state.inlinable_functions.insert(func_name.to_string());
    }

    // Function Prologue
    writeln!(writer, "\n# Function: @{}", func_name)?;
    if is_exported || is_main {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, ".type {}, @function", asm_label)?;
    writeln!(writer, ".align 16")?; // Ensure function entry is aligned
    writeln!(writer, "{}:", asm_label)?;
    writeln!(writer, "    pushq %rbp")?;
    writeln!(writer, "    movq %rsp, %rbp")?;

    // Only save callee-saved registers we actually use
    let mut saved_regs = Vec::new();
    if required_regs.contains("rbx") {
        writeln!(writer, "    pushq %rbx")?;
        saved_regs.push("rbx");
    }
    if required_regs.contains("r12") {
        writeln!(writer, "    pushq %r12")?;
        saved_regs.push("r12");
    }
    if required_regs.contains("r13") {
        writeln!(writer, "    pushq %r13")?;
        saved_regs.push("r13");
    }
    if required_regs.contains("r14") {
        writeln!(writer, "    pushq %r14")?;
        saved_regs.push("r14");
    }
    if required_regs.contains("r15") {
        writeln!(writer, "    pushq %r15")?;
        saved_regs.push("r15");
    }

    // Calculate stack layout (offsets relative to %rbp)
    precompute_function_layout(func, &mut func_ctx, state, &required_regs)?; // Populate context

    // Calculate required stack size and ensure alignment for calls
    let needed_bytes = func_ctx.total_stack_size;

    // Calculate proper alignment based on saved registers
    // Each saved reg is 8 bytes, plus rbp is 8 bytes, total must be 16-byte aligned
    let _saved_bytes = (saved_regs.len() + 1) * 8; // +1 for rbp
    let frame_size = (needed_bytes + 15) & !15; // Round up to multiple of 16

    if frame_size > 0 {
        writeln!(writer, "    subq ${}, %rsp", frame_size)?;
    }

    // Spill argument registers to stack slots if necessary
    writeln!(writer, "    # Spill argument registers to stack slots")?;
    for (i, arg) in func.signature.params.iter().enumerate() {
        if let Some(loc) = func_ctx.value_locations.get(arg.name)
            && let ValueLocation::StackOffset(offset) = loc
        {
            if i < ARG_REGISTERS.len() {
                // Note: Currently only handles 64-bit arg types (movq)
                writeln!(
                    writer,
                    "        movq {}, {}(%rbp) # Spill arg {}",
                    ARG_REGISTERS[i], offset, arg.name
                )?;
            } else {
                // Arguments passed on the stack are already above RBP,
                // copy them down if needed? Or adjust offsets?
                // For now, assume stack args are accessed relative to old RBP if needed,
                // but ideally they are copied to local stack slots if reused.
                // Stack args are at %rbp + 16, %rbp + 24, ...
                let stack_arg_offset = 16 + (i - ARG_REGISTERS.len()) * 8;
                writeln!(
                    writer,
                    "        movq {}(%rbp), %r10 # Load stack arg {}",
                    stack_arg_offset, arg.name
                )?;
                writeln!(
                    writer,
                    "        movq %r10, {}(%rbp) # Spill stack arg {} to local slot",
                    offset, arg.name
                )?;
            }
        }
    }

    // Layout function stack and assign value locations
    precompute_function_layout(func, &mut func_ctx, state, &required_regs)?;

    // Function Body Generation
    writeln!(writer, "    # Function Body Start")?;

    // Process the entry block first to ensure correct control flow
    if let Some(entry_block) = func.basic_blocks.get(&func.entry_block) {
        let asm_label = func_ctx.get_block_label(&func.entry_block)?;
        writeln!(writer, "{}:", asm_label)?;
        generate_basic_block(entry_block, writer, state, &mut func_ctx, func_name)?;
    }

    // Process the remaining blocks (sorted for deterministic order)
    let mut sorted_blocks: Vec<_> = func.basic_blocks.keys().collect();
    sorted_blocks.sort();
    for ir_label in sorted_blocks {
        // Skip the entry block since we've already processed it
        if *ir_label != func.entry_block {
            let block = &func.basic_blocks[ir_label];
            let asm_label = func_ctx.get_block_label(ir_label)?;
            writeln!(writer, "{}:", asm_label)?;
            generate_basic_block(block, writer, state, &mut func_ctx, func_name)?;
        }
    }

    writeln!(writer, "    # Function Body End")?;

    // Function Epilogue
    writeln!(writer, "{}:", func_ctx.epilogue_label)?;

    // Restore stack pointer directly if needed
    if frame_size > 0 {
        writeln!(writer, "    addq ${}, %rsp", frame_size)?;
    }

    // Write restore of callee-saved registers (in reverse order)
    writeln!(writer, "    # Restore callee-saved registers")?;
    for reg in saved_regs.iter().rev() {
        writeln!(writer, "    popq %{}", reg)?;
    }

    // Return directly without leave instruction
    writeln!(writer, "    popq %rbp")?;
    writeln!(writer, "    ret")?;

    // Add size directive for debugging/analysis
    writeln!(writer, ".size {}, .-{}\n", asm_label, asm_label)?;

    Ok(())
}

// Analyze which registers this function actually requires
fn analyze_register_requirements<'a>(func: &'a Function<'a>) -> (HashSet<&'static str>, bool) {
    let mut required_regs = HashSet::new();

    // Count instructions to estimate complexity
    let total_instructions: usize = func
        .basic_blocks
        .values()
        .map(|block| block.instructions.len())
        .sum();

    // Check for recursive calls
    let has_recursion = func.basic_blocks.values().any(|block| {
        block.instructions.iter().any(|instr| {
            if let Instruction::Call { func_name, .. } = instr {
                *func_name == func.name
            } else {
                false
            }
        })
    });

    // Check for function calls (non-recursive)
    let has_function_calls = func.basic_blocks.values().any(|block| {
        block
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instruction::Call { .. }))
    });

    // Count number of binary operations
    let binary_op_count = func
        .basic_blocks
        .values()
        .map(|block| {
            block
                .instructions
                .iter()
                .filter(|instr| matches!(instr, Instruction::Binary { .. }))
                .count()
        })
        .sum::<usize>();

    // Count number of comparisons
    let cmp_count = func
        .basic_blocks
        .values()
        .map(|block| {
            block
                .instructions
                .iter()
                .filter(|instr| matches!(instr, Instruction::Cmp { .. }))
                .count()
        })
        .sum::<usize>();

    // If the function is recursive or has many instructions, it'll need rbx
    if has_recursion || total_instructions > 15 || binary_op_count > 5 {
        required_regs.insert("rbx");
    }

    // If it has more operations or calls other functions, allocate r12
    if has_function_calls || binary_op_count > 10 || cmp_count > 3 {
        required_regs.insert("r12");
    }

    // More complex functions with many instructions and function calls
    if total_instructions > 30 || (has_function_calls && total_instructions > 20) {
        required_regs.insert("r13");
    }

    // Only for very complex functions
    if total_instructions > 40 || (has_recursion && has_function_calls) {
        required_regs.insert("r14");
    }

    // Only for the most complex functions
    if total_instructions > 60 || (has_recursion && total_instructions > 40) {
        required_regs.insert("r15");
    }

    // Determine if function could be inlined
    // Small, non-recursive functions with few instructions are good candidates
    let should_inline = !has_recursion
        && total_instructions <= 10
        && func.basic_blocks.len() <= 2
        && !func.signature.params.is_empty()
        && func.signature.params.len() <= 3;

    (required_regs, should_inline)
}

// Pass to calculate stack layout and assign assembly labels to IR blocks
fn precompute_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
    required_regs: &HashSet<&'static str>,
) -> Result<()> {
    let mut current_stack_offset: i64;
    let mut current_param_stack_offset = 16i64; // For args passed via stack (> 6th arg)

    // Calculate size taken by saved callee registers
    let saved_callee_regs_size = (required_regs.len() as i64) * 8; // Each saved reg is 8 bytes

    // 1. Assign locations for parameters (registers first, then stack)
    let mut temp_param_locations = Vec::new(); // Store temp locations before assigning spill slots
    for (i, param) in func.signature.params.iter().enumerate() {
        let location = if i < ARG_REGISTERS.len() {
            ValueLocation::Register(ARG_REGISTERS[i].to_string())
        } else {
            let loc = ValueLocation::StackOffset(current_param_stack_offset);
            let (_, size) = get_type_size_directive_and_bytes(&param.ty)?;
            current_param_stack_offset += ((size + 7) & !7) as i64;
            loc
        };
        temp_param_locations.push((param.name, location));
    }

    // 2. Assign assembly labels to IR blocks (sorted for deterministic order)
    let mut sorted_labels: Vec<_> = func.basic_blocks.keys().collect();
    sorted_labels.sort();
    for ir_label in sorted_labels {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    // 3. Calculate stack space for locals (sorted for deterministic order)
    let mut _local_size = 0u64;
    let mut local_allocations = Vec::new(); // Store (result_name, size)
    let mut sorted_blocks: Vec<_> = func.basic_blocks.keys().collect();
    sorted_blocks.sort();
    for block_label in sorted_blocks {
        let block = &func.basic_blocks[block_label];
        for instr in &block.instructions {
            let result_info: Option<(&Identifier<'a>, u64)> = match instr {
                Instruction::Alloc { result, .. } => {
                    // Alloc always results in a pointer, regardless of allocated type
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?;
                    Some((result, s))
                }
                Instruction::Binary { result, ty, .. } | Instruction::Cmp { result, ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(&Type::Primitive(*ty))?;
                    Some((result, s))
                }
                Instruction::ZeroExtend {
                    result,
                    target_type,
                    ..
                } => {
                    let (_, s) = get_type_size_directive_and_bytes(&Type::Primitive(*target_type))?;
                    Some((result, s))
                }
                Instruction::Load { result, ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(ty)?;
                    Some((result, s))
                }
                Instruction::GetFieldPtr { result, .. }
                | Instruction::GetElemPtr { result, .. } => {
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?;
                    Some((result, s))
                }
                Instruction::Tuple { result, .. } => {
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?;
                    Some((result, s))
                }
                Instruction::ExtractTuple { result, .. } => {
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?;
                    Some((result, s))
                }
                Instruction::Phi { result, ty, .. } => {
                    let (_, s) = get_type_size_directive_and_bytes(ty)?;
                    Some((result, s))
                }
                Instruction::Call {
                    result: Some(res), ..
                } => {
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?;
                    Some((res, s))
                }
                Instruction::Write { result, .. }
                | Instruction::Read { result, .. }
                | Instruction::WriteByte { result, .. }
                | Instruction::ReadByte { result, .. }
                | Instruction::WritePtr { result, .. }
                | Instruction::ReadPtr { result, .. } => {
                    let (_, s) =
                        get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::I64))?;
                    Some((result, s))
                }
                _ => None, // Only care about instructions with results
            };

            if let Some((result, size)) = result_info {
                let aligned_size = (size + 7) & !7;
                _local_size += aligned_size;
                local_allocations.push((result, aligned_size));
            }
        }
    }

    // 4. Assign stack offsets for locals, starting below the saved callee registers
    // Start allocating from -8 (below saved rbp), then go more negative for each variable
    current_stack_offset = -(saved_callee_regs_size + 8); // Start below saved rbp
    let _local_start_offset = current_stack_offset;

    // Assign offsets to local values based on the aligned layout
    for (result, aligned_size) in local_allocations {
        func_ctx
            .value_locations
            .insert(result, ValueLocation::StackOffset(current_stack_offset));
        current_stack_offset -= aligned_size as i64; // Move offset *down* (more negative) for next variable
    }

    // Continue allocating spill slots below the locals we just allocated
    // current_stack_offset is already at the bottom of the locals, so continue from there
    // No need to reset - we want to allocate spills below the locals

    // 5. Allocate stack spill slots for register parameters and record them
    for (param_name, initial_location) in temp_param_locations {
        if let ValueLocation::Register(reg_name) = initial_location {
            let param_sig = func
                .signature
                .params
                .iter()
                .find(|p| p.name == param_name)
                .ok_or(LaminaError::CodegenError(CodegenError::InternalError))?; // Find param to get type
            let (_, size) = get_type_size_directive_and_bytes(&param_sig.ty)?;
            let aligned_size = (size + 7) & !7;
            current_stack_offset -= aligned_size as i64; // Allocate space below locals/previous spills
            func_ctx
                .arg_register_spills
                .insert(reg_name.clone(), current_stack_offset);
            // Final location for parameters (even register ones) is their spill slot
            func_ctx
                .value_locations
                .insert(param_name, ValueLocation::StackOffset(current_stack_offset));
        } else {
            // Parameter was already on the stack, just insert its location
            func_ctx
                .value_locations
                .insert(param_name, initial_location);
        }
    }

    // Calculate total stack size: from saved rbp to bottom of allocated space
    // current_stack_offset is now at the bottom of all allocated space (locals + spills)
    let total_allocated = -current_stack_offset as u64;
    func_ctx.total_stack_size = (total_allocated + 15) & !15; // 16-byte align

    Ok(())
}

// Generate assembly for a single basic block
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

// Optimized function layout that considers register allocation results
fn precompute_optimized_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    // Use simplified layout for register-allocated variables
    let mut current_stack_offset = -8i64; // Start below RBP

    // Assign assembly labels to IR blocks (sorted for deterministic order)
    let mut sorted_labels: Vec<_> = func.basic_blocks.keys().collect();
    sorted_labels.sort();
    for ir_label in sorted_labels {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    // Only allocate stack space for spilled variables and function parameters
    for param in &func.signature.params {
        if allocation_result.spilled_vars.contains(param.name) {
            // Spilled parameter gets stack location
            func_ctx
                .value_locations
                .insert(param.name, ValueLocation::StackOffset(current_stack_offset));
            current_stack_offset -= 8; // Move down for next allocation
        } else if let Some(reg) = allocation_result.assignments.get(param.name) {
            // Register-allocated parameter
            func_ctx
                .value_locations
                .insert(param.name, ValueLocation::Register(reg.clone()));
        } else {
            // Fallback: parameter not handled by register allocator, assign stack location
            func_ctx
                .value_locations
                .insert(param.name, ValueLocation::StackOffset(current_stack_offset));
            current_stack_offset -= 8;
        }
    }

    // Handle local variables (sorted for deterministic order)
    let mut sorted_blocks: Vec<_> = func.basic_blocks.keys().collect();
    sorted_blocks.sort();
    for block_label in sorted_blocks {
        let block = &func.basic_blocks[block_label];
        for instr in &block.instructions {
            if let Some(result_var) = get_instruction_result(instr) {
                if allocation_result.spilled_vars.contains(result_var) {
                    func_ctx
                        .value_locations
                        .insert(result_var, ValueLocation::StackOffset(current_stack_offset));
                    current_stack_offset -= 8;
                } else if let Some(reg) = allocation_result.assignments.get(result_var) {
                    func_ctx
                        .value_locations
                        .insert(result_var, ValueLocation::Register(reg.clone()));
                } else {
                    // Fallback: variable not handled by register allocator, assign stack location
                    func_ctx
                        .value_locations
                        .insert(result_var, ValueLocation::StackOffset(current_stack_offset));
                    current_stack_offset -= 8;
                }
            }
        }
    }

    func_ctx.total_stack_size = (-current_stack_offset) as u64;
    Ok(())
}

// Generate optimized basic block that uses register allocation results
fn generate_optimized_basic_block<'a, W: Write>(
    block: &BasicBlock<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    func_name: Identifier<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    for instr in &block.instructions {
        // Use optimized instruction generation that prefers registers
        generate_optimized_instruction(
            instr,
            writer,
            state,
            func_ctx,
            func_name,
            allocation_result,
        )?;
    }
    Ok(())
}

// Optimized instruction generation that leverages register allocation
fn generate_optimized_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    func_name: Identifier<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    // Try to generate highly optimized assembly based on register allocation
    match instr {
        Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => generate_optimized_binary(
            op,
            result,
            ty,
            lhs,
            rhs,
            writer,
            state,
            func_ctx,
            allocation_result,
        ),
        Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => generate_optimized_cmp(
            op,
            result,
            ty,
            lhs,
            rhs,
            writer,
            state,
            func_ctx,
            allocation_result,
        ),
        Instruction::Load { result, ty, ptr } => {
            generate_optimized_load(result, ty, ptr, writer, state, func_ctx, allocation_result)
        }
        Instruction::Store { ty, ptr, value } => {
            generate_optimized_store(ty, ptr, value, writer, state, func_ctx, allocation_result)
        }
        // For other instructions, use the existing generator
        _ => generate_instruction(instr, writer, state, func_ctx, func_name),
    }
}

// Helper to get result variable from instruction
fn get_instruction_result<'a>(instr: &Instruction<'a>) -> Option<&'a str> {
    match instr {
        Instruction::Binary { result, .. } => Some(result),
        Instruction::Cmp { result, .. } => Some(result),
        Instruction::Load { result, .. } => Some(result),
        Instruction::Alloc { result, .. } => Some(result),
        Instruction::Call {
            result: Some(result),
            ..
        } => Some(result),
        Instruction::ZeroExtend { result, .. } => Some(result),
        Instruction::GetFieldPtr { result, .. } => Some(result),
        Instruction::GetElemPtr { result, .. } => Some(result),
        Instruction::Tuple { result, .. } => Some(result),
        Instruction::ExtractTuple { result, .. } => Some(result),
        Instruction::Phi { result, .. } => Some(result),
        Instruction::Write { result, .. } => Some(result),
        Instruction::Read { result, .. } => Some(result),
        Instruction::WriteByte { result, .. } => Some(result),
        Instruction::ReadByte { result, .. } => Some(result),
        Instruction::WritePtr { result, .. } => Some(result),
        Instruction::ReadPtr { result, .. } => Some(result),
        _ => None,
    }
}

// Highly optimized binary operation generation using register allocation results
fn generate_optimized_binary<'a, W: Write>(
    op: &crate::ir::instruction::BinaryOp,
    result: &'a str,
    ty: &crate::ir::types::PrimitiveType,
    lhs: &Value<'a>,
    rhs: &Value<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    use crate::ir::instruction::BinaryOp;
    use crate::ir::types::{Literal, PrimitiveType};

    // Get register/memory locations for operands
    let lhs_in_reg = if let Value::Variable(var) = lhs {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    let rhs_in_reg = if let Value::Variable(var) = rhs {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    let result_in_reg = allocation_result.assignments.get(result);

    // Determine size suffix and instruction format
    let (op_suffix, _reg_prefix) = match ty {
        PrimitiveType::I32 => ("l", "e"),
        PrimitiveType::I64 | PrimitiveType::Ptr => ("q", "r"),
        _ => {
            return generate_instruction(
                &crate::Instruction::Binary {
                    op: op.clone(),
                    result,
                    ty: *ty,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                writer,
                state,
                func_ctx,
                "",
            );
        }
    };

    // Fast path: both operands in registers, result in register
    if let (Some(lhs_reg), Some(rhs_reg), Some(result_reg)) =
        (lhs_in_reg, rhs_in_reg, result_in_reg)
    {
        let op_name = match op {
            BinaryOp::Add => format!("add{}", op_suffix),
            BinaryOp::Sub => format!("sub{}", op_suffix),
            BinaryOp::Mul => format!("imul{}", op_suffix),
            BinaryOp::Div => {
                // Division is more complex, fall back to regular implementation
                return generate_instruction(
                    &crate::Instruction::Binary {
                        op: op.clone(),
                        result,
                        ty: *ty,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    },
                    writer,
                    state,
                    func_ctx,
                    "",
                );
            }
        };

        // Ultra-fast register-to-register operations
        if lhs_reg == result_reg {
            // result = lhs op rhs, and result is already in lhs register
            writeln!(
                writer,
                "    {} {}, {} # Optimized: {} = {} {} {}",
                op_name, rhs_reg, lhs_reg, result, lhs_reg, op_name, rhs_reg
            )?;
        } else if rhs_reg == result_reg && matches!(op, BinaryOp::Add | BinaryOp::Mul) {
            // For commutative operations, we can do result = rhs op lhs
            writeln!(
                writer,
                "    {} {}, {} # Optimized: {} = {} {} {}",
                op_name, lhs_reg, rhs_reg, result, rhs_reg, op_name, lhs_reg
            )?;
        } else {
            // Move one operand to result register, then operate
            writeln!(
                writer,
                "    mov{} {}, {} # Optimized move",
                op_suffix, lhs_reg, result_reg
            )?;
            writeln!(
                writer,
                "    {} {}, {} # Optimized: {} = {} {} {}",
                op_name, rhs_reg, result_reg, result, lhs_reg, op_name, rhs_reg
            )?;
        }
        return Ok(());
    }

    // Constant folding for immediate values
    if let (Value::Constant(Literal::I64(lhs_val)), Value::Constant(Literal::I64(rhs_val))) =
        (lhs, rhs)
    {
        let folded_result = match op {
            BinaryOp::Add => lhs_val + rhs_val,
            BinaryOp::Sub => lhs_val - rhs_val,
            BinaryOp::Mul => lhs_val * rhs_val,
            BinaryOp::Div => {
                if *rhs_val != 0 {
                    lhs_val / rhs_val
                } else {
                    0
                }
            }
        };

        if let Some(result_reg) = result_in_reg {
            writeln!(
                writer,
                "    mov{} ${}, {} # Constant folding: {} = {}",
                op_suffix, folded_result, result_reg, result, folded_result
            )?;
            return Ok(());
        }
    }

    // Optimized immediate operations
    if let (Some(lhs_reg), Value::Constant(Literal::I64(rhs_val))) = (lhs_in_reg, rhs)
        && let Some(result_reg) = result_in_reg
    {
        let op_name = match op {
            BinaryOp::Add => format!("add{}", op_suffix),
            BinaryOp::Sub => format!("sub{}", op_suffix),
            BinaryOp::Mul => {
                // Check for power-of-2 multiplication
                if *rhs_val > 0 && (*rhs_val & (rhs_val - 1)) == 0 {
                    let shift_amount = rhs_val.trailing_zeros();
                    if lhs_reg == result_reg {
                        writeln!(
                            writer,
                            "    shl{} ${}, {} # Optimized multiply by power of 2",
                            op_suffix, shift_amount, result_reg
                        )?;
                    } else {
                        writeln!(
                            writer,
                            "    mov{} {}, {} # Move for shift",
                            op_suffix, lhs_reg, result_reg
                        )?;
                        writeln!(
                            writer,
                            "    shl{} ${}, {} # Optimized multiply by power of 2",
                            op_suffix, shift_amount, result_reg
                        )?;
                    }
                    return Ok(());
                }
                format!("imul{}", op_suffix)
            }
            BinaryOp::Div => {
                // Fall back to regular division
                return generate_instruction(
                    &crate::Instruction::Binary {
                        op: op.clone(),
                        result,
                        ty: *ty,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    },
                    writer,
                    state,
                    func_ctx,
                    "",
                );
            }
        };

        if lhs_reg == result_reg {
            writeln!(
                writer,
                "    {} ${}, {} # Optimized immediate operation",
                op_name, rhs_val, result_reg
            )?;
        } else {
            writeln!(
                writer,
                "    mov{} {}, {} # Move for immediate op",
                op_suffix, lhs_reg, result_reg
            )?;
            writeln!(
                writer,
                "    {} ${}, {} # Optimized immediate operation",
                op_name, rhs_val, result_reg
            )?;
        }
        return Ok(());
    }

    // Fall back to regular instruction generation
    generate_instruction(
        &crate::Instruction::Binary {
            op: op.clone(),
            result,
            ty: *ty,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        writer,
        state,
        func_ctx,
        "",
    )
}

// Optimized comparison generation
fn generate_optimized_cmp<'a, W: Write>(
    op: &crate::ir::instruction::CmpOp,
    result: &'a str,
    ty: &crate::ir::types::PrimitiveType,
    lhs: &Value<'a>,
    rhs: &Value<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    use crate::ir::instruction::CmpOp;
    use crate::ir::types::PrimitiveType;

    // Get register locations
    let lhs_in_reg = if let Value::Variable(var) = lhs {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    let rhs_in_reg = if let Value::Variable(var) = rhs {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    let result_in_reg = allocation_result.assignments.get(result);

    // Fast path: both operands in registers
    if let (Some(lhs_reg), Some(rhs_reg), Some(result_reg)) =
        (lhs_in_reg, rhs_in_reg, result_in_reg)
    {
        let cmp_suffix = match ty {
            PrimitiveType::I32 => "l",
            PrimitiveType::I64 | PrimitiveType::Ptr => "q",
            PrimitiveType::I8 => "b",
            _ => {
                return generate_instruction(
                    &crate::Instruction::Cmp {
                        op: op.clone(),
                        result,
                        ty: *ty,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    },
                    writer,
                    state,
                    func_ctx,
                    "",
                );
            }
        };

        let set_instr = match op {
            CmpOp::Eq => "sete",
            CmpOp::Ne => "setne",
            CmpOp::Gt => "setg",
            CmpOp::Ge => "setge",
            CmpOp::Lt => "setl",
            CmpOp::Le => "setle",
        };

        writeln!(
            writer,
            "        cmp{} {}, {} # Optimized register comparison",
            cmp_suffix, rhs_reg, lhs_reg
        )?;
        writeln!(writer, "        {} %al # Set AL based on flags", set_instr)?;
        // Convert to appropriate register size for the result
        let result_reg_sized = if cmp_suffix == "l" {
            // For 32-bit comparison, use 32-bit destination register
            if result_reg.starts_with("%r") {
                result_reg.replace("%r", "%e").replace("ax", "eax")
            } else {
                result_reg.clone()
            }
        } else {
            result_reg.clone()
        };
        writeln!(
            writer,
            "        movzbq %al, {} # Zero-extend to result register",
            result_reg_sized
        )?;
        return Ok(());
    }

    // Fall back to regular instruction generation
    generate_instruction(
        &crate::Instruction::Cmp {
            op: op.clone(),
            result,
            ty: *ty,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        writer,
        state,
        func_ctx,
        "",
    )
}

// Optimized load generation
fn generate_optimized_load<'a, W: Write>(
    result: &'a str,
    ty: &crate::ir::types::Type<'a>,
    ptr: &Value<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    use crate::ir::types::Type;

    // Check if result is in a register
    if let Some(result_reg) = allocation_result.assignments.get(result) {
        // Check if pointer is also in a register
        if let Value::Variable(ptr_var) = ptr
            && let Some(ptr_reg) = allocation_result.assignments.get(*ptr_var)
        {
            let load_suffix = match ty {
                Type::Primitive(crate::ir::types::PrimitiveType::I32) => "l",
                Type::Primitive(crate::ir::types::PrimitiveType::I64)
                | Type::Primitive(crate::ir::types::PrimitiveType::Ptr) => "q",
                Type::Primitive(crate::ir::types::PrimitiveType::I8) => "b",
                _ => {
                    return generate_instruction(
                        &crate::Instruction::Load {
                            result,
                            ty: ty.clone(),
                            ptr: ptr.clone(),
                        },
                        writer,
                        state,
                        func_ctx,
                        "",
                    );
                }
            };

            writeln!(
                writer,
                "        mov{} ({}), {} # Optimized register-to-register load",
                load_suffix, ptr_reg, result_reg
            )?;
            return Ok(());
        }
    }

    // Fall back to regular instruction generation
    generate_instruction(
        &crate::Instruction::Load {
            result,
            ty: ty.clone(),
            ptr: ptr.clone(),
        },
        writer,
        state,
        func_ctx,
        "",
    )
}

// Optimized store generation
fn generate_optimized_store<'a, W: Write>(
    ty: &crate::ir::types::Type<'a>,
    ptr: &Value<'a>,
    value: &Value<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    allocation_result: &AllocationResult,
) -> Result<()> {
    use crate::ir::types::Type;

    // Check if both pointer and value are in registers
    let ptr_in_reg = if let Value::Variable(var) = ptr {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    let value_in_reg = if let Value::Variable(var) = value {
        allocation_result.assignments.get(*var)
    } else {
        None
    };

    if let (Some(ptr_reg), Some(value_reg)) = (ptr_in_reg, value_in_reg) {
        let store_suffix = match ty {
            Type::Primitive(crate::ir::types::PrimitiveType::I32) => "l",
            Type::Primitive(crate::ir::types::PrimitiveType::I64)
            | Type::Primitive(crate::ir::types::PrimitiveType::Ptr) => "q",
            Type::Primitive(crate::ir::types::PrimitiveType::I8) => "b",
            _ => {
                return generate_instruction(
                    &crate::Instruction::Store {
                        ty: ty.clone(),
                        ptr: ptr.clone(),
                        value: value.clone(),
                    },
                    writer,
                    state,
                    func_ctx,
                    "",
                );
            }
        };

        writeln!(
            writer,
            "        mov{} {}, ({}) # Optimized register-to-register store",
            store_suffix, value_reg, ptr_reg
        )?;
        return Ok(());
    }

    // Fall back to regular instruction generation
    generate_instruction(
        &crate::Instruction::Store {
            ty: ty.clone(),
            ptr: ptr.clone(),
            value: value.clone(),
        },
        writer,
        state,
        func_ctx,
        "",
    )
}

#[cfg(test)]
mod tests {
    use crate::codegen::x86_64::generate_x86_64_assembly;
    use crate::parser::parse_module;

    fn compile_to_asm(ir_code: &str) -> String {
        let module = parse_module(ir_code).expect("Test IR parsing failed");
        let mut asm_buffer = Vec::new();
        generate_x86_64_assembly(&module, &mut asm_buffer).expect("Codegen failed");
        String::from_utf8(asm_buffer).expect("Assembly output is not valid UTF-8")
    }

    #[test]
    fn test_function_argument_spilling() {
        let ir = r#"
            fn @simple_args(i64 %a, i64 %b, i64 %c) -> i64 {
                entry:
                    %sum = add.i64 %a, %b
                    %sum2 = add.i64 %sum, %c
                    ret.i64 %sum2
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check that arguments are properly spilled to stack
        assert!(asm.contains("movq %rdi, "));
        assert!(asm.contains("# Spill arg a"));
        assert!(asm.contains("movq %rsi, "));
        assert!(asm.contains("# Spill arg b"));
        assert!(asm.contains("movq %rdx, "));
        assert!(asm.contains("# Spill arg c"));
    }

    #[test]
    fn test_function_with_multiple_blocks() {
        let ir = r#"
            fn @conditional(i64 %value) -> i64 {
                entry:
                    %is_positive = gt.i64 %value, 0
                    br %is_positive, positive, negative
                
                positive:
                    ret.i64 1
                
                negative:
                    ret.i64 0
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check for multiple blocks with labels
        assert!(asm.contains("L_block_entry_"));
        assert!(asm.contains("L_block_positive_"));
        assert!(asm.contains("L_block_negative_"));

        // Check for branch instruction
        assert!(asm.contains("cmpq"));
        assert!(asm.contains("setg %al"));
        assert!(asm.contains("testb %al, %al"));
        assert!(asm.contains("jne"));
        assert!(asm.contains("jmp"));
    }

    #[test]
    fn test_function_calls() {
        let ir = r#"
            fn @callee(i64 %x) -> i64 {
                entry:
                    ret.i64 %x
            }

            fn @caller() -> i64 {
                entry:
                    %result = call @callee(42)
                    ret.i64 %result
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check for function call setup
        assert!(asm.contains("# Setup register arguments for call @callee"));
        assert!(asm.contains("movq $42, %rdi # Arg 0"));
        assert!(asm.contains("call func_callee"));
        assert!(asm.contains("movq %rax, "));
        assert!(asm.contains("# Store call result"));
    }

    #[test]
    fn test_binary_operations() {
        let ir = r#"
            fn @binary_ops(i64 %a, i64 %b) -> i64 {
                entry:
                    %sum = add.i64 %a, %b
                    %diff = sub.i64 %sum, 10
                    %product = mul.i64 %diff, 3
                    %quotient = div.i64 %product, 5
                    ret.i64 %quotient
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check for binary operations
        assert!(asm.contains("addq"));
        assert!(asm.contains("subq"));
        assert!(asm.contains("imulq"));
        assert!(asm.contains("idivq"));
    }

    #[test]
    fn test_matmul_consistency() {
        let ir = r#"
            fn @get_matrix_a_element(i64 %i, i64 %k) -> i64 {
              entry:
                %result = mul.i64 %i, %k
                %result = add.i64 %result, 1
                ret.i64 %result
            }

            fn @get_matrix_b_element(i64 %k, i64 %j) -> i64 {
              entry:
                %result = mul.i64 %k, %j
                %result = add.i64 %result, 1
                ret.i64 %result
            }

            fn @compute_matrix_cell(i64 %i, i64 %j, i64 %k_dim) -> i64 {
              entry:
                %sum = add.i64 0, 0
                %k = add.i64 0, 0
                jmp k_loop
                
              k_loop:
                %k_done = eq.i64 %k, %k_dim
                br %k_done, loop_exit, loop_body
                
              loop_body:
                %a_elem = call @get_matrix_a_element(%i, %k)
                %b_elem = call @get_matrix_b_element(%k, %j)
                %product = mul.i64 %a_elem, %b_elem
                %sum = add.i64 %sum, %product
                %k = add.i64 %k, 1
                jmp k_loop
                
              loop_exit:
                ret.i64 %sum
            }

            fn @test_matmul() -> i64 {
              entry:
                # Test compute_matrix_cell for specific values
                # For k_dim=2, i=1, j=1, we should get:
                # A[1,0] * B[0,1] + A[1,1] * B[1,1]
                # (1*0+1) * (0*1+1) + (1*1+1) * (1*1+1)
                # 1 * 1 + 2 * 2 = 1 + 4 = 5
                %result = call @compute_matrix_cell(1, 1, 2)
                ret.i64 %result
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check that the compiler generated the function correctly
        assert!(asm.contains("func_get_matrix_a_element:"));
        assert!(asm.contains("func_get_matrix_b_element:"));
        assert!(asm.contains("func_compute_matrix_cell:"));
        assert!(asm.contains("func_test_matmul:"));
    }

    #[test]
    fn test_global_variables() {
        let ir = r#"
            global @counter: i64 = 100
            global @message: [5 x i8] = "hello"
            
            fn @increment_counter() -> i64 {
                entry:
                    %current = load.i64 @counter
                    %incremented = add.i64 %current, 1
                    store.i64 %incremented, @counter
                    ret.i64 %incremented
            }
            
            fn @get_char(i64 %idx) -> i8 {
                entry:
                    %ptr = getelementptr @message, %idx
                    %char = load.i8 %ptr
                    ret.i8 %char
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check global data section
        assert!(asm.contains(".section .data"));
        assert!(asm.contains("global_counter:"));
        assert!(asm.contains(".quad 100"));
        assert!(asm.contains("global_message:"));

        // Check load from global
        assert!(asm.contains("global_counter(%rip)"));

        // Check getelementptr for array access
        assert!(asm.contains("global_message(%rip)"));
    }

    #[test]
    fn test_different_primitive_types() {
        let ir = r#"
            fn @test_types(i32 %a, i32 %b) -> i64 {
                entry:
                    %bool_val = eq.i32 %a, %b
                    %i32_val = add.i32 %a, 42
                    %extended = zext.i32.i64 %i32_val
                    
                    br %bool_val, true_branch, false_branch
                    
                true_branch:
                    ret.i64 %extended
                    
                false_branch:
                    ret.i64 0
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check i32 operations
        assert!(asm.contains("movl"));
        assert!(asm.contains("addl"));

        // Check boolean operations
        assert!(asm.contains("sete %al"));
        assert!(asm.contains("testb %al, %al"));

        // Check zero extension
        assert!(asm.contains("movslq"));
    }

    #[test]
    fn test_simple_loop() {
        let ir = r#"
            fn @sum_up_to(i64 %n) -> i64 {
                entry:
                    %sum = add.i64 0, 0
                    %i = add.i64 0, 0
                    jmp loop_start
                    
                loop_start:
                    %continue = lt.i64 %i, %n
                    br %continue, loop_body, loop_exit
                    
                loop_body:
                    %sum = add.i64 %sum, %i
                    %i = add.i64 %i, 1
                    jmp loop_start
                    
                loop_exit:
                    ret.i64 %sum
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check loop structure
        assert!(asm.contains("jmp"));
        assert!(asm.contains("L_block_loop_start_"));
        assert!(asm.contains("L_block_loop_body_"));
        assert!(asm.contains("L_block_loop_exit_"));

        // Check comparison and branching
        assert!(asm.contains("cmpq"));
        assert!(asm.contains("setl %al"));
    }

    #[test]
    fn test_nested_if_statements() {
        let ir = r#"
            fn @test_nested_if(i64 %x, i64 %y) -> i64 {
                entry:
                    %cond1 = gt.i64 %x, 10
                    br %cond1, if_true, if_false
                    
                if_true:
                    %cond2 = lt.i64 %y, 5
                    br %cond2, nested_true, nested_false
                    
                nested_true:
                    ret.i64 1
                    
                nested_false:
                    ret.i64 2
                    
                if_false:
                    %cond3 = eq.i64 %y, 0
                    br %cond3, if_false_true, if_false_false
                    
                if_false_true:
                    ret.i64 3
                    
                if_false_false:
                    ret.i64 4
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check all the basic blocks exist
        assert!(asm.contains("L_block_entry_"));
        assert!(asm.contains("L_block_if_true_"));
        assert!(asm.contains("L_block_if_false_"));
        assert!(asm.contains("L_block_nested_true_"));
        assert!(asm.contains("L_block_nested_false_"));
        assert!(asm.contains("L_block_if_false_true_"));
        assert!(asm.contains("L_block_if_false_false_"));

        // Check different comparison operators
        assert!(asm.contains("setg %al"));
        assert!(asm.contains("setl %al"));
        assert!(asm.contains("sete %al"));
    }

    #[test]
    fn test_exported_function() {
        let ir = r#"
            @export
            fn @main() -> i64 {
                entry:
                    ret.i64 42
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check that main is exported properly
        assert!(asm.contains(".globl main"));
        assert!(asm.contains("main:"));
        assert!(asm.contains("movq $42, %rax"));
    }

    #[test]
    fn test_many_function_args() {
        let ir = r#"
            fn @many_args(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) -> i64 {
                entry:
                    %sum1 = add.i64 %a, %b
                    %sum2 = add.i64 %c, %d
                    %sum3 = add.i64 %e, %f
                    %sum4 = add.i64 %g, %h
                    
                    %sum12 = add.i64 %sum1, %sum2
                    %sum34 = add.i64 %sum3, %sum4
                    
                    %result = add.i64 %sum12, %sum34
                    ret.i64 %result
            }

            fn @call_many_args() -> i64 {
                entry:
                    %result = call @many_args(1, 2, 3, 4, 5, 6, 7, 8)
                    ret.i64 %result
            }
        "#;
        let asm = compile_to_asm(ir);

        // Check register args
        assert!(asm.contains("movq %rdi,")); // arg a
        assert!(asm.contains("movq %rsi,")); // arg b
        assert!(asm.contains("movq %rdx,")); // arg c
        assert!(asm.contains("movq %rcx,")); // arg d
        assert!(asm.contains("movq %r8,")); // arg e
        assert!(asm.contains("movq %r9,")); // arg f

        // Check stack args (g and h)
        assert!(asm.contains("16(%rbp)") || asm.contains("24(%rbp)"));

        // Check calling with many args
        assert!(asm.contains("movq $1, %rdi"));
        assert!(asm.contains("movq $2, %rsi"));
        assert!(asm.contains("movq $7, ")); // Stack arg
        assert!(asm.contains("movq $8, ")); // Stack arg
    }

    #[test]
    fn test_block_ordering() {
        let ir = r#"
            fn @fibonacci(i64 %n) -> i64 {
                entry:
                    %cmp_le1 = le.i64 %n, 1
                    br %cmp_le1, base_case, recursive_step

                base_case:
                    ret.i64 %n

                recursive_step:
                    %n_minus_1 = sub.i64 %n, 1
                    %n_minus_2 = sub.i64 %n, 2
                    %fib1 = call @fibonacci(%n_minus_1)
                    %fib2 = call @fibonacci(%n_minus_2)
                    %sum = add.i64 %fib1, %fib2
                    ret.i64 %sum
            }
        "#;
        let asm = compile_to_asm(ir);

        // Extract all block labels for the fibonacci function
        let _block_labels: Vec<&str> = asm
            .lines()
            .filter(|line| {
                line.contains(".L_block_")
                    && !line.contains(":")
                    && !line.contains("jmp")
                    && !line.contains("je")
            })
            .collect();

        // Find the first occurrence of each unique block type
        let mut first_entry = None;
        let mut first_base_case = None;
        let mut first_recursive_step = None;

        for (i, line) in asm.lines().enumerate() {
            if line.contains(".L_block_entry_") && line.ends_with(":") && first_entry.is_none() {
                first_entry = Some(i);
            } else if line.contains(".L_block_base_case_")
                && line.ends_with(":")
                && first_base_case.is_none()
            {
                first_base_case = Some(i);
            } else if line.contains(".L_block_recursive_step_")
                && line.ends_with(":")
                && first_recursive_step.is_none()
            {
                first_recursive_step = Some(i);
            }
        }

        // Check that we found all block types
        assert!(first_entry.is_some(), "Entry block not found");
        assert!(first_base_case.is_some(), "Base case block not found");
        assert!(
            first_recursive_step.is_some(),
            "Recursive step block not found"
        );

        // Check that entry block is first
        // Safe to unwrap because we asserted these are Some above
        let entry_pos = first_entry.expect("Entry block position should be available");
        let base_case_pos = first_base_case.expect("Base case block position should be available");
        let recursive_step_pos =
            first_recursive_step.expect("Recursive step block position should be available");

        // Entry should come before both others
        assert!(
            entry_pos < base_case_pos && entry_pos < recursive_step_pos,
            "Entry block should come before other blocks"
        );
    }
}
