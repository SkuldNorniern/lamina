use crate::{
    Function, BasicBlock, Identifier, Type, PrimitiveType, Instruction, FunctionAnnotation, Result
};
use super::state::{CodegenState, FunctionContext, ValueLocation, ARG_REGISTERS};
use super::instructions::generate_instruction;
use super::util::get_type_size_directive_and_bytes;
use std::io::Write;

// Generate the assembly for all functions in the module
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

// Generate assembly for a single function
pub fn generate_function<'a, W: Write>(
    func_name: Identifier<'a>,
    func: &'a Function<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    // Determine assembly label
    let is_exported = func.annotations.contains(&FunctionAnnotation::Export);
    let asm_label = if is_exported && func_name == "main" {
        "main".to_string() // Special case for C runtime entry point
    } else {
        format!("func_{}", func_name)
    };

    // Create context and determine epilogue label *before* generating blocks
    let mut func_ctx = FunctionContext::new();
    func_ctx.epilogue_label = format!(".Lfunc_epilogue_{}", func_name); // Set epilogue label early

    // Function Prologue
    writeln!(writer, "\n# Function: @{}", func_name)?;
    if is_exported {
        writeln!(writer, ".globl {}", asm_label)?;
    }
    writeln!(writer, ".type {}, @function", asm_label)?;
    writeln!(writer, ".align 16")?; // Ensure function entry is aligned
    writeln!(writer, "{}:", asm_label)?;
    writeln!(writer, "    pushq %rbp")?;
    writeln!(writer, "    movq %rsp, %rbp")?;
    // Save callee-saved registers
    writeln!(writer, "    pushq %rbx")?;
    writeln!(writer, "    pushq %r12")?;
    writeln!(writer, "    pushq %r13")?;
    writeln!(writer, "    pushq %r14")?;
    writeln!(writer, "    pushq %r15")?;

    // Calculate stack layout (offsets relative to %rbp)
    precompute_function_layout(func, &mut func_ctx, state)?; // Populate context

    // Calculate required stack size and ensure alignment for calls
    let needed_bytes = func_ctx.total_stack_size;
    // We need the stack pointer (%rsp) to be 16-byte aligned *before* the call instruction.
    // RSP is 16-byte aligned after pushing rbp + 5 regs (48 bytes total).
    // Subtracting frame_size must *maintain* this 16-byte alignment.
    // Therefore, frame_size must be a multiple of 16.
    // We need the smallest `frame_size >= needed_bytes` that is a multiple of 16.
    let frame_size = (needed_bytes + 15) & !15; // Round up to multiple of 16

    if frame_size > 0 {
        writeln!(writer, "    subq ${}, %rsp", frame_size)?;
    }

    // Spill argument registers to stack slots if necessary
    writeln!(writer, "    # Spill argument registers to stack slots")?;
    for (i, arg) in func.signature.params.iter().enumerate() {
        if let Some(loc) = func_ctx.value_locations.get(arg.name) {
            if let ValueLocation::StackOffset(offset) = loc {
                if i < ARG_REGISTERS.len() {
                    // TODO: Handle non-64bit arg types (movl etc.)
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
    }

    // Function Body Generation
    writeln!(writer, "    # Function Body Start")?;
    
    // Process the entry block first to ensure correct control flow
    if let Some(entry_block) = func.basic_blocks.get(&func.entry_block) {
        let asm_label = func_ctx.get_block_label(&func.entry_block)?;
        writeln!(writer, "{}:", asm_label)?;
        generate_basic_block(entry_block, writer, state, &func_ctx, func_name)?;
    }
    
    // Process the remaining blocks
    for (ir_label, block) in &func.basic_blocks {
        // Skip the entry block since we've already processed it
        if *ir_label != func.entry_block {
            let asm_label = func_ctx.get_block_label(ir_label)?;
            writeln!(writer, "{}:", asm_label)?;
            generate_basic_block(block, writer, state, &func_ctx, func_name)?;
        }
    }
    
    writeln!(writer, "    # Function Body End")?;

    // Unified Function Epilogue
    writeln!(writer, "{}:", func_ctx.epilogue_label)?;
    writeln!(writer, "    # Restore callee-saved registers")?;
    writeln!(writer, "    popq %r15")?;
    writeln!(writer, "    popq %r14")?;
    writeln!(writer, "    popq %r13")?;
    writeln!(writer, "    popq %r12")?;
    writeln!(writer, "    popq %rbx")?;
    // Restore stack pointer and base pointer
    writeln!(writer, "    leave")?;
    writeln!(writer, "    ret")?;

    // Add size directive for debugging/analysis
    writeln!(writer, ".size {}, .-{}\n", asm_label, asm_label)?;

    Ok(())
}

// Pass to calculate stack layout and assign assembly labels to IR blocks
fn precompute_function_layout<'a>(
    func: &'a Function<'a>,
    func_ctx: &mut FunctionContext<'a>,
    state: &mut CodegenState<'a>,
) -> Result<()> {
    let mut current_stack_offset: i64;
    let mut current_param_stack_offset = 16i64; // For args passed via stack (> 6th arg)

    let _saved_regs_size = 48i64; // rbx, r12-r15 (5 * 8 = 40) + rbp (8) - Unused for now
    // NOTE: The calculation here assumes we only save rbx, r12-r15. If more are saved, update saved_regs_size.
    // We actually push rbp separately before saving these, so the space used below the *new* rbp is 40 bytes (rbp-8 to rbp-40)
    let saved_callee_regs_size = 40i64; // rbx, r12-r15 pushed *after* new rbp established.

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

    // 2. Assign assembly labels to IR blocks
    for ir_label in func.basic_blocks.keys() {
        let asm_label = state.new_label(&format!("block_{}", ir_label));
        func_ctx.block_labels.insert(ir_label, asm_label);
    }

    // 3. Calculate stack space for locals
    let mut local_size = 0u64;
    let mut local_allocations = Vec::new(); // Store (result_name, size)
    for block in func.basic_blocks.values() {
        for instr in &block.instructions {
             let result_info: Option<(&Identifier<'a>, u64)> = match instr {
                 Instruction::Alloc { result, allocated_ty, .. } => { let (_,s)=get_type_size_directive_and_bytes(allocated_ty)?; Some((result,s)) },
                 Instruction::Binary { result, ty, .. } | Instruction::Cmp { result, ty, .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(*ty))?; Some((result,s)) },
                 Instruction::ZeroExtend { result, target_type, .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(*target_type))?; Some((result,s)) },
                 Instruction::Load { result, ty, .. } => { let (_,s)=get_type_size_directive_and_bytes(ty)?; Some((result,s)) },
                 Instruction::GetFieldPtr { result, .. } | Instruction::GetElemPtr { result, .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?; Some((result,s)) },
                 Instruction::Tuple { result, .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?; Some((result,s)) },
                 Instruction::ExtractTuple { result, .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?; Some((result,s)) },
                 Instruction::Phi { result, ty, .. } => { let (_,s)=get_type_size_directive_and_bytes(ty)?; Some((result,s)) },
                 Instruction::Call { result: Some(res), .. } => { let (_,s)=get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr))?; Some((res,s)) },
                 _ => None, // Only care about instructions with results
            };

            if let Some((result, size)) = result_info {
                 let aligned_size = (size + 7) & !7;
                 local_size += aligned_size;
                 local_allocations.push((result, aligned_size));
            }
        }
    }

    // 4. Assign stack offsets for locals, starting below the saved callee registers
    current_stack_offset = -(saved_callee_regs_size + local_size as i64);
    let local_start_offset = current_stack_offset;
    for (result, aligned_size) in local_allocations {
         func_ctx.value_locations.insert(result, ValueLocation::StackOffset(current_stack_offset));
         current_stack_offset += aligned_size as i64; // Move offset *up* towards -saved_callee_regs_size
    }
    
    // Reset offset to start allocating spill slots below locals
    current_stack_offset = local_start_offset;

    // 5. Allocate stack spill slots for register parameters and record them
    for (param_name, initial_location) in temp_param_locations {
        if let ValueLocation::Register(reg_name) = initial_location {
            let param_sig = func.signature.params.iter().find(|p| p.name == param_name).unwrap(); // Find param to get type
            let (_, size) = get_type_size_directive_and_bytes(&param_sig.ty)?;
            let aligned_size = (size + 7) & !7;
            current_stack_offset -= aligned_size as i64; // Allocate space below locals/previous spills
            func_ctx.arg_register_spills.insert(reg_name.clone(), current_stack_offset);
            // Final location for parameters (even register ones) is their spill slot
            func_ctx.value_locations.insert(param_name, ValueLocation::StackOffset(current_stack_offset));
        } else {
            // Parameter was already on the stack, just insert its location
             func_ctx.value_locations.insert(param_name, initial_location);
        }
    }

    // total_stack_size represents the number of bytes to allocate *below* the saved callee registers.
    // It's the difference between the base offset (-40) and the lowest offset reached.
    func_ctx.total_stack_size = (-current_stack_offset - saved_callee_regs_size) as u64;

    Ok(())
}

// Generate assembly for a single basic block
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

#[cfg(test)]
mod tests {
    use crate::parser::parse_module;
    use crate::codegen::x86_64::generate_x86_64_assembly;
    use super::*;
    // Add imports needed specifically for test_codegen_tensor_benchmark
    use std::fs;
    use std::io::{BufWriter, Write};
    use crate::codegen::x86_64; // To call x86_64::generate_x86_64_assembly
    use crate::Result; // Needed for the test function signature

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
                    %product = mul.i64 %diff, 2
                    %quotient = div.i64 %product, 3
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
            fn @test_types() -> i64 {
                entry:
                    %bool_val = eq.i32 5, 5
                    %i32_val = add.i32 42, 10
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
        assert!(asm.contains("movq %rdi,"));  // arg a
        assert!(asm.contains("movq %rsi,"));  // arg b
        assert!(asm.contains("movq %rdx,"));  // arg c
        assert!(asm.contains("movq %rcx,"));  // arg d
        assert!(asm.contains("movq %r8,"));   // arg e
        assert!(asm.contains("movq %r9,"));   // arg f
        
        // Check stack args (g and h)
        assert!(asm.contains("16(%rbp)") || asm.contains("24(%rbp)"));
        
        // Check calling with many args
        assert!(asm.contains("movq $1, %rdi"));
        assert!(asm.contains("movq $2, %rsi"));
        assert!(asm.contains("movq $7, "));   // Stack arg
        assert!(asm.contains("movq $8, "));   // Stack arg
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
        let block_labels: Vec<&str> = asm.lines()
            .filter(|line| line.contains(".L_block_") && !line.contains(":") && !line.contains("jmp") && !line.contains("je"))
            .collect();
            
        // Find the first occurrence of each unique block type
        let mut first_entry = None;
        let mut first_base_case = None;
        let mut first_recursive_step = None;
        
        for (i, line) in asm.lines().enumerate() {
            if line.contains(".L_block_entry_") && line.ends_with(":") && first_entry.is_none() {
                first_entry = Some(i);
            } else if line.contains(".L_block_base_case_") && line.ends_with(":") && first_base_case.is_none() {
                first_base_case = Some(i);
            } else if line.contains(".L_block_recursive_step_") && line.ends_with(":") && first_recursive_step.is_none() {
                first_recursive_step = Some(i);
            }
        }
        
        // Check that we found all block types
        assert!(first_entry.is_some(), "Entry block not found");
        assert!(first_base_case.is_some(), "Base case block not found");
        assert!(first_recursive_step.is_some(), "Recursive step block not found");
        
        // Check that entry block is first
        let entry_pos = first_entry.unwrap();
        let base_case_pos = first_base_case.unwrap();
        let recursive_step_pos = first_recursive_step.unwrap();
        
        // Entry should come before both others
        assert!(entry_pos < base_case_pos && entry_pos < recursive_step_pos, 
                "Entry block should come before other blocks");
    }
}
