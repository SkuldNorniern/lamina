use super::state::{CodegenState, FunctionContext};
use crate::{Instruction, Result, Value};
use std::collections::{HashMap, HashSet};

/// Represents an optimized load/store operation
#[derive(Debug)]
pub enum OptimizedOperation<'a> {
    OriginalInstruction(&'a Instruction<'a>),
    RegisterLoad {
        dest_reg: String,
        src_mem: String,
        comment: String,
    },
    RegisterStore {
        src_reg: String,
        dest_mem: String,
        comment: String,
    },
    RegisterToRegister {
        dest_reg: String,
        src_reg: String,
        comment: String,
    },
    DirectOperation {
        op: String,
        dst: String,
        src: String,
        comment: String,
    },
    Eliminated {
        comment: String,
    },
}

/// Optimizes load-store operations for a sequence of instructions
pub fn optimize_load_store<'a>(
    instructions: &'a [Instruction<'a>],
    func_ctx: &FunctionContext<'a>,
    _state: &mut CodegenState<'a>,
) -> Result<Vec<OptimizedOperation<'a>>> {
    let mut optimized_ops = Vec::with_capacity(instructions.len());

    // Maps memory locations to registers containing their values
    let mut memory_to_reg: HashMap<&'a str, String> = HashMap::new();
    // Maps variable names to their last computed value
    let mut var_loaded_values: HashMap<&'a str, &'a str> = HashMap::new();
    // Maps registers to the variables they contain
    let mut reg_to_var: HashMap<String, &'a str> = HashMap::new();
    // Track which values are dead (no longer used)
    let mut live_vars: HashSet<&'a str> = HashSet::new();

    // Pre-analyze to find all variable uses and collect live variables
    for instr in instructions {
        match instr {
            Instruction::Load { result, ptr, .. } => {
                live_vars.insert(result);
                if let Value::Variable(ptr_var) = ptr {
                    live_vars.insert(ptr_var);
                }
            }
            Instruction::Store { ptr, value, .. } => {
                if let Value::Variable(ptr_var) = ptr {
                    live_vars.insert(ptr_var);
                }
                if let Value::Variable(value_var) = value {
                    live_vars.insert(value_var);
                }
            }
            Instruction::Binary {
                result, lhs, rhs, ..
            } => {
                live_vars.insert(result);
                if let Value::Variable(lhs_var) = lhs {
                    live_vars.insert(lhs_var);
                }
                if let Value::Variable(rhs_var) = rhs {
                    live_vars.insert(rhs_var);
                }
            }
            Instruction::Call { result, args, .. } => {
                if let Some(res) = result {
                    live_vars.insert(res);
                }
                for arg in args {
                    if let Value::Variable(arg_var) = arg {
                        live_vars.insert(arg_var);
                    }
                }
            }
            Instruction::Cmp {
                result, lhs, rhs, ..
            } => {
                live_vars.insert(result);
                if let Value::Variable(lhs_var) = lhs {
                    live_vars.insert(lhs_var);
                }
                if let Value::Variable(rhs_var) = rhs {
                    live_vars.insert(rhs_var);
                }
            }
            // Instead of trying to pattern match all variants with exact fields,
            // let's collect all variables from all instructions to be safe
            _ => {
                // Extract all variables from any instruction type
                collect_variables_from_instruction(instr, &mut live_vars);
            }
        }
    }

    for instr in instructions {
        match instr {
            Instruction::Load { result, ptr, .. } => {
                // Try to optimize loads by reusing already loaded values
                if let Value::Variable(ptr_var) = ptr {
                    // Check if we already have the value in a register
                    if let Some(reg) = memory_to_reg.get(ptr_var) {
                        let result_loc = func_ctx.get_value_location(result)?.to_operand_string();

                        // Direct register move - major optimization
                        optimized_ops.push(OptimizedOperation::RegisterToRegister {
                            dest_reg: "%rax".to_string(),
                            src_reg: reg.clone(),
                            comment: format!("Reusing cached value for {}", result),
                        });
                        optimized_ops.push(OptimizedOperation::RegisterStore {
                            src_reg: "%rax".to_string(),
                            dest_mem: result_loc,
                            comment: format!("Store to result {}", result),
                        });

                        // Remember that this result variable now has the value from ptr_var
                        var_loaded_values.insert(*result, *ptr_var);
                        continue;
                    }

                    // Check if this is just reloading a previously stored value
                    if let Some(src_var) = var_loaded_values.get(ptr_var) {
                        // Fix the reference pattern problem by using a different approach
                        let mut found_reg = None;
                        for (reg, var) in &reg_to_var {
                            if var == src_var {
                                found_reg = Some(reg.clone());
                                break;
                            }
                        }

                        if let Some(src_reg) = found_reg {
                            let result_loc =
                                func_ctx.get_value_location(result)?.to_operand_string();

                            optimized_ops.push(OptimizedOperation::RegisterToRegister {
                                dest_reg: "%rax".to_string(),
                                src_reg,
                                comment: format!(
                                    "Reusing value chained through memory for {}",
                                    result
                                ),
                            });
                            optimized_ops.push(OptimizedOperation::RegisterStore {
                                src_reg: "%rax".to_string(),
                                dest_mem: result_loc,
                                comment: format!("Store to result {}", result),
                            });

                            var_loaded_values.insert(*result, *src_var);
                            continue;
                        }
                    }
                }

                // If we couldn't optimize, use the original instruction
                optimized_ops.push(OptimizedOperation::OriginalInstruction(instr));

                // After a load, remember the relation: result = load(ptr)
                if let Value::Variable(ptr_var) = ptr {
                    var_loaded_values.insert(*result, *ptr_var);
                }
            }
            Instruction::Store { ptr, value, .. } => {
                // Check if we're storing to a location that nobody reads from
                if let Value::Variable(ptr_var) = ptr
                    && !live_vars.contains(ptr_var) {
                        optimized_ops.push(OptimizedOperation::Eliminated {
                            comment: format!("Eliminated dead store to {}", ptr_var),
                        });
                        continue;
                    }

                // For Store, remember that the register now contains this memory value
                if let (Value::Variable(ptr_var), Value::Variable(value_var)) = (ptr, value) {
                    memory_to_reg.insert(*ptr_var, format!("%{}", value_var));

                    // Also remember which variable each register holds
                    reg_to_var.insert(format!("%{}", value_var), *value_var);
                }

                // Use the original instruction
                optimized_ops.push(OptimizedOperation::OriginalInstruction(instr));
            }
            // Control flow changes clear our knowledge about registers
            Instruction::Call { .. }
            | Instruction::Br { .. }
            | Instruction::Jmp { .. }
            | Instruction::Ret { .. } => {
                memory_to_reg.clear();
                var_loaded_values.clear();
                reg_to_var.clear();
                optimized_ops.push(OptimizedOperation::OriginalInstruction(instr));
            }
            // For all other instructions, just use the original
            _ => {
                optimized_ops.push(OptimizedOperation::OriginalInstruction(instr));
            }
        }
    }

    Ok(optimized_ops)
}

// Helper function to collect variable identifiers from any instruction type
fn collect_variables_from_instruction<'a>(
    instr: &Instruction<'a>,
    live_vars: &mut HashSet<&'a str>,
) {
    match instr {
        Instruction::Br { condition, .. } => {
            // Add the condition variable if it's a variable
            if let Value::Variable(var) = condition {
                live_vars.insert(var);
            }
        }
        Instruction::Ret { value, .. } => {
            // Check if there's a return value that's a variable
            if let Some(val) = value
                && let Value::Variable(var) = val {
                    live_vars.insert(var);
                }
        }
        // We already handle Load, Store, Binary, Call, and Cmp separately
        // Add any other cases here as needed
        _ => {
            // For instructions we don't explicitly handle, don't add any variables
        }
    }
}
