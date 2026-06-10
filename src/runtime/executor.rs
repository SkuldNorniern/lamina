//! JIT function execution
//!
//! Functions to execute JIT-compiled functions with dynamic argument counts
//! using the platform C ABI.

use crate::mir::{
    Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register, ScalarType,
    Signature,
};
use crate::runtime::c_abi_dynamic::{MAX_JIT_ARGS, call_function_dynamic};
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::collections::HashMap;
use std::env;
use std::error::Error;

fn evaluate_operand(
    operand: &Operand,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, Box<dyn Error>> {
    match operand {
        Operand::Immediate(Immediate::I8(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I16(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I32(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I64(value)) => Ok(*value),
        Operand::Immediate(Immediate::F32(_)) => {
            Err("Interpreter: f32 immediate not supported".into())
        }
        Operand::Immediate(Immediate::F64(_)) => {
            Err("Interpreter: f64 immediate not supported".into())
        }
        Operand::Register(register) => register_values
            .get(register)
            .copied()
            .ok_or_else(|| format!("Interpreter: missing register value for {register}").into()),
    }
}

fn interpret_mir_function(
    function: &Function,
    args: &[i64],
) -> Result<Option<i64>, Box<dyn Error>> {
    if function.sig.params.len() != args.len() {
        return Err(format!(
            "Interpreter: expected {} arguments, got {}",
            function.sig.params.len(),
            args.len()
        )
        .into());
    }

    let mut register_values: HashMap<Register, i64> = HashMap::new();
    for (param, value) in function.sig.params.iter().zip(args.iter().copied()) {
        register_values.insert(param.reg.clone(), value);
    }

    let mut block_index_by_label: HashMap<String, usize> = HashMap::new();
    for (index, block) in function.blocks.iter().enumerate() {
        block_index_by_label.insert(block.label.clone(), index);
    }

    let mut current_block_index = block_index_by_label
        .get(&function.entry)
        .copied()
        .ok_or_else(|| format!("Interpreter: entry block '{}' not found", function.entry))?;

    loop {
        let block = &function.blocks[current_block_index];
        let mut next_label: Option<String> = None;

        for instruction in &block.instructions {
            match instruction {
                Instruction::IntBinary {
                    op, dst, lhs, rhs, ..
                } => {
                    let left_value = evaluate_operand(lhs, &register_values)?;
                    let right_value = evaluate_operand(rhs, &register_values)?;
                    let result = match op {
                        IntBinOp::Add => left_value.wrapping_add(right_value),
                        IntBinOp::Sub => left_value.wrapping_sub(right_value),
                        IntBinOp::Mul => left_value.wrapping_mul(right_value),
                        IntBinOp::UDiv => {
                            if right_value == 0 {
                                return Err("Interpreter: division by zero".into());
                            }
                            (left_value as u64 / right_value as u64) as i64
                        }
                        IntBinOp::SDiv => {
                            if right_value == 0 {
                                return Err("Interpreter: division by zero".into());
                            }
                            left_value.wrapping_div(right_value)
                        }
                        IntBinOp::URem => {
                            if right_value == 0 {
                                return Err("Interpreter: remainder by zero".into());
                            }
                            (left_value as u64 % right_value as u64) as i64
                        }
                        IntBinOp::SRem => {
                            if right_value == 0 {
                                return Err("Interpreter: remainder by zero".into());
                            }
                            left_value.wrapping_rem(right_value)
                        }
                        IntBinOp::And => left_value & right_value,
                        IntBinOp::Or => left_value | right_value,
                        IntBinOp::Xor => left_value ^ right_value,
                        IntBinOp::Shl => left_value.wrapping_shl(right_value as u32),
                        IntBinOp::LShr => ((left_value as u64) >> (right_value as u32)) as i64,
                        IntBinOp::AShr => left_value.wrapping_shr(right_value as u32),
                    };
                    register_values.insert(dst.clone(), result);
                }
                Instruction::IntCmp {
                    op, dst, lhs, rhs, ..
                } => {
                    let left_value = evaluate_operand(lhs, &register_values)?;
                    let right_value = evaluate_operand(rhs, &register_values)?;
                    let result = match op {
                        IntCmpOp::Eq => left_value == right_value,
                        IntCmpOp::Ne => left_value != right_value,
                        IntCmpOp::ULt => (left_value as u64) < (right_value as u64),
                        IntCmpOp::ULe => (left_value as u64) <= (right_value as u64),
                        IntCmpOp::UGt => (left_value as u64) > (right_value as u64),
                        IntCmpOp::UGe => (left_value as u64) >= (right_value as u64),
                        IntCmpOp::SLt => left_value < right_value,
                        IntCmpOp::SLe => left_value <= right_value,
                        IntCmpOp::SGt => left_value > right_value,
                        IntCmpOp::SGe => left_value >= right_value,
                    };
                    register_values.insert(dst.clone(), if result { 1 } else { 0 });
                }
                Instruction::Ret { value } => {
                    let result = match value {
                        Some(operand) => Some(evaluate_operand(operand, &register_values)?),
                        None => None,
                    };
                    return Ok(result);
                }
                Instruction::Jmp { target } => {
                    next_label = Some(target.clone());
                    break;
                }
                Instruction::Br {
                    cond,
                    true_target,
                    false_target,
                } => {
                    let condition_value = register_values
                        .get(cond)
                        .copied()
                        .ok_or_else(|| format!("Interpreter: missing condition register {cond}"))?;
                    let target = if condition_value != 0 {
                        true_target.clone()
                    } else {
                        false_target.clone()
                    };
                    next_label = Some(target);
                    break;
                }
                Instruction::Switch {
                    value,
                    cases,
                    default,
                } => {
                    let switch_value = register_values
                        .get(value)
                        .copied()
                        .ok_or_else(|| format!("Interpreter: missing switch register {value}"))?;
                    let mut target = default.clone();
                    for (case_value, label) in cases {
                        if *case_value == switch_value {
                            target = label.clone();
                            break;
                        }
                    }
                    next_label = Some(target);
                    break;
                }
                Instruction::Unreachable => {
                    return Err("Interpreter: unreachable executed".into());
                }
                Instruction::Call { .. } => {
                    return Err("Interpreter: call not supported".into());
                }
                Instruction::TailCall { .. } => {
                    return Err("Interpreter: tail call not supported".into());
                }
                Instruction::Comment { .. } => {}
                Instruction::Load { .. }
                | Instruction::Store { .. }
                | Instruction::Lea { .. }
                | Instruction::FloatBinary { .. }
                | Instruction::FloatUnary { .. }
                | Instruction::FloatCmp { .. }
                | Instruction::Select { .. }
                | Instruction::VectorOp { .. }
                | Instruction::SafePoint
                | Instruction::StackMap { .. }
                | Instruction::PatchPoint { .. } => {
                    return Err("Interpreter: unsupported instruction".into());
                }
                #[cfg(feature = "nightly")]
                Instruction::SimdBinary { .. }
                | Instruction::SimdUnary { .. }
                | Instruction::SimdTernary { .. }
                | Instruction::SimdShuffle { .. }
                | Instruction::SimdExtract { .. }
                | Instruction::SimdInsert { .. }
                | Instruction::SimdLoad { .. }
                | Instruction::SimdStore { .. }
                | Instruction::AtomicLoad { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::AtomicBinary { .. }
                | Instruction::AtomicCompareExchange { .. }
                | Instruction::Fence { .. } => {
                    return Err("Interpreter: SIMD/Atomic instructions not supported".into());
                }
            }
        }

        if let Some(label) = next_label {
            current_block_index = block_index_by_label
                .get(&label)
                .copied()
                .ok_or_else(|| format!("Interpreter: unknown block label {label}"))?;
        } else {
            return Err(format!("Interpreter: block '{}' missing terminator", block.label).into());
        }
    }
}

/// Executes a JIT-compiled function using its signature to determine calling convention.
///
/// Validates the function signature and calls it with the provided arguments.
/// Currently, handles only i64 parameters and i64 or void return types.
///
/// # Arguments
///
/// * `sig` - Function signature from MIR
/// * `function_ptr` - Raw pointer to the compiled function
/// * `args` - Arguments to pass to the function (defaults to zeros if not provided)
/// * `verbose` - Whether to print execution details
///
/// # Errors
///
/// Returns an error if:
/// - Function has non-i64 parameters
/// - Function has unsupported return type
/// - Function should return a value but didn't
///
/// # Safety
///
/// This function is unsafe because it calls code at an arbitrary function pointer.
/// The caller must ensure:
/// - `function_ptr` points to valid, executable code
/// - The function signature matches `sig`
/// - The function pointer remains valid for the duration of the call
pub unsafe fn execute_jit_function(
    sig: &Signature,
    function_ptr: *const u8,
    args: Option<&[i64]>,
    verbose: bool,
    function: Option<&Function>,
) -> Result<Option<i64>, Box<dyn Error>> {
    let param_count = sig.params.len();

    if param_count > MAX_JIT_ARGS {
        return Err(format!(
            "JIT execution: Handles up to {MAX_JIT_ARGS} parameters, got {param_count}"
        )
        .into());
    }

    let all_i64 = sig
        .params
        .iter()
        .all(|p| matches!(p.ty, MirType::Scalar(ScalarType::I64)));

    let returns_i64 = matches!(sig.ret_ty.as_ref(), Some(MirType::Scalar(ScalarType::I64)));
    let returns_void = sig.ret_ty.is_none();

    if !all_i64 {
        return Err(format!(
            "JIT execution: Function has non-i64 parameters, not yet supported. Parameter types: {:?}",
            sig.params.iter().map(|p| &p.ty).collect::<Vec<_>>()
        ).into());
    }

    if !returns_i64 && !returns_void {
        return Err(format!("JIT execution: Unsupported return type: {:?}", sig.ret_ty).into());
    }

    #[cfg(target_arch = "aarch64")]
    {
        let sp: usize;
        unsafe {
            asm!("mov {}, sp", out(reg) sp);
        }
        if !sp.is_multiple_of(16) && verbose {
            eprintln!(
                "[WARNING] Stack pointer is not 16-byte aligned: SP={:p}, alignment={} bytes",
                sp as *const u8,
                sp % 16
            );
            eprintln!("[WARNING] This may cause the STP instruction to fault!");
        }
    }

    let default_args: Vec<i64> = vec![0; param_count];
    let args = args.unwrap_or(&default_args);

    if args.len() != param_count {
        return Err(format!(
            "JIT execution: Argument count mismatch. Expected {}, got {}",
            param_count,
            args.len()
        )
        .into());
    }

    // Historical safety valve: the AArch64 encoder used to be incomplete, and we interpreted MIR.
    // Keep this behavior opt-in for debugging, but default to executing the generated code.
    if let Some(function) = function
        && env::var_os("LAMINA_JIT_INTERPRET").is_some()
    {
        if verbose {
            eprintln!(
                "[JIT] LAMINA_JIT_INTERPRET=1 set; interpreting MIR instead of executing JIT code"
            );
        }
        return interpret_mir_function(function, args);
    }

    if verbose {
        println!(
            "[JIT] Calling function with {} i64 parameter(s), {} return...",
            param_count,
            if returns_i64 { "i64" } else { "void" }
        );
    }

    unsafe {
        let result = call_function_dynamic(function_ptr, args, returns_i64);

        if returns_i64 {
            if let Some(value) = result {
                if verbose {
                    println!("[JIT] Function returned: {value}");
                }
                Ok(Some(value))
            } else {
                Err("JIT execution: Function should have returned a value but didn't".into())
            }
        } else {
            if verbose {
                println!("[JIT] Function executed successfully (void return)");
            }
            Ok(None)
        }
    }
}
