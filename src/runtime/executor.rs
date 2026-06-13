//! JIT function execution
//!
//! Functions to execute JIT-compiled functions with dynamic argument counts
//! using the platform C ABI.

use crate::error::LaminaError;
use crate::mir::{
    Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register, ScalarType,
    Signature,
};
use crate::runtime::c_abi_dynamic::{MAX_JIT_ARGS, call_function_dynamic};
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::collections::HashMap;
use std::env;

fn evaluate_operand(
    operand: &Operand,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, LaminaError> {
    match operand {
        Operand::Immediate(Immediate::I8(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I16(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I32(value)) => Ok(*value as i64),
        Operand::Immediate(Immediate::I64(value)) => Ok(*value),
        Operand::Immediate(Immediate::F32(_)) => Err(LaminaError::RuntimeError(
            "Interpreter: f32 immediate not supported".to_owned(),
        )),
        Operand::Immediate(Immediate::F64(_)) => Err(LaminaError::RuntimeError(
            "Interpreter: f64 immediate not supported".to_owned(),
        )),
        Operand::Register(register) => register_values.get(register).copied().ok_or_else(|| {
            LaminaError::RuntimeError(format!(
                "Interpreter: missing register value for {register}"
            ))
        }),
    }
}

fn interpret_mir_function(function: &Function, args: &[i64]) -> Result<Option<i64>, LaminaError> {
    if function.sig.params.len() != args.len() {
        return Err(LaminaError::RuntimeError(format!(
            "Interpreter: expected {} arguments, got {}",
            function.sig.params.len(),
            args.len()
        )));
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
        .ok_or_else(|| {
            LaminaError::RuntimeError(format!(
                "Interpreter: entry block '{}' not found",
                function.entry
            ))
        })?;

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
                                return Err(LaminaError::RuntimeError(
                                    "Interpreter: division by zero".to_owned(),
                                ));
                            }
                            (left_value as u64 / right_value as u64) as i64
                        }
                        IntBinOp::SDiv => {
                            if right_value == 0 {
                                return Err(LaminaError::RuntimeError(
                                    "Interpreter: division by zero".to_owned(),
                                ));
                            }
                            left_value.wrapping_div(right_value)
                        }
                        IntBinOp::URem => {
                            if right_value == 0 {
                                return Err(LaminaError::RuntimeError(
                                    "Interpreter: remainder by zero".to_owned(),
                                ));
                            }
                            (left_value as u64 % right_value as u64) as i64
                        }
                        IntBinOp::SRem => {
                            if right_value == 0 {
                                return Err(LaminaError::RuntimeError(
                                    "Interpreter: remainder by zero".to_owned(),
                                ));
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
                    let condition_value = register_values.get(cond).copied().ok_or_else(|| {
                        LaminaError::RuntimeError(format!(
                            "Interpreter: missing condition register {cond}"
                        ))
                    })?;
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
                    let switch_value = register_values.get(value).copied().ok_or_else(|| {
                        LaminaError::RuntimeError(format!(
                            "Interpreter: missing switch register {value}"
                        ))
                    })?;
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
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: unreachable executed".to_owned(),
                    ));
                }
                Instruction::Call { .. } => {
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: call not supported".to_owned(),
                    ));
                }
                Instruction::TailCall { .. } => {
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: tail call not supported".to_owned(),
                    ));
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
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: unsupported instruction".to_owned(),
                    ));
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
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: SIMD/Atomic instructions not supported".to_owned(),
                    ));
                }
            }
        }

        if let Some(label) = next_label {
            current_block_index = block_index_by_label.get(&label).copied().ok_or_else(|| {
                LaminaError::RuntimeError(format!("Interpreter: unknown block label {label}"))
            })?;
        } else {
            return Err(LaminaError::RuntimeError(format!(
                "Interpreter: block '{}' missing terminator",
                block.label
            )));
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
) -> Result<Option<i64>, LaminaError> {
    let param_count = sig.params.len();

    if param_count > MAX_JIT_ARGS {
        return Err(LaminaError::RuntimeError(format!(
            "JIT execution: Handles up to {MAX_JIT_ARGS} parameters, got {param_count}"
        )));
    }

    let all_i64 = sig
        .params
        .iter()
        .all(|p| matches!(p.ty, MirType::Scalar(ScalarType::I64)));

    let returns_i64 = matches!(sig.ret_ty.as_ref(), Some(MirType::Scalar(ScalarType::I64)));
    let returns_void = sig.ret_ty.is_none();

    if !all_i64 {
        return Err(LaminaError::RuntimeError(format!(
            "JIT execution: Function has non-i64 parameters, not yet supported. Parameter types: {:?}",
            sig.params.iter().map(|p| &p.ty).collect::<Vec<_>>()
        )));
    }

    if !returns_i64 && !returns_void {
        return Err(LaminaError::RuntimeError(format!(
            "JIT execution: Unsupported return type: {:?}",
            sig.ret_ty
        )));
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
        return Err(LaminaError::RuntimeError(format!(
            "JIT execution: Argument count mismatch. Expected {}, got {}",
            param_count,
            args.len()
        )));
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
                Err(LaminaError::RuntimeError(
                    "JIT execution: Function should have returned a value but didn't".to_owned(),
                ))
            }
        } else {
            if verbose {
                println!("[JIT] Function executed successfully (void return)");
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        Block, Function, Immediate, Instruction, IntBinOp, MirType, Operand, Parameter, Register,
        ScalarType, Signature, VirtualReg,
    };

    fn vreg(id: u32) -> Register {
        Register::Virtual(VirtualReg::gpr(id))
    }

    #[test]
    fn evaluate_operand_integer_immediates() {
        let regs = HashMap::new();
        assert_eq!(
            evaluate_operand(&Operand::Immediate(Immediate::I8(5)), &regs).unwrap(),
            5
        );
        assert_eq!(
            evaluate_operand(&Operand::Immediate(Immediate::I16(-3)), &regs).unwrap(),
            -3
        );
        assert_eq!(
            evaluate_operand(&Operand::Immediate(Immediate::I32(100)), &regs).unwrap(),
            100
        );
        assert_eq!(
            evaluate_operand(&Operand::Immediate(Immediate::I64(i64::MAX)), &regs).unwrap(),
            i64::MAX
        );
    }

    #[test]
    fn evaluate_operand_float_immediate_is_error() {
        let regs = HashMap::new();
        assert!(evaluate_operand(&Operand::Immediate(Immediate::F32(1.0)), &regs).is_err());
        assert!(evaluate_operand(&Operand::Immediate(Immediate::F64(1.0)), &regs).is_err());
    }

    #[test]
    fn evaluate_operand_register_found() {
        let mut regs = HashMap::new();
        let r0 = vreg(0);
        regs.insert(r0.clone(), 42_i64);
        assert_eq!(evaluate_operand(&Operand::Register(r0), &regs).unwrap(), 42);
    }

    #[test]
    fn evaluate_operand_register_missing_is_error() {
        let regs = HashMap::new();
        assert!(evaluate_operand(&Operand::Register(vreg(0)), &regs).is_err());
    }

    fn make_const_return_func(value: i64) -> Function {
        let sig = Signature::new("test_fn");
        let mut func = Function::new(sig);
        let mut block = Block::new("entry");
        block.push(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(value))),
        });
        func.blocks.push(block);
        func
    }

    #[test]
    fn interpret_returns_constant() {
        let func = make_const_return_func(99);
        assert_eq!(interpret_mir_function(&func, &[]).unwrap(), Some(99));
    }

    #[test]
    fn interpret_wrong_arg_count_is_error() {
        let func = make_const_return_func(0);
        assert!(interpret_mir_function(&func, &[1]).is_err());
    }

    #[test]
    fn interpret_add_two_params() {
        let sig = Signature::new("add_fn")
            .with_params(vec![
                Parameter::new(vreg(0), MirType::Scalar(ScalarType::I64)),
                Parameter::new(vreg(1), MirType::Scalar(ScalarType::I64)),
            ])
            .with_return(MirType::Scalar(ScalarType::I64));
        let dst = vreg(2);
        let mut func = Function::new(sig);
        let mut block = Block::new("entry");
        block.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: Operand::Register(vreg(0)),
            rhs: Operand::Register(vreg(1)),
        });
        block.push(Instruction::Ret {
            value: Some(Operand::Register(dst)),
        });
        func.blocks.push(block);
        assert_eq!(interpret_mir_function(&func, &[10, 32]).unwrap(), Some(42));
    }

    #[test]
    fn interpret_division_by_zero_is_error() {
        let sig = Signature::new("div_fn")
            .with_params(vec![Parameter::new(
                vreg(0),
                MirType::Scalar(ScalarType::I64),
            )])
            .with_return(MirType::Scalar(ScalarType::I64));
        let dst = vreg(1);
        let mut func = Function::new(sig);
        let mut block = Block::new("entry");
        block.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: dst.clone(),
            lhs: Operand::Register(vreg(0)),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        block.push(Instruction::Ret {
            value: Some(Operand::Register(dst)),
        });
        func.blocks.push(block);
        assert!(interpret_mir_function(&func, &[10]).is_err());
    }

    #[test]
    fn execute_jit_arg_count_mismatch_is_error() {
        let sig = Signature::new("fn");
        // SAFETY: error is triggered during argument validation before the pointer is called.
        let result =
            unsafe { execute_jit_function(&sig, std::ptr::null(), Some(&[1, 2]), false, None) };
        assert!(result.is_err());
    }

    #[test]
    fn execute_jit_non_i64_param_is_error() {
        let sig = Signature::new("fn").with_params(vec![Parameter::new(
            vreg(0),
            MirType::Scalar(ScalarType::F32),
        )]);
        // SAFETY: error is triggered during type validation before the pointer is called.
        let result =
            unsafe { execute_jit_function(&sig, std::ptr::null(), Some(&[1]), false, None) };
        assert!(result.is_err());
    }

    #[test]
    fn execute_jit_unsupported_return_type_is_error() {
        let sig = Signature::new("fn").with_return(MirType::Scalar(ScalarType::F64));
        // SAFETY: error is triggered during return-type validation before the pointer is called.
        let result = unsafe { execute_jit_function(&sig, std::ptr::null(), None, false, None) };
        assert!(result.is_err());
    }
}
