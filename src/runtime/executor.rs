//! JIT function execution
//!
//! Functions to execute JIT-compiled functions with dynamic argument counts
//! using the platform C ABI.

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, AtomicBinOp, Function, Immediate, Instruction, IntCmpOp, MemoryOrdering,
        MirType, Operand, Register, ScalarType, Signature, VirtualReg,
    },
    runtime::c_abi_dynamic::{MAX_JIT_ARGS, call_function_dynamic},
};
use lamina_platform::PointerWidth;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::{collections::HashMap, env, ops::Range};

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

fn register_value(
    register: &Register,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, LaminaError> {
    evaluate_operand(&Operand::Register(register.clone()), register_values)
}

fn operand_bits(
    operand: &Operand,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, LaminaError> {
    match operand {
        Operand::Immediate(Immediate::F32(value)) => Ok(i64::from(value.to_bits())),
        Operand::Immediate(Immediate::F64(value)) => Ok(value.to_bits() as i64),
        _ => evaluate_operand(operand, register_values),
    }
}

fn effective_address(
    addr: &AddressMode,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            Ok(register_value(base, register_values)?.wrapping_add(i64::from(*offset)))
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => Ok(register_value(base, register_values)?
            .wrapping_add(register_value(index, register_values)?.wrapping_mul(i64::from(*scale)))
            .wrapping_add(i64::from(*offset))),
    }
}

fn atomic_address(
    addr: &AddressMode,
    register_values: &HashMap<Register, i64>,
) -> Result<i64, LaminaError> {
    match addr {
        AddressMode::BaseOffset {
            base, offset: 0, ..
        } => register_value(base, register_values),
        other => Err(LaminaError::RuntimeError(format!(
            "Interpreter: atomic address mode {other:?} not supported"
        ))),
    }
}

fn access_size(ty: &MirType, pointer_width: PointerWidth) -> Result<usize, LaminaError> {
    let size = ty.size_bytes(pointer_width);
    if ty.is_vector() || !(1..=8).contains(&size) {
        return Err(LaminaError::RuntimeError(format!(
            "Interpreter: memory access of {ty:?} not supported"
        )));
    }
    Ok(size)
}

fn loaded_value(ty: &MirType, raw: u64, pointer_width: PointerWidth) -> i64 {
    if ty.is_float() {
        raw as i64
    } else {
        ty.signed_integer(raw as i64, pointer_width)
    }
}

fn unsupported_ordering(op: &str, ordering: &MemoryOrdering) -> LaminaError {
    LaminaError::RuntimeError(format!(
        "Interpreter: {ordering} ordering not supported for {op}"
    ))
}

/// Byte-addressed memory for one interpreted call. Addresses below `NULL_GUARD`
/// are never allocated, so a null or small integer pointer traps.
struct Memory {
    bytes: Vec<u8>,
    slots: HashMap<VirtualReg, i64>,
}

const NULL_GUARD: usize = 16;

impl Memory {
    fn new() -> Self {
        Self {
            bytes: vec![0; NULL_GUARD],
            slots: HashMap::new(),
        }
    }

    fn stack_slot(&mut self, slot: VirtualReg, size: usize, align: usize) -> i64 {
        if let Some(address) = self.slots.get(&slot) {
            return *address;
        }
        let start = self.bytes.len().next_multiple_of(align.max(8));
        self.bytes.resize(start + size, 0);
        let address = start as i64;
        self.slots.insert(slot, address);
        address
    }

    fn range(&self, address: i64, size: usize) -> Result<Range<usize>, LaminaError> {
        let start = usize::try_from(address)
            .ok()
            .filter(|start| *start >= NULL_GUARD);
        start
            .and_then(|start| {
                let end = start.checked_add(size)?;
                (end <= self.bytes.len()).then_some(start..end)
            })
            .ok_or_else(|| {
                LaminaError::RuntimeError(format!(
                    "Interpreter: invalid {size}-byte memory access at {address:#x}"
                ))
            })
    }

    fn load(&self, address: i64, size: usize) -> Result<u64, LaminaError> {
        let range = self.range(address, size)?;
        let mut raw = [0u8; 8];
        raw[..size].copy_from_slice(&self.bytes[range]);
        Ok(u64::from_le_bytes(raw))
    }

    fn store(&mut self, address: i64, size: usize, value: u64) -> Result<(), LaminaError> {
        let range = self.range(address, size)?;
        self.bytes[range].copy_from_slice(&value.to_le_bytes()[..size]);
        Ok(())
    }
}

pub(crate) fn interpret_mir_function(
    function: &Function,
    args: &[i64],
) -> Result<Option<i64>, LaminaError> {
    if function.sig.params.len() != args.len() {
        return Err(LaminaError::RuntimeError(format!(
            "Interpreter: expected {} arguments, got {}",
            function.sig.params.len(),
            args.len()
        )));
    }

    let mut register_values: HashMap<Register, i64> = HashMap::new();
    for (param, value) in function.sig.params.iter().zip(args.iter().copied()) {
        register_values.insert(
            param.reg.clone(),
            param.ty.signed_integer(value, function.pointer_width),
        );
    }

    let mut memory = Memory::new();
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
            match &instruction.instr {
                Instruction::Copy { ty, dst, src }
                | Instruction::Bitcast {
                    to: ty, dst, src, ..
                } => {
                    let value = operand_bits(src, &register_values)?;
                    let value = if ty.is_float() {
                        value
                    } else {
                        ty.signed_integer(value, function.pointer_width)
                    };
                    register_values.insert(dst.clone(), value);
                }
                Instruction::IntConvert {
                    from,
                    to,
                    signed,
                    dst,
                    src,
                } => {
                    let value = evaluate_operand(src, &register_values)?;
                    let value = if *signed {
                        from.signed_integer(value, function.pointer_width)
                    } else {
                        from.unsigned_integer(value, function.pointer_width) as i64
                    };
                    let value = to.signed_integer(value, function.pointer_width);
                    register_values.insert(dst.clone(), value);
                }
                Instruction::IntToFloat {
                    from,
                    to,
                    signed,
                    dst,
                    src,
                } => {
                    let bits = evaluate_operand(src, &register_values)?;
                    let result = match (to.size_bytes(function.pointer_width), *signed) {
                        (4, true) => i64::from(
                            (from.signed_integer(bits, function.pointer_width) as f32).to_bits(),
                        ),
                        (4, false) => i64::from(
                            (from.unsigned_integer(bits, function.pointer_width) as f32).to_bits(),
                        ),
                        (_, true) => (from.signed_integer(bits, function.pointer_width) as f64)
                            .to_bits() as i64,
                        (_, false) => (from.unsigned_integer(bits, function.pointer_width) as f64)
                            .to_bits() as i64,
                    };
                    register_values.insert(dst.clone(), result);
                }
                Instruction::FloatConvert { to, dst, src, .. } => {
                    let bits = evaluate_operand(src, &register_values)?;
                    let result = if matches!(to, MirType::Scalar(ScalarType::F32)) {
                        i64::from((f64::from_bits(bits as u64) as f32).to_bits())
                    } else {
                        f64::from(f32::from_bits(bits as u32)).to_bits() as i64
                    };
                    register_values.insert(dst.clone(), result);
                }
                Instruction::FloatToInt {
                    from,
                    to,
                    signed,
                    dst,
                    src,
                } => {
                    let bits = evaluate_operand(src, &register_values)?;
                    let value = if matches!(from, MirType::Scalar(ScalarType::F32)) {
                        f64::from(f32::from_bits(bits as u32))
                    } else {
                        f64::from_bits(bits as u64)
                    };
                    let converted = match (*signed, to.size_bytes(function.pointer_width)) {
                        (true, 1) => value as i8 as i64,
                        (true, 2) => value as i16 as i64,
                        (true, 4) => value as i32 as i64,
                        (true, _) => value as i64,
                        (false, 1) => value as u8 as i64,
                        (false, 2) => value as u16 as i64,
                        (false, 4) => value as u32 as i64,
                        (false, _) => value as u64 as i64,
                    };
                    register_values.insert(
                        dst.clone(),
                        to.signed_integer(converted, function.pointer_width),
                    );
                }
                Instruction::IntBinary {
                    op,
                    ty,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let left = evaluate_operand(lhs, &register_values)?;
                    let right = evaluate_operand(rhs, &register_values)?;
                    let result = op
                        .evaluate(*ty, left, right, function.pointer_width)
                        .ok_or_else(|| {
                            LaminaError::RuntimeError(format!(
                                "Interpreter: {op} integer division trap"
                            ))
                        })?;
                    register_values.insert(dst.clone(), result);
                }
                Instruction::IntOverflow {
                    op,
                    ty,
                    signed,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let left = evaluate_operand(lhs, &register_values)?;
                    let right = evaluate_operand(rhs, &register_values)?;
                    let overflowed =
                        op.overflows(*ty, *signed, left, right, function.pointer_width);
                    register_values.insert(dst.clone(), i64::from(overflowed));
                }
                Instruction::IntUnary { op, ty, dst, src } => {
                    let value = evaluate_operand(src, &register_values)?;
                    register_values
                        .insert(dst.clone(), op.evaluate(*ty, value, function.pointer_width));
                }
                Instruction::IntRotate {
                    op,
                    ty,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let value = evaluate_operand(lhs, &register_values)?;
                    let count = evaluate_operand(rhs, &register_values)?;
                    register_values.insert(
                        dst.clone(),
                        op.evaluate(*ty, value, count, function.pointer_width),
                    );
                }
                Instruction::IntCmp {
                    op,
                    ty,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let left_value = ty.signed_integer(
                        evaluate_operand(lhs, &register_values)?,
                        function.pointer_width,
                    );
                    let right_value = ty.signed_integer(
                        evaluate_operand(rhs, &register_values)?,
                        function.pointer_width,
                    );
                    let result = match op {
                        IntCmpOp::Eq => left_value == right_value,
                        IntCmpOp::Ne => left_value != right_value,
                        IntCmpOp::ULt => {
                            ty.unsigned_integer(left_value, function.pointer_width)
                                < ty.unsigned_integer(right_value, function.pointer_width)
                        }
                        IntCmpOp::ULe => {
                            ty.unsigned_integer(left_value, function.pointer_width)
                                <= ty.unsigned_integer(right_value, function.pointer_width)
                        }
                        IntCmpOp::UGt => {
                            ty.unsigned_integer(left_value, function.pointer_width)
                                > ty.unsigned_integer(right_value, function.pointer_width)
                        }
                        IntCmpOp::UGe => {
                            ty.unsigned_integer(left_value, function.pointer_width)
                                >= ty.unsigned_integer(right_value, function.pointer_width)
                        }
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
                Instruction::Call { .. } | Instruction::CallIndirect { .. } => {
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
                Instruction::Load { ty, dst, addr, .. } => {
                    let size = access_size(ty, function.pointer_width)?;
                    let address = effective_address(addr, &register_values)?;
                    let raw = memory.load(address, size)?;
                    register_values
                        .insert(dst.clone(), loaded_value(ty, raw, function.pointer_width));
                }
                Instruction::Store { ty, src, addr, .. } => {
                    let size = access_size(ty, function.pointer_width)?;
                    let address = effective_address(addr, &register_values)?;
                    let value = operand_bits(src, &register_values)?;
                    memory.store(address, size, value as u64)?;
                }
                Instruction::StackAddr { dst, slot } => {
                    let reservations = function.stack_reservations.get(slot).copied().unwrap_or(0);
                    let align = function.stack_alignments.get(slot).copied().unwrap_or(8);
                    let address = memory.stack_slot(*slot, 8 * (reservations + 1), align);
                    register_values.insert(dst.clone(), address);
                }
                Instruction::Lea { dst, base, offset } => {
                    let value =
                        register_value(base, &register_values)?.wrapping_add(i64::from(*offset));
                    register_values.insert(dst.clone(), value);
                }
                Instruction::AtomicLoad {
                    ty,
                    dst,
                    addr,
                    ordering,
                } => {
                    if matches!(ordering, MemoryOrdering::Release | MemoryOrdering::AcqRel) {
                        return Err(unsupported_ordering("atomic_load", ordering));
                    }
                    let size = access_size(ty, function.pointer_width)?;
                    let address = atomic_address(addr, &register_values)?;
                    let raw = memory.load(address, size)?;
                    register_values
                        .insert(dst.clone(), loaded_value(ty, raw, function.pointer_width));
                }
                Instruction::AtomicStore {
                    ty,
                    src,
                    addr,
                    ordering,
                } => {
                    if matches!(ordering, MemoryOrdering::Acquire | MemoryOrdering::AcqRel) {
                        return Err(unsupported_ordering("atomic_store", ordering));
                    }
                    let size = access_size(ty, function.pointer_width)?;
                    let address = atomic_address(addr, &register_values)?;
                    let value = operand_bits(src, &register_values)?;
                    memory.store(address, size, value as u64)?;
                }
                Instruction::AtomicBinary {
                    op,
                    ty,
                    dst,
                    addr,
                    value,
                    ..
                } => {
                    let size = access_size(ty, function.pointer_width)?;
                    let address = atomic_address(addr, &register_values)?;
                    let operand = operand_bits(value, &register_values)? as u64;
                    let old = memory.load(address, size)?;
                    let new = match op {
                        AtomicBinOp::Add => old.wrapping_add(operand),
                        AtomicBinOp::Sub => old.wrapping_sub(operand),
                        AtomicBinOp::And => old & operand,
                        AtomicBinOp::Or => old | operand,
                        AtomicBinOp::Xor => old ^ operand,
                        AtomicBinOp::Exchange => operand,
                    };
                    memory.store(address, size, new)?;
                    register_values
                        .insert(dst.clone(), loaded_value(ty, old, function.pointer_width));
                }
                Instruction::AtomicCompareExchange {
                    ty,
                    dst,
                    addr,
                    expected,
                    desired,
                    ..
                } => {
                    let size = access_size(ty, function.pointer_width)?;
                    let address = atomic_address(addr, &register_values)?;
                    let expected = operand_bits(expected, &register_values)? as u64;
                    let desired = operand_bits(desired, &register_values)? as u64;
                    let old = memory.load(address, size)?;
                    let mask = u64::MAX >> (64 - 8 * size);
                    if old & mask == expected & mask {
                        memory.store(address, size, desired)?;
                    }
                    register_values
                        .insert(dst.clone(), loaded_value(ty, old, function.pointer_width));
                }
                Instruction::Fence { .. } => {}
                Instruction::SymbolAddress { .. }
                | Instruction::ThreadLocalAddr { .. }
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
                Instruction::InlineAsm { .. } => {
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: inline assembly not supported".to_owned(),
                    ));
                }
                Instruction::VaStart { .. }
                | Instruction::VaArg { .. }
                | Instruction::VaEnd { .. }
                | Instruction::VaCopy { .. } => {
                    return Err(LaminaError::RuntimeError(
                        "Interpreter: variadic arguments not supported".to_owned(),
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

    // LAMINA_JIT_INTERPRET walks the MIR instead of running the code, for comparing the
    // two when a backend is suspect.
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
    use std::ptr::null;

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

    fn narrow_compare_func(ty: MirType, op: IntCmpOp, rhs: i64) -> Function {
        let param = vreg(0);
        let dst = vreg(1);
        let mut func = Function::new(
            Signature::new("narrow_cmp")
                .with_params(vec![Parameter::new(param.clone(), ty)])
                .with_return(MirType::Scalar(ScalarType::I64)),
        );
        let mut block = Block::new("entry");
        block.push(Instruction::IntCmp {
            op,
            ty,
            dst: dst.clone(),
            lhs: Operand::Register(param),
            rhs: Operand::Immediate(Immediate::I64(rhs)),
        });
        block.push(Instruction::Ret {
            value: Some(Operand::Register(dst)),
        });
        func.blocks.push(block);
        func
    }

    #[test]
    fn narrow_arguments_compare_at_their_declared_width() {
        let signed = narrow_compare_func(MirType::Scalar(ScalarType::I8), IntCmpOp::SLt, 0);
        assert_eq!(interpret_mir_function(&signed, &[255]).unwrap(), Some(1));
        let unsigned = narrow_compare_func(MirType::Scalar(ScalarType::I8), IntCmpOp::ULt, 128);
        assert_eq!(interpret_mir_function(&unsigned, &[255]).unwrap(), Some(0));
        let unsigned_byte =
            narrow_compare_func(MirType::Scalar(ScalarType::I8), IntCmpOp::UGt, 127);
        assert_eq!(
            interpret_mir_function(&unsigned_byte, &[-1]).unwrap(),
            Some(1)
        );
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
        let result = unsafe { execute_jit_function(&sig, null(), Some(&[1, 2]), false, None) };
        assert!(result.is_err());
    }

    #[test]
    fn execute_jit_non_i64_param_is_error() {
        let sig = Signature::new("fn").with_params(vec![Parameter::new(
            vreg(0),
            MirType::Scalar(ScalarType::F32),
        )]);
        // SAFETY: error is triggered during type validation before the pointer is called.
        let result = unsafe { execute_jit_function(&sig, null(), Some(&[1]), false, None) };
        assert!(result.is_err());
    }

    #[test]
    fn execute_jit_unsupported_return_type_is_error() {
        let sig = Signature::new("fn").with_return(MirType::Scalar(ScalarType::F64));
        // SAFETY: error is triggered during return-type validation before the pointer is called.
        let result = unsafe { execute_jit_function(&sig, null(), None, false, None) };
        assert!(result.is_err());
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[path = "integer_tests.rs"]
mod integer_tests;

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[path = "atomic_tests.rs"]
mod atomic_tests;
