//! Instruction parsing for Lamina IR.

use crate::mir::asm;
use crate::{
    LaminaError,
    ir::{
        AllocType, AtomicBinOp, BinaryOp, CmpOp, Identifier, Instruction, Literal, MemoryOrdering,
        OverflowOp, PrimitiveType, Type, UnaryOp, Value, instruction::AsmOperand,
    },
    parser::{
        functions::alignment_from_bytes,
        globals::parse_value_with_type_hint,
        state::{CallSite, ParserState},
        types::{parse_type, parse_vector_type, removed_char_error},
        values::parse_value,
    },
};

/// Parses a single instruction from the input.
pub fn parse_instruction<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    let _start_pos = state.position();

    if state.current_char() == Some('%') {
        let result = state.parse_value_identifier()?;
        if state.current_char() == Some(',') {
            return Err(state.error(
                "Instructions with two results are not supported yet\n  Hint: Each instruction has one result. atomic_cmpxchg yields the old value; compare it with the expected value (eq) to learn whether the exchange happened",
            ));
        }
        state.expect_char('=')?;
        state.skip_whitespace_and_comments();
        let opcode_str = state.parse_identifier_str()?;

        match opcode_str {
            "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr"
            | "rotl" | "rotr" => parse_binary_op(state, result, opcode_str),
            "splat" => parse_splat(state, result),
            "extractlane" => parse_extract_lane(state, result),
            "insertlane" => parse_insert_lane(state, result),
            "shuffle" => parse_shuffle(state, result),
            "neg" | "not" | "sqrt" | "popcount" | "ctlz" | "cttz" | "bswap" => {
                parse_unary_op(state, result, opcode_str)
            }
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" => parse_cmp_op(state, result, opcode_str),
            "extend" => parse_extend(state, result),
            "trunc" => parse_trunc(state, result),
            "zext" | "sext" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'extend.<source-type>.<target-type>' (a signed source sign-extends, an unsigned or bool source zero-extends)",
            )),
            "bitcast" => parse_bitcast(state, result),
            "inttofp" => parse_int_to_float(state, result),
            "sitofp" | "uitofp" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'inttofp.<int-type>.<float-type>'",
            )),
            "fptoint" => parse_float_to_int(state, result),
            "fptosi" | "fptoui" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'fptoint.<float-type>.<int-type>'",
            )),
            "fpcast" => parse_float_convert(state, result),
            "fpext" | "fptrunc" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'fpcast.<float-type>.<float-type>'",
            )),
            "select" => parse_select(state, result),
            "stackalloc" => parse_alloc(state, result, AllocType::Stack),
            "heapalloc" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'%p = call @malloc(<size>)' after declaring '@extern fn @malloc(i64 %n) -> ptr'",
            )),
            "alloc" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'stackalloc.<type>'",
            )),
            "load" => parse_load(state, result, false),
            "volatile" => parse_volatile_assignment(state, result),
            "getfieldptr" => parse_getfield(state, result),
            "getfield" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'getfieldptr.<struct-type> <ptr>, <index>'",
            )),
            "getelem" => parse_getelem(state, result),
            "getelementptr" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'getelem.<type> <ptr>, <index>'",
            )),
            "ptrtoint" => parse_ptrtoint(state, result),
            "inttoptr" => parse_inttoptr(state, result),
            "tuple" | "extract" => Err(removed_opcode_error(
                state,
                opcode_str,
                "a struct in memory: 'stackalloc.<type>' then 'getfieldptr.<type> <ptr>, <index>'",
            )),
            "call" => parse_call(state, Some(result)),
            "asm" => parse_asm(state, Some(result)),
            "funcaddr" => parse_func_addr(state, result),
            "callptr" => parse_call_indirect(state, Some(result)),
            "copy" => parse_copy(state, result),
            "phi" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'%x = copy.<type> <value>' in each predecessor block, assigning the same %x on every incoming edge",
            )),
            "write" => parse_write_assignment(state, result),
            "read" => parse_read_assignment(state, result),
            "writebyte" => parse_writebyte_assignment(state, result),
            "readbyte" => parse_readbyte_assignment(state, result),
            "va_arg" => parse_va_arg(state, result),
            "writeptr" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'%v = load.<type> <ptr>' then '%r = writebyte %v' for one byte, or '%r = write <ptr>, <size>' for raw bytes",
            )),
            "atomic_load" => parse_atomic_load(state, result),
            "atomic_add" => parse_atomic_rmw(state, result, AtomicBinOp::Add),
            "atomic_sub" => parse_atomic_rmw(state, result, AtomicBinOp::Sub),
            "atomic_and" => parse_atomic_rmw(state, result, AtomicBinOp::And),
            "atomic_or" => parse_atomic_rmw(state, result, AtomicBinOp::Or),
            "atomic_xor" => parse_atomic_rmw(state, result, AtomicBinOp::Xor),
            "atomic_exchange" => parse_atomic_rmw(state, result, AtomicBinOp::Exchange),
            "atomic_cmpxchg" => parse_atomic_cmpxchg(state, result),
            "atomic_store" | "fence" => Err(state.error(format!(
                "{opcode_str} produces no result\n  Hint: Drop the '%name =' prefix"
            ))),
            _ => {
                if let Some(replacement) = statement_opcode_replacement(opcode_str) {
                    return Err(removed_opcode_error(state, opcode_str, replacement));
                }
                let valid_ops = super::get_assignment_opcode_names();
                let suggestions = super::suggest_alternatives(opcode_str, valid_ops);

                let hint = if !suggestions.is_empty() {
                    if suggestions.len() == 1 {
                        format!("Did you mean '{}'?", suggestions[0])
                    } else {
                        format!(
                            "Did you mean one of: {}?",
                            suggestions
                                .iter()
                                .take(3)
                                .map(|s| format!("'{s}'"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                } else {
                    "Valid opcodes include: add, sub, mul, div, load, store, call, alloc, and many others".to_string()
                };

                Err(state.error(format!(
                    "Unknown instruction opcode '{opcode_str}' after assignment\n  Hint: {hint}"
                )))
            }
        }
    } else {
        let opcode_str = state.parse_identifier_str()?;
        match opcode_str {
            "store" => parse_store(state, false),
            "volatile" => parse_volatile_store(state),
            "switch" => parse_switch(state),
            "br" => parse_br(state),
            "jmp" => parse_jmp(state),
            "ret" => parse_ret(state),
            "unreachable" => Ok(Instruction::Unreachable),
            "call" => parse_call(state, None),
            "asm" => parse_asm(state, None),
            "callptr" => parse_call_indirect(state, None),
            "va_start" => parse_va_start(state),
            "va_end" => parse_va_end(state),
            "va_copy" => parse_va_copy(state),
            "write" => parse_write(state),
            "read" => parse_read(state),
            "writebyte" => parse_writebyte(state),
            "readbyte" => parse_readbyte(state),
            "writeptr" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'%v = load.<type> <ptr>' then '%r = writebyte %v' for one byte, or '%r = write <ptr>, <size>' for raw bytes",
            )),
            "atomic_store" => parse_atomic_store(state),
            "fence" => parse_fence(state),
            "atomic_load" | "atomic_add" | "atomic_sub" | "atomic_and" | "atomic_or"
            | "atomic_xor" | "atomic_exchange" | "atomic_cmpxchg" => Err(state.error(format!(
                "{opcode_str} produces a value\n  Hint: Write '%name = {opcode_str}.<type> ...'"
            ))),
            _ => {
                if let Some(replacement) = statement_opcode_replacement(opcode_str) {
                    return Err(removed_opcode_error(state, opcode_str, replacement));
                }
                let valid_ops = super::get_non_assignment_opcode_names();
                let suggestions = super::suggest_alternatives(opcode_str, valid_ops);

                let hint = if !suggestions.is_empty() {
                    if suggestions.len() == 1 {
                        format!("Did you mean '{}'?", suggestions[0])
                    } else {
                        format!(
                            "Did you mean one of: {}?",
                            suggestions
                                .iter()
                                .take(3)
                                .map(|s| format!("'{s}'"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                } else {
                    "Valid instruction opcodes include: ret, jmp, br, call, store, and many others"
                        .to_string()
                };

                Err(state.error(format!(
                    "Unknown instruction opcode '{opcode_str}'\n  Hint: {hint}"
                )))
            }
        }
    }
}

fn parse_memory_ordering_name(state: &mut ParserState<'_>) -> Result<MemoryOrdering, LaminaError> {
    let name = state.parse_identifier_str()?;
    memory_ordering_from_name(state, name)
}

fn memory_ordering_from_name(
    state: &ParserState<'_>,
    name: &str,
) -> Result<MemoryOrdering, LaminaError> {
    match name {
        "relaxed" => Ok(MemoryOrdering::Relaxed),
        "acquire" => Ok(MemoryOrdering::Acquire),
        "release" => Ok(MemoryOrdering::Release),
        "acqrel" => Ok(MemoryOrdering::AcqRel),
        "seqcst" => Ok(MemoryOrdering::SeqCst),
        _ => Err(state.error(format!(
            "Unknown memory ordering '{name}'\n  Hint: Valid orderings are: relaxed, acquire, release, acqrel, seqcst"
        ))),
    }
}

fn parse_memory_ordering_operand(
    state: &mut ParserState<'_>,
) -> Result<MemoryOrdering, LaminaError> {
    state.expect_char(',')?;
    parse_memory_ordering_name(state)
}

fn parse_atomic_type(state: &mut ParserState<'_>) -> Result<PrimitiveType, LaminaError> {
    let ty = parse_primitive_type_suffix(state)?;
    if matches!(ty, PrimitiveType::F32 | PrimitiveType::F64) {
        return Err(state.error(format!(
            "Atomic operations take an integer or pointer type, found {ty}"
        )));
    }
    Ok(ty)
}

fn parse_atomic_load<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_atomic_type(state)?;
    let ptr = parse_value(state)?;
    let ordering = parse_memory_ordering_operand(state)?;
    if matches!(ordering, MemoryOrdering::Release | MemoryOrdering::AcqRel) {
        return Err(state.error(format!(
            "atomic_load cannot use {ordering} ordering\n  Hint: Use relaxed, acquire or seqcst"
        )));
    }
    Ok(Instruction::AtomicLoad {
        result,
        ty: Type::Primitive(ty),
        ptr,
        ordering,
    })
}

fn parse_atomic_store<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_atomic_type(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let value = parse_typed_operand(state, &Type::Primitive(ty))?;
    let ordering = parse_memory_ordering_operand(state)?;
    if matches!(ordering, MemoryOrdering::Acquire | MemoryOrdering::AcqRel) {
        return Err(state.error(format!(
            "atomic_store cannot use {ordering} ordering\n  Hint: Use relaxed, release or seqcst"
        )));
    }
    Ok(Instruction::AtomicStore {
        ty: Type::Primitive(ty),
        ptr,
        value,
        ordering,
    })
}

fn parse_atomic_rmw<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    op: AtomicBinOp,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_atomic_type(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let value = parse_typed_operand(state, &Type::Primitive(ty))?;
    let ordering = parse_memory_ordering_operand(state)?;
    Ok(Instruction::AtomicBinary {
        op,
        result,
        ty: Type::Primitive(ty),
        ptr,
        value,
        ordering,
    })
}

/// Parses `atomic_cmpxchg.T ptr, expected, desired, success, failure`. The result is
/// the value that was in memory before the operation.
fn parse_atomic_cmpxchg<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_atomic_type(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let expected = parse_typed_operand(state, &Type::Primitive(ty))?;
    state.expect_char(',')?;
    let desired = parse_typed_operand(state, &Type::Primitive(ty))?;
    let success_ordering = parse_memory_ordering_operand(state)?;
    let failure_ordering = parse_memory_ordering_operand(state)?;
    if let Some(reason) =
        MemoryOrdering::cmpxchg_failure_rejection(success_ordering, failure_ordering)
    {
        return Err(state.error(format!(
            "atomic_cmpxchg cannot use {failure_ordering} failure ordering with {success_ordering} success ordering\n  Hint: {reason}"
        )));
    }
    Ok(Instruction::AtomicCompareExchange {
        result,
        ty: Type::Primitive(ty),
        ptr,
        expected,
        desired,
        success_ordering,
        failure_ordering,
    })
}

/// Parses `asm.T "template", operands...` (with a result) or `asm.void "template", operands...`.
///
/// Operands are `in(location) %value`, `out(location)` and `clobber(name, ...)`. The
/// template is kept as written; `$n` names the n-th operand and `$$` is a literal dollar.
fn parse_asm<'a>(
    state: &mut ParserState<'a>,
    result: Option<Identifier<'a>>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let suffix = state.parse_identifier_str()?;
    let result = match (result, suffix) {
        (None, "void") => None,
        (Some(name), "void") => {
            return Err(state.error(format!(
                "asm.void produces no value, so '%{name}' cannot be bound
  Hint: Write '%{name} = asm.T' with an out(...) operand to produce a value"
            )));
        }
        (None, other) => {
            return Err(state.error(format!(
                "asm.{other} needs a result
  Hint: Write '%x = asm.{other} \"...\", out(reg)', or use asm.void when there is no result"
            )));
        }
        (Some(name), other) => {
            let ty = parse_primitive_from_ident(state, other)?;
            if !matches!(
                ty,
                PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64
                    | PrimitiveType::Ptr
            ) {
                return Err(state.error(
                    "asm results must be an integer or ptr type\n  Hint: Floating-point and 128-bit asm results are not supported yet",
                ));
            }
            Some((name, ty))
        }
    };

    state.skip_inline_space();
    let template = state.parse_string_literal()?;

    let mut operands = Vec::new();
    let mut clobbers = Vec::new();
    loop {
        state.skip_inline_space();
        if state.current_char() != Some(',') {
            break;
        }
        state.advance();
        state.skip_inline_space();
        let keyword = state.parse_identifier_str()?;
        match keyword {
            "in" => {
                let location = parse_asm_group(state, "in")?;
                state.skip_inline_space();
                let value = parse_value(state)?;
                if !matches!(value, Value::Variable(_)) {
                    return Err(state
                        .error("asm inputs must be values such as %x, not literals or globals"));
                }
                operands.push(AsmOperand::In { location, value });
            }
            "out" => {
                let location = parse_asm_group(state, "out")?;
                operands.push(AsmOperand::Out { location });
            }
            "clobber" => {
                state.skip_inline_space();
                state.expect_char('(')?;
                loop {
                    clobbers.push(state.parse_identifier_str()?);
                    state.skip_inline_space();
                    if state.current_char() == Some(',') {
                        state.advance();
                        continue;
                    }
                    state.expect_char(')')?;
                    break;
                }
            }
            other => {
                return Err(state.error(format!(
                    "Unknown asm operand '{other}'\n  Hint: Operands are in(location) %value, out(location) and clobber(names)"
                )));
            }
        }
    }

    check_asm_template(state, template, operands.len())?;
    check_asm_operands(state, result.is_some(), &operands, &clobbers)?;

    Ok(Instruction::InlineAsm {
        result,
        template,
        operands,
        clobbers,
    })
}

fn parse_asm_group<'a>(state: &mut ParserState<'a>, keyword: &str) -> Result<&'a str, LaminaError> {
    state.skip_inline_space();
    state.expect_char('(')?;
    state.skip_inline_space();
    let location = state.parse_identifier_str()?;
    state.skip_inline_space();
    if state.current_char() != Some(')') {
        return Err(state.error(format!("{keyword}(...) takes a single location")));
    }
    state.advance();
    Ok(location)
}

fn check_asm_template(
    state: &ParserState<'_>,
    template: &str,
    operand_count: usize,
) -> Result<(), LaminaError> {
    let bytes = template.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'$' {
            index += 1;
            continue;
        }
        match bytes.get(index + 1) {
            Some(b'$') => index += 2,
            Some(digit) if digit.is_ascii_digit() => {
                let mut end = index + 1;
                let mut position = 0usize;
                while let Some(&byte) = bytes.get(end)
                    && byte.is_ascii_digit()
                {
                    position = position
                        .saturating_mul(10)
                        .saturating_add(usize::from(byte - b'0'));
                    end += 1;
                }
                if position >= operand_count {
                    return Err(state.error(format!(
                        "asm template uses ${position}, but the instruction has {operand_count} operand(s)\n  Hint: Operands are numbered from $0 in declaration order; write $$ for a literal dollar sign"
                    )));
                }
                index = end;
            }
            _ => {
                return Err(state.error(
                    "stray '$' in asm template\n  Hint: $0, $1, ... name operands; write $$ for a literal dollar sign",
                ));
            }
        }
    }
    Ok(())
}

fn check_asm_operands(
    state: &ParserState<'_>,
    has_result: bool,
    operands: &[AsmOperand<'_>],
    clobbers: &[&str],
) -> Result<(), LaminaError> {
    let outputs = operands
        .iter()
        .filter(|operand| matches!(operand, AsmOperand::Out { .. }))
        .count();
    if has_result && outputs != 1 {
        return Err(state.error(format!(
            "asm.T needs exactly one out(...) operand to hold its result, found {outputs}\n  Hint: Registers written without a result belong in clobber(...)"
        )));
    }
    if !has_result && outputs != 0 {
        return Err(state.error(
            "asm.void takes no out(...) operand\n  Hint: Use clobber(...) for registers the template writes, or asm.T for a result",
        ));
    }

    let mut used: Vec<&str> = Vec::new();
    for operand in operands {
        let location = match operand {
            AsmOperand::In { location, .. } | AsmOperand::Out { location } => *location,
        };
        if location == asm_class() {
            continue;
        }
        if asm::is_frame_register(location) {
            return Err(state.error(format!(
                "cannot use the stack or frame pointer '{location}' as an asm operand"
            )));
        }
        if matches!(location, "memory" | "flags") {
            return Err(state.error(format!(
                "'{location}' is a clobber, not an operand location\n  Hint: Write clobber({location})"
            )));
        }
        let canonical = asm_canonical(location);
        if used.contains(&canonical) {
            return Err(state.error(format!(
                "register '{location}' is used by two asm operands\n  Hint: Each operand needs its own register"
            )));
        }
        used.push(canonical);
    }

    for &clobber in clobbers {
        if asm::is_frame_register(clobber) {
            return Err(state.error(format!(
                "cannot clobber the stack or frame pointer '{clobber}'\n  Hint: The frame is not available to inline assembly"
            )));
        }
        if matches!(clobber, "memory" | "flags") {
            continue;
        }
        if used.contains(&asm_canonical(clobber)) {
            return Err(state.error(format!(
                "register '{clobber}' is both an asm operand and a clobber\n  Hint: Remove it from clobber(...); operands are already destroyed"
            )));
        }
    }
    Ok(())
}

fn asm_class() -> &'static str {
    asm::GPR_CLASS
}

fn asm_canonical(name: &str) -> &str {
    asm::x86_64_register(name).map_or(name, |(canonical, _)| canonical)
}

fn parse_fence<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let name = state.parse_same_line_identifier(
        "fence needs a memory ordering on the same line\n  Hint: Write 'fence seqcst', with relaxed, acquire, release, acqrel or seqcst",
    )?;
    let ordering = memory_ordering_from_name(state, name)?;
    Ok(Instruction::Fence { ordering })
}

/// Parses a primitive type suffix after a dot (e.g., `.i32` in `add.i32`).
fn parse_primitive_type_suffix(state: &mut ParserState<'_>) -> Result<PrimitiveType, LaminaError> {
    state.expect_char('.')?;
    let type_str = state.parse_identifier_str()?;
    match type_str {
        "i8" => Ok(PrimitiveType::I8),
        "i16" => Ok(PrimitiveType::I16),
        "i32" => Ok(PrimitiveType::I32),
        "i64" => Ok(PrimitiveType::I64),
        "u8" => Ok(PrimitiveType::U8),
        "u16" => Ok(PrimitiveType::U16),
        "u32" => Ok(PrimitiveType::U32),
        "u64" => Ok(PrimitiveType::U64),
        "i128" => Ok(PrimitiveType::I128),
        "u128" => Ok(PrimitiveType::U128),
        "f32" => Ok(PrimitiveType::F32),
        "f64" => Ok(PrimitiveType::F64),
        "bool" => Ok(PrimitiveType::Bool),
        "char" => Err(removed_char_error(state)),
        "ptr" => Ok(PrimitiveType::Ptr),
        _ => {
            let valid_types =
                "i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64, bool, ptr";
            Err(state.error(format!(
                "Expected primitive type suffix, found '.{type_str}'\n  Hint: Valid type suffixes are: {valid_types}"
            )))
        }
    }
}

/// Parses a type suffix after a dot (e.g., `.i32`, `.struct`, or named types).
fn parse_type_suffix<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.expect_char('.')?;
    parse_type(state) // Can be any type after the dot
}

fn require_type_suffix(state: &ParserState<'_>, message: &str) -> Result<(), LaminaError> {
    if state.current_char() == Some('.') {
        Ok(())
    } else {
        Err(state.error(message))
    }
}

/// What replaced an opcode that was removed and never took a result, so that writing
/// it in assignment form is told the same thing writing it as a statement is.
fn statement_opcode_replacement(opcode: &str) -> Option<&'static str> {
    match opcode {
        "heapfree" | "dealloc" => {
            Some("'call @free(<ptr>)' after declaring '@extern fn @free(ptr %p) -> void'")
        }
        "print" => Some(
            "'writebyte <value>' for one byte, 'write <ptr>, <size>' for raw bytes, or a call to a printf-style @extern for formatted output",
        ),
        "memcpy" => Some(
            "'call @memcpy(<dst>, <src>, <size>)' after declaring '@extern fn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr'",
        ),
        "memmove" => Some(
            "'call @memmove(<dst>, <src>, <size>)' after declaring '@extern fn @memmove(ptr %d, ptr %s, i64 %n) -> ptr'",
        ),
        "memset" => Some(
            "'call @memset(<dst>, <value>, <size>)' after declaring '@extern fn @memset(ptr %d, i32 %v, i64 %n) -> ptr'",
        ),
        _ => None,
    }
}

fn removed_opcode_error(state: &ParserState<'_>, opcode: &str, replacement: &str) -> LaminaError {
    state.error(format!(
        "Unknown instruction opcode '{opcode}'\n  Hint: Did you mean {replacement}?"
    ))
}

fn parse_unary_op<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    op_str: &str,
) -> Result<Instruction<'a>, LaminaError> {
    let op = match op_str {
        "neg" => UnaryOp::Neg,
        "not" => UnaryOp::Not,
        "sqrt" => UnaryOp::Sqrt,
        "popcount" => UnaryOp::Popcount,
        "ctlz" => UnaryOp::Ctlz,
        "cttz" => UnaryOp::Cttz,
        "bswap" => UnaryOp::Bswap,
        _ => return Err(state.error(format!("Unknown unary operation '{op_str}'"))),
    };
    let ty = parse_primitive_type_suffix(state)?;
    let float = matches!(ty, PrimitiveType::F32 | PrimitiveType::F64);
    match op {
        UnaryOp::Not if float => {
            return Err(state.error(format!("not takes an integer type, found {ty}")));
        }
        UnaryOp::Sqrt if !float => {
            return Err(state.error(format!("sqrt takes f32 or f64, found {ty}")));
        }
        UnaryOp::Popcount | UnaryOp::Ctlz | UnaryOp::Cttz | UnaryOp::Bswap
            if !is_integer_type(ty) =>
        {
            return Err(state.error(format!("{op_str} takes an integer type, found {ty}")));
        }
        UnaryOp::Neg
        | UnaryOp::Not
        | UnaryOp::Sqrt
        | UnaryOp::Popcount
        | UnaryOp::Ctlz
        | UnaryOp::Cttz
        | UnaryOp::Bswap => {}
    }
    let value = parse_typed_operand(state, &Type::Primitive(ty))?;
    Ok(Instruction::Unary {
        op,
        result,
        ty,
        value,
    })
}

fn parse_binary_op<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    op_str: &str,
) -> Result<Instruction<'a>, LaminaError> {
    let op = match op_str {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        "div" => BinaryOp::Div,
        "rem" => BinaryOp::Rem,
        "and" => BinaryOp::And,
        "or" => BinaryOp::Or,
        "xor" => BinaryOp::Xor,
        "shl" => BinaryOp::Shl,
        "shr" => BinaryOp::Shr,
        "rotl" => BinaryOp::Rotl,
        "rotr" => BinaryOp::Rotr,
        _ => return Err(state.error(format!("Unknown binary operation '{op_str}'"))),
    };
    if take_ovf_suffix(state) {
        let op = match op {
            BinaryOp::Add => OverflowOp::Add,
            BinaryOp::Sub => OverflowOp::Sub,
            BinaryOp::Mul => OverflowOp::Mul,
            _ => {
                return Err(state.error(format!(
                    "'.ovf' applies to add, sub and mul only, found '{op_str}.ovf'\n  Hint: Division traps instead of overflowing; shifts and bitwise operations cannot overflow"
                )));
            }
        };
        let ty = parse_primitive_type_suffix(state)?;
        if !is_integer_type(ty) {
            return Err(state.error(format!(
                "{op_str}.ovf takes an integer type, found {ty}\n  Hint: Overflow is defined for integers; a float operation rounds or goes to infinity instead"
            )));
        }
        if matches!(ty, PrimitiveType::I128 | PrimitiveType::U128) {
            return Err(state.error(format!(
                "{op_str}.ovf.{ty} is not supported: 128-bit arithmetic is lowered to word pairs and no backend reports its overflow\n  Hint: Compare the operands before the operation, or use a 64-bit type"
            )));
        }
        let lhs = parse_typed_operand(state, &Type::Primitive(ty))?;
        state.expect_char(',')?;
        let rhs = parse_typed_operand(state, &Type::Primitive(ty))?;
        return Ok(Instruction::OverflowFlag {
            op,
            result,
            ty,
            lhs,
            rhs,
        });
    }
    if let Some((element_type, lanes)) = peek_vector_suffix(state)? {
        let vector = Type::Vector {
            element_type,
            lanes,
        };
        let lhs = parse_typed_operand(state, &vector)?;
        state.expect_char(',')?;
        let rhs = parse_typed_operand(state, &vector)?;
        reject_out_of_range_shift(state, &op, element_type, &rhs)?;
        return Ok(Instruction::VectorBinary {
            op,
            result,
            element_type,
            lanes,
            lhs,
            rhs,
        });
    }
    let ty = parse_primitive_type_suffix(state)?;
    if matches!(op, BinaryOp::Rotl | BinaryOp::Rotr) && !is_integer_type(ty) {
        return Err(state.error(format!("{op_str} takes an integer type, found {ty}")));
    }
    let lhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    state.expect_char(',')?;
    let rhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    reject_out_of_range_shift(state, &op, ty, &rhs)?;
    Ok(Instruction::Binary {
        op,
        result,
        ty,
        lhs,
        rhs,
    })
}

fn integer_bit_width(ty: PrimitiveType) -> Option<u32> {
    Some(match ty {
        PrimitiveType::I8 | PrimitiveType::U8 => 8,
        PrimitiveType::I16 | PrimitiveType::U16 => 16,
        PrimitiveType::I32 | PrimitiveType::U32 => 32,
        PrimitiveType::I64 | PrimitiveType::U64 => 64,
        PrimitiveType::I128 | PrimitiveType::U128 => 128,
        _ => return None,
    })
}

/// Rejects a shift whose count is a literal the compiler can see to fall outside
/// `0..bit_width(ty)`. A count computed at run time keeps the masked semantic.
fn reject_out_of_range_shift(
    state: &ParserState<'_>,
    op: &BinaryOp,
    ty: PrimitiveType,
    rhs: &Value<'_>,
) -> Result<(), LaminaError> {
    if !matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        return Ok(());
    }
    let Some(width) = integer_bit_width(ty) else {
        return Ok(());
    };
    let Value::Constant(literal) = rhs else {
        return Ok(());
    };
    let (in_range, shown) = match literal {
        Literal::U128(value) => (*value < u128::from(width), value.to_string()),
        _ => match literal.as_integer() {
            Some(count) => ((0..i128::from(width)).contains(&count), count.to_string()),
            None => return Ok(()),
        },
    };
    if in_range {
        return Ok(());
    }
    Err(state.error(format!(
        "{op}.{ty} shifts by {shown}, which is out of range for a {width}-bit operand\n  Hint: A constant count must lie in 0..{width}; only a count computed at run time is masked with {}",
        width - 1
    )))
}

/// Consumes a `.<N x T>` suffix when the opcode carries one, leaving the input
/// untouched when the suffix names a scalar type instead.
fn peek_vector_suffix(
    state: &mut ParserState<'_>,
) -> Result<Option<(PrimitiveType, u32)>, LaminaError> {
    state.skip_whitespace_and_comments();
    if state.current_char() != Some('.') || state.peek_byte(1) != Some(b'<') {
        return Ok(None);
    }
    state.expect_char('.')?;
    match parse_vector_type(state)? {
        Type::Vector {
            element_type,
            lanes,
        } => Ok(Some((element_type, lanes))),
        _ => unreachable!("parse_vector_type yields Type::Vector"),
    }
}

/// Parses the `.<N x T>` suffix an explicit vector instruction always carries.
fn parse_vector_suffix(
    state: &mut ParserState<'_>,
    opcode: &str,
) -> Result<(PrimitiveType, u32), LaminaError> {
    peek_vector_suffix(state)?.ok_or_else(|| {
        state.error(format!(
            "'{opcode}' takes a vector type suffix\n  Hint: Write '{opcode}.<4 x i32> ...'"
        ))
    })
}

fn parse_splat<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let (element_type, lanes) = parse_vector_suffix(state, "splat")?;
    let value = parse_typed_operand(state, &Type::Primitive(element_type))?;
    Ok(Instruction::VectorSplat {
        result,
        element_type,
        lanes,
        value,
    })
}

fn parse_lane_index(state: &mut ParserState<'_>, opcode: &str) -> Result<u32, LaminaError> {
    state.skip_whitespace_and_comments();
    if state.current_char() == Some('%') {
        return Err(state.error(format!(
            "'{opcode}' takes a constant lane index\n  Hint: A lane chosen at run time is a different operation; branch over the lanes, or go through memory"
        )));
    }
    let index = state.parse_integer()?;
    u32::try_from(index).map_err(|_| state.error(format!("Invalid lane index: {index}")))
}

fn parse_extract_lane<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let (element_type, lanes) = parse_vector_suffix(state, "extractlane")?;
    let vector = parse_value(state)?;
    state.expect_char(',')?;
    let lane = parse_lane_index(state, "extractlane")?;
    Ok(Instruction::VectorExtractLane {
        result,
        element_type,
        lanes,
        vector,
        lane,
    })
}

fn parse_insert_lane<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let (element_type, lanes) = parse_vector_suffix(state, "insertlane")?;
    let vector = parse_value(state)?;
    state.expect_char(',')?;
    let scalar = parse_typed_operand(state, &Type::Primitive(element_type))?;
    state.expect_char(',')?;
    let lane = parse_lane_index(state, "insertlane")?;
    Ok(Instruction::VectorInsertLane {
        result,
        element_type,
        lanes,
        vector,
        scalar,
        lane,
    })
}

fn parse_shuffle<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let (element_type, lanes) = parse_vector_suffix(state, "shuffle")?;
    let vector = Type::Vector {
        element_type,
        lanes,
    };
    let lhs = parse_typed_operand(state, &vector)?;
    state.expect_char(',')?;
    let rhs = parse_typed_operand(state, &vector)?;
    state.expect_char(',')?;
    state.skip_whitespace_and_comments();
    if state.current_char() != Some('[') {
        return Err(state.error(
            "shuffle takes a constant lane mask\n  Hint: Write the lanes out, as in 'shuffle.<4 x i32> %a, %b, [0, 5, 2, 7]'. A mask chosen at run time is a table lookup, not a shuffle, and costs far more than one instruction",
        ));
    }
    state.expect_char('[')?;
    let mut mask = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(']') {
            state.advance();
            break;
        }
        if state.current_char() == Some('%') {
            return Err(state.error(
                "shuffle takes a constant lane mask\n  Hint: Every mask entry is a literal lane number. A mask chosen at run time is a table lookup, not a shuffle",
            ));
        }
        let index = state.parse_integer()?;
        let index = u32::try_from(index)
            .map_err(|_| state.error(format!("Invalid shuffle lane index: {index}")))?;
        mask.push(index);
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(']') {
            state.advance();
            break;
        }
        state.expect_char(',')?;
    }
    Ok(Instruction::VectorShuffle {
        result,
        element_type,
        lanes,
        lhs,
        rhs,
        mask,
    })
}

/// Consumes a `.ovf` written between the opcode and its type suffix.
fn take_ovf_suffix(state: &mut ParserState<'_>) -> bool {
    let word_ends = !state
        .peek_byte(4)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if state.peek_slice(4) != Some(".ovf") || !word_ends {
        return false;
    }
    state.advance_by(4);
    true
}

fn is_integer_type(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
            | PrimitiveType::U128
    )
}

fn parse_cmp_op<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    op_str: &str,
) -> Result<Instruction<'a>, LaminaError> {
    let op = match op_str {
        "eq" => CmpOp::Eq,
        "ne" => CmpOp::Ne,
        "gt" => CmpOp::Gt,
        "ge" => CmpOp::Ge,
        "lt" => CmpOp::Lt,
        "le" => CmpOp::Le,
        _ => unreachable!(),
    };
    let ty = parse_primitive_type_suffix(state)?;
    let lhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    state.expect_char(',')?;
    let rhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    Ok(Instruction::Cmp {
        op,
        result,
        ty,
        lhs,
        rhs,
    })
}

/// Parses `stackalloc.T` with an optional trailing `align n`.
fn parse_alloc<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    alloc_type: AllocType,
) -> Result<Instruction<'a>, LaminaError> {
    let allocated_ty = parse_type_suffix(state)?;
    let align = parse_align_modifier(state, "stackalloc.i64 align 16")?;
    Ok(Instruction::Alloc {
        result,
        alloc_type,
        allocated_ty,
        align,
    })
}

/// Parses the optional `align n` that may follow a stackalloc type, a load or a
/// store. Anything else is left unread, so the next line still parses as its own
/// instruction.
fn parse_align_modifier(
    state: &mut ParserState<'_>,
    example: &str,
) -> Result<Option<u32>, LaminaError> {
    let start = state.position();
    state.skip_inline_space();
    let word_ends = !state
        .peek_byte(5)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if state.peek_slice(5) != Some("align") || !word_ends {
        state.set_position(start);
        return Ok(None);
    }
    state.advance_by(5);
    state.skip_inline_space();
    if !state
        .current_char()
        .is_some_and(|c| c.is_ascii_digit() || c == '-')
    {
        state.set_position(start);
        return Ok(None);
    }
    let bytes = state.parse_integer()?;
    match alignment_from_bytes(bytes) {
        Some(alignment) => Ok(Some(alignment)),
        None => Err(state.error(format!(
            "Invalid alignment: align {bytes}\n  Hint: Alignment must be a power of two, e.g. {example}"
        ))),
    }
}

fn parse_volatile_assignment<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let opcode = state.parse_identifier_str()?;
    match opcode {
        "load" => parse_load(state, result, true),
        "store" => Err(state.error(
            "store produces no result\n  Hint: Write 'volatile store.<type> <ptr>, <value>'",
        )),
        _ => Err(volatile_opcode_error(state, opcode)),
    }
}

fn parse_volatile_store<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let opcode = state.parse_identifier_str()?;
    match opcode {
        "store" => parse_store(state, true),
        _ => Err(volatile_opcode_error(state, opcode)),
    }
}

fn volatile_opcode_error(state: &ParserState<'_>, opcode: &str) -> LaminaError {
    state.error(format!(
        "'volatile' applies to load and store only, found '{opcode}'\n  Hint: Write 'volatile load.<type> <ptr>' or 'volatile store.<type> <ptr>, <value>'"
    ))
}

fn parse_load<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    volatile: bool,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let ptr = parse_value(state)?;
    let align = parse_align_modifier(state, "load.i32 %p align 1")?;
    Ok(Instruction::Load {
        result,
        ty,
        ptr,
        volatile,
        align,
    })
}

fn parse_store<'a>(
    state: &mut ParserState<'a>,
    volatile: bool,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let value = parse_typed_operand(state, &ty)?;
    let align = parse_align_modifier(state, "store.i32 %p, %v align 1")?;
    Ok(Instruction::Store {
        ty,
        ptr,
        value,
        volatile,
        align,
    })
}

fn parse_getfield<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    require_type_suffix(
        state,
        "getfieldptr needs its struct layout as a suffix, because the field offset depends on it\n  Hint: Did you mean 'getfieldptr.<struct-type> <ptr>, <index>', e.g. 'getfieldptr.!Pair %p, 1'?",
    )?;
    let struct_type = parse_type_suffix(state)?;
    let struct_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let field_index_val = state.parse_integer()?;

    if field_index_val < 0 {
        return Err(state.error(format!("Invalid field index: {field_index_val}")));
    }

    let field_index = field_index_val as usize;
    Ok(Instruction::GetFieldPtr {
        result,
        struct_ptr,
        struct_type,
        field_index,
    })
}

fn parse_getelem<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    require_type_suffix(
        state,
        "getelem takes its element type as a suffix, because the index is scaled by the element size\n  Hint: Write 'getelem.<type> <ptr>, <index>', e.g. 'getelem.i8 %buf, 0'. For a struct field, use 'getfieldptr.<struct-type> <ptr>, <index>'",
    )?;
    let element_type = parse_type_suffix(state)?;
    let array_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let index = parse_value(state)?;
    state.skip_inline_space();
    if state.current_char() == Some(',') {
        return Err(state.error(
            "getelem no longer takes an element type operand\n  Hint: Put the element type in the suffix: 'getelem.<type> <ptr>, <index>'",
        ));
    }

    Ok(Instruction::GetElemPtr {
        result,
        array_ptr,
        index,
        element_type,
    })
}

fn parse_ptrtoint<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    require_type_suffix(
        state,
        "ptrtoint takes its integer type as a suffix\n  Hint: Write 'ptrtoint.i64 %ptr' instead of 'ptrtoint %ptr, .i64'",
    )?;
    let target_type = parse_type_suffix(state)?;

    let target_primitive_type = match target_type {
        Type::Primitive(pt) => pt,
        _ => {
            return Err(LaminaError::ParsingError(
                "Expected primitive type for ptrtoint".to_string(),
            ));
        }
    };
    if !is_integer_type(target_primitive_type) {
        return Err(state.error(format!(
            "ptrtoint.{target_primitive_type} names {target_primitive_type}, which is not an integer type\n  Hint: The suffix names the integer the pointer becomes, as in 'ptrtoint.i64 %ptr'"
        )));
    }
    let ptr_value = parse_typed_operand(state, &Type::Primitive(PrimitiveType::Ptr))?;

    Ok(Instruction::PtrToInt {
        result,
        ptr_value,
        target_type: target_primitive_type,
    })
}

fn parse_inttoptr<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    require_type_suffix(
        state,
        "inttoptr takes its integer type as a suffix\n  Hint: Write 'inttoptr.i64 %addr' instead of 'inttoptr %addr, .i64'",
    )?;
    let target_type = parse_type_suffix(state)?;

    let target_primitive_type = match target_type {
        Type::Primitive(pt) => pt,
        _ => {
            return Err(LaminaError::ParsingError(
                "Expected primitive type for inttoptr".to_string(),
            ));
        }
    };
    if !is_integer_type(target_primitive_type) {
        return Err(state.error(format!(
            "inttoptr.{target_primitive_type} names {target_primitive_type}, which is not an integer type\n  Hint: The suffix names the integer the pointer is built from, as in 'inttoptr.i64 %addr'; the result is always ptr"
        )));
    }
    let int_value = parse_typed_operand(state, &Type::Primitive(target_primitive_type))?;

    Ok(Instruction::IntToPtr {
        result,
        int_value,
        target_type: target_primitive_type,
    })
}

fn parse_call<'a>(
    state: &mut ParserState<'a>,
    result: Option<Identifier<'a>>,
) -> Result<Instruction<'a>, LaminaError> {
    let call_position = state.position();
    let func_name = state.parse_function_identifier()?;
    state.call_sites.push(CallSite {
        callee: func_name,
        position: call_position,
    });
    state.expect_char('(')?;
    let mut args = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance();
            break;
        }
        if starts_untyped_value(state) {
            args.push((None, parse_value(state)?));
        } else {
            let ty = parse_type(state).map_err(|_| untyped_call_error(state))?;
            state.skip_whitespace_and_comments();
            if matches!(state.current_char(), None | Some(')') | Some(',')) {
                return Err(untyped_call_error(state));
            }
            let value = parse_typed_operand(state, &ty)?;
            args.push((Some(ty), value));
        }
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance();
            break;
        }
        state.expect_char(',')?;
    }
    Ok(Instruction::Call {
        result,
        func_name,
        args,
    })
}

fn starts_untyped_value(state: &ParserState<'_>) -> bool {
    match state.current_char() {
        Some('%' | '$' | '"' | '@' | '-') => true,
        Some(c) if c.is_ascii_digit() => true,
        _ => ["true", "false"].iter().any(|word| {
            state.peek_slice(word.len()) == Some(word)
                && !state
                    .peek_char(word.len())
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
        }),
    }
}

fn untyped_call_error(state: &ParserState<'_>) -> LaminaError {
    state.error(
        "call arguments are values, optionally preceded by their type\n  Hint: Write 'call @f(7)' or 'call @f(i64 %x)'; for a call through a function pointer, use 'callptr.<ret-ty> %f(<ty> <arg>, ...)'",
    )
}

fn parse_func_addr<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let func_name = state.parse_function_identifier()?;
    Ok(Instruction::FuncAddr { result, func_name })
}

fn parse_call_indirect<'a>(
    state: &mut ParserState<'a>,
    result: Option<Identifier<'a>>,
) -> Result<Instruction<'a>, LaminaError> {
    let return_type = parse_type_suffix(state)?;
    let target = parse_value(state)?;
    state.expect_char('(')?;
    let mut args = Vec::new();
    let mut variadic_fixed = None;
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance();
            break;
        }
        if state.peek_slice(3) == Some("...") {
            if variadic_fixed.is_some() {
                return Err(state.error(
                    "callptr argument list has more than one '...'\n  Hint: One '...' marks where the callee's named parameters end",
                ));
            }
            for _ in 0..3 {
                state.advance();
            }
            variadic_fixed = Some(args.len());
            state.skip_whitespace_and_comments();
            if state.current_char() == Some(')') {
                state.advance();
                break;
            }
            state.expect_char(',')?;
            continue;
        }
        let ty = parse_type(state).map_err(|_| untyped_call_indirect_error(state))?;
        state.skip_whitespace_and_comments();
        if matches!(state.current_char(), None | Some(')') | Some(',')) {
            return Err(untyped_call_indirect_error(state));
        }
        let value = parse_typed_operand(state, &ty)?;
        args.push((ty, value));
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance();
            break;
        }
        state.expect_char(',')?;
    }
    Ok(Instruction::CallIndirect {
        result,
        target,
        variadic_fixed,
        args,
        return_type,
    })
}

fn untyped_call_indirect_error(state: &ParserState<'_>) -> LaminaError {
    state.error(
        "callptr arguments must each carry a type
  Hint: Did you mean 'callptr.<ret-ty> <ptr>(<ty> <arg>, <ty> <arg>)', as in 'callptr.i64 %f(f32 1.5, i64 %n)'?",
    )
}

fn parse_copy<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    if !matches!(ty, Type::Primitive(_)) {
        return Err(state.error(format!(
            "copy moves one scalar value, found {ty}\n  Hint: Copy an aggregate through memory one scalar at a time with 'getfieldptr' and 'load'/'store'"
        )));
    }
    let value = parse_typed_operand(state, &ty)?;
    Ok(Instruction::Copy { result, ty, value })
}

fn parse_br<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let condition = parse_value(state)?;
    state.expect_char(',')?;
    let true_label = state.parse_label_identifier()?;
    state.expect_char(',')?;
    let false_label = state.parse_label_identifier()?;
    Ok(Instruction::Br {
        condition,
        true_label,
        false_label,
    })
}

fn parse_jmp<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let target_label = state.parse_label_identifier()?;
    Ok(Instruction::Jmp { target_label })
}

fn parse_ret<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    if state.current_char() != Some('.') {
        return Err(state.error(
            "'ret' needs a type suffix\n  Hint: Use 'ret.void' to return nothing, or 'ret.<type> <value>' (e.g. 'ret.i64 0')",
        ));
    }
    state.expect_char('.')?;
    state.skip_whitespace_and_comments();
    if state.peek_slice(4) == Some("void") {
        state.consume_keyword("void")?;
        Ok(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    } else {
        let ty = parse_type(state)?;
        let value = parse_typed_operand(state, &ty)?;
        Ok(Instruction::Ret {
            ty,
            value: Some(value),
        })
    }
}

fn parse_va_start<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ap = parse_value(state)?;
    Ok(Instruction::VaStart { ap })
}

fn parse_va_arg<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_primitive_type_suffix(state)?;
    let ap = parse_value(state)?;
    Ok(Instruction::VaArg { result, ty, ap })
}

fn parse_va_end<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ap = parse_value(state)?;
    Ok(Instruction::VaEnd { ap })
}

fn parse_va_copy<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let dst = parse_value(state)?;
    state.expect_char(',')?;
    let src = parse_value(state)?;
    Ok(Instruction::VaCopy { dst, src })
}

fn parse_write_assignment<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let buffer = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    Ok(Instruction::Write {
        buffer,
        size,
        result,
    })
}

fn parse_read_assignment<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let buffer = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    Ok(Instruction::Read {
        buffer,
        size,
        result,
    })
}

fn parse_writebyte_assignment<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let value = parse_value(state)?;
    Ok(Instruction::WriteByte { value, result })
}

fn parse_readbyte_assignment<'a>(
    _state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    Ok(Instruction::ReadByte { result })
}

/// Parses the `, <name>` that closes a statement form, which must share the line.
fn parse_trailing_name<'a>(
    state: &mut ParserState<'a>,
    missing: &str,
) -> Result<&'a str, LaminaError> {
    state.skip_inline_space();
    if state.current_char() != Some(',') {
        return Err(state.error(missing));
    }
    state.advance();
    state.parse_same_line_identifier(missing)
}

fn parse_write<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let buffer = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    let result = parse_trailing_name(
        state,
        "write needs a result name on the same line\n  Hint: Write 'write <ptr>, <size>, <name>'",
    )?;
    Ok(Instruction::Write {
        buffer,
        size,
        result,
    })
}

fn parse_read<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let buffer = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    let result = parse_trailing_name(
        state,
        "read needs a result name on the same line\n  Hint: Write 'read <ptr>, <size>, <name>'",
    )?;
    Ok(Instruction::Read {
        buffer,
        size,
        result,
    })
}

fn parse_writebyte<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let value = parse_value(state)?;
    let result = parse_trailing_name(
        state,
        "writebyte needs a result name on the same line\n  Hint: Write 'writebyte <value>, <name>'",
    )?;
    Ok(Instruction::WriteByte { value, result })
}

fn parse_readbyte<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let result = state.parse_same_line_identifier(
        "readbyte needs a result name on the same line\n  Hint: Write 'readbyte <name>'",
    )?;
    Ok(Instruction::ReadByte { result })
}

fn parse_int_to_float<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_str)?;
    state.expect_char('.')?;
    let target_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_str)?;

    let signed = match source_type {
        PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::I128 => true,
        PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64
        | PrimitiveType::U128 => false,
        _ => {
            return Err(state.error(format!(
                "inttofp source must be an integer type, found {source_str}"
            )));
        }
    };
    if !matches!(target_type, PrimitiveType::F32 | PrimitiveType::F64) {
        return Err(state.error(format!(
            "inttofp target must be f32 or f64, found {target_str}"
        )));
    }

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    Ok(Instruction::IntToFloat {
        result,
        signed,
        source_type,
        target_type,
        value,
    })
}

fn parse_float_to_int<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_str)?;
    state.expect_char('.')?;
    let target_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_str)?;

    if !matches!(source_type, PrimitiveType::F32 | PrimitiveType::F64) {
        return Err(state.error(format!(
            "fptoint source must be f32 or f64, found {source_str}"
        )));
    }
    let signed = match target_type {
        PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::I128 => true,
        PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64
        | PrimitiveType::U128 => false,
        _ => {
            return Err(state.error(format!(
                "fptoint target must be an integer type, found {target_str}"
            )));
        }
    };

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    Ok(Instruction::FloatToInt {
        result,
        signed,
        source_type,
        target_type,
        value,
    })
}

/// Parses `fpcast.f32.f64` and `fpcast.f64.f32`.
fn parse_float_convert<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_str)?;
    state.expect_char('.')?;
    let target_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_str)?;

    let widths_ok = matches!(
        (source_type, target_type),
        (PrimitiveType::F32, PrimitiveType::F64) | (PrimitiveType::F64, PrimitiveType::F32)
    );
    if !widths_ok {
        return Err(state.error(format!(
            "fpcast converts between f32 and f64, found {source_str} to {target_str}"
        )));
    }

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    Ok(Instruction::FloatConvert {
        result,
        source_type,
        target_type,
        value,
    })
}

fn parse_trunc<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_type_str)?;

    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_type_str)?;

    if source_type == target_type {
        return Err(state.error(format!(
            "Source and target types must be different for conversion: {source_type_str}"
        )));
    }

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    Ok(Instruction::Trunc {
        result,
        source_type,
        target_type,
        value,
    })
}

fn parse_extend<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_str)?;
    state.expect_char('.')?;
    let target_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_str)?;

    let Some(source_bits) = integer_bits(source_type) else {
        return Err(state.error(format!(
            "extend source must be an integer or bool, found {source_str}"
        )));
    };
    let target_bits = match integer_bits(target_type) {
        Some(bits) if target_type != PrimitiveType::Bool => bits,
        _ => {
            return Err(state.error(format!(
                "extend target must be an integer, found {target_str}"
            )));
        }
    };
    if target_bits == source_bits {
        return Err(state.error(format!(
            "extend from {source_str} to {target_str} keeps the width\n  Hint: Use 'bitcast.{source_str}.{target_str}' to reinterpret the bits"
        )));
    }
    if target_bits < source_bits {
        return Err(state.error(format!(
            "extend from {source_str} to {target_str} narrows\n  Hint: Use 'trunc.{source_str}.{target_str}' to narrow"
        )));
    }

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    let signed = matches!(
        source_type,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
    );
    if signed {
        Ok(Instruction::SignExtend {
            result,
            source_type,
            target_type,
            value,
        })
    } else {
        Ok(Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }
}

/// Bit width of an integer or bool type. Floats and pointers have no integer width.
fn integer_bits(ty: PrimitiveType) -> Option<u32> {
    match ty {
        PrimitiveType::Bool => Some(1),
        PrimitiveType::I8 | PrimitiveType::U8 => Some(8),
        PrimitiveType::I16 | PrimitiveType::U16 => Some(16),
        PrimitiveType::I32 | PrimitiveType::U32 => Some(32),
        PrimitiveType::I64 | PrimitiveType::U64 => Some(64),
        PrimitiveType::I128 | PrimitiveType::U128 => Some(128),
        PrimitiveType::F32 | PrimitiveType::F64 | PrimitiveType::Ptr => None,
    }
}

fn parse_bitcast<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_type_str)?;

    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_type_str)?;

    let value = parse_typed_operand(state, &Type::Primitive(source_type))?;
    Ok(Instruction::Bitcast {
        result,
        source_type,
        target_type,
        value,
    })
}

fn parse_switch<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_primitive_type_suffix(state)?;
    let value = parse_typed_operand(state, &Type::Primitive(ty))?;
    state.expect_char(',')?;
    let default = state.parse_label_identifier()?;

    let mut cases = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() != Some(',') {
            break;
        }
        state.expect_char(',')?;
        state.skip_whitespace_and_comments();
        if state.current_char() != Some('[') {
            break;
        }
        state.expect_char('[')?;
        let lit_value = parse_typed_operand(state, &Type::Primitive(ty))?;
        let lit = match lit_value {
            Value::Constant(l) => l,
            _ => {
                return Err(
                    state.error("switch case expects a literal constant, found non-constant value")
                );
            }
        };
        state.expect_char(',')?;
        let label = state.parse_label_identifier()?;
        state.expect_char(']')?;
        cases.push((lit, label));
    }

    Ok(Instruction::Switch {
        ty,
        value,
        default,
        cases,
    })
}

fn parse_select<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let cond = parse_value(state)?;
    state.expect_char(',')?;
    let true_val = parse_typed_operand(state, &ty)?;
    state.expect_char(',')?;
    let false_val = parse_typed_operand(state, &ty)?;
    Ok(Instruction::Select {
        result,
        ty,
        cond,
        true_val,
        false_val,
    })
}

fn parse_primitive_from_ident(
    state: &ParserState<'_>,
    ident: &str,
) -> Result<PrimitiveType, LaminaError> {
    match ident {
        "i8" => Ok(PrimitiveType::I8),
        "i16" => Ok(PrimitiveType::I16),
        "i32" => Ok(PrimitiveType::I32),
        "i64" => Ok(PrimitiveType::I64),
        "u8" => Ok(PrimitiveType::U8),
        "u16" => Ok(PrimitiveType::U16),
        "u32" => Ok(PrimitiveType::U32),
        "u64" => Ok(PrimitiveType::U64),
        "i128" => Ok(PrimitiveType::I128),
        "u128" => Ok(PrimitiveType::U128),
        "f32" => Ok(PrimitiveType::F32),
        "f64" => Ok(PrimitiveType::F64),
        "bool" => Ok(PrimitiveType::Bool),
        "char" => Err(removed_char_error(state)),
        "ptr" => Ok(PrimitiveType::Ptr),
        other => Err(state.error(format!("Invalid primitive type in conversion: {other}"))),
    }
}

/// Parses an operand whose type the instruction states. A primitive type makes an
/// integer literal range-checked against it, the same check a global initializer gets.
fn parse_typed_operand<'a>(
    state: &mut ParserState<'a>,
    ty: &Type<'a>,
) -> Result<Value<'a>, LaminaError> {
    match ty {
        Type::Primitive(_) => parse_value_with_type_hint(state, ty),
        _ => parse_value(state),
    }
}
