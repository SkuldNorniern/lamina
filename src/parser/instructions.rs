//! Instruction parsing for Lamina IR.

use crate::{
    LaminaError,
    ir::{
        AllocType, AtomicBinOp, BinaryOp, CmpOp, Identifier, Instruction, MemoryOrdering,
        PrimitiveType, Type, UnaryOp, Value,
    },
    parser::{
        globals::parse_value_with_type_hint,
        state::{ParserState, TypedCallArg},
        types::{parse_type, removed_char_error},
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
                "Instructions with two results are not supported yet\n  Hint: atomic_cmpxchg yields a value and a success flag, which the text syntax cannot express yet",
            ));
        }
        state.expect_char('=')?;
        state.skip_whitespace_and_comments();
        let opcode_str = state.parse_identifier_str()?;

        match opcode_str {
            "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr" | "rotl"
            | "rotr" => parse_binary_op(state, result, opcode_str),
            "neg" | "not" | "sqrt" | "popcount" | "ctlz" | "cttz" | "bswap" => {
                parse_unary_op(state, result, opcode_str)
            }
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" => parse_cmp_op(state, result, opcode_str),
            "zext" => parse_zext(state, result),
            "trunc" => parse_trunc(state, result),
            "sext" => parse_sext(state, result),
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
                "'getfieldptr <ptr>, <index>'",
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
                "a struct in memory: 'stackalloc.<type>' then 'getfieldptr <ptr>, <index>'",
            )),
            "call" => parse_call(state, Some(result)),
            "funcaddr" => parse_func_addr(state, result),
            "callptr" => parse_call_indirect(state, Some(result)),
            "phi" => parse_phi(state, result),
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
            "atomic_cmpxchg" => Err(state.error(
                "atomic_cmpxchg is not supported yet: it yields two results\n  Hint: Use atomic_exchange or an atomic_add-style operation for now",
            )),
            "atomic_store" | "fence" => Err(state.error(format!(
                "{opcode_str} produces no result\n  Hint: Drop the '%name =' prefix"
            ))),
            _ => {
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
            "heapfree" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'call @free(<ptr>)' after declaring '@extern fn @free(ptr %p) -> void'",
            )),
            "dealloc" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'call @free(<ptr>)' after declaring '@extern fn @free(ptr %p) -> void'",
            )),
            "call" => parse_call(state, None),
            "callptr" => parse_call_indirect(state, None),
            "print" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'writebyte <value>' for one byte, 'write <ptr>, <size>' for raw bytes, or a call to a printf-style @extern for formatted output",
            )),
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
            "memcpy" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'call @memcpy(<dst>, <src>, <size>)' after declaring '@extern fn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr'",
            )),
            "memmove" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'call @memmove(<dst>, <src>, <size>)' after declaring '@extern fn @memmove(ptr %d, ptr %s, i64 %n) -> ptr'",
            )),
            "memset" => Err(removed_opcode_error(
                state,
                opcode_str,
                "'call @memset(<dst>, <value>, <size>)' after declaring '@extern fn @memset(ptr %d, i32 %v, i64 %n) -> ptr'",
            )),
            "atomic_store" => parse_atomic_store(state),
            "fence" => parse_fence(state),
            "atomic_load" | "atomic_add" | "atomic_sub" | "atomic_and" | "atomic_or"
            | "atomic_xor" | "atomic_exchange" | "atomic_cmpxchg" => Err(state.error(format!(
                "{opcode_str} produces a value\n  Hint: Write '%name = {opcode_str}.<type> ...'"
            ))),
            _ => {
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

fn parse_fence<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ordering = parse_memory_ordering_name(state)?;
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
    let ty = parse_primitive_type_suffix(state)?;
    if matches!(op, BinaryOp::Rotl | BinaryOp::Rotr) && !is_integer_type(ty) {
        return Err(state.error(format!("{op_str} takes an integer type, found {ty}")));
    }
    let lhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    state.expect_char(',')?;
    let rhs = parse_typed_operand(state, &Type::Primitive(ty))?;
    Ok(Instruction::Binary {
        op,
        result,
        ty,
        lhs,
        rhs,
    })
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

/// Parses `stackalloc.T`.
fn parse_alloc<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
    alloc_type: AllocType,
) -> Result<Instruction<'a>, LaminaError> {
    let allocated_ty = parse_type_suffix(state)?;
    Ok(Instruction::Alloc {
        result,
        alloc_type,
        allocated_ty,
    })
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
    Ok(Instruction::Load {
        result,
        ty,
        ptr,
        volatile,
    })
}

fn parse_store<'a>(
    state: &mut ParserState<'a>,
    volatile: bool,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let value = match &ty {
        Type::Primitive(PrimitiveType::I128 | PrimitiveType::U128) => {
            parse_typed_operand(state, &ty)?
        }
        _ => parse_value(state)?,
    };
    Ok(Instruction::Store {
        ty,
        ptr,
        value,
        volatile,
    })
}

fn parse_getfield<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    if state.current_char() == Some('.') {
        return Err(state.error(
            "getfieldptr takes no type suffix, because its offset comes from the struct layout\n  Hint: Write 'getfieldptr <ptr>, <index>'. For an array element, use 'getelem.<type> <ptr>, <index>', which needs the element type to scale the index",
        ));
    }

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
        field_index,
        field_byte_offset: None,
    })
}

fn parse_getelem<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    require_type_suffix(
        state,
        "getelem takes its element type as a suffix, because the index is scaled by the element size\n  Hint: Write 'getelem.<type> <ptr>, <index>', e.g. 'getelem.i8 %buf, 0'. For a struct field, use 'getfieldptr <ptr>, <index>', which takes no suffix",
    )?;
    let element_type = parse_type_suffix(state)?;
    let array_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let index = parse_value(state)?;
    state.skip_whitespace_and_comments();
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
        "ptrtoint takes its target type as a suffix\n  Hint: Write 'ptrtoint.i64 %ptr' instead of 'ptrtoint %ptr, .i64'",
    )?;
    let target_type = parse_type_suffix(state)?;
    let ptr_value = parse_value(state)?;

    let target_primitive_type = match target_type {
        Type::Primitive(pt) => pt,
        _ => {
            return Err(LaminaError::ParsingError(
                "Expected primitive type for ptrtoint".to_string(),
            ));
        }
    };

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
        "inttoptr takes its target type as a suffix\n  Hint: Write 'inttoptr.i64 %addr' instead of 'inttoptr %addr, .i64'",
    )?;
    let target_type = parse_type_suffix(state)?;
    let int_value = parse_value(state)?;

    let target_primitive_type = match target_type {
        Type::Primitive(pt) => pt,
        _ => {
            return Err(LaminaError::ParsingError(
                "Expected primitive type for inttoptr".to_string(),
            ));
        }
    };

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
    let func_name = state.parse_function_identifier()?;
    state.expect_char('(')?;
    let mut args = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance();
            break;
        }
        let index = args.len();
        if starts_untyped_value(state) {
            args.push(parse_value(state)?);
        } else {
            let position = state.position();
            let ty = parse_type(state).map_err(|_| untyped_call_error(state))?;
            state.skip_whitespace_and_comments();
            if matches!(state.current_char(), None | Some(')') | Some(',')) {
                return Err(untyped_call_error(state));
            }
            args.push(parse_value(state)?);
            state.typed_call_args.push(TypedCallArg {
                callee: func_name,
                index,
                ty,
                position,
            });
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
        let value = parse_value(state)?;
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

fn parse_phi<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let mut incoming = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() != Some('[') {
            break; // End of incoming values
        }
        state.expect_char('[')?;
        let value = parse_typed_operand(state, &ty)?;
        state.expect_char(',')?;
        let label = state.parse_label_identifier()?;
        state.expect_char(']')?;
        incoming.push((value, label));

        state.skip_whitespace_and_comments();
        if state.current_char() != Some(',') {
            break;
        }
        state.expect_char(',')?;
    }
    if incoming.is_empty() {
        return Err(state.error("phi instruction requires at least one incoming value"));
    }
    Ok(Instruction::Phi {
        result,
        ty,
        incoming,
    })
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

fn parse_write<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let buffer = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    state.expect_char(',')?;
    let result = state.parse_identifier_str()?;
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
    state.expect_char(',')?;
    let result = state.parse_identifier_str()?;
    Ok(Instruction::Read {
        buffer,
        size,
        result,
    })
}

fn parse_writebyte<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let value = parse_value(state)?;
    state.expect_char(',')?;
    let result = state.parse_identifier_str()?;
    Ok(Instruction::WriteByte { value, result })
}

fn parse_readbyte<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let result = state.parse_identifier_str()?;
    Ok(Instruction::ReadByte { result })
}

fn parse_zext<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = match source_type_str {
        "i8" => PrimitiveType::I8,
        "i16" => PrimitiveType::I16,
        "i32" => PrimitiveType::I32,
        "i64" => PrimitiveType::I64,
        "u8" => PrimitiveType::U8,
        "u16" => PrimitiveType::U16,
        "u32" => PrimitiveType::U32,
        "u64" => PrimitiveType::U64,
        "i128" => PrimitiveType::I128,
        "u128" => PrimitiveType::U128,
        "f32" => PrimitiveType::F32,
        "f64" => PrimitiveType::F64,
        "bool" => PrimitiveType::Bool,
        "char" => return Err(removed_char_error(state)),
        _ => return Err(state.error(format!("Invalid source type for zext: {source_type_str}"))),
    };

    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = match target_type_str {
        "i8" => PrimitiveType::I8,
        "i16" => PrimitiveType::I16,
        "i32" => PrimitiveType::I32,
        "i64" => PrimitiveType::I64,
        "u8" => PrimitiveType::U8,
        "u16" => PrimitiveType::U16,
        "u32" => PrimitiveType::U32,
        "u64" => PrimitiveType::U64,
        "i128" => PrimitiveType::I128,
        "u128" => PrimitiveType::U128,
        "f32" => PrimitiveType::F32,
        "f64" => PrimitiveType::F64,
        "bool" => PrimitiveType::Bool,
        "char" => return Err(removed_char_error(state)),
        _ => return Err(state.error(format!("Invalid target type for zext: {target_type_str}"))),
    };

    if source_type == target_type {
        return Err(state.error(format!(
            "Source and target types must be different for conversion: {source_type_str}"
        )));
    }

    let value = parse_value(state)?;
    Ok(Instruction::ZeroExtend {
        result,
        source_type,
        target_type,
        value,
    })
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

    let value = parse_value(state)?;
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

    let value = parse_value(state)?;
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

    let value = parse_value(state)?;
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

    let value = parse_value(state)?;
    Ok(Instruction::Trunc {
        result,
        source_type,
        target_type,
        value,
    })
}

fn parse_sext<'a>(
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

    let value = parse_value(state)?;
    Ok(Instruction::SignExtend {
        result,
        source_type,
        target_type,
        value,
    })
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

    let value = parse_value(state)?;
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
    let true_val = parse_value(state)?;
    state.expect_char(',')?;
    let false_val = parse_value(state)?;
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
