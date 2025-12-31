//! Instruction parsing for Lamina IR.

use super::state::ParserState;
use super::types::parse_type;
use super::values::parse_value;
use crate::{
    AllocType, BinaryOp, CmpOp, Identifier, Instruction, LaminaError, PrimitiveType, Type, Value,
};

/// Parses a single instruction from the input.
pub fn parse_instruction<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    let _start_pos = state.position();

    if state.current_char() == Some('%') {
        let result = state.parse_value_identifier()?;
        state.expect_char('=')?;
        state.skip_whitespace_and_comments();
        let opcode_str = state.parse_identifier_str()?;

        match opcode_str {
            "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr" => {
                parse_binary_op(state, result, opcode_str)
            }
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" => parse_cmp_op(state, result, opcode_str),
            "zext" => parse_zext(state, result),
            "trunc" => parse_trunc(state, result),
            "sext" => parse_sext(state, result),
            "bitcast" => parse_bitcast(state, result),
            "select" => parse_select(state, result),
            "alloc" => parse_alloc(state, result),
            "load" => parse_load(state, result),
            "getfield" => parse_getfield(state, result),
            "getfieldptr" => parse_getfield(state, result),
            "getelem" => parse_getelem(state, result),
            "getelementptr" => parse_getelem(state, result),
            "ptrtoint" => parse_ptrtoint(state, result),
            "inttoptr" => parse_inttoptr(state, result),
            "tuple" => parse_tuple(state, result),
            "extract" => parse_extract_tuple(state, result),
            "call" => parse_call(state, Some(result)),
            "phi" => parse_phi(state, result),
            "write" => parse_write_assignment(state, result),
            "read" => parse_read_assignment(state, result),
            "writebyte" => parse_writebyte_assignment(state, result),
            "readbyte" => parse_readbyte_assignment(state, result),
            "writeptr" => parse_writeptr_assignment(state, result),
            _ => {
                let common_ops = vec!["add", "sub", "mul", "div", "load", "store", "call", "alloc"];
                let suggestion = if common_ops.iter().any(|&op| opcode_str.starts_with(op)) {
                    format!("Did you mean one of: {}? (check spelling)", common_ops.join(", "))
                } else {
                    "Valid opcodes include: add, sub, mul, div, load, store, call, alloc, and many others".to_string()
                };
                Err(state.error(format!(
                    "Unknown opcode after assignment: '{}'\n  Hint: {}",
                    opcode_str, suggestion
                )))
            },
        }
    } else {
        let opcode_str = state.parse_identifier_str()?;
        match opcode_str {
            "store" => parse_store(state),
            "switch" => parse_switch(state),
            "br" => parse_br(state),
            "jmp" => parse_jmp(state),
            "ret" => parse_ret(state),
            "dealloc" => parse_dealloc(state),
            "call" => parse_call(state, None),
            "print" => parse_print(state),
            "write" => parse_write(state),
            "read" => parse_read(state),
            "writebyte" => parse_writebyte(state),
            "readbyte" => parse_readbyte(state),
            "writeptr" => parse_writeptr(state),
            "memcpy" => parse_memcpy(state),
            "memmove" => parse_memmove(state),
            "memset" => parse_memset(state),
            _ => {
                let common_ops = vec!["ret", "jmp", "br", "call", "print", "store"];
                let suggestion = if common_ops.iter().any(|&op| opcode_str.starts_with(op)) {
                    format!("Did you mean one of: {}? (check spelling)", common_ops.join(", "))
                } else {
                    "Valid instruction opcodes include: ret, jmp, br, call, print, store, and many others".to_string()
                };
                Err(state.error(format!(
                    "Unknown instruction opcode: '{}'\n  Hint: {}",
                    opcode_str, suggestion
                )))
            },
        }
    }
}

/// Parses a primitive type suffix (e.g., `.i32`).
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
        "f32" => Ok(PrimitiveType::F32),
        "f64" => Ok(PrimitiveType::F64),
        "bool" => Ok(PrimitiveType::Bool),
        "char" => Ok(PrimitiveType::Char),
        "ptr" => Ok(PrimitiveType::Ptr),
        _ => {
            let valid_types = "i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, char, ptr";
            Err(state.error(format!(
                "Expected primitive type suffix, found '.{}'\n  Hint: Valid type suffixes are: {}",
                type_str, valid_types
            )))
        },
    }
}

/// Parses a type suffix (e.g., `.i32`, `.struct`).
fn parse_type_suffix<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.expect_char('.')?;
    parse_type(state) // Can be any type after the dot
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
        _ => unreachable!(),
    };
    let ty = parse_primitive_type_suffix(state)?;
    let lhs = parse_value(state)?;
    state.expect_char(',')?;
    let rhs = parse_value(state)?;
    Ok(Instruction::Binary {
        op,
        result,
        ty,
        lhs,
        rhs,
    })
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
    let lhs = parse_value(state)?;
    state.expect_char(',')?;
    let rhs = parse_value(state)?;
    Ok(Instruction::Cmp {
        op,
        result,
        ty,
        lhs,
        rhs,
    })
}

/// Parses an allocation instruction.
fn parse_alloc<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;

    let peek_str = state.peek_slice(5).unwrap_or("");
    if peek_str.starts_with("stack") {
        state.consume_keyword("stack")?;
        let allocated_ty = parse_type(state)?;
        return Ok(Instruction::Alloc {
            result,
            alloc_type: AllocType::Stack,
            allocated_ty,
        });
    } else if peek_str.starts_with("heap") {
        state.consume_keyword("heap")?;
        let allocated_ty = parse_type(state)?;
        return Ok(Instruction::Alloc {
            result,
            alloc_type: AllocType::Heap,
            allocated_ty,
        });
    }

    state.consume_keyword("ptr")?;
    state.expect_char('.')?;
    let alloc_type_str = state.parse_identifier_str()?;
    let alloc_type = match alloc_type_str {
        "stack" => AllocType::Stack,
        "heap" => AllocType::Heap,
        _ => return Err(state.error(format!("Invalid allocation type: {}", alloc_type_str))),
    };
    let allocated_ty = parse_type(state)?;
    state.skip_whitespace_and_comments();
    Ok(Instruction::Alloc {
        result,
        alloc_type,
        allocated_ty,
    })
}

/// Parses a load instruction.
fn parse_load<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let ptr = parse_value(state)?;
    Ok(Instruction::Load { result, ty, ptr })
}

/// Parses a store instruction.
fn parse_store<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let ty = parse_type_suffix(state)?;
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let value = parse_value(state)?;
    Ok(Instruction::Store { ty, ptr, value })
}

/// Parses a getfieldptr instruction.
fn parse_getfield<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let _current_pos = state.position();
    let has_dot = state.current_char() == Some('.');

    if has_dot {
        state.expect_char('.')?;
        state.consume_keyword("ptr")?;
    }

    let struct_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let field_index = state.parse_integer()? as usize;
    Ok(Instruction::GetFieldPtr {
        result,
        struct_ptr,
        field_index,
    })
}

/// Parses a getelementptr instruction.
fn parse_getelem<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let _current_pos = state.position();
    let has_dot = state.current_char() == Some('.');

    if has_dot {
        state.expect_char('.')?;
        state.consume_keyword("ptr")?;
    }

    let array_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let index = parse_value(state)?;
    state.expect_char(',')?;
    let element_type = if state.current_char() == Some('.') {
        parse_primitive_type_suffix(state)?
    } else {
        let type_str = state.parse_identifier_str()?;
        match type_str {
            "i8" => PrimitiveType::I8,
            "i16" => PrimitiveType::I16,
            "i32" => PrimitiveType::I32,
            "i64" => PrimitiveType::I64,
            "u8" => PrimitiveType::U8,
            "u16" => PrimitiveType::U16,
            "u32" => PrimitiveType::U32,
            "u64" => PrimitiveType::U64,
            "f32" => PrimitiveType::F32,
            "f64" => PrimitiveType::F64,
            "bool" => PrimitiveType::Bool,
            "char" => PrimitiveType::Char,
            "ptr" => PrimitiveType::Ptr,
            _ => return Err(state.error(format!("Expected primitive type, found '{}'", type_str))),
        }
    };

    Ok(Instruction::GetElemPtr {
        result,
        array_ptr,
        index,
        element_type,
    })
}

/// Parses a ptrtoint instruction.
fn parse_ptrtoint<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let ptr_value = parse_value(state)?;
    state.expect_char(',')?;
    let target_type = parse_type_suffix(state)?;

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

/// Parses an inttoptr instruction.
fn parse_inttoptr<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    let int_value = parse_value(state)?;
    state.expect_char(',')?;
    let target_type = parse_type_suffix(state)?;

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

fn parse_tuple<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // tuple.T element1, element2, ... (Simplified: just parse elements)
    let mut elements = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        // Check if we're at end of input or block
        if state.is_eof() || state.current_char() == Some('}') {
            break; // No more elements
        }
        // Check if next token could be an operand
        let _current_pos = state.position(); // Prefixed with _
        if parse_value(state).is_ok() {
            // Check if this value is followed by '=' (next instruction) or ',' (tuple element)
            let _after_value_pos = state.position();
            state.skip_whitespace_and_comments();
            if state.current_char() == Some('=') {
                // This is the start of the next instruction, not a tuple element
                state.set_position(_current_pos); // Backtrack to before the value
                break;
            }
            // It's a tuple element
            state.set_position(_current_pos); // Backtrack
            let elem = parse_value(state)?;
            elements.push(elem);
            state.skip_whitespace_and_comments();
            if state.current_char() != Some(',') {
                break; // No more elements
            }
            state.expect_char(',')?;
        } else {
            state.set_position(_current_pos); // Backtrack
            break; // No more operands
        }
    }
    // Allow empty tuples (elements can be empty)
    // Skip any remaining whitespace/comments after tuple elements
    state.skip_whitespace_and_comments();
    Ok(Instruction::Tuple { result, elements })
}

/// Parses an extract tuple instruction.
fn parse_extract_tuple<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    state.expect_char('.')?;
    state.consume_keyword("tuple")?;
    let tuple_val = parse_value(state)?;
    state.expect_char(',')?;
    let index = state.parse_integer()? as usize;
    Ok(Instruction::ExtractTuple {
        result,
        tuple_val,
        index,
    })
}

/// Parses a call instruction.
fn parse_call<'a>(
    state: &mut ParserState<'a>,
    result: Option<Identifier<'a>>,
) -> Result<Instruction<'a>, LaminaError> {
    let func_name = state.parse_type_identifier()?; // @func
    state.expect_char('(')?;
    let mut args = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance(); // Consume ')'
            break;
        }
        args.push(parse_value(state)?);
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance(); // Consume ')'
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

fn parse_phi<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // phi.T [val1, label1], [val2, label2], ...
    let ty = parse_type_suffix(state)?;
    let mut incoming = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() != Some('[') {
            break; // End of incoming values
        }
        state.expect_char('[')?;
        let value = parse_value(state)?;
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
        return Err(state.error("phi instruction requires at least one incoming value".to_string()));
    }
    Ok(Instruction::Phi {
        result,
        ty,
        incoming,
    })
}

fn parse_br<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // br cond, label1, label2
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
    // jmp label
    let target_label = state.parse_label_identifier()?;
    Ok(Instruction::Jmp { target_label })
}

fn parse_ret<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // ret.T val  OR ret.void
    state.expect_char('.')?;
    state.skip_whitespace_and_comments();
    // Peek ahead to check for void
    if state.peek_slice(4) == Some("void") {
        state.consume_keyword("void")?;
        Ok(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    } else {
        let ty = parse_type(state)?;
        let value = parse_value(state)?;
        Ok(Instruction::Ret {
            ty,
            value: Some(value),
        })
    }
}

fn parse_dealloc<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // dealloc.heap ptr
    state.expect_char('.')?;
    state.consume_keyword("heap")?;
    let ptr = parse_value(state)?;
    Ok(Instruction::Dealloc { ptr })
}

/// Parses a print instruction.
fn parse_print<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    let value = parse_value(state)?;
    Ok(Instruction::Print { value })
}

/// Parses a write instruction (assignment form).
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

/// Parses a read instruction (assignment form).
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
    // Format: %result = writebyte %value
    let value = parse_value(state)?;
    Ok(Instruction::WriteByte { value, result })
}

fn parse_readbyte_assignment<'a>(
    _state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // Format: %result = readbyte
    Ok(Instruction::ReadByte { result })
}

fn parse_writeptr_assignment<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // Format: %result = writeptr %ptr
    let ptr = parse_value(state)?;
    Ok(Instruction::WritePtr { ptr, result })
}

// I/O instruction parsers - Non-assignment form (opcode ...)
fn parse_write<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: write %buffer, %size, %result
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
    // Format: read %buffer, %size, %result
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
    // Format: writebyte %value, %result
    let value = parse_value(state)?;
    state.expect_char(',')?;
    let result = state.parse_identifier_str()?;
    Ok(Instruction::WriteByte { value, result })
}

fn parse_readbyte<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: readbyte %result
    let result = state.parse_identifier_str()?;
    Ok(Instruction::ReadByte { result })
}

fn parse_writeptr<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: writeptr %ptr, %result
    let ptr = parse_value(state)?;
    state.expect_char(',')?;
    let result = state.parse_identifier_str()?;
    Ok(Instruction::WritePtr { ptr, result })
}

fn parse_memcpy<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: memcpy dst, src, size
    let dst = parse_value(state)?;
    state.expect_char(',')?;
    let src = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    Ok(Instruction::MemCpy { dst, src, size })
}

fn parse_memmove<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: memmove dst, src, size
    let dst = parse_value(state)?;
    state.expect_char(',')?;
    let src = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    Ok(Instruction::MemMove { dst, src, size })
}

fn parse_memset<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>, LaminaError> {
    // Format: memset dst, value, size
    let dst = parse_value(state)?;
    state.expect_char(',')?;
    let value = parse_value(state)?;
    state.expect_char(',')?;
    let size = parse_value(state)?;
    Ok(Instruction::MemSet { dst, value, size })
}

// Add parse function for zero extend
fn parse_zext<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // Format: zext.i32.i64 %value
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
        "f32" => PrimitiveType::F32,
        "f64" => PrimitiveType::F64,
        "bool" => PrimitiveType::Bool,
        "char" => PrimitiveType::Char,
        _ => return Err(state.error(format!("Invalid source type for zext: {}", source_type_str))),
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
        "f32" => PrimitiveType::F32,
        "f64" => PrimitiveType::F64,
        "bool" => PrimitiveType::Bool,
        "char" => PrimitiveType::Char,
        _ => return Err(state.error(format!("Invalid target type for zext: {}", target_type_str))),
    };

    // Allow all conversions for now - the IR will handle validation
    // Source type must be different from target type for meaningful conversion
    if source_type == target_type {
        return Err(state.error(format!(
            "Source and target types must be different for conversion: {}",
            source_type_str
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

fn parse_trunc<'a>(
    state: &mut ParserState<'a>,
    result: Identifier<'a>,
) -> Result<Instruction<'a>, LaminaError> {
    // Format: trunc.src_tgt %value  e.g. trunc.i64.i32 %x
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_type_str)?;

    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_type_str)?;

    // The IR validator is responsible for enforcing that target_type is narrower.
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
    // Format: sext.src_tgt %value  e.g. sext.i32.i64 %x
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = parse_primitive_from_ident(state, source_type_str)?;

    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = parse_primitive_from_ident(state, target_type_str)?;

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
    // Format: bitcast.src_tgt %value  e.g. bitcast.i32.f32 %x
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
    // Format: switch.T value, default, [lit1, label1], [lit2, label2], ...
    let ty = parse_primitive_type_suffix(state)?;
    let value = parse_value(state)?;
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
        // Switch cases restrict literals to those supported by IR Literal and MIR lowering.
        let lit_value = parse_value(state)?;
        let lit = match lit_value {
            Value::Constant(l) => l,
            _ => {
                return Err(state.error(
                    "switch case expects a literal constant, found non-constant value".to_string(),
                ));
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
    // Format: select.T cond, true_val, false_val
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
        "f32" => Ok(PrimitiveType::F32),
        "f64" => Ok(PrimitiveType::F64),
        "bool" => Ok(PrimitiveType::Bool),
        "char" => Ok(PrimitiveType::Char),
        "ptr" => Ok(PrimitiveType::Ptr),
        other => Err(state.error(format!("Invalid primitive type in conversion: {}", other))),
    }
}

// Helper to check if an instruction is a terminator
impl Instruction<'_> {
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Ret { .. } | Instruction::Jmp { .. } | Instruction::Br { .. }
        )
    }
}
