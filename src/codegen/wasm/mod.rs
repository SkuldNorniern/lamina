pub mod generate;
pub mod state;

use crate::{BinaryOp, Instruction, Literal, Module, PrimitiveType, Result, Type, Value};
use generate::{FloatType, IntegerType, ModuleExpression, NumericConstant, NumericType};
use state::Register;
use std::{collections::HashMap, io::Write};

pub fn get_wasm_type_primitive(
    ty: PrimitiveType,
    is_wasm64: bool,
) -> (NumericType, Option<FloatType>, Option<IntegerType>) {
    match ty {
        PrimitiveType::Bool
        | PrimitiveType::Char
        | PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32 => (NumericType::I32, None, Some(IntegerType::I32)),
        PrimitiveType::Ptr => {
            if is_wasm64 {
                (NumericType::I64, None, Some(IntegerType::I64))
            } else {
                (NumericType::I32, None, Some(IntegerType::I32))
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => (NumericType::I64, None, Some(IntegerType::I64)),
        PrimitiveType::F32 => (NumericType::F32, Some(FloatType::F32), None),
        PrimitiveType::F64 => (NumericType::F64, Some(FloatType::F64), None),
    }
}

pub fn get_wasm_type<'a>(
    ty: &Type<'a>,
    is_wasm64: bool,
) -> (NumericType, Option<FloatType>, Option<IntegerType>) {
    match ty {
        Type::Primitive(prim) => get_wasm_type_primitive(*prim, is_wasm64),
        Type::Array {
            element_type: _,
            size: _,
        }
        | Type::Struct(_)
        | Type::Tuple(_)
        | Type::Named(_) => get_wasm_type_primitive(PrimitiveType::Ptr, is_wasm64),
        Type::Void => (NumericType::I32, None, Some(IntegerType::I32)), // probably the best option for now
    }
}

pub fn get_size_primitive(ty: PrimitiveType, is_wasm64: bool) -> u8 {
    match ty {
        PrimitiveType::Bool | PrimitiveType::I8 | PrimitiveType::U8 => 8,
        PrimitiveType::I16 | PrimitiveType::U16 => 16,
        PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::Char => 32,
        PrimitiveType::Ptr => {
            if is_wasm64 {
                64
            } else {
                32
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => 64,
        PrimitiveType::F32 => 32,
        PrimitiveType::F64 => 64,
    }
}

pub fn get_size<'a>(ty: &Type<'a>, is_wasm64: bool, module: &'a Module<'a>) -> u64 {
    match ty {
        Type::Primitive(ty) => get_size_primitive(*ty, is_wasm64) as u64,
        Type::Array { element_type, size } => {
            get_size(element_type.as_ref(), is_wasm64, module) * *size
        }
        Type::Named(id) => get_size(
            &module.type_declarations.get(id).unwrap().ty,
            is_wasm64,
            module,
        ),
        Type::Struct(fields) => fields
            .iter()
            .map(|v| get_size(&v.ty, is_wasm64, module))
            .fold(0u64, |acc, v| acc + v),
        Type::Tuple(fields) => fields
            .iter()
            .map(|v| get_size(&v, is_wasm64, module))
            .fold(0u64, |acc, v| acc + v),
        Type::Void => 0,
    }
}

pub fn get_wasm_size_primitive(ty: PrimitiveType, is_wasm64: bool) -> u8 {
    match ty {
        PrimitiveType::Bool
        | PrimitiveType::Char
        | PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32 => 32,
        PrimitiveType::Ptr => {
            if is_wasm64 {
                64
            } else {
                32
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => 64,
        PrimitiveType::F32 => 32,
        PrimitiveType::F64 => 64,
    }
}

pub fn get_wasm_size<'a>(ty: &Type<'a>, is_wasm64: bool, module: &'a Module<'a>) -> u64 {
    match ty {
        Type::Primitive(prim) => get_wasm_size_primitive(*prim, is_wasm64) as u64,
        Type::Array { element_type, size } => get_wasm_size(element_type, is_wasm64, module) * size,
        Type::Struct(fields) => fields
            .iter()
            .map(|v| get_wasm_size(&v.ty, is_wasm64, module))
            .sum(),
        Type::Tuple(types) => types
            .iter()
            .map(|v| get_wasm_size(v, is_wasm64, module))
            .sum(),
        Type::Named(name) => get_wasm_size(
            &module.type_declarations.get(name).unwrap().ty,
            is_wasm64,
            module,
        ),
        Type::Void => 0,
    }
}

pub fn get_align_primitive(ty: PrimitiveType, is_wasm64: bool) -> u64 {
    match ty {
        PrimitiveType::Bool | PrimitiveType::I8 | PrimitiveType::U8 => 1,
        PrimitiveType::I16 | PrimitiveType::U16 => 2,
        PrimitiveType::Char | PrimitiveType::I32 | PrimitiveType::U32 => 4,
        PrimitiveType::Ptr => {
            if is_wasm64 {
                8
            } else {
                4
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => 8,
        PrimitiveType::F32 => 4,
        PrimitiveType::F64 => 8,
    }
}

pub fn get_align<'a>(ty: &Type<'a>, is_wasm64: bool, module: &'a Module<'a>) -> u64 {
    match ty {
        Type::Primitive(prim) => get_align_primitive(*prim, is_wasm64) as u64,
        Type::Array {
            element_type,
            size: _,
        } => get_align(element_type, is_wasm64, module),
        Type::Struct(fields) => fields
            .iter()
            .map(|v| get_align(&v.ty, is_wasm64, module))
            .max()
            .unwrap(),
        Type::Tuple(types) => types
            .iter()
            .map(|v| get_align(v, is_wasm64, module))
            .max()
            .unwrap(),
        Type::Named(name) => get_align(
            &module.type_declarations.get(name).unwrap().ty,
            is_wasm64,
            module,
        ),
        Type::Void => 1,
    }
}

pub fn get_wasm_align_primitive(ty: PrimitiveType, is_wasm64: bool) -> u64 {
    match ty {
        PrimitiveType::Bool
        | PrimitiveType::Char
        | PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32 => 4,
        PrimitiveType::Ptr => {
            if is_wasm64 {
                8
            } else {
                4
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => 8,
        PrimitiveType::F32 => 4,
        PrimitiveType::F64 => 8,
    }
}

pub fn get_wasm_align<'a>(ty: &Type<'a>, is_wasm64: bool, module: &'a Module<'a>) -> u64 {
    match ty {
        Type::Primitive(prim) => get_wasm_align_primitive(*prim, is_wasm64) as u64,
        Type::Array { element_type, size } => {
            get_wasm_align(element_type, is_wasm64, module) * size
        }
        Type::Struct(fields) => fields
            .iter()
            .map(|v| get_wasm_align(&v.ty, is_wasm64, module))
            .sum(),
        Type::Tuple(types) => types
            .iter()
            .map(|v| get_wasm_align(v, is_wasm64, module))
            .sum(),
        Type::Named(name) => get_wasm_align(
            &module.type_declarations.get(name).unwrap().ty,
            is_wasm64,
            module,
        ),
        Type::Void => 0,
    }
}

pub fn get_wasm_size_value<'a>(
    val: &Value<'a>,
    is_wasm64: bool,
    state: &'a state::CodegenState,
    locals: &HashMap<&'a str, (u64, NumericType)>,
) -> u64 {
    match val {
        Value::Constant(c) => match c {
            Literal::Bool(_) => 1,
            Literal::Char(_) => 4,
            Literal::I8(_) => 1,
            Literal::U8(_) => 1,
            Literal::I16(_) => 2,
            Literal::U16(_) => 2,
            Literal::I32(_) => 4,
            Literal::U32(_) => 4,

            Literal::String(s) => s.as_bytes().len() as u64,

            Literal::I64(_) => 8,
            Literal::U64(_) => 8,

            Literal::F32(_) => 4,
            Literal::F64(_) => 8,
        },
        Value::Global(id) => get_wasm_size_value(
            state.get_global_value(id).as_ref().unwrap(),
            is_wasm64,
            state,
            locals,
        ),
        Value::Variable(id) => locals
            .get(id)
            .map(|v| match v.1 {
                NumericType::F32 | NumericType::I32 => 4,
                NumericType::F64 | NumericType::I64 => 8,
            })
            .unwrap(),
    }
}

pub fn get_wasm_type_value<'a>(
    val: &Value<'a>,
    is_wasm64: bool,
    state: &'a state::CodegenState,
    locals: &HashMap<&'a str, (u64, NumericType)>,
) -> NumericType {
    match val {
        Value::Constant(c) => match c {
            Literal::Bool(_) => NumericType::I32,
            Literal::Char(_) => NumericType::I32,
            Literal::I8(_) => NumericType::I32,
            Literal::U8(_) => NumericType::I32,
            Literal::I16(_) => NumericType::I32,
            Literal::U16(_) => NumericType::I32,
            Literal::I32(_) => NumericType::I32,
            Literal::U32(_) => NumericType::I32,

            Literal::String(_) => {
                if is_wasm64 {
                    NumericType::I64
                } else {
                    NumericType::I32
                }
            }

            Literal::I64(_) => NumericType::I64,
            Literal::U64(_) => NumericType::I64,

            Literal::F32(_) => NumericType::F32,
            Literal::F64(_) => NumericType::F64,
        },
        Value::Global(id) => get_wasm_type_value(
            state.get_global_value(id).as_ref().unwrap(),
            is_wasm64,
            state,
            locals,
        ),
        Value::Variable(id) => locals.get(id).unwrap().1,
    }
}

fn get_wasm_type_for_return_primitive(ty: PrimitiveType, is_wasm64: bool) -> NumericType {
    match ty {
        PrimitiveType::Bool
        | PrimitiveType::Char
        | PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32 => NumericType::I32,
        PrimitiveType::Ptr => {
            if is_wasm64 {
                NumericType::I64
            } else {
                NumericType::I32
            }
        }
        PrimitiveType::I64 | PrimitiveType::U64 => NumericType::I64,
        PrimitiveType::F32 => NumericType::F32,
        PrimitiveType::F64 => NumericType::F64,
    }
}

fn get_wasm_type_for_return<'a>(
    ty: &Type<'a>,
    is_wasm64: bool,
    module: &'a Module<'a>,
) -> Option<(NumericType, bool)> {
    match ty {
        Type::Primitive(ty) => Some((
            get_wasm_type_for_return_primitive(*ty, is_wasm64),
            *ty == PrimitiveType::Ptr,
        )),
        Type::Array {
            element_type: _,
            size: _,
        }
        | Type::Struct(_)
        | Type::Tuple(_) => Some((
            get_wasm_type_for_return_primitive(PrimitiveType::Ptr, is_wasm64),
            true,
        )),
        Type::Named(id) => get_wasm_type_for_return(
            &module.type_declarations.get(id).unwrap().ty,
            is_wasm64,
            module,
        ),
        Type::Void => None,
    }
}

fn is_const(
    value: &Value,
    ty: Option<&PrimitiveType>,
    is_wasm64: bool,
) -> Option<generate::NumericConstant> {
    match value {
        Value::Constant(lit) => Some(if let Some(ty) = ty {
            match ty {
                PrimitiveType::Bool => match lit {
                    Literal::Bool(v) => {
                        if *v {
                            generate::NumericConstant::I32(1)
                        } else {
                            generate::NumericConstant::I32(0)
                        }
                    }
                    _ => panic!("ICE: Attempted to assign non-bool to bool!"),
                },

                PrimitiveType::Char
                | PrimitiveType::U32
                | PrimitiveType::I32
                | PrimitiveType::I8
                | PrimitiveType::U8
                | PrimitiveType::I16
                | PrimitiveType::U16 => match lit {
                    Literal::Char(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::U32(v) => generate::NumericConstant::I32(*v),
                    Literal::I32(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::I8(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::U8(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::I16(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::U16(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::I64(v) => generate::NumericConstant::I32(*v as u32),
                    Literal::U64(v) => generate::NumericConstant::I32(*v as u32),
                    _ => return None,
                },

                PrimitiveType::Ptr => match lit {
                    Literal::I8(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v as u32)
                        }
                    }
                    Literal::U8(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v as u32)
                        }
                    }
                    Literal::I16(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v as u32)
                        }
                    }
                    Literal::U16(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v as u32)
                        }
                    }
                    Literal::I32(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v as u32)
                        }
                    }
                    Literal::U32(v) => {
                        if is_wasm64 {
                            generate::NumericConstant::I64(*v as u64)
                        } else {
                            generate::NumericConstant::I32(*v)
                        }
                    }
                    Literal::I64(v) if is_wasm64 => generate::NumericConstant::I64(*v as u64),
                    Literal::U64(v) if is_wasm64 => generate::NumericConstant::I64(*v),
                    _ => return None,
                },

                PrimitiveType::I64 | PrimitiveType::U64 => match lit {
                    Literal::I64(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::U64(v) => generate::NumericConstant::I64(*v),
                    Literal::U32(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::I32(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::I8(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::U8(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::I16(v) => generate::NumericConstant::I64(*v as u64),
                    Literal::U16(v) => generate::NumericConstant::I64(*v as u64),
                    _ => return None,
                },

                PrimitiveType::F32 => match lit {
                    Literal::I64(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::U64(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::U32(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::I32(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::I8(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::U8(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::I16(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::U16(v) => generate::NumericConstant::F32(*v as f32),
                    Literal::F32(v) => generate::NumericConstant::F32(*v),
                    Literal::F64(v) => generate::NumericConstant::F32(*v as f32),
                    _ => return None,
                },

                PrimitiveType::F64 => match lit {
                    Literal::I64(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::U64(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::U32(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::I32(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::I8(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::U8(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::I16(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::U16(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::F32(v) => generate::NumericConstant::F64(*v as f64),
                    Literal::F64(v) => generate::NumericConstant::F64(*v),
                    _ => return None,
                },
            }
        } else {
            match lit {
                Literal::Bool(v) => {
                    if *v {
                        generate::NumericConstant::I32(1)
                    } else {
                        generate::NumericConstant::I32(0)
                    }
                }
                Literal::Char(v) => generate::NumericConstant::I32(*v as u32),
                Literal::U8(v) => generate::NumericConstant::I32(*v as u32),
                Literal::I8(v) => generate::NumericConstant::I32(*v as u32),
                Literal::U16(v) => generate::NumericConstant::I32(*v as u32),
                Literal::I16(v) => generate::NumericConstant::I32(*v as u32),
                Literal::U32(v) => generate::NumericConstant::I32(*v as u32),
                Literal::I32(v) => generate::NumericConstant::I32(*v as u32),
                Literal::U64(v) => generate::NumericConstant::I64(*v as u64),
                Literal::I64(v) => generate::NumericConstant::I64(*v as u64),

                Literal::F32(v) => generate::NumericConstant::F32(*v),
                Literal::F64(v) => generate::NumericConstant::F64(*v),

                Literal::String(_) => return None,
            }
        }),
        _ => None,
    }
}

fn get_pointer(value: &Value, state: &state::CodegenState) -> Option<u64> {
    match value {
        Value::Constant(_) => None,
        Value::Global(id) | Value::Variable(id) => match state.get_global(id) {
            state::GlobalRef::Wasm(_) => None,
            state::GlobalRef::Memory(ptr) => Some(ptr),
        },
    }
}

fn create_load_reg<'a>(
    state: &state::CodegenState,
    value: &Value,
    ty: Option<&PrimitiveType>,
    locals: &HashMap<&'a str, (u64, NumericType)>,
    is_wasm64: bool,
) -> generate::WasmInstruction<'a> {
    match value {
        Value::Constant(_) => {
            generate::WasmInstruction::Const(is_const(value, ty, is_wasm64).unwrap())
        }
        Value::Global(id) => match state.get_global(id) {
            state::GlobalRef::Wasm(id) => {
                generate::WasmInstruction::GlobalGet(generate::Identifier::Index(id))
            }
            state::GlobalRef::Memory(_) => unreachable!(), // currently no integers larger then 64 bits
        },
        Value::Variable(id) => generate::WasmInstruction::LocalGet(generate::Identifier::Index(
            locals.get(id).unwrap().0,
        )),
    }
}

fn generate_load_reg<'a>(
    instructions: &mut Vec<generate::WasmInstruction<'a>>,
    state: &state::CodegenState,
    value: &Value,
    ty: Option<&PrimitiveType>,
    locals: &HashMap<&'a str, (u64, NumericType)>,
    is_wasm64: bool,
) {
    instructions.push(create_load_reg(state, value, ty, locals, is_wasm64))
}

fn generate_result<'a>(
    instructions: &mut Vec<generate::WasmInstruction>,
    locals: &mut HashMap<&'a str, (u64, NumericType)>,
    next_local_i: &mut u64,
    ty: NumericType,
    result: &'a str,
) {
    let output_i = locals
        .entry(result)
        .or_insert_with(|| {
            let output_i = *next_local_i;
            *next_local_i += 1;
            (output_i, ty)
        })
        .0;
    instructions.push(generate::WasmInstruction::LocalSet(
        generate::Identifier::Index(output_i),
    ))
}

fn generate_memory_write<'a>(
    instructions: &mut Vec<generate::WasmInstruction<'a>>,
    address: generate::WasmInstruction<'a>,
    value: generate::WasmInstruction<'a>,
    ty: NumericType,
    size: u8,
    mem: Option<generate::Identifier<'a>>,
) {
    instructions.push(address);
    instructions.push(value);
    match (ty, size) {
        (NumericType::I32 | NumericType::F32, 32) | (NumericType::I64 | NumericType::F64, 64) => {
            instructions.push(generate::WasmInstruction::Store(ty, mem))
        }
        (NumericType::I32 | NumericType::I64, 8) => instructions.push(
            generate::WasmInstruction::Store8(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I32 | NumericType::I64, 16) => instructions.push(
            generate::WasmInstruction::Store16(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I64, 32) => instructions.push(generate::WasmInstruction::I64_Store32(mem)),
        _ => panic!("ICE: Invalid type, size pair passed to generate_memory_write! Please report!"),
    }
}

fn generate_memory_read<'a>(
    instructions: &mut Vec<generate::WasmInstruction<'a>>,
    address: Option<generate::WasmInstruction<'a>>,
    ty: NumericType,
    size: u8,
    signed: bool,
    mem: Option<generate::Identifier<'a>>,
) {
    if let Some(address) = address {
        instructions.push(address);
    }
    match (ty, size, signed) {
        (NumericType::I32 | NumericType::F32, 32, _)
        | (NumericType::I64 | NumericType::F64, 64, _) => {
            instructions.push(generate::WasmInstruction::Load(ty, mem))
        }
        (NumericType::I32 | NumericType::I64, 8, false) => instructions.push(
            generate::WasmInstruction::Load8U(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I32 | NumericType::I64, 8, true) => instructions.push(
            generate::WasmInstruction::Load8S(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I32 | NumericType::I64, 16, false) => instructions.push(
            generate::WasmInstruction::Load16U(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I32 | NumericType::I64, 16, true) => instructions.push(
            generate::WasmInstruction::Load16S(ty.try_into().unwrap(), mem),
        ),
        (NumericType::I64, 32, false) => {
            instructions.push(generate::WasmInstruction::I64_Load32U(mem))
        }
        (NumericType::I64, 32, true) => {
            instructions.push(generate::WasmInstruction::I64_Load32S(mem))
        }
        _ => panic!(
            "ICE: Invalid type, size, signed triplet passed to generate_memory_read! Please report!"
        ),
    }
}

pub fn generate_wasm_assembly<'a, W: Write>(
    module: &'a Module<'a>,
    writer: &mut W,
    is_wasm64: bool,
) -> Result<()> {
    let mut state = state::CodegenState::new();

    for (name, decl) in &module.global_declarations {
        if decl.ty == Type::Primitive(PrimitiveType::F32) {
            state.add_wasm_global(decl.name, NumericType::F32, decl.initializer.clone());
            continue;
        }
        if decl.ty == Type::Primitive(PrimitiveType::F64) {
            state.add_wasm_global(decl.name, NumericType::F64, decl.initializer.clone());
            continue;
        }
        let size = get_wasm_size(&decl.ty, is_wasm64, module);

        // force compound types to be in memory for individual field access
        if size <= 32 && matches!(decl.ty, Type::Primitive(_)) {
            state.add_wasm_global(decl.name, NumericType::I32, decl.initializer.clone());
            continue;
        }
        if size <= 64 && matches!(decl.ty, Type::Primitive(_)) {
            state.add_wasm_global(decl.name, NumericType::I64, decl.initializer.clone());
            continue;
        }

        let align = get_wasm_align(&decl.ty, is_wasm64, module);

        let address = state.get_next_address(size, align);

        state.add_memory_register(
            Register {
                name,
                size,
                address,
                reg_type: state::RegType::Global,
            },
            decl.initializer.clone(),
        );
    }

    let mut func_mapping = HashMap::new();

    for (func_name, _) in &module.functions {
        func_mapping.insert(func_name, format!("f{}", func_mapping.len()));
    }

    let mut needs_console_log = false;

    for (name, func) in &module.functions {
        let mut func_instructions: Vec<generate::Instructions> = Vec::new();

        let mut blocks = vec![func.basic_blocks.get(func.entry_block).unwrap()];
        let mut block_mapping = HashMap::new();
        block_mapping.insert(func.entry_block.to_string(), 0);

        for (block_name, block) in &func.basic_blocks {
            if *block_name == func.entry_block {
                continue;
            }
            blocks.push(block);
            block_mapping.insert(block_name.to_string(), blocks.len() - 1);
        }

        let mut locals = HashMap::new();
        let mut next_local_i = 0u64;

        for param in &func.signature.params {
            let wasm_ty = get_wasm_type(&param.ty, is_wasm64).0;
            locals.entry(param.name).or_insert_with(|| {
                let output_i = next_local_i;
                next_local_i += 1;
                (output_i, wasm_ty)
            });
        }

        locals.insert("pc", (next_local_i, NumericType::I64));
        next_local_i += 1;

        for (i, block) in blocks.iter().enumerate() {
            let mut instructions: Vec<generate::WasmInstruction> =
                vec![generate::WasmInstruction::Comment(format!("Block {i}"))];
            for instr in &block.instructions {
                match instr {
                    Instruction::Binary {
                        op,
                        result,
                        ty,
                        lhs,
                        rhs,
                    } => {
                        let (wasm_ty, float_ty, int_ty) = get_wasm_type_primitive(*ty, is_wasm64);

                        generate_load_reg(
                            &mut instructions,
                            &state,
                            lhs,
                            Some(ty),
                            &locals,
                            is_wasm64,
                        );
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            rhs,
                            Some(ty),
                            &locals,
                            is_wasm64,
                        );

                        match op {
                            BinaryOp::Add => {
                                instructions.push(generate::WasmInstruction::Add(wasm_ty))
                            }
                            BinaryOp::Div => instructions.push(match wasm_ty {
                                NumericType::F32 | NumericType::F64 => {
                                    generate::WasmInstruction::DivF(float_ty.unwrap())
                                }
                                NumericType::I32 | NumericType::I64 => match ty {
                                    PrimitiveType::I8
                                    | PrimitiveType::I16
                                    | PrimitiveType::I32
                                    | PrimitiveType::I64 => {
                                        generate::WasmInstruction::DivS(int_ty.unwrap())
                                    }
                                    PrimitiveType::U8
                                    | PrimitiveType::U16
                                    | PrimitiveType::U32
                                    | PrimitiveType::U64 => {
                                        generate::WasmInstruction::DivU(int_ty.unwrap())
                                    }
                                    _ => unreachable!(),
                                },
                            }),
                            BinaryOp::Mul => {
                                instructions.push(generate::WasmInstruction::Mul(wasm_ty))
                            }
                            BinaryOp::Sub => {
                                instructions.push(generate::WasmInstruction::Sub(wasm_ty))
                            }
                        }

                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            wasm_ty,
                            result,
                        );
                    }
                    Instruction::Alloc {
                        result: _,
                        alloc_type: _,
                        allocated_ty: _,
                    }
                    | Instruction::Dealloc { ptr: _ } => {
                        return Err(crate::LaminaError::CodegenError(
                            crate::codegen::CodegenError::UnsupportedFeature(
                                crate::codegen::FeatureType::HeapAllocation,
                            ),
                        ));
                    }
                    Instruction::Br {
                        condition,
                        true_label,
                        false_label,
                    } => {
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            condition,
                            None,
                            &locals,
                            is_wasm64,
                        );

                        // WASM blocks are weird so we need to do a dispatch loop.

                        match get_wasm_type_value(condition, is_wasm64, &state, &locals) {
                            NumericType::I32 => {}
                            NumericType::I64 => {
                                instructions.push(generate::WasmInstruction::I32_WrapI64)
                            }
                            NumericType::F32 => {
                                instructions.push(generate::WasmInstruction::F64_Promote_F32);
                                instructions
                                    .push(generate::WasmInstruction::Reinterpret(NumericType::F64));
                            }
                            NumericType::F64 => {
                                instructions
                                    .push(generate::WasmInstruction::Reinterpret(NumericType::F64));
                            }
                        };

                        instructions.push(generate::WasmInstruction::If {
                            identifier: None,
                            result: None,
                            then: vec![
                                generate::WasmInstruction::Const(NumericConstant::I64(
                                    *block_mapping.get(&true_label.to_string()).unwrap() as u64,
                                )),
                                generate::WasmInstruction::LocalSet(generate::Identifier::Index(
                                    locals.get("pc").unwrap().0,
                                )),
                            ]
                            .into(),
                            r#else: Some(
                                vec![
                                    generate::WasmInstruction::Const(NumericConstant::I64(
                                        *block_mapping.get(&false_label.to_string()).unwrap()
                                            as u64,
                                    )),
                                    generate::WasmInstruction::LocalSet(
                                        generate::Identifier::Index(locals.get("pc").unwrap().0),
                                    ),
                                ]
                                .into(),
                            ),
                        });

                        instructions.push(generate::WasmInstruction::Br(
                            generate::Identifier::Name("l".into()),
                        ));
                    }
                    Instruction::Jmp { target_label } => {
                        instructions.push(generate::WasmInstruction::Br(
                            generate::Identifier::Name(
                                format!(
                                    "{}",
                                    block_mapping.get(&target_label.to_string()).unwrap()
                                )
                                .into(),
                            ),
                        ));
                    }
                    Instruction::Call {
                        result,
                        func_name,
                        args,
                    } => {
                        let mut reversed_args = args.clone();
                        reversed_args.reverse();

                        for arg in reversed_args {
                            generate_load_reg(
                                &mut instructions,
                                &state,
                                &arg,
                                None,
                                &locals,
                                is_wasm64,
                            );
                            instructions.push(generate::WasmInstruction::Comment(format!(
                                "Argument {arg}"
                            )));
                        }

                        instructions.push(generate::WasmInstruction::Call(
                            generate::Identifier::Name(func_mapping.get(func_name).unwrap().into()),
                        ));

                        if let Some(result) = result {
                            let ret = &module
                                .functions
                                .get(func_name)
                                .unwrap()
                                .signature
                                .return_type;

                            let (wasm_ty, is_ptr) =
                                get_wasm_type_for_return(&ret, is_wasm64, &module).unwrap();

                            if !is_ptr {
                                generate_result(
                                    &mut instructions,
                                    &mut locals,
                                    &mut next_local_i,
                                    wasm_ty,
                                    result,
                                );
                            } else {
                                let size = get_size(&ret, is_wasm64, &module);
                                let align = get_align(&ret, is_wasm64, &module);

                                let dest = state.get_next_address(size, align);

                                state.add_memory_register(
                                    Register {
                                        size,
                                        address: dest,
                                        name: result,
                                        reg_type: state::RegType::Local,
                                    },
                                    None,
                                );

                                // store the from location to a temporary location
                                generate_result(
                                    &mut instructions,
                                    &mut locals,
                                    &mut next_local_i,
                                    wasm_ty,
                                    "%%$#memresult0",
                                );

                                instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                    NumericConstant::I64(dest)
                                } else {
                                    NumericConstant::I32(dest as u32)
                                }));

                                instructions.push(generate::WasmInstruction::LocalGet(
                                    generate::Identifier::Index(
                                        locals.get("%%$#memresult0").unwrap().0,
                                    ),
                                ));

                                instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                    NumericConstant::I64(size)
                                } else {
                                    NumericConstant::I32(size as u32)
                                }));

                                instructions.push(generate::WasmInstruction::MemoryCopy);
                            }
                            instructions.push(generate::WasmInstruction::Comment(format!(
                                "Result to {result}"
                            )));
                        }
                    }
                    Instruction::Cmp {
                        op,
                        result,
                        ty,
                        lhs,
                        rhs,
                    } => {
                        let (wasm_ty, float_ty, int_ty) = get_wasm_type_primitive(*ty, is_wasm64);

                        let (is_lhs_zero, is_rhs_zero) = (
                            is_const(lhs, Some(ty), is_wasm64)
                                .map(|v| v.is_zero())
                                .unwrap_or(false),
                            is_const(rhs, Some(ty), is_wasm64)
                                .map(|v| v.is_zero())
                                .unwrap_or(false),
                        );

                        if !is_lhs_zero || *op != crate::CmpOp::Eq || !int_ty.is_some() {
                            generate_load_reg(
                                &mut instructions,
                                &state,
                                lhs,
                                Some(ty),
                                &locals,
                                is_wasm64,
                            );
                        }
                        if !is_rhs_zero || *op != crate::CmpOp::Eq || !int_ty.is_some() {
                            generate_load_reg(
                                &mut instructions,
                                &state,
                                rhs,
                                Some(ty),
                                &locals,
                                is_wasm64,
                            );
                        }

                        instructions.push(match op {
                            crate::CmpOp::Eq => match wasm_ty {
                                NumericType::I32 => {
                                    if is_lhs_zero || is_rhs_zero {
                                        generate::WasmInstruction::Eqz(IntegerType::I32)
                                    } else {
                                        generate::WasmInstruction::Eq(NumericType::I32)
                                    }
                                }
                                NumericType::I64 => {
                                    if is_lhs_zero || is_rhs_zero {
                                        generate::WasmInstruction::Eqz(IntegerType::I64)
                                    } else {
                                        generate::WasmInstruction::Eq(NumericType::I64)
                                    }
                                }
                                NumericType::F32 | NumericType::F64 => {
                                    generate::WasmInstruction::Eq(wasm_ty)
                                }
                            },
                            crate::CmpOp::Ge => match ty {
                                PrimitiveType::F32 | PrimitiveType::F64 => {
                                    generate::WasmInstruction::GeF(float_ty.unwrap())
                                }
                                PrimitiveType::I8
                                | PrimitiveType::I16
                                | PrimitiveType::I32
                                | PrimitiveType::I64 => {
                                    generate::WasmInstruction::GeS(int_ty.unwrap())
                                }
                                PrimitiveType::Bool
                                | PrimitiveType::Char
                                | PrimitiveType::Ptr
                                | PrimitiveType::U8
                                | PrimitiveType::U16
                                | PrimitiveType::U32
                                | PrimitiveType::U64 => {
                                    generate::WasmInstruction::GeU(int_ty.unwrap())
                                }
                            },
                            crate::CmpOp::Gt => match ty {
                                PrimitiveType::F32 | PrimitiveType::F64 => {
                                    generate::WasmInstruction::GtF(float_ty.unwrap())
                                }
                                PrimitiveType::I8
                                | PrimitiveType::I16
                                | PrimitiveType::I32
                                | PrimitiveType::I64 => {
                                    generate::WasmInstruction::GtS(int_ty.unwrap())
                                }
                                PrimitiveType::Bool
                                | PrimitiveType::Char
                                | PrimitiveType::Ptr
                                | PrimitiveType::U8
                                | PrimitiveType::U16
                                | PrimitiveType::U32
                                | PrimitiveType::U64 => {
                                    generate::WasmInstruction::GtU(int_ty.unwrap())
                                }
                            },
                            crate::CmpOp::Le => match ty {
                                PrimitiveType::F32 | PrimitiveType::F64 => {
                                    generate::WasmInstruction::LeF(float_ty.unwrap())
                                }
                                PrimitiveType::I8
                                | PrimitiveType::I16
                                | PrimitiveType::I32
                                | PrimitiveType::I64 => {
                                    generate::WasmInstruction::LeS(int_ty.unwrap())
                                }
                                PrimitiveType::Bool
                                | PrimitiveType::Char
                                | PrimitiveType::Ptr
                                | PrimitiveType::U8
                                | PrimitiveType::U16
                                | PrimitiveType::U32
                                | PrimitiveType::U64 => {
                                    generate::WasmInstruction::LeU(int_ty.unwrap())
                                }
                            },
                            crate::CmpOp::Lt => match ty {
                                PrimitiveType::F32 | PrimitiveType::F64 => {
                                    generate::WasmInstruction::LtF(float_ty.unwrap())
                                }
                                PrimitiveType::I8
                                | PrimitiveType::I16
                                | PrimitiveType::I32
                                | PrimitiveType::I64 => {
                                    generate::WasmInstruction::LtS(int_ty.unwrap())
                                }
                                PrimitiveType::Bool
                                | PrimitiveType::Char
                                | PrimitiveType::Ptr
                                | PrimitiveType::U8
                                | PrimitiveType::U16
                                | PrimitiveType::U32
                                | PrimitiveType::U64 => {
                                    generate::WasmInstruction::LtU(int_ty.unwrap())
                                }
                            },
                            crate::CmpOp::Ne => generate::WasmInstruction::Ne(wasm_ty),
                        });

                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            NumericType::I32,
                            result,
                        ); // boolean output
                    }
                    Instruction::ZeroExtend {
                        result,
                        source_type,
                        target_type,
                        value,
                    } => {
                        let (from_wasm_ty, _, _) = get_wasm_type_primitive(*source_type, is_wasm64);
                        let (to_wasm_ty, float_ty, int_ty) =
                            get_wasm_type_primitive(*target_type, is_wasm64);

                        if from_wasm_ty == to_wasm_ty {
                            continue;
                        }

                        generate_load_reg(
                            &mut instructions,
                            &state,
                            value,
                            Some(source_type),
                            &locals,
                            is_wasm64,
                        );

                        instructions.push(match (source_type, target_type) {
                            (_, PrimitiveType::I64) if from_wasm_ty == NumericType::I32 => {
                                generate::WasmInstruction::I64_ExtendI32S
                            }
                            (_, PrimitiveType::U64) if from_wasm_ty == NumericType::I32 => {
                                generate::WasmInstruction::I64_ExtendI32U
                            }

                            (PrimitiveType::I8 | PrimitiveType::U8, PrimitiveType::I32) => {
                                generate::WasmInstruction::Extend8S(IntegerType::I32)
                            }
                            (PrimitiveType::I8 | PrimitiveType::U8, PrimitiveType::I64) => {
                                generate::WasmInstruction::Extend8S(IntegerType::I64)
                            }

                            (PrimitiveType::I16 | PrimitiveType::U16, PrimitiveType::I32) => {
                                generate::WasmInstruction::Extend16S(IntegerType::I32)
                            }
                            (PrimitiveType::I16 | PrimitiveType::U16, PrimitiveType::I64) => {
                                generate::WasmInstruction::Extend16S(IntegerType::I64)
                            }

                            _ if from_wasm_ty == NumericType::I32
                                && to_wasm_ty == NumericType::I64 =>
                            {
                                generate::WasmInstruction::I32_WrapI64
                            }

                            (PrimitiveType::F32, _) if int_ty == Some(IntegerType::I32) => {
                                generate::WasmInstruction::Reinterpret(NumericType::F32)
                            }
                            (PrimitiveType::F64, _) if int_ty == Some(IntegerType::I64) => {
                                generate::WasmInstruction::Reinterpret(NumericType::F64)
                            }

                            (PrimitiveType::I32, _) if float_ty == Some(FloatType::F32) => {
                                generate::WasmInstruction::Reinterpret(NumericType::I32)
                            }
                            (PrimitiveType::I64, _) if float_ty == Some(FloatType::F64) => {
                                generate::WasmInstruction::Reinterpret(NumericType::I64)
                            }

                            (PrimitiveType::F32, PrimitiveType::F64) => {
                                generate::WasmInstruction::F64_Promote_F32
                            }
                            (PrimitiveType::F64, PrimitiveType::F32) => {
                                generate::WasmInstruction::F32_Demote_F64
                            }

                            _ => unreachable!("uh oh something is broken"),
                        });

                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            to_wasm_ty,
                            result,
                        );
                    }
                    Instruction::Tuple { result, elements } => {
                        let mut align = 1u64;
                        let size = elements.iter().fold(0u64, |last, v| {
                            let size = get_wasm_size_value(v, is_wasm64, &state, &locals);
                            let e_align = (size as f64).log2();
                            if e_align.is_normal()
                                && e_align.fract() == 0.0
                                && e_align as u64 > align
                            {
                                align = e_align as u64;
                            }
                            last + size
                        });

                        let mut address = state.get_next_address(size, align);

                        state.add_memory_register(
                            Register {
                                size,
                                address,
                                name: result,
                                reg_type: state::RegType::Local,
                            },
                            None,
                        );
                        for value in elements {
                            let size = 64; // for now we just need a constant item size for extracting tuples
                            generate_memory_write(
                                &mut instructions,
                                generate::WasmInstruction::Const(if is_wasm64 {
                                    NumericConstant::I64(address)
                                } else {
                                    NumericConstant::I32(address as u32)
                                }),
                                create_load_reg(&state, value, None, &locals, is_wasm64),
                                match size {
                                    ..=32 => NumericType::I32,
                                    ..=64 => NumericType::I64,
                                    _ => todo!(
                                        "Implement tuples with compound types larger than 64 bits"
                                    ),
                                },
                                size,
                                None,
                            );
                            address += (size / 8) as u64;
                        }
                    }
                    Instruction::ExtractTuple {
                        result,
                        tuple_val: ptr,
                        index,
                    }
                    | Instruction::GetFieldPtr {
                        result,
                        struct_ptr: ptr,
                        field_index: index,
                    } => {
                        let addr = match state.get_global(match ptr {
                            Value::Global(id) | Value::Variable(id) => id,
                            _ => todo!(
                                "Implement extracting values from a tuple constant immediately"
                            ),
                        }) {
                            state::GlobalRef::Memory(addr) => addr,
                            _ => unreachable!(),
                        } + (*index as u64 * 8);

                        let wasm_size = get_wasm_size_value(ptr, is_wasm64, &state, &locals);
                        let wasm_ty = get_wasm_type_value(ptr, is_wasm64, &state, &locals);

                        generate_memory_read(
                            &mut instructions,
                            Some(generate::WasmInstruction::Const(if is_wasm64 {
                                NumericConstant::I64(addr)
                            } else {
                                NumericConstant::I32(addr as u32)
                            })),
                            wasm_ty,
                            wasm_size as u8,
                            false,
                            None,
                        );

                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            wasm_ty,
                            result,
                        );
                    }
                    Instruction::GetElemPtr {
                        result,
                        array_ptr,
                        index,
                        element_type,
                    } => {
                        let elem_size = get_size_primitive(*element_type, is_wasm64);
                        let (ty, _, _) = get_wasm_type_primitive(*element_type, is_wasm64);

                        let base_addr = match state.get_global(match array_ptr {
                            Value::Global(id) | Value::Variable(id) => id,
                            _ => todo!(
                                "Implement extracting values from an array constant immediately"
                            ),
                        }) {
                            state::GlobalRef::Memory(addr) => addr,
                            _ => unreachable!(),
                        };

                        instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                            NumericConstant::I64(base_addr)
                        } else {
                            NumericConstant::I32(base_addr as u32)
                        }));

                        // generate offset by loading the value, the size of each element, and multiplying
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            index,
                            None,
                            &locals,
                            is_wasm64,
                        );
                        instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                            NumericConstant::I64(elem_size as u64)
                        } else {
                            NumericConstant::I32(elem_size as u32)
                        }));
                        instructions.push(generate::WasmInstruction::Mul(if is_wasm64 {
                            NumericType::I64
                        } else {
                            NumericType::I32
                        }));

                        instructions.push(generate::WasmInstruction::Add(if is_wasm64 {
                            NumericType::I64
                        } else {
                            NumericType::I32
                        }));

                        generate_memory_read(
                            &mut instructions,
                            None,
                            ty,
                            elem_size,
                            match element_type {
                                PrimitiveType::I8
                                | PrimitiveType::I16
                                | PrimitiveType::I32
                                | PrimitiveType::I64 => true,
                                _ => false,
                            },
                            None,
                        );
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            ty,
                            result,
                        );
                    }
                    Instruction::Print { value } => {
                        needs_console_log = true;
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            value,
                            Some(&PrimitiveType::I64),
                            &locals,
                            is_wasm64,
                        );
                        instructions.push(generate::WasmInstruction::Call(
                            generate::Identifier::Name("i0".into()),
                        ));
                    }
                    Instruction::Write {
                        buffer,
                        size,
                        result,
                    } => {
                        needs_console_log = true;
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            buffer,
                            Some(&PrimitiveType::Ptr),
                            &locals,
                            is_wasm64,
                        );
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            size,
                            Some(&if is_wasm64 {
                                PrimitiveType::U64
                            } else {
                                PrimitiveType::U32
                            }),
                            &locals,
                            is_wasm64,
                        );
                        instructions.push(generate::WasmInstruction::Call(
                            generate::Identifier::Name("i1".into()),
                        ));
                        instructions
                            .push(generate::WasmInstruction::Const(NumericConstant::I64(1)));
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            NumericType::I64,
                            result,
                        );
                    }
                    Instruction::WriteByte { value, result } => {
                        needs_console_log = true;
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            value,
                            Some(&PrimitiveType::U8),
                            &locals,
                            is_wasm64,
                        );
                        match get_wasm_type_value(value, is_wasm64, &state, &locals) {
                            NumericType::I32 => instructions.push(generate::WasmInstruction::Call(
                                generate::Identifier::Name("i4".into()),
                            )),
                            NumericType::I64 => instructions.push(generate::WasmInstruction::Call(
                                generate::Identifier::Name("i2".into()),
                            )),
                            _ => unreachable!(),
                        }
                        instructions
                            .push(generate::WasmInstruction::Const(NumericConstant::I64(1)));
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            NumericType::I64,
                            result,
                        );
                    }
                    Instruction::WritePtr { ptr, result } => {
                        needs_console_log = true;
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            ptr,
                            Some(&if is_wasm64 {
                                PrimitiveType::U64
                            } else {
                                PrimitiveType::U32
                            }),
                            &locals,
                            is_wasm64,
                        );
                        instructions.push(generate::WasmInstruction::Call(
                            generate::Identifier::Name("i3".into()),
                        ));
                        instructions
                            .push(generate::WasmInstruction::Const(NumericConstant::I64(1)));
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            NumericType::I64,
                            result,
                        );
                    }

                    Instruction::Read {
                        buffer: _,
                        size: _,
                        result: _,
                    }
                    | Instruction::ReadByte { result: _ } => {
                        return Err(crate::LaminaError::CodegenError(
                            crate::codegen::CodegenError::UnsupportedFeature(
                                crate::codegen::FeatureType::Custom("reading data".to_string()),
                            ),
                        ));
                    }

                    Instruction::Ret { ty, value } => {
                        let wasm_ty = get_wasm_type_for_return(ty, is_wasm64, module);
                        if let Some((_, is_ptr)) = wasm_ty {
                            println!("returning with type {ty}");
                            if is_ptr {
                                instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                    NumericConstant::I64(
                                        get_pointer(value.as_ref().unwrap(), &state).unwrap(),
                                    )
                                } else {
                                    NumericConstant::I32(
                                        get_pointer(value.as_ref().unwrap(), &state).unwrap()
                                            as u32,
                                    )
                                }));
                            } else {
                                generate_load_reg(
                                    &mut instructions,
                                    &state,
                                    value.as_ref().unwrap(),
                                    match ty {
                                        Type::Primitive(ty) => Some(ty),
                                        _ => None,
                                    },
                                    &locals,
                                    is_wasm64,
                                );
                            }
                        }

                        instructions.push(generate::WasmInstruction::Return);
                    }

                    Instruction::PtrToInt {
                        result,
                        ptr_value,
                        target_type,
                    } => {
                        if *target_type
                            != if is_wasm64 {
                                PrimitiveType::U64
                            } else {
                                PrimitiveType::U32
                            }
                            && *target_type
                                != if is_wasm64 {
                                    PrimitiveType::I64
                                } else {
                                    PrimitiveType::I32
                                }
                        {
                            return Err(crate::LaminaError::CodegenError(
                                crate::codegen::CodegenError::UnsupportedTypeForOperation(
                                    crate::codegen::OperationType::Custom("ptr to int".to_string()),
                                ),
                            ));
                        }

                        generate_load_reg(
                            &mut instructions,
                            &state,
                            ptr_value,
                            Some(&PrimitiveType::Ptr),
                            &locals,
                            is_wasm64,
                        );
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            get_wasm_type_for_return_primitive(*target_type, is_wasm64),
                            result,
                        );
                    }

                    Instruction::IntToPtr {
                        result,
                        int_value,
                        target_type,
                    } => {
                        generate_load_reg(
                            &mut instructions,
                            &state,
                            int_value,
                            Some(target_type),
                            &locals,
                            is_wasm64,
                        );
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            get_wasm_type_for_return_primitive(*target_type, is_wasm64),
                            result,
                        );
                    }

                    Instruction::Load { result, ty, ptr } => {
                        generate_memory_read(
                            &mut instructions,
                            Some(create_load_reg(&state, ptr, None, &locals, is_wasm64)),
                            get_wasm_type(ty, is_wasm64).0,
                            get_wasm_size(ty, is_wasm64, &module) as u8,
                            matches!(
                                ty,
                                Type::Primitive(
                                    PrimitiveType::I8
                                        | PrimitiveType::I16
                                        | PrimitiveType::I32
                                        | PrimitiveType::I64
                                )
                            ),
                            None,
                        );
                        generate_result(
                            &mut instructions,
                            &mut locals,
                            &mut next_local_i,
                            get_wasm_type(ty, is_wasm64).0,
                            result,
                        );
                    }

                    Instruction::Store { ty, ptr, value } => {
                        let addr = get_pointer(ptr, &state).unwrap();

                        if let Some(from) = get_pointer(value, &state) {
                            instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                NumericConstant::I64(addr)
                            } else {
                                NumericConstant::I32(addr as u32)
                            }));

                            instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                NumericConstant::I64(from)
                            } else {
                                NumericConstant::I32(from as u32)
                            }));

                            instructions.push(generate::WasmInstruction::Const(if is_wasm64 {
                                NumericConstant::I64(get_size(ty, is_wasm64, &module))
                            } else {
                                NumericConstant::I32(get_size(ty, is_wasm64, &module) as u32)
                            }));

                            instructions.push(generate::WasmInstruction::MemoryCopy);

                            continue;
                        }

                        generate_memory_write(
                            &mut instructions,
                            generate::WasmInstruction::Const(if is_wasm64 {
                                generate::NumericConstant::I64(addr)
                            } else {
                                generate::NumericConstant::I32(addr as u32)
                            }),
                            create_load_reg(&state, value, None, &locals, is_wasm64),
                            get_wasm_type(ty, is_wasm64).0,
                            get_size(ty, is_wasm64, &module) as u8,
                            None,
                        );
                    }

                    Instruction::Phi {
                        result: _,
                        ty: _,
                        incoming: _,
                    } => {
                        return Err(crate::LaminaError::CodegenError(
                            crate::codegen::CodegenError::InvalidInstructionForTarget(
                                crate::codegen::InstructionType::Custom("phi".to_string()),
                            ),
                        ));
                    }
                }
                instructions.push(generate::WasmInstruction::Comment(format!("{instr:?}")));
            }
            func_instructions.push(instructions.into());
        }

        let mut locals_vec = locals
            .iter()
            .map(|v| (*v.0, (*v.1).0, (*v.1).1))
            .collect::<Vec<_>>();

        locals_vec.sort_by(|v1, v2| v1.1.cmp(&v2.1));

        for _ in &func.signature.params {
            locals_vec.remove(0);
        }

        let mut paths = [
            generate::WasmInstruction::Nop,
            generate::WasmInstruction::LocalGet(generate::Identifier::Index(
                locals.get("pc").unwrap().0,
            )),
            generate::WasmInstruction::Eqz(IntegerType::I64),
            generate::WasmInstruction::Comment(format!("Test for entry block")),
            generate::WasmInstruction::If {
                identifier: None,
                result: None,
                then: func_instructions[0].clone(),
                r#else: Some(vec![].into()),
            },
        ]
        .to_vec()
        .into();
        let mut path: &mut generate::Instructions<'_> = &mut paths;

        for (i, block) in func_instructions[1..].iter().enumerate() {
            path = match (*path)[4] {
                generate::WasmInstruction::If {
                    identifier: _,
                    result: _,
                    then: _,
                    ref mut r#else,
                } => r#else.as_mut().unwrap(),
                _ => unreachable!(),
            };
            (*path) = vec![
                generate::WasmInstruction::LocalGet(generate::Identifier::Index(
                    locals.get("pc").unwrap().0,
                )),
                generate::WasmInstruction::Const(NumericConstant::I64(i as u64 + 1)),
                generate::WasmInstruction::Eq(NumericType::I64),
                generate::WasmInstruction::Comment(format!("Test for block {i}")),
                generate::WasmInstruction::If {
                    identifier: None,
                    result: None,
                    then: block.clone(),
                    r#else: Some(vec![].into()),
                },
            ]
            .into();
        }
        match (*path)[4] {
            generate::WasmInstruction::If {
                identifier: _,
                result: _,
                then: _,
                ref mut r#else,
            } => *r#else = None,
            _ => unreachable!(),
        };
        let new_mod = ModuleExpression::Function {
            name: Some(func_mapping.get(name).unwrap()),
            export: Some(name),
            parameters: func
                .signature
                .params
                .iter()
                .map(|v| (None, get_wasm_type(&v.ty, is_wasm64).0))
                .collect(),
            result: match func.signature.return_type {
                Type::Void => None,
                _ => Some(vec![
                    get_wasm_type(&func.signature.return_type, is_wasm64).0,
                ]),
            },
            locals: locals_vec.iter().map(|v| (None, v.2)).collect(),
            instructions: vec![
                generate::WasmInstruction::Comment(format!(
                    "{next_local_i} locals: {locals_vec:?}; {} block(s)", blocks.len()
                )),
                generate::WasmInstruction::Const(NumericConstant::I64(0)),
                generate::WasmInstruction::LocalSet(generate::Identifier::Index(
                    locals.get("pc").unwrap().0,
                )),
                generate::WasmInstruction::Comment("PC setup".to_string()),
                generate::WasmInstruction::Loop {
                    identifier: Some("l"),
                    contents: paths.to_vec().into(),
                },
                generate::WasmInstruction::Comment("Primary loop".to_string()),
                match get_wasm_type(&func.signature.return_type, is_wasm64).0 {
                    // implicit return
                    NumericType::I32 => generate::WasmInstruction::Const(NumericConstant::I32(0)),
                    NumericType::I64 => generate::WasmInstruction::Const(NumericConstant::I64(0)),
                    NumericType::F32 => generate::WasmInstruction::Const(NumericConstant::F32(0.0)),
                    NumericType::F64 => generate::WasmInstruction::Const(NumericConstant::F64(0.0)),
                },
                generate::WasmInstruction::Comment("Implicit return".to_string()),
            ].into(),
        };
        state.out_expressions.push(new_mod);
    }

    if needs_console_log {
        let ptr_ty = if is_wasm64 {
            NumericType::I64
        } else {
            NumericType::I32
        };
        state.out_expressions.insert(
            0,
            ModuleExpression::ImportFunc {
                namespace: "console",
                name: "log",
                import_name: Some("i0"),
                parameters: vec![NumericType::I64],
                result: None,
            },
        );

        state.out_expressions.insert(
            1,
            ModuleExpression::ImportFunc {
                // string
                namespace: "console",
                name: "log",
                import_name: Some("i1"),
                parameters: vec![ptr_ty, ptr_ty],
                result: None,
            },
        );

        state.out_expressions.insert(
            2,
            ModuleExpression::ImportFunc {
                // byte
                namespace: "console",
                name: "log",
                import_name: Some("i2"),
                parameters: vec![NumericType::I64],
                result: None,
            },
        );

        state.out_expressions.insert(
            3,
            ModuleExpression::ImportFunc {
                // ptr
                namespace: "console",
                name: "log",
                import_name: Some("i3"),
                parameters: vec![ptr_ty],
                result: None,
            },
        );

        state.out_expressions.insert(
            2,
            ModuleExpression::ImportFunc {
                // byte
                namespace: "console",
                name: "log",
                import_name: Some("i4"),
                parameters: vec![NumericType::I32],
                result: None,
            },
        );
    }

    if state.has_any_mem_regs() {
        state.out_expressions.push(ModuleExpression::Memory(
            state.output_memory.len() / (64 * 1024),
        ));
    }

    let last_nonzero = state
        .output_memory
        .iter()
        .rposition(|x| *x != 0)
        .map(|i| i + 1) // slices are exclusive on the right bound
        .unwrap_or(0);

    let shrink_fit_mem = &state.output_memory[..last_nonzero];

    if !shrink_fit_mem.is_empty() {
        state.out_expressions.push(ModuleExpression::Data {
            memory: generate::Identifier::Index(0),
            offset: vec![generate::WasmInstruction::Const(if is_wasm64 {
                NumericConstant::I64(0)
            } else {
                NumericConstant::I32(0)
            })]
            .into(),
            bytes: shrink_fit_mem,
        });
    }

    write!(
        writer,
        "{}",
        state
            .out_expressions
            .iter()
            .map(|v| format!("{v}\n"))
            .collect::<String>()
    )?;

    Ok(())
}
