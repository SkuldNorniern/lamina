//! Fills in `field_byte_offset` on `getfieldptr` parsed from text.
//!
//! The text syntax carries no type on a `getfieldptr`, so the lowering falls
//! back to one 8-byte slot per field. That is wrong for any struct with
//! narrower fields, and makes neighbouring elements of a struct array alias.
//! Pointee types are recoverable from the instructions that produce them, so
//! resolve them here and record the real offsets.

use lamina_ir::function::Function;
use lamina_ir::instruction::Instruction;
use lamina_ir::module::{Module, TypeDeclaration};
use lamina_ir::types::{Identifier, StructField, Type, Value, struct_field_byte_offset};
use std::collections::HashMap;

pub fn resolve_field_offsets(module: &mut Module<'_>) {
    let types = module.type_declarations.clone();
    for func in module.functions.values_mut() {
        resolve_in_function(func, &types);
    }
}

fn resolve_in_function<'a>(
    func: &mut Function<'a>,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) {
    let mut pointee: HashMap<Identifier<'a>, Type<'a>> = HashMap::new();

    let labels: Vec<_> = func.basic_blocks.keys().copied().collect();
    for label in labels {
        let Some(block) = func.basic_blocks.get_mut(label) else {
            continue;
        };
        for inst in &mut block.instructions {
            match inst {
                Instruction::Alloc {
                    result,
                    allocated_ty,
                    ..
                } => {
                    pointee.insert(*result, allocated_ty.clone());
                }
                Instruction::GetElemPtr {
                    result,
                    array_ptr,
                    element_type,
                    ..
                } => {
                    // The array's own element type is authoritative; the
                    // written suffix cannot name a struct.
                    let elem = pointee_of(array_ptr, &pointee, types).and_then(|t| match t {
                        Type::Array { element_type, .. } => Some((*element_type).clone()),
                        _ => None,
                    });
                    pointee.insert(*result, elem.unwrap_or(Type::Primitive(*element_type)));
                }
                Instruction::GetFieldPtr {
                    result,
                    struct_ptr,
                    field_index,
                    field_byte_offset,
                } => {
                    let Some(fields) = pointee_of(struct_ptr, &pointee, types)
                        .and_then(|t| struct_fields(&t, types).cloned())
                    else {
                        continue;
                    };
                    if field_byte_offset.is_none() {
                        *field_byte_offset = struct_field_byte_offset(&fields, *field_index);
                    }
                    if let Some(field) = fields.get(*field_index) {
                        pointee.insert(*result, field.ty.clone());
                    }
                }
                _ => {}
            }
        }
    }
}

fn pointee_of<'a>(
    value: &Value<'a>,
    pointee: &HashMap<Identifier<'a>, Type<'a>>,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Option<Type<'a>> {
    let Value::Variable(name) = value else {
        return None;
    };
    let ty = pointee.get(name)?;
    Some(resolve_named(ty, types))
}

fn resolve_named<'a>(
    ty: &Type<'a>,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Type<'a> {
    match ty {
        Type::Named(name) => types
            .get(name)
            .map(|d| d.ty.clone())
            .unwrap_or_else(|| ty.clone()),
        other => other.clone(),
    }
}

fn struct_fields<'b, 'a>(
    ty: &'b Type<'a>,
    types: &'b HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Option<&'b Vec<StructField<'a>>> {
    match ty {
        Type::Struct(fields) => Some(fields),
        Type::Named(name) => match &types.get(name)?.ty {
            Type::Struct(fields) => Some(fields),
            _ => None,
        },
        _ => None,
    }
}
