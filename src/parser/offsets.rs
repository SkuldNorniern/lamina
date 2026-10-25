//! Fills in the layout facts that the text syntax leaves out: `field_byte_offset`
//! on `getfieldptr`, and the pointee width of `writeptr`.
//!
//! The text syntax carries no type on a `getfieldptr`, so the lowering falls
//! back to one 8-byte slot per field. That is wrong for any struct with
//! narrower fields, and makes neighbouring elements of a struct array alias.
//! Pointee types are recoverable from the instructions that produce them, so
//! resolve them here and record the real offsets.

use lamina_ir::function::Function;
use lamina_ir::instruction::Instruction;
use lamina_ir::module::{Module, TypeDeclaration};
use lamina_ir::types::{
    Identifier, StructField, Type, Value, resolve_type_names, sizeof_type, struct_field_byte_offset,
};
use std::collections::HashMap;

pub fn resolve_field_offsets(module: &mut Module<'_>) {
    let types = module.type_declarations.clone();
    let returns: HashMap<Identifier<'_>, Type<'_>> = module
        .functions
        .iter()
        .map(|(name, func)| (*name, func.signature.return_type.clone()))
        .collect();
    for func in module.functions.values_mut() {
        resolve_in_function(func, &types, &returns);
    }
}

fn resolve_in_function<'a>(
    func: &mut Function<'a>,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
    returns: &HashMap<Identifier<'a>, Type<'a>>,
) {
    let mut pointee: HashMap<Identifier<'a>, Type<'a>> = HashMap::new();

    // An aggregate parameter reaches the body as a pointer to the callee's copy, so its
    // fields are addressed the same way an alloc's are.
    for param in &func.signature.params {
        let Some(ty) = resolve_type_names(&param.ty, types) else {
            continue;
        };
        if struct_fields(&ty).is_some() {
            pointee.insert(param.name, ty);
        }
    }

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
                    let elem = pointee_of(array_ptr, &pointee, types).and_then(|t| match t {
                        Type::Array { element_type, .. } => Some((*element_type).clone()),
                        _ => None,
                    });
                    pointee.insert(*result, elem.unwrap_or_else(|| element_type.clone()));
                }
                Instruction::Call {
                    result: Some(result),
                    func_name,
                    ..
                } => {
                    // An aggregate result reaches the caller as a pointer to its copy.
                    if let Some(ty) = returns
                        .get(func_name)
                        .and_then(|ty| resolve_type_names(ty, types))
                        && struct_fields(&ty).is_some()
                    {
                        pointee.insert(*result, ty);
                    }
                }
                Instruction::WritePtr {
                    ptr, pointee_size, ..
                } => {
                    if pointee_size.is_none() {
                        *pointee_size =
                            pointee_of(ptr, &pointee, types).and_then(|t| sizeof_type(&t));
                    }
                }
                Instruction::GetFieldPtr {
                    result,
                    struct_ptr,
                    field_index,
                    field_byte_offset,
                } => {
                    let Some(fields) =
                        pointee_of(struct_ptr, &pointee, types).and_then(|t| struct_fields(&t))
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
    resolve_type_names(ty, types)
}

fn struct_fields<'a>(ty: &Type<'a>) -> Option<Vec<StructField<'a>>> {
    match ty {
        Type::Struct(fields) => Some(fields.clone()),
        Type::Tuple(elements) => Some(
            elements
                .iter()
                .map(|ty| StructField {
                    name: "",
                    ty: ty.clone(),
                })
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;

    fn writeptr_widths(source: &str) -> Vec<Option<u64>> {
        let module = parse_module(source).expect("parse module");
        let func = &module.functions["main"];
        func.basic_blocks["entry"]
            .instructions
            .iter()
            .filter_map(|inst| match inst {
                Instruction::WritePtr { pointee_size, .. } => Some(*pointee_size),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn writeptr_pointee_width_follows_alloc_type() {
        let widths = writeptr_widths(
            r#"
fn @main() -> i32 {
entry:
    %b8 = alloc.stack i8
    %b16 = alloc.stack i16
    %b32 = alloc.stack i32
    %b64 = alloc.stack i64
    %w8 = writeptr %b8
    %w16 = writeptr %b16
    %w32 = writeptr %b32
    %w64 = writeptr %b64
    ret.i32 0
}
"#,
        );
        assert_eq!(widths, [Some(1), Some(2), Some(4), Some(8)]);
    }

    #[test]
    fn writeptr_pointee_width_unknown_for_parameters() {
        let widths = writeptr_widths(
            r#"
fn @main(ptr %p) -> i32 {
entry:
    %w = writeptr %p
    ret.i32 0
}
"#,
        );
        assert_eq!(widths, [None]);
    }

    #[test]
    fn writeptr_suffix_states_width_for_parameters() {
        let widths = writeptr_widths(
            r#"
fn @main(ptr %p, ptr %q) -> i32 {
entry:
    %a = writeptr.i32 %p
    %b = writeptr.i8 %q
    ret.i32 0
}
"#,
        );
        assert_eq!(widths, [Some(4), Some(1)]);
    }
}
