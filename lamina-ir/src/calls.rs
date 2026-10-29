//! Boundary checking: a call's arguments against the callee's signature, and a `ret`
//! against the signature it returns from.
//!
//! This is the one layer that sees both a boundary and the signature it crosses, so
//! every argument and result form is checked here and nowhere else: register, literal,
//! global address, function address, pointer and aggregate.
//!
//! An aggregate is still passed and returned as the address of its storage, so the
//! site must state the aggregate type (`call @sum(!Pair %p)`, `ret.!Pair %p`). That
//! annotation is what separates an aggregate from a plain pointer, which an operand's
//! own type cannot do.

use std::collections::HashMap;
use std::fmt;

use crate::function::Function;
use crate::instruction::Instruction;
use crate::module::Module;
use crate::types::{Literal, PrimitiveType, Type, Value, resolve_type_names};

pub(crate) fn validate_calls(module: &Module<'_>, errors: &mut Vec<String>) {
    let mut functions: Vec<&Function<'_>> = module.functions.values().collect();
    functions.sort_by_key(|func| func.name);
    for func in functions {
        let env = value_types(module, func);
        let mut blocks: Vec<_> = func.basic_blocks.iter().collect();
        blocks.sort_by_key(|(label, _)| **label);
        for (label, block) in blocks {
            for (index, instr) in block.instructions.iter().enumerate() {
                let site = Site {
                    caller: func.name,
                    label,
                    index: index + 1,
                };
                match instr {
                    Instruction::Call {
                        func_name, args, ..
                    } => check_direct(module, &env, site, func_name, args, errors),
                    Instruction::CallIndirect { args, .. } => {
                        for (position, (ty, value)) in args.iter().enumerate() {
                            if let Some(found) = mismatch_against(module, &env, ty, value, true) {
                                errors.push(format!(
                                    "argument {} of the indirect call in '@{}' ({}) is declared {ty}, but {found}\n  Hint: Write the type the value has, or convert it before the call",
                                    position + 1,
                                    site.caller,
                                    site,
                                ));
                            }
                        }
                    }
                    Instruction::Ret { ty, value } => {
                        check_return(module, &env, site, func, ty, value.as_ref(), errors)
                    }
                    _ => {}
                }
            }
        }
    }
}

/// The result side of the same rule the arguments follow: the type written on `ret`
/// must be the function's declared return type, and the value must have that type.
/// An aggregate result is the address of its storage, so `ret` names the aggregate
/// and a `ptr` operand is then what it holds.
fn check_return<'a>(
    module: &Module<'a>,
    env: &HashMap<&'a str, Type<'a>>,
    site: Site<'_>,
    func: &Function<'a>,
    written: &Type<'a>,
    value: Option<&Value<'a>>,
    errors: &mut Vec<String>,
) {
    let declared = &func.signature.return_type;
    if !same_type(module, written, declared) {
        let hint = if is_aggregate(module, declared) {
            format!(
                "  Hint: An aggregate result is the address of its storage, so the return must name the type: write 'ret.{declared} <value>'"
            )
        } else {
            format!("  Hint: Write 'ret.{declared} <value>', or change the signature")
        };
        errors.push(format!(
            "return in '@{}' ({site}) is written ret.{written}, but '@{}' declares {declared}\n{hint}",
            site.caller, site.caller,
        ));
        return;
    }
    let Some(value) = value else {
        return;
    };
    if let Some(found) = mismatch_against(module, env, declared, value, true) {
        errors.push(format!(
            "return in '@{}' ({site}) must be {declared}, but {found}\n  Hint: Return a {declared}, or convert the value before the return",
            site.caller,
        ));
    }
}

struct Site<'a> {
    caller: &'a str,
    label: &'a str,
    index: usize,
}

impl fmt::Display for Site<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block '{}' instruction {}", self.label, self.index)
    }
}

fn check_direct<'a>(
    module: &Module<'a>,
    env: &HashMap<&'a str, Type<'a>>,
    site: Site<'_>,
    callee_name: &str,
    args: &[(Option<Type<'a>>, Value<'a>)],
    errors: &mut Vec<String>,
) {
    let Some(callee) = module.functions.get(callee_name) else {
        return;
    };
    let params = &callee.signature.params;
    let variadic = callee.signature.variadic;
    if args.len() < params.len() || (!variadic && args.len() > params.len()) {
        errors.push(format!(
            "call to '@{callee_name}' in '@{}' ({site}) passes {} argument{}, but '@{callee_name}' declares {}{}\n  Hint: Match the signature '{}'",
            site.caller,
            args.len(),
            if args.len() == 1 { "" } else { "s" },
            params.len(),
            if variadic { " named parameters" } else { "" },
            callee.signature,
        ));
        return;
    }

    for (position, ((annotation, value), param)) in args.iter().zip(params).enumerate() {
        let declared = &param.ty;
        if let Some(annotation) = annotation
            && !same_type(module, annotation, declared)
        {
            errors.push(format!(
                "argument {} of call to '@{callee_name}' in '@{}' ({site}) is declared {annotation}, but parameter '%{}' of '@{callee_name}' has type {declared}\n  Hint: Write '{declared} <value>' at the call site, or change the parameter",
                position + 1,
                site.caller,
                param.name,
            ));
            continue;
        }
        let expected = annotation.as_ref().unwrap_or(declared);
        if let Some(found) = mismatch_against(module, env, expected, value, annotation.is_some()) {
            let hint = if annotation.is_none() && is_aggregate(module, declared) {
                format!(
                    "  Hint: An aggregate argument is the address of its storage, so the call site must name the type: write '{declared} {value}'"
                )
            } else {
                format!("  Hint: Pass a {declared}, or convert the value before the call")
            };
            errors.push(format!(
                "argument {} of call to '@{callee_name}' in '@{}' ({site}) must be {declared} for parameter '%{}', but {found}\n{hint}",
                position + 1,
                site.caller,
                param.name,
            ));
        }
    }
}

/// Describes why `value` cannot be passed where `expected` is wanted, or `None` when it
/// can. `address_ok` holds where the call site wrote the type itself, which is how an
/// aggregate argument names the type of the storage its pointer operand addresses.
fn mismatch_against<'a>(
    module: &Module<'a>,
    env: &HashMap<&'a str, Type<'a>>,
    expected: &Type<'a>,
    value: &Value<'a>,
    address_ok: bool,
) -> Option<String> {
    let resolved = resolve_type_names(expected, &module.type_declarations)?;
    if let Value::Constant(literal) = value {
        return literal_mismatch(literal, &resolved);
    }
    let supplied = value_type(env, value)?;
    if same_type(module, &supplied, expected) {
        return None;
    }
    if matches!(resolved, Type::Primitive(PrimitiveType::Ptr))
        && is_aggregate_resolved(&value_type_resolved(module, env, value)?)
    {
        return Some(format!("{value} holds the aggregate {supplied}"));
    }
    if is_aggregate_resolved(&resolved) && matches!(supplied, Type::Primitive(PrimitiveType::Ptr)) {
        return if address_ok {
            None
        } else {
            Some(format!("{value} is a plain ptr"))
        };
    }
    Some(format!("{value} has type {supplied}"))
}

fn literal_mismatch(literal: &Literal<'_>, resolved: &Type<'_>) -> Option<String> {
    let Type::Primitive(primitive) = resolved else {
        return Some(format!("{literal} is a literal"));
    };
    match literal {
        Literal::String(_) if *primitive == PrimitiveType::Ptr => None,
        Literal::String(_) => Some(format!("{literal} is a string literal")),
        Literal::Bool(_) if *primitive == PrimitiveType::Bool => None,
        Literal::Bool(_) => Some(format!("{literal} is a bool literal")),
        Literal::F32(_) | Literal::F64(_)
            if matches!(primitive, PrimitiveType::F32 | PrimitiveType::F64) =>
        {
            None
        }
        Literal::F32(_) | Literal::F64(_) => Some(format!("{literal} is a float literal")),
        other => {
            let value = other.as_integer()?;
            if *primitive == PrimitiveType::Bool {
                return Some(format!(
                    "{value} is an integer literal, and a bool takes 'true' or 'false'"
                ));
            }
            let (min, max) = primitive.integer_bounds()?;
            (value < min || value > max).then(|| {
                format!("{value} is out of range for {primitive}, which holds {min} to {max}")
            })
        }
    }
}

fn value_type<'a>(env: &HashMap<&'a str, Type<'a>>, value: &Value<'a>) -> Option<Type<'a>> {
    match value {
        Value::Variable(name) => env.get(name).cloned(),
        Value::Global(_) => Some(Type::Primitive(PrimitiveType::Ptr)),
        Value::Constant(Literal::String(_)) => Some(Type::Primitive(PrimitiveType::Ptr)),
        Value::Constant(_) => None,
    }
}

fn value_type_resolved<'a>(
    module: &Module<'a>,
    env: &HashMap<&'a str, Type<'a>>,
    value: &Value<'a>,
) -> Option<Type<'a>> {
    resolve_type_names(&value_type(env, value)?, &module.type_declarations)
}

fn same_type<'a>(module: &Module<'a>, left: &Type<'a>, right: &Type<'a>) -> bool {
    if left == right {
        return true;
    }
    match (
        resolve_type_names(left, &module.type_declarations),
        resolve_type_names(right, &module.type_declarations),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

fn is_aggregate<'a>(module: &Module<'a>, ty: &Type<'a>) -> bool {
    resolve_type_names(ty, &module.type_declarations).is_some_and(|ty| is_aggregate_resolved(&ty))
}

fn is_aggregate_resolved(ty: &Type<'_>) -> bool {
    matches!(ty, Type::Struct { .. } | Type::Array { .. })
}

/// The type of every named value in `func`, taken from the parameters and from each
/// defining instruction. A value whose type the instruction does not state is absent.
pub fn value_types<'a>(module: &Module<'a>, func: &Function<'a>) -> HashMap<&'a str, Type<'a>> {
    let mut env: HashMap<&'a str, Type<'a>> = func
        .signature
        .params
        .iter()
        .map(|param| (param.name, param.ty.clone()))
        .collect();
    let mut blocks: Vec<_> = func.basic_blocks.iter().collect();
    blocks.sort_by_key(|(label, _)| **label);
    for (_, block) in blocks {
        for instr in &block.instructions {
            let Some(defined) = instr.defined_var() else {
                continue;
            };
            let ty = match instr {
                Instruction::Call { func_name, .. } => module
                    .functions
                    .get(func_name)
                    .map(|callee| callee.signature.return_type.clone()),
                Instruction::CallIndirect { return_type, .. } => Some(return_type.clone()),
                other => other.result_type(),
            };
            if let Some(ty) = ty {
                env.entry(defined).or_insert(ty);
            }
        }
    }
    env
}
