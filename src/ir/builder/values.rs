use crate::ir::types::{Literal, Value};

/// Creates a variable reference
pub fn var(name: &str) -> Value<'_> {
    Value::Variable(name)
}

/// Creates an i8 constant
pub fn i8<'a>(val: i8) -> Value<'a> {
    Value::Constant(Literal::I8(val))
}

/// Creates an i32 constant
pub fn i32<'a>(val: i32) -> Value<'a> {
    Value::Constant(Literal::I32(val))
}

/// Creates an i64 constant
pub fn i64<'a>(val: i64) -> Value<'a> {
    Value::Constant(Literal::I64(val))
}

/// Creates an f32 constant
pub fn f32<'a>(val: f32) -> Value<'a> {
    Value::Constant(Literal::F32(val))
}

/// Creates a boolean constant
pub fn bool<'a>(val: bool) -> Value<'a> {
    Value::Constant(Literal::Bool(val))
}

/// Creates a string constant
pub fn string<'a>(val: &'a str) -> Value<'a> {
    Value::Constant(Literal::String(val))
}

/// Creates a global reference
pub fn global(name: &str) -> Value<'_> {
    Value::Global(name)
}

