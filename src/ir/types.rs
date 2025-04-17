use std::fmt;

// Using &'a str for identifiers to avoid allocating Strings
// Assumes the IR lives as long as the source it was parsed from.
pub type Identifier<'a> = &'a str;
pub type Label<'a> = &'a str;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    I8,
    I32,
    I64,
    F32,
    Bool,
    Ptr, // Generic pointer type
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField<'a> {
    pub name: Identifier<'a>,
    pub ty: Type<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type<'a> {
    Primitive(PrimitiveType),
    Named(Identifier<'a>), // Like "@Vec2"
    // Using Box to avoid recursive type definition issues
    Array {
        element_type: Box<Type<'a>>,
        size: u64,
    },
    Struct(Vec<StructField<'a>>),
    Tuple(Vec<Type<'a>>),
    Void, // For functions that don't return a value
}

#[derive(Debug, Clone, PartialEq)] // Float comparison requires PartialEq only
pub enum Literal<'a> {
    I32(i32),
    I64(i64),
    F32(f32),
    Bool(bool),
    String(&'a str),
    // Null pointers or other constants might be needed later
}

#[derive(Debug, Clone, PartialEq)] // Removed Eq, Hash because f32 doesn't support them
pub enum Value<'a> {
    // SSA register/variable, like "%result"
    Variable(Identifier<'a>),
    // Literal values used directly in instructions
    Constant(Literal<'a>),
    // Reference to a global variable, like "@message"
    Global(Identifier<'a>),
}

// --- Display implementations for better readability ---

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveType::I8 => write!(f, "i8"),
            PrimitiveType::I32 => write!(f, "i32"),
            PrimitiveType::I64 => write!(f, "i64"),
            PrimitiveType::F32 => write!(f, "f32"),
            PrimitiveType::Bool => write!(f, "bool"),
            PrimitiveType::Ptr => write!(f, "ptr"),
        }
    }
}

impl<'a> fmt::Display for Type<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Primitive(pt) => write!(f, "{}", pt),
            Type::Named(id) => write!(f, "@{}", id),
            Type::Array { element_type, size } => write!(f, "[{} x {}]", size, element_type),
            Type::Struct(fields) => {
                write!(f, "struct {{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    write!(f, "{}: {}", field.name, field.ty)?;
                    if i < fields.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " }}")
            }
            Type::Tuple(types) => {
                write!(f, "tuple(")?;
                for (i, ty) in types.iter().enumerate() {
                    write!(f, "{}", ty)?;
                    if i < types.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
            Type::Void => write!(f, "void"),
        }
    }
}

impl<'a> fmt::Display for Literal<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::I32(v) => write!(f, "{}", v),
            Literal::I64(v) => write!(f, "{}", v),
            Literal::F32(v) => write!(f, "{}", v),
            Literal::Bool(v) => write!(f, "{}", v),
            Literal::String(s) => write!(f, "\"{}\"", s), // Note: needs escaping for proper output
        }
    }
}

impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Variable(id) => write!(f, "%{id}"),
            Value::Constant(lit) => write!(f, "{}", lit),
            Value::Global(id) => write!(f, "@{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_primitive_type() {
        assert_eq!(format!("{}", PrimitiveType::I32), "i32");
        assert_eq!(format!("{}", PrimitiveType::F32), "f32");
        assert_eq!(format!("{}", PrimitiveType::Bool), "bool");
        assert_eq!(format!("{}", PrimitiveType::Ptr), "ptr");
    }

    #[test]
    fn test_display_type() {
        assert_eq!(format!("{}", Type::Primitive(PrimitiveType::I64)), "i64");
        assert_eq!(format!("{}", Type::Named("MyStruct")), "@MyStruct");
        let arr_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
            size: 10,
        };
        assert_eq!(format!("{}", arr_type), "[10 x i8]");
        let struct_type = Type::Struct(vec![
            StructField { name: "x", ty: Type::Primitive(PrimitiveType::F32) },
            StructField { name: "y", ty: Type::Primitive(PrimitiveType::F32) },
        ]);
        assert_eq!(format!("{}", struct_type), "struct { x: f32, y: f32 }");
        let tuple_type = Type::Tuple(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Named("MyBool"),
        ]);
        assert_eq!(format!("{}", tuple_type), "tuple(i32, @MyBool)");
        assert_eq!(format!("{}", Type::Void), "void");
    }

     #[test]
    fn test_display_literal() {
        assert_eq!(format!("{}", Literal::I32(123)), "123");
        assert_eq!(format!("{}", Literal::I64(-456)), "-456");
        assert_eq!(format!("{}", Literal::F32(1.25)), "1.25");
        assert_eq!(format!("{}", Literal::Bool(true)), "true");
        assert_eq!(format!("{}", Literal::Bool(false)), "false");
        assert_eq!(format!("{}", Literal::String("hello")), "\"hello\"");
    }

    #[test]
    fn test_display_value() {
        assert_eq!(format!("{}", Value::Variable("tmp1")), "%tmp1");
        assert_eq!(format!("{}", Value::Constant(Literal::I32(42))), "42");
         assert_eq!(
            format!("{}", Value::Constant(Literal::String("world"))),
            "\"world\""
        );
        assert_eq!(format!("{}", Value::Global("my_global_var")), "@my_global_var");
    }
} 