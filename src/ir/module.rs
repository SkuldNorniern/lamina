use std::collections::HashMap; // Using HashMap for functions and types
use std::fmt;

use super::function::Function;
use super::types::{Identifier, Type, Value};

// Represents a named type declaration (e.g., type @Vec2 = struct { ... })
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Eq might be too restrictive if Type contains f32 indirectly, but should be okay here.
pub struct TypeDeclaration<'a> {
    pub name: Identifier<'a>, // e.g., "@Vec2"
    pub ty: Type<'a>,
}

// Represents a global variable declaration (e.g., global @message: [5 x i8] = "hello")
#[derive(Debug, Clone, PartialEq)] // PartialEq due to Value potentially holding f32
pub struct GlobalDeclaration<'a> {
    pub name: Identifier<'a>, // e.g., "@message"
    pub ty: Type<'a>,
    pub initializer: Option<Value<'a>>, // Globals might be uninitialized (extern)
}

// Represents a complete Lamina IR module (file)
#[derive(Debug, Clone, PartialEq)]
pub struct Module<'a> {
    // Using HashMap for efficient lookup by name.
    // Order is not preserved, but typically not critical for declarations.
    pub type_declarations: HashMap<Identifier<'a>, TypeDeclaration<'a>>,
    pub global_declarations: HashMap<Identifier<'a>, GlobalDeclaration<'a>>,
    pub functions: HashMap<Identifier<'a>, Function<'a>>,
}

impl<'a> Module<'a> {
    // Basic constructor
    pub fn new() -> Self {
        Module {
            type_declarations: HashMap::new(),
            global_declarations: HashMap::new(),
            functions: HashMap::new(),
        }
    }
}

impl<'a> Default for Module<'a> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Display Implementations ---

impl<'a> fmt::Display for TypeDeclaration<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type @{} = {}", self.name, self.ty)
    }
}

impl<'a> fmt::Display for GlobalDeclaration<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "global @{}: {}", self.name, self.ty)?;
        if let Some(init) = &self.initializer {
            write!(f, " = {}", init)?;
        }
        Ok(())
    }
}

impl<'a> fmt::Display for Module<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print type declarations first
        for decl in self.type_declarations.values() {
            writeln!(f, "{}", decl)?;
        }
        if !self.type_declarations.is_empty() {
            writeln!(f)?;
        }

        // Print global declarations
        for decl in self.global_declarations.values() {
            writeln!(f, "{}", decl)?;
        }
        if !self.global_declarations.is_empty() {
            writeln!(f)?;
        }

        // Print functions
        let mut function_names: Vec<_> = self.functions.keys().collect();
        function_names.sort(); // Sort by name for consistent output order

        for (i, name) in function_names.iter().enumerate() {
             // Dereference name twice (&&str -> &str) to match the key type for lookup
             if let Some(func) = self.functions.get(*name) { 
                write!(f, "{}", func)?;
                if i < function_names.len() - 1 {
                    writeln!(f)?;
                }
             } 
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{Literal, PrimitiveType, Type, Value}; // Assuming crate root
    use crate::ir::function::{Function, FunctionSignature, FunctionParameter, BasicBlock};
    use crate::ir::instruction::{Instruction, BinaryOp};
    use std::collections::HashMap;

    #[test]
    fn test_display_type_declaration() {
        let decl = TypeDeclaration {
            name: "MyType",
            ty: Type::Struct(vec![]), // Empty struct
        };
        // Expect two spaces for empty struct {  }
        assert_eq!(format!("{}", decl), "type @MyType = struct {  }");
    }

    #[test]
    fn test_display_global_declaration() {
        // Initialized
        let decl1 = GlobalDeclaration {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I32),
            initializer: Some(Value::Constant(Literal::I32(0))),
        };
        assert_eq!(format!("{}", decl1), "global @count: i32 = 0");

        // Uninitialized (Extern)
        let decl2 = GlobalDeclaration {
            name: "external_data",
            ty: Type::Array { element_type: Box::new(Type::Primitive(PrimitiveType::I8)), size: 256 },
            initializer: None,
        };
        assert_eq!(format!("{}", decl2), "global @external_data: [256 x i8]");
    }

    #[test]
    fn test_display_module() {
        let mut module = Module::new();

        // Add type declaration
        module.type_declarations.insert("Vec2", TypeDeclaration {
            name: "Vec2",
            ty: Type::Struct(vec![
                crate::ir::types::StructField { name: "x", ty: Type::Primitive(PrimitiveType::F32) },
                crate::ir::types::StructField { name: "y", ty: Type::Primitive(PrimitiveType::F32) },
            ]),
        });

        // Add global declaration
        module.global_declarations.insert("PI", GlobalDeclaration {
            name: "PI",
            ty: Type::Primitive(PrimitiveType::F32),
            initializer: Some(Value::Constant(Literal::F32(3.14159))),
        });

        // Add a simple function
        let func_sig = FunctionSignature {
            params: vec![FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };
        let mut func_blocks = HashMap::new();
        func_blocks.insert("entry", BasicBlock {
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Add,
                    result: "res",
                    ty: PrimitiveType::I32,
                    lhs: Value::Variable("a"),
                    rhs: Value::Constant(Literal::I32(1)),
                },
                Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Variable("res")),
                },
            ],
        });
        let func = Function {
            name: "add_one",
            signature: func_sig,
            annotations: vec![],
            basic_blocks: func_blocks,
            entry_block: "entry",
        };
        module.functions.insert("add_one", func);

        // Match actual output - no indentation for function instructions
        let expected_output = 
"type @Vec2 = struct { x: f32, y: f32 }\n\nglobal @PI: f32 = 3.14159\n\nfn @add_one(i32 %a) -> i32 {\nentry:\n  %res = add.i32 %a, 1\n  ret.i32 %res\n}\n";

        // Note: Hashmap iteration order isn't guaranteed, but Display impl sorts function keys.
        // Type/Global order isn't sorted, so this test might be fragile if more are added.
        // A more robust test might parse the output or check for substrings.
        assert_eq!(format!("{}", module), expected_output);

    }

     #[test]
    fn test_display_empty_module() {
         let module = Module::<'static>::new();
         assert_eq!(format!("{}", module), ""); // Empty module should print nothing
    }
} 