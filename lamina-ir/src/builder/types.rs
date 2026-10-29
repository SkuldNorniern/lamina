//! Named type declarations for IR builder.

use crate::{builder::IRBuilder, module::TypeDeclaration, types::Type};

impl<'a> IRBuilder<'a> {
    /// Declares a named type (`type !name = ty`) that instructions can refer to as `Type::Named`.
    ///
    /// Panics when the name is already declared.
    pub fn declare_type(&mut self, name: &'a str, ty: Type<'a>) -> &mut Self {
        assert!(
            !self.module.type_declarations.contains_key(name),
            "type '!{name}' is already declared"
        );
        self.module
            .type_declarations
            .insert(name, TypeDeclaration { name, ty });
        self
    }
}
