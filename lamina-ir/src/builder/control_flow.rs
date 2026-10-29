//! Control flow operations for IR builder.
//!
//! This module provides methods for branching, jumping, and function return
//! operations in the IR builder API.

use crate::{
    builder::IRBuilder,
    instruction::Instruction,
    types::{Literal, PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Creates a conditional branch instruction
    pub fn branch(
        &mut self,
        condition: Value<'a>,
        true_label: &'a str,
        false_label: &'a str,
    ) -> &mut Self {
        self.inst(Instruction::Br {
            condition,
            true_label,
            false_label,
        })
    }

    /// Creates an unconditional jump instruction
    pub fn jump(&mut self, target: &'a str) -> &mut Self {
        self.inst(Instruction::Jmp {
            target_label: target,
        })
    }

    /// Creates a switch instruction for multi-way branching on an integer value.
    pub fn switch(
        &mut self,
        ty: PrimitiveType,
        value: Value<'a>,
        default: &'a str,
        cases: &[(Literal<'a>, &'a str)],
    ) -> &mut Self {
        let mapped_cases = cases.iter().map(|(lit, lbl)| (lit.clone(), *lbl)).collect();
        self.inst(Instruction::Switch {
            ty,
            value,
            default,
            cases: mapped_cases,
        })
    }

    /// Creates a return instruction with a value
    pub fn ret(&mut self, ty: Type<'a>, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Ret {
            ty,
            value: Some(value),
        })
    }

    /// Creates an unreachable terminator. Reaching it is undefined behavior.
    pub fn unreachable(&mut self) -> &mut Self {
        self.inst(Instruction::Unreachable)
    }

    /// Creates a void return instruction (returns nothing)
    pub fn ret_void(&mut self) -> &mut Self {
        self.inst(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    }

    /// Creates a scalar value copy (`%result = copy.<ty> <value>`).
    ///
    /// # Panics
    /// Panics unless `ty` is a primitive type. Aggregates move through memory one scalar at a time.
    pub fn copy(&mut self, result: &'a str, ty: Type<'a>, value: Value<'a>) -> &mut Self {
        assert!(
            matches!(ty, Type::Primitive(_)),
            "copy moves one scalar value, found {ty}; move an aggregate through memory with getfieldptr and load/store"
        );
        self.inst(Instruction::Copy { result, ty, value })
    }

    /// Creates a function call instruction
    pub fn call(
        &mut self,
        result: Option<&'a str>,
        func_name: &'a str,
        args: Vec<Value<'a>>,
    ) -> &mut Self {
        self.inst(Instruction::Call {
            result,
            func_name,
            args,
        })
    }

    /// Creates an indirect call through a function pointer (`callptr.<ret-ty> <ptr>(<ty> <arg>, ...)`).
    ///
    /// Each argument carries the type the call site declares for it. `variadic_fixed` is
    /// the number of named parameters when the call is variadic, which places the `...`
    /// marker after that many arguments; pass `None` for a non-variadic call.
    ///
    /// # Panics
    /// Panics if `variadic_fixed` exceeds the number of arguments.
    pub fn callptr(
        &mut self,
        result: Option<&'a str>,
        return_type: Type<'a>,
        target: Value<'a>,
        args: Vec<(Type<'a>, Value<'a>)>,
        variadic_fixed: Option<usize>,
    ) -> &mut Self {
        if let Some(fixed) = variadic_fixed {
            assert!(
                fixed <= args.len(),
                "callptr variadic_fixed ({fixed}) exceeds the argument count ({})",
                args.len()
            );
        }
        self.inst(Instruction::CallIndirect {
            result,
            target,
            args,
            variadic_fixed,
            return_type,
        })
    }
}
