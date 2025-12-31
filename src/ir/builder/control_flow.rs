use super::IRBuilder;
use crate::ir::instruction::Instruction;
use crate::ir::types::{Literal, PrimitiveType, Type, Value};

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

    /// Creates a void return instruction (returns nothing)
    pub fn ret_void(&mut self) -> &mut Self {
        self.inst(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    }

    /// Creates a phi node for SSA form
    pub fn phi(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        incoming: Vec<(Value<'a>, &'a str)>,
    ) -> &mut Self {
        self.inst(Instruction::Phi {
            result,
            ty,
            incoming,
        })
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
}

