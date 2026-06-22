//! Sanity checks and validation for MIR transforms.
//!
//! Validation functions to check that MIR structures are
//! well-formed after transformations. These checks catch bugs in
//! transform implementations.

use std::collections::HashSet;

use crate::mir::transform::TransformError;
use crate::mir::{Function, Instruction};

/// Validate that all branch and jump targets reference existing blocks.
pub fn validate_cfg(func: &Function) -> Result<(), TransformError> {
    let labels: HashSet<&str> = func.blocks.iter().map(|b| b.label.as_str()).collect();
    for block in &func.blocks {
        let invalid_edge = |target: &str, edge: &'static str| TransformError::InvalidCfg {
            block: block.label.clone(),
            target: target.to_string(),
            edge,
        };
        for inst in &block.instructions {
            match inst {
                Instruction::Jmp { target } => {
                    if !labels.contains(target.as_str()) {
                        return Err(invalid_edge(target, "jumps to"));
                    }
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if !labels.contains(true_target.as_str()) {
                        return Err(invalid_edge(true_target, "true branch"));
                    }
                    if !labels.contains(false_target.as_str()) {
                        return Err(invalid_edge(false_target, "false branch"));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, Instruction, MirType, Operand, Register, ScalarType, VirtualReg,
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn ret_func_with_jmp(target: &str) -> Function {
        FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: target.to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build()
    }

    #[test]
    fn valid_jmp_passes() {
        let func = ret_func_with_jmp("exit");
        assert!(validate_cfg(&func).is_ok());
    }

    #[test]
    fn jmp_to_missing_block_fails() {
        let func = ret_func_with_jmp("nowhere");
        assert!(validate_cfg(&func).is_err());
    }

    #[test]
    fn valid_br_passes() {
        let cond: Register = VirtualReg::gpr(0).into();
        let func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Br {
                cond: cond.clone(),
                true_target: "yes".to_owned(),
                false_target: "no".to_owned(),
            })
            .block("yes")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(1))),
            })
            .block("no")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();
        assert!(validate_cfg(&func).is_ok());
    }

    #[test]
    fn br_missing_true_target_fails() {
        let cond: Register = VirtualReg::gpr(0).into();
        let func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Br {
                cond: cond.clone(),
                true_target: "ghost".to_owned(),
                false_target: "no".to_owned(),
            })
            .block("no")
            .instr(Instruction::Ret { value: None })
            .build();
        assert!(validate_cfg(&func).is_err());
    }

    #[test]
    fn br_missing_false_target_fails() {
        let cond: Register = VirtualReg::gpr(0).into();
        let func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Br {
                cond,
                true_target: "yes".to_owned(),
                false_target: "ghost".to_owned(),
            })
            .block("yes")
            .instr(Instruction::Ret { value: None })
            .build();
        assert!(validate_cfg(&func).is_err());
    }
}
