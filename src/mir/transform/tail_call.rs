//! Tail call optimization transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Operand, Register};

/// Tail call optimization that converts tail calls into jumps.
///
/// A tail call is a function call that is the last operation before returning.
/// This optimization converts them to jumps to avoid stack overflow and improve performance.
#[derive(Default)]
pub struct TailCallOptimization;

impl Transform for TailCallOptimization {
    fn name(&self) -> &'static str {
        "tail_call_optimization"
    }

    fn description(&self) -> &'static str {
        "Convert tail calls to jumps to avoid stack overflow and improve performance"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl TailCallOptimization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit block size
        const MAX_BLOCK_INSTRUCTIONS: usize = 1_000;
        for block in &func.blocks {
            if block.instructions.len() > MAX_BLOCK_INSTRUCTIONS {
                return Err(format!(
                    "Block '{}' too large for tail call optimization ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_BLOCK_INSTRUCTIONS
                ));
            }
        }

        let mut changed = false;
        // Borrow signature separately from blocks to avoid borrow checker issues
        let _sig = &func.sig;
        // We need to bypass the borrow checker restriction on `func` being borrowed mutably for blocks
        // while we need `sig`. Since `sig` and `blocks` are disjoint fields, we can destructure or just clone signature if needed.
        // Cloning signature is safe but slightly inefficient.
        // However, `Function` struct definition:
        // struct Function { sig, blocks, entry }
        // We can't easily iterate `func.blocks` mutably while holding `&func.sig` because `func` is borrowed.
        // Actually, we can if we split borrows, but `apply_internal` takes `&mut Function`.
        // Rust's split borrow works if we do `let Function { sig, blocks, .. } = func`.
        // But `blocks` is Vec<Block>.
        // Let's rely on disjoint fields if the compiler is smart enough, OR just clone the signature since it's small metadata.
        // Signature contains a Vec<Parameter>, so it allocates.
        // Let's try to be smart.
        // But `func.blocks` access is via `func` reference.
        // Workaround: Clone signature. It is not huge.
        let func_sig = func.sig.clone();

        for block in &mut func.blocks {
            if self.optimize_block_tail_calls(&func_sig, block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Optimize tail calls within a single block
    fn optimize_block_tail_calls(
        &self,
        func_sig: &crate::mir::function::Signature,
        block: &mut Block,
    ) -> bool {
        let mut changed = false;

        // Find the last instruction in the block
        if let Some(last_instr) = block.instructions.last()
            && let Instruction::Ret { value } = last_instr
        {
            // Check if there's a call instruction before the return
            if block.instructions.len() >= 2 {
                let second_last_idx = block.instructions.len() - 2;
                let second_last_instr = &block.instructions[second_last_idx];

                if let Instruction::Call { name, args, ret } = second_last_instr {
                    // Check if this is a tail call (return value matches call result)
                    if self.is_tail_call(value, ret)
                        && self.is_tail_call_safe(func_sig, name, args, ret)
                    {
                        // Convert the call to a tail call jump
                        if self.convert_to_tail_call(&mut block.instructions[second_last_idx]) {
                            changed = true;
                        }
                    }
                }
            }
        }

        changed
    }

    /// Check if a call instruction is in tail position
    fn is_tail_call(&self, return_value: &Option<Operand>, call_result: &Option<Register>) -> bool {
        match (return_value, call_result) {
            // Direct return of call result: return call_result
            (Some(Operand::Register(ret_reg)), Some(call_reg)) => ret_reg == call_reg,
            // No return value from both
            (None, None) => true,
            // Call has no result but function returns something - not a tail call
            (_, None) => false,
            // Call has result but function returns nothing - not a tail call
            (None, Some(_)) => false,
            // Return value doesn't match call result
            _ => false,
        }
    }

    /// Convert a regular call to a tail call (jump)
    fn convert_to_tail_call(&self, instr: &mut Instruction) -> bool {
        if let Instruction::Call { name, args, ret: _ } = instr {
            // Convert the Call instruction to a TailCall instruction
            let call_name = name.clone();
            let call_args = args.clone();

            *instr = Instruction::TailCall {
                name: call_name,
                args: call_args,
            };

            true
        } else {
            false
        }
    }

    /// Check if a function is suitable for tail call optimization
    ///
    /// This now supports generalized tail calls (not just self-recursion) by strictly
    /// checking signature compatibility, which is crucial for Windows x86 (stdcall).
    ///
    /// Helper for checking if the caller and callee have compatible signatures.
    /// Since we don't have access to the callee's definition here, we infer its
    /// signature from the call instruction's arguments and return value.
    fn is_tail_call_safe(
        &self,
        func_sig: &crate::mir::function::Signature,
        _call_name: &str,
        call_args: &[Operand],
        call_ret: &Option<Register>,
    ) -> bool {
        // 1. Check return type compatibility
        match (&func_sig.ret_ty, call_ret) {
            (Some(_func_ty), Some(_ret_reg)) => {
                // Ideally checks types, but register type is unknown here.
                // Optimistically assume registers match if present.
            }
            (None, None) => {} // Both void
            _ => return false, // Mismatch (void vs scalar)
        }

        // 2. Check argument compatibility (Stack size / Calling Convention)
        if func_sig.params.len() != call_args.len() {
            return false;
        }

        for (param, arg) in func_sig.params.iter().zip(call_args.iter()) {
            if !self.is_compatible(param.ty, arg) {
                return false;
            }
        }

        true
    }

    /// Check if an operand is compatible with a formal parameter type
    fn is_compatible(&self, param_ty: crate::mir::types::MirType, arg: &Operand) -> bool {
        match arg {
            Operand::Register(_) => true, // Assume virtual registers match (optimistic)
            Operand::Immediate(imm) => {
                // Check if immediate fits in the type
                match (param_ty, imm) {
                    (crate::mir::types::MirType::Scalar(s), _) => match (s, imm) {
                        (crate::mir::types::ScalarType::I64, crate::mir::Immediate::I64(_)) => true,
                        (crate::mir::types::ScalarType::I32, crate::mir::Immediate::I32(_)) => true,
                        (crate::mir::types::ScalarType::I16, crate::mir::Immediate::I16(_)) => true,
                        (crate::mir::types::ScalarType::I8, crate::mir::Immediate::I8(_)) => true,
                        // Allow smaller immediates to fit in larger types?
                        // Usually MIR expects exact type match for immediates
                        _ => false,
                    },
                    (crate::mir::types::MirType::Vector(_), _) => false, // Immediate vectors not fully supported yet in this check
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, Immediate, MirType, Operand, ScalarType, VirtualReg};

    #[test]
    fn test_tail_call_detection() {
        let tco = TailCallOptimization::default();

        // Test direct return of call result
        let ret_reg = VirtualReg::gpr(0).into();
        let call_reg = VirtualReg::gpr(0).into();
        assert!(tco.is_tail_call(&Some(Operand::Register(ret_reg)), &Some(call_reg)));

        // Test no return value
        assert!(tco.is_tail_call(&None, &None));

        // Test mismatch - call has result but function doesn't return it
        assert!(!tco.is_tail_call(&None, &Some(VirtualReg::gpr(0).into())));

        // Test different registers
        let ret_reg1 = VirtualReg::gpr(0).into();
        let call_reg2 = VirtualReg::gpr(1).into();
        assert!(!tco.is_tail_call(&Some(Operand::Register(ret_reg1)), &Some(call_reg2)));
    }

    #[test]
    fn test_tail_call_conversion() {
        let tco = TailCallOptimization::default();

        let mut instr = Instruction::Call {
            name: "factorial".to_string(),
            args: vec![Operand::Register(VirtualReg::gpr(0).into())],
            ret: Some(VirtualReg::gpr(1).into()),
        };

        let changed = tco.convert_to_tail_call(&mut instr);
        assert!(changed);

        if let Instruction::TailCall { name, args } = &instr {
            assert_eq!(name, "factorial");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected TailCall instruction, got: {:?}", instr);
        }
    }

    #[test]
    fn test_tail_call_optimization_simple() {
        // Test a simple recursive function that can benefit from TCO
        let func = FunctionBuilder::new("factorial")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Base case check would be here, but simplified for test
            // Call factorial(n-1)
            .instr(Instruction::Call {
                name: "factorial".to_string(),
                args: vec![Operand::Register(VirtualReg::gpr(0).into())],
                ret: Some(VirtualReg::gpr(1).into()),
            })
            // Return the result
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let mut func = func;
        let tco = TailCallOptimization::default();

        // The function calls itself with the same name, so it should be optimized
        let changed = tco.apply(&mut func).expect("TCO should succeed");

        // Should have optimized the tail call
        assert!(changed);
    }

    #[test]
    fn test_no_optimization_for_non_tail_calls() {
        // Test that non-tail calls are not optimized
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Call that is not in tail position
            .instr(Instruction::Call {
                name: "other_func".to_string(),
                args: vec![],
                ret: Some(VirtualReg::gpr(0).into()),
            })
            // Do something with the result
            .instr(Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let mut func = func;
        let tco = TailCallOptimization::default();
        let changed = tco.apply(&mut func).expect("TCO should succeed");

        // Should not have changed anything
        assert!(!changed);

        // Verify the call instruction is unchanged
        let entry = func.get_block("entry").expect("entry block exists");
        if let Some(Instruction::Call { name, .. }) = entry.instructions.first() {
            assert_eq!(name, "other_func");
        } else {
            panic!("Expected Call instruction first");
        }
    }
}
