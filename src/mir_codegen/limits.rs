//! Upper bounds shared by AOT assembly emission, RAS JIT encoding, and the runtime call shim.
//!
//! [`validate_module_call_parameters`] runs at the start of each AOT backend
//! (`generate_mir_*_with_units_and_settings` / WASM `generate_mir_wasm_with_units`) so limits
//! cannot be skipped by calling a per-arch emitter directly. JIT entry points also validate
//! before RAS encoding.

use crate::error::LaminaError;
use crate::mir::{Instruction, Module};
use lamina_platform::TargetArchitecture;

/// Maximum number of scalar parameters on any single function and on any `Call` / `TailCall`.
///
/// Matches the dynamic JIT shim cap on AArch64 and SysV x86_64. Toolchains accept large arities
/// in practice; this is a Lamina policy limit (stack use, encoder immediates, test surface).
pub const MAX_MIR_CALL_PARAMETERS: usize = 256;

/// Rejects modules whose signatures or call sites exceed [`MAX_MIR_CALL_PARAMETERS`].
///
/// `target_arch` is reserved for future per-target caps (e.g. stricter WASM tooling).
pub fn validate_module_call_parameters(
    module: &Module,
    target_arch: TargetArchitecture,
) -> Result<(), LaminaError> {
    let _ = target_arch;
    let max = MAX_MIR_CALL_PARAMETERS;
    for func in module.functions.values() {
        if func.sig.params.len() > max {
            return Err(LaminaError::ValidationError(format!(
                "Function '{}' has {} parameters (maximum is {})",
                func.sig.name,
                func.sig.params.len(),
                max
            )));
        }
        for block in &func.blocks {
            for instr in &block.instructions {
                match instr {
                    Instruction::Call { name, args, .. } => {
                        if args.len() > max {
                            return Err(LaminaError::ValidationError(format!(
                                "Call to '{}' passes {} arguments (maximum is {})",
                                name,
                                args.len(),
                                max
                            )));
                        }
                    }
                    Instruction::TailCall { name, args } => {
                        if args.len() > max {
                            return Err(LaminaError::ValidationError(format!(
                                "TailCall to '{}' passes {} arguments (maximum is {})",
                                name,
                                args.len(),
                                max
                            )));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        Block, Function, MirType, Operand, Parameter, Register, ScalarType, Signature, VirtualReg,
    };

    #[test]
    fn rejects_call_over_limit() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let callee_sig = Signature::new("callee").with_return(i64_ty.clone());
        let mut callee = Function::new(callee_sig);
        let mut cb = Block::new("entry");
        cb.push(Instruction::Ret { value: None });
        callee.add_block(cb);

        let mut args = Vec::new();
        for i in 0..MAX_MIR_CALL_PARAMETERS {
            args.push(Operand::Immediate(lamina_mir::Immediate::I64(i as i64)));
        }
        args.push(Operand::Immediate(lamina_mir::Immediate::I64(0)));

        let caller_sig = Signature::new("caller").with_return(i64_ty.clone());
        let mut caller = Function::new(caller_sig);
        let mut eb = Block::new("entry");
        eb.push(Instruction::Call {
            name: "callee".to_string(),
            args,
            ret: None,
        });
        eb.push(Instruction::Ret { value: None });
        caller.add_block(eb);

        let mut module = Module::new("m");
        module.add_function(callee);
        module.add_function(caller);

        let err = validate_module_call_parameters(&module, TargetArchitecture::Aarch64)
            .expect_err("should reject");
        assert!(
            err.to_string().contains("maximum is 256"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn accepts_boundary_arity() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let params: Vec<Parameter> = (0..MAX_MIR_CALL_PARAMETERS)
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty.clone()))
            .collect();
        let sig = Signature::new("big")
            .with_params(params)
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut b = Block::new("entry");
        b.push(Instruction::Ret { value: None });
        f.add_block(b);
        let mut module = Module::new("m");
        module.add_function(f);
        validate_module_call_parameters(&module, TargetArchitecture::Aarch64).expect("ok at limit");
    }
}
