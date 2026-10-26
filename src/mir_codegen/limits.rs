//! Upper bounds shared by AOT assembly emission, RAS JIT encoding, and the runtime call shim.
//!
//! [`validate_module_call_parameters`] runs at the start of each AOT backend
//! (`generate_mir_*_with_units_and_settings` / WASM `generate_mir_wasm_with_units`) so limits
//! cannot be skipped by calling a per-arch emitter directly. JIT entry points also validate
//! before RAS encoding.

use crate::{
    error::LaminaError,
    mir::{Instruction, Module, codegen::MAX_REGISTER_AGGREGATE, types::Aggregate},
    mir_codegen::CodegenError,
};
use lamina_platform::TargetArchitecture;

/// Maximum number of scalar parameters on any single function and on any `Call` / `TailCall`.
///
/// Matches the dynamic JIT shim cap on AArch64 and SysV x86_64. Toolchains accept large arities
/// in practice; this is a Lamina policy limit (stack use, encoder immediates, test surface).
pub const MAX_MIR_CALL_PARAMETERS: usize = 256;

/// Architectures with an aggregate classification. The others reject a module that
/// passes an aggregate by value rather than guess at an ABI for it.
fn classifies_aggregates(target_arch: TargetArchitecture) -> bool {
    matches!(
        target_arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64
    )
}

/// Rejects modules whose signatures or call sites exceed [`MAX_MIR_CALL_PARAMETERS`],
/// and those passing an aggregate by value to a backend that cannot classify one.
pub fn validate_module_call_parameters(
    module: &Module,
    target_arch: TargetArchitecture,
) -> Result<(), LaminaError> {
    if !classifies_aggregates(target_arch) {
        let mut names: Vec<&String> = module
            .functions
            .iter()
            .filter(|(_, func)| {
                func.sig.ret_aggregate.is_some()
                    || func.sig.params.iter().any(|p| p.aggregate.is_some())
            })
            .map(|(name, _)| name)
            .collect();
        names.sort();
        if let Some(name) = names.first() {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("{target_arch:?} does not yet pass aggregates by value, as @{name} does"),
            )));
        }
    }
    if target_arch == TargetArchitecture::Aarch64 {
        // aapcs64 passes a composite over 16 bytes by reference, to a copy the caller
        // owns. Nothing makes that copy yet, so say so rather than share the original.
        let oversized = |aggregate: &Option<Aggregate>| {
            aggregate.is_some_and(|a| a.size as u64 > MAX_REGISTER_AGGREGATE)
        };
        let mut names: Vec<&String> = module
            .functions
            .iter()
            .filter(|(_, func)| {
                oversized(&func.sig.ret_aggregate)
                    || func.sig.params.iter().any(|p| oversized(&p.aggregate))
            })
            .map(|(name, _)| name)
            .collect();
        names.sort();
        if let Some(name) = names.first() {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!(
                    "Aarch64 does not yet pass an aggregate over {MAX_REGISTER_AGGREGATE} bytes, as @{name} does"
                ),
            )));
        }
    }
    if !matches!(
        target_arch,
        TargetArchitecture::X86_64
            | TargetArchitecture::Aarch64
            | TargetArchitecture::Riscv64
            | TargetArchitecture::PowerPC64
    ) {
        let mut names: Vec<&String> = module
            .functions
            .iter()
            .filter(|(_, func)| {
                func.blocks.iter().any(|block| {
                    block
                        .instructions
                        .iter()
                        .any(|i| matches!(i, Instruction::CallIndirect { .. }))
                })
            })
            .map(|(name, _)| name)
            .collect();
        names.sort();
        if let Some(name) = names.first() {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("{target_arch:?} does not yet call through a pointer, as @{name} does"),
            )));
        }
    }
    module
        .validate_in_pipeline()
        .map_err(CodegenError::InvalidMir)
        .map_err(LaminaError::CodegenError)?;
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
                    Instruction::TailCall { name, args } if args.len() > max => {
                        return Err(LaminaError::ValidationError(format!(
                            "TailCall to '{}' passes {} arguments (maximum is {})",
                            name,
                            args.len(),
                            max
                        )));
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
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::mir::{
        Block, Function, MirType, Module as MirModule, Operand, Parameter, Register, ScalarType,
        Signature, VirtualReg,
    };
    use lamina_mir::Immediate;

    #[test]
    fn only_classifying_backends_accept_aggregates_at_boundaries() {
        use lamina_mir::{Aggregate, AggregateClass};
        let slot = VirtualReg::gpr(0);
        let aggregate = Aggregate {
            size: 16,
            align: 8,
            class: AggregateClass::Integer,
        };
        let parameter =
            Parameter::new(Register::Virtual(slot), MirType::ptr()).with_aggregate(aggregate);
        let by_parameter = Signature::new("take")
            .with_params(vec![parameter])
            .with_return(MirType::i64());
        let by_return = Signature::new("take")
            .with_return(MirType::ptr())
            .with_return_aggregate(aggregate);

        let pointer = Operand::Register(Register::Virtual(VirtualReg::gpr(1)));
        let returned = [
            (by_parameter, Operand::Immediate(Immediate::I64(0))),
            (by_return, pointer),
        ];
        for (signature, value) in returned {
            let mut func = Function::new(signature);
            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr {
                dst: Register::Virtual(VirtualReg::gpr(1)),
                slot,
            });
            entry.push(Instruction::Ret { value: Some(value) });
            func.add_block(entry);
            let mut module = MirModule::new("aggregates");
            module.add_function(func);

            for arch in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
                validate_module_call_parameters(&module, arch).expect("classifying backend");
            }
            for arch in [
                TargetArchitecture::Riscv64,
                TargetArchitecture::PowerPC64,
                TargetArchitecture::Wasm64,
            ] {
                let error = validate_module_call_parameters(&module, arch)
                    .expect_err("backend without classification");
                assert!(
                    error
                        .to_string()
                        .contains("does not yet pass aggregates by value"),
                    "{arch:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn rejects_call_over_limit() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut args = Vec::new();
        for i in 0..MAX_MIR_CALL_PARAMETERS {
            args.push(Operand::Immediate(Immediate::I64(i as i64)));
        }
        args.push(Operand::Immediate(Immediate::I64(0)));

        let caller_sig = Signature::new("caller").with_return(i64_ty);
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
            .map(|i| Parameter::new(Register::Virtual(VirtualReg::gpr(i as u32)), i64_ty))
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
