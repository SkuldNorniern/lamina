//! Tail call optimization transform for MIR.

use std::collections::HashSet;

use crate::mir::{
    Block, Function, Immediate, Instruction, Operand, Register,
    function::Signature,
    gpu::GpuIntrinsic,
    transform::{
        CalleeShape, CalleeSignatures, ParamShape, Transform, TransformCategory, TransformError,
        TransformLevel,
    },
    types::{MirType, ScalarType},
};

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
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func, &CalleeSignatures::new())
    }

    fn apply_with_callees(
        &self,
        func: &mut Function,
        callees: &CalleeSignatures,
    ) -> Result<bool, TransformError> {
        self.apply_internal(func, callees)
    }
}

impl TailCallOptimization {
    fn apply_internal(
        &self,
        func: &mut Function,
        callees: &CalleeSignatures,
    ) -> Result<bool, TransformError> {
        if func.sig.variadic {
            return Ok(false);
        }

        let mut changed = false;
        let func_sig = func.sig.clone();
        let (stack_addresses, escaped) = Self::stack_addresses(func);
        if escaped {
            return Ok(false);
        }

        for block in &mut func.blocks {
            if self.optimize_block_tail_calls(&func_sig, block, &stack_addresses, callees) {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Registers holding an address in this frame, and whether any of them reached
    /// memory or a callee. Once one has, the frame outlives the call that let it out,
    /// and no call in the function can become a tail call.
    fn stack_addresses(func: &Function) -> (HashSet<Register>, bool) {
        let instructions: Vec<_> = func.blocks.iter().flat_map(|b| &b.instructions).collect();
        let mut addresses = HashSet::new();
        let mut escaped = false;
        loop {
            let old_len = addresses.len();
            let old_escaped = escaped;
            for instruction in &instructions {
                let uses_address = instruction
                    .use_regs()
                    .iter()
                    .any(|r| addresses.contains(*r));
                if matches!(&instruction.instr, Instruction::Store { src: Operand::Register(r), .. } if addresses.contains(r))
                    || matches!(
                        &instruction.instr,
                        Instruction::AtomicStore { src: Operand::Register(r), .. }
                            | Instruction::AtomicBinary { value: Operand::Register(r), .. }
                        if addresses.contains(r)
                    )
                    || matches!(
                        &instruction.instr,
                        Instruction::Call { .. } | Instruction::CallIndirect { .. }
                        if uses_address
                    )
                {
                    escaped = true;
                }
                if (matches!(&instruction.instr, Instruction::StackAddr { .. })
                    || uses_address
                    || (escaped
                        && matches!(
                            &instruction.instr,
                            Instruction::Load { .. }
                                | Instruction::AtomicLoad { .. }
                                | Instruction::AtomicBinary { .. }
                                | Instruction::Call { .. }
                                | Instruction::CallIndirect { .. }
                        )))
                    && let Some(dst) = instruction.def_reg()
                {
                    addresses.insert(dst.clone());
                }
            }
            if addresses.len() == old_len && escaped == old_escaped {
                return (addresses, escaped);
            }
        }
    }

    fn optimize_block_tail_calls(
        &self,
        func_sig: &Signature,
        block: &mut Block,
        stack_addresses: &HashSet<Register>,
        callees: &CalleeSignatures,
    ) -> bool {
        let mut changed = false;

        // Find the last instruction in the block
        if let Some(last_instr) = block.instructions.last()
            && let Instruction::Ret { value } = &last_instr.instr
        {
            // Check if there's a call instruction before the return
            if block.instructions.len() >= 2 {
                let second_last_idx = block.instructions.len() - 2;
                let second_last_instr = &block.instructions[second_last_idx];

                if let Instruction::Call { name, args, ret } = &second_last_instr.instr {
                    // Check if this is a tail call (return value matches call result)
                    if !args.iter().any(
                        |arg| matches!(arg, Operand::Register(r) if stack_addresses.contains(r)),
                    ) && self.is_tail_call(value, ret)
                        && self.is_tail_call_safe(func_sig, name, args, ret, callees)
                        && self.convert_to_tail_call(&mut block.instructions[second_last_idx])
                    {
                        // The call no longer defines the returned register, so the return
                        // that followed it is both unreachable and undefined.
                        block.instructions.pop();
                        changed = true;
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
    /// Handles generalized tail calls (not just self-recursion) by strictly
    /// checking signature compatibility. This is required for Windows x86 (stdcall).
    ///
    /// The callee's return shape and parameters come from the module when it defines
    /// the callee; otherwise only a returned call result can be trusted, and the
    /// arguments are checked against the caller's own parameters.
    fn is_tail_call_safe(
        &self,
        func_sig: &Signature,
        call_name: &str,
        call_args: &[Operand],
        call_ret: &Option<Register>,
        callees: &CalleeSignatures,
    ) -> bool {
        // An intrinsic is not a symbol a jump can reach: each one lowers to its own
        // sequence, and `print` loads a format string before calling printf.
        if GpuIntrinsic::from_name(call_name).is_some() || call_name.starts_with("lamina.") {
            return false;
        }
        let caller = CalleeShape::of(func_sig);
        // 1. The callee must hand back exactly what the caller does, and take exactly
        // what the caller takes, so both sides agree on the registers and the stack
        // bytes the arguments occupy.
        let params: &[ParamShape] = match callees.get(call_name) {
            Some(callee) => {
                if *callee != caller {
                    return false;
                }
                &callee.params
            }
            None => {
                if call_ret.is_none() || func_sig.ret_aggregate.is_some() {
                    return false;
                }
                &caller.params
            }
        };

        // 2. Check argument compatibility against the callee's parameters
        if params.len() != call_args.len() {
            return false;
        }

        for (param, arg) in params.iter().zip(call_args.iter()) {
            if !self.is_compatible(param.ty, arg) {
                return false;
            }
        }

        true
    }

    /// Check if an operand is compatible with a formal parameter type
    fn is_compatible(&self, param_ty: MirType, arg: &Operand) -> bool {
        match arg {
            Operand::Register(_) => true, // Assume virtual registers match (optimistic)
            Operand::Immediate(imm) => {
                // Check if immediate fits in the type
                match (param_ty, imm) {
                    (MirType::Scalar(s), _) => match (s, imm) {
                        (ScalarType::I64, Immediate::I64(_))
                        | (ScalarType::I32, Immediate::I32(_))
                        | (ScalarType::I16, Immediate::I16(_))
                        | (ScalarType::I8, Immediate::I8(_)) => true,
                        // Allow smaller immediates to fit in larger types?
                        // Usually MIR expects exact type match for immediates
                        _ => false,
                    },
                    (MirType::Vector(_), _) => false, // Immediate vectors not fully supported yet in this check
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
        transform::ReturnShape,
    };
    use lamina_mir::Located;

    #[test]
    fn stack_derived_arguments_keep_caller_frame() {
        use crate::mir::AddressMode;
        let reg = |id| Register::from(VirtualReg::gpr(id));
        let operand = |id| Operand::Register(reg(id));
        let ptr = MirType::Scalar(ScalarType::Ptr);
        let derivations = vec![
            vec![],
            vec![Instruction::Copy {
                ty: ptr,
                dst: reg(2),
                src: operand(1),
            }],
            vec![Instruction::Lea {
                dst: reg(2),
                base: reg(1),
                offset: 8,
            }],
            vec![Instruction::IntConvert {
                from: ptr,
                to: MirType::Scalar(ScalarType::I64),
                signed: false,
                dst: reg(2),
                src: operand(1),
            }],
            vec![Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: reg(2),
                lhs: operand(1),
                rhs: Operand::Immediate(Immediate::I64(8)),
            }],
            vec![Instruction::Select {
                ty: ptr,
                dst: reg(2),
                cond: reg(0),
                true_val: operand(0),
                false_val: operand(1),
            }],
            vec![
                Instruction::Load {
                    attrs: Default::default(),
                    ty: ptr,
                    dst: reg(2),
                    addr: AddressMode::BaseOffset {
                        base: reg(0),
                        offset: 0,
                        address_space: Default::default(),
                    },
                },
                Instruction::Store {
                    attrs: Default::default(),
                    ty: ptr,
                    src: operand(1),
                    addr: AddressMode::BaseOffset {
                        base: reg(0),
                        offset: 0,
                        address_space: Default::default(),
                    },
                },
            ],
            vec![
                Instruction::Copy {
                    ty: ptr,
                    dst: reg(2),
                    src: operand(3),
                },
                Instruction::Copy {
                    ty: ptr,
                    dst: reg(3),
                    src: operand(1),
                },
            ],
        ];
        for derivation in derivations {
            for name in ["caller", "callee"] {
                let argument = if derivation.is_empty() { 1 } else { 2 };
                let mut function = FunctionBuilder::new("caller")
                    .param(reg(0), ptr)
                    .returns(MirType::Scalar(ScalarType::I64))
                    .block("entry")
                    .instr(Instruction::StackAddr {
                        dst: reg(1),
                        slot: VirtualReg::gpr(20),
                    })
                    .build();
                function.blocks[0]
                    .instructions
                    .extend(derivation.iter().cloned().map(Located::from));
                function.blocks[0].instructions.extend(
                    [
                        Instruction::Call {
                            name: name.into(),
                            args: vec![operand(argument)],
                            ret: Some(reg(10)),
                        },
                        Instruction::Ret {
                            value: Some(operand(10)),
                        },
                    ]
                    .map(Located::from),
                );
                assert!(!TailCallOptimization.apply(&mut function).expect("optimize"));
                assert!(matches!(
                    &function.blocks[0].instructions.last().map(|i| &i.instr),
                    Some(Instruction::Ret { .. })
                ));
            }
        }
    }

    #[test]
    fn gpu_barrier_keeps_its_return() {
        let mut function = FunctionBuilder::new("barrier")
            .block("entry")
            .instr(Instruction::Call {
                name: "__lamina_gpu_workgroup_barrier".into(),
                args: vec![],
                ret: None,
            })
            .instr(Instruction::Ret { value: None })
            .build();
        let optimization = TailCallOptimization;
        assert!(!optimization.apply(&mut function).expect("optimize barrier"));
        function.validate().expect("barrier retains one terminator");
        assert!(matches!(
            &function.blocks[0].instructions[0].instr,
            Instruction::Call { .. }
        ));
    }

    #[test]
    fn test_tail_call_detection() {
        let tco = TailCallOptimization;

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
        let tco = TailCallOptimization;

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
        let tco = TailCallOptimization;

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
                op: IntBinOp::Add,
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
        let tco = TailCallOptimization;
        let changed = tco.apply(&mut func).expect("TCO should succeed");

        // Should not have changed anything
        assert!(!changed);

        // Verify the call instruction is unchanged
        let entry = func.get_block("entry").expect("entry block exists");
        if let Some(Instruction::Call { name, .. }) = entry.instructions.first().map(|i| &i.instr) {
            assert_eq!(name, "other_func");
        } else {
            panic!("Expected Call instruction first");
        }
    }

    fn tail_call_candidate(
        caller_ret: Option<MirType>,
        callee: &str,
        call_ret: Option<Register>,
        ret_value: Option<Operand>,
    ) -> Function {
        let mut builder = FunctionBuilder::new("caller")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .block("entry");
        if let Some(ty) = caller_ret {
            builder = builder.returns(ty);
        }
        builder
            .instr(Instruction::Call {
                name: callee.into(),
                args: vec![Operand::Register(VirtualReg::gpr(0).into())],
                ret: call_ret,
            })
            .instr(Instruction::Ret { value: ret_value })
            .build()
    }

    fn callees(entries: &[(&str, Option<MirType>)]) -> CalleeSignatures {
        entries
            .iter()
            .map(|(name, ty)| {
                (
                    (*name).to_string(),
                    CalleeShape {
                        ret: ReturnShape {
                            ty: *ty,
                            aggregate: false,
                        },
                        params: vec![ParamShape {
                            ty: MirType::Scalar(ScalarType::I64),
                            aggregate: None,
                        }],
                        variadic: false,
                    },
                )
            })
            .collect()
    }

    fn assert_stays_call(func: &Function) {
        let instructions = &func.blocks[0].instructions;
        assert!(matches!(
            &instructions[instructions.len() - 2].instr,
            Instruction::Call { .. }
        ));
        assert!(matches!(
            &instructions[instructions.len() - 1].instr,
            Instruction::Ret { .. }
        ));
    }

    fn assert_becomes_tail_call(func: &Function) {
        let instructions = &func.blocks[0].instructions;
        assert!(matches!(
            &instructions.last().map(|i| &i.instr),
            Some(Instruction::TailCall { .. })
        ));
    }

    #[test]
    fn void_caller_discarding_nonvoid_result_is_not_a_tail_call() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let known = callees(&[("f", Some(i64_ty))]);
        let mut func = tail_call_candidate(None, "f", None, None);
        assert!(
            !TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_stays_call(&func);

        let mut unknown = tail_call_candidate(None, "f", None, None);
        assert!(!TailCallOptimization.apply(&mut unknown).expect("optimize"));
        assert_stays_call(&unknown);
    }

    #[test]
    fn nonvoid_caller_with_void_callee_is_not_a_tail_call() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let known = callees(&[("g", None)]);
        let mut func = tail_call_candidate(
            Some(i64_ty),
            "g",
            None,
            Some(Operand::Register(VirtualReg::gpr(0).into())),
        );
        assert!(
            !TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_stays_call(&func);
    }

    #[test]
    fn matching_scalar_return_stays_a_tail_call() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let known = callees(&[("f", Some(i64_ty))]);
        let mut func = tail_call_candidate(
            Some(i64_ty),
            "f",
            Some(VirtualReg::gpr(1).into()),
            Some(Operand::Register(VirtualReg::gpr(1).into())),
        );
        assert!(
            TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_becomes_tail_call(&func);
    }

    #[test]
    fn matching_void_return_stays_a_tail_call() {
        let known = callees(&[("g", None)]);
        let mut func = tail_call_candidate(None, "g", None, None);
        assert!(
            TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_becomes_tail_call(&func);
    }

    #[test]
    fn mismatched_scalar_widths_are_not_a_tail_call() {
        let known = callees(&[("f", Some(MirType::Scalar(ScalarType::I64)))]);
        let mut func = tail_call_candidate(
            Some(MirType::Scalar(ScalarType::I32)),
            "f",
            Some(VirtualReg::gpr(1).into()),
            Some(Operand::Register(VirtualReg::gpr(1).into())),
        );
        assert!(
            !TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_stays_call(&func);
    }

    fn callee_with_params(name: &str, params: Vec<ParamShape>) -> CalleeSignatures {
        let shape = CalleeShape {
            ret: ReturnShape {
                ty: None,
                aggregate: false,
            },
            params,
            variadic: false,
        };
        [(name.to_string(), shape)].into_iter().collect()
    }

    #[test]
    fn callee_with_more_parameters_than_the_call_is_not_a_tail_call() {
        let i64_param = ParamShape {
            ty: MirType::Scalar(ScalarType::I64),
            aggregate: None,
        };
        let known = callee_with_params("g", vec![i64_param, i64_param]);
        let mut func = tail_call_candidate(None, "g", None, None);
        assert!(
            !TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_stays_call(&func);
    }

    #[test]
    fn callee_parameter_type_differing_from_the_caller_is_not_a_tail_call() {
        let known = callee_with_params(
            "g",
            vec![ParamShape {
                ty: MirType::Scalar(ScalarType::I32),
                aggregate: None,
            }],
        );
        let mut func = tail_call_candidate(None, "g", None, None);
        assert!(
            !TailCallOptimization
                .apply_with_callees(&mut func, &known)
                .expect("optimize")
        );
        assert_stays_call(&func);
    }
}
