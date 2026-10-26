//! Function inlining transforms for MIR.

use crate::mir::{
    AddressMode, Block, Function, Instruction, Module, Operand, Register, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
};

/// Compatibility placeholder; cross-function inlining requires module context.
#[derive(Default)]
pub struct FunctionInlining;

impl Transform for FunctionInlining {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn description(&self) -> &'static str {
        "Module-level function inlining (see ModuleInlining for full implementation)"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::Inlining
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, _func: &mut Function) -> Result<bool, TransformError> {
        Ok(false)
    }
}

/// Module-level function inlining that analyzes the entire program.
#[derive(Default)]
pub struct ModuleInlining {
    inline_counter: Cell<usize>,
}

impl ModuleInlining {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inline_functions(&self, module: &mut Module) -> Result<usize, TransformError> {
        module
            .validate_in_pipeline()
            .map_err(|errors| TransformError::VerificationFailed { errors })?;
        let mut count = 0;
        loop {
            let mut sites = Vec::new();
            let mut names: Vec<_> = module.functions.keys().cloned().collect();
            names.sort();
            for name in names {
                let caller = &module.functions[&name];
                if caller.instruction_count() > 1000 {
                    continue;
                }
                for (block_index, block) in caller.blocks.iter().enumerate() {
                    for (index, instruction) in block.instructions.iter().enumerate() {
                        if let Instruction::Call {
                            name: callee, args, ..
                        } = instruction
                            && !module.is_external(callee)
                            && let Some(function) = module.functions.get(callee)
                            && function.sig.kernel.is_none()
                            && !function.sig.no_inline
                            && function.instruction_count() <= 30
                            && function.sig.params.len() == args.len()
                            && function.sig.ret_aggregate.is_none()
                            && function
                                .sig
                                .params
                                .iter()
                                .all(|param| param.aggregate.is_none())
                            && !Self::is_recursive(callee, callee, module, &mut HashSet::new())
                            && function
                                .blocks
                                .iter()
                                .flat_map(|b| &b.instructions)
                                .all(|i| {
                                    let mut clone = i.clone();
                                    !matches!(i, Instruction::TailCall { .. })
                                        && Self::map_registers(&mut clone, &mut Clone::clone)
                                })
                        {
                            sites.push((name.clone(), block_index, index, callee.clone()));
                        }
                    }
                }
            }
            let Some((caller_name, block_index, index, callee_name)) = sites.into_iter().next()
            else {
                break;
            };
            let callee = module.functions[&callee_name].clone();
            let caller = module
                .functions
                .get_mut(&caller_name)
                .ok_or(TransformError::InvalidState("missing caller"))?;
            self.expand(caller, block_index, index, &callee)?;
            count += 1;
            if count >= 1000 {
                break;
            }
        }
        module
            .validate_in_pipeline()
            .map_err(|errors| TransformError::VerificationFailed { errors })?;
        Ok(count)
    }

    fn is_recursive(
        root: &str,
        current: &str,
        module: &Module,
        seen: &mut HashSet<String>,
    ) -> bool {
        if !seen.insert(current.to_string()) {
            return false;
        }
        module.functions.get(current).is_some_and(|function| {
            function
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|i| {
                    if let Instruction::Call { name, .. } | Instruction::TailCall { name, .. } = i {
                        name == root || Self::is_recursive(root, name, module, seen)
                    } else {
                        false
                    }
                })
        })
    }

    fn expand(
        &self,
        caller: &mut Function,
        block_index: usize,
        index: usize,
        callee: &Function,
    ) -> Result<(), TransformError> {
        let Instruction::Call { args, ret, .. } =
            caller.blocks[block_index].instructions[index].clone()
        else {
            return Err(TransformError::InvalidState("expected call"));
        };
        let mut max_id = 0;
        let mut record = |reg: &Register| {
            if let Register::Virtual(v) = reg {
                max_id = max_id.max(v.id);
            }
            reg.clone()
        };
        for param in &caller.sig.params {
            record(&param.reg);
        }
        for slot in caller.stack_reservations.keys() {
            record(&Register::Virtual(*slot));
        }
        for block in &caller.blocks {
            for instruction in &block.instructions {
                if let Some(dst) = instruction.def_reg() {
                    record(dst);
                }
                for used in instruction.use_regs() {
                    record(used);
                }
                if let Instruction::StackAddr { slot, .. } = instruction {
                    record(&Register::Virtual(*slot));
                }
            }
        }
        let mut next_id = max_id + 1;
        let mut registers = HashMap::new();
        let mut rename = |reg: &Register| -> Register {
            match reg {
                Register::Virtual(v) => registers
                    .entry(reg.clone())
                    .or_insert_with(|| {
                        let fresh = Register::Virtual(VirtualReg::new(next_id, v.class));
                        next_id += 1;
                        fresh
                    })
                    .clone(),
                Register::Physical(_) => reg.clone(),
            }
        };
        let mut copies = Vec::new();
        for (param, arg) in callee.sig.params.iter().zip(args) {
            let mut copy = Instruction::Copy {
                ty: param.ty,
                dst: rename(&param.reg),
                src: arg,
            };
            copy.normalize_integer_immediates(None);
            copies.push(copy);
        }
        for (slot, size) in &callee.stack_reservations {
            if let Register::Virtual(fresh) = rename(&Register::Virtual(*slot)) {
                caller.stack_reservations.insert(fresh, *size);
            }
        }
        let id = self.inline_counter.get();
        self.inline_counter.set(id + 1);
        let prefix = format!("__inline_{id}_");
        let mut labels = HashMap::new();
        for block in &callee.blocks {
            let mut label = format!("{prefix}{}", block.label);
            while caller.blocks.iter().any(|b| b.label == label)
                || labels.values().any(|l| l == &label)
            {
                label.push('_');
            }
            labels.insert(block.label.clone(), label);
        }
        let mut continuation = format!("{prefix}continuation");
        while caller.blocks.iter().any(|b| b.label == continuation)
            || labels.values().any(|l| l == &continuation)
        {
            continuation.push('_');
        }
        let mut blocks = callee.blocks.clone();
        for block in &mut blocks {
            block.label = labels[&block.label].clone();
            let mut instructions = Vec::new();
            for mut instruction in block.instructions.drain(..) {
                Self::map_registers(&mut instruction, &mut rename);
                match &mut instruction {
                    Instruction::Jmp { target } => *target = labels[target].clone(),
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        *true_target = labels[true_target].clone();
                        *false_target = labels[false_target].clone();
                    }
                    Instruction::Switch { cases, default, .. } => {
                        for (_, target) in cases {
                            *target = labels[target].clone();
                        }
                        *default = labels[default].clone();
                    }
                    Instruction::Ret { value } => {
                        if let (Some(dst), Some(src), Some(ty)) = (&ret, value, callee.sig.ret_ty) {
                            let mut copy = Instruction::Copy {
                                dst: dst.clone(),
                                src: src.clone(),
                                ty,
                            };
                            copy.normalize_integer_immediates(None);
                            instructions.push(copy);
                        }
                        instruction = Instruction::Jmp {
                            target: continuation.clone(),
                        };
                    }
                    _ => {}
                }
                instruction.normalize_integer_immediates(callee.sig.ret_ty);
                instructions.push(instruction);
            }
            block.instructions = instructions;
        }
        let block = &mut caller.blocks[block_index];
        if blocks.len() == 1
            && matches!(blocks[0].instructions.last(), Some(Instruction::Jmp { target }) if target == &continuation)
        {
            blocks[0].instructions.pop();
            copies.append(&mut blocks[0].instructions);
            block.instructions.splice(index..=index, copies);
            return Ok(());
        }
        let after = block.instructions.split_off(index + 1);
        block.instructions.pop();
        block.instructions.extend(copies);
        block.instructions.push(Instruction::Jmp {
            target: labels[&callee.entry].clone(),
        });
        blocks.push(Block {
            label: continuation,
            instructions: after,
        });
        caller
            .blocks
            .splice(block_index + 1..block_index + 1, blocks);
        Ok(())
    }

    pub(crate) fn map_registers(
        instruction: &mut Instruction,
        map: &mut impl FnMut(&Register) -> Register,
    ) -> bool {
        fn operand(value: &mut Operand, map: &mut impl FnMut(&Register) -> Register) {
            if let Operand::Register(reg) = value {
                *reg = map(reg);
            }
        }
        fn address(addr: &mut AddressMode, map: &mut impl FnMut(&Register) -> Register) {
            match addr {
                AddressMode::BaseOffset { base, .. } => *base = map(base),
                AddressMode::BaseIndexScale { base, index, .. } => {
                    *base = map(base);
                    *index = map(index);
                }
            }
        }
        match instruction {
            Instruction::Copy { dst, src, .. }
            | Instruction::IntConvert { dst, src, .. }
            | Instruction::IntToFloat { dst, src, .. }
            | Instruction::FloatToInt { dst, src, .. }
            | Instruction::FloatConvert { dst, src, .. }
            | Instruction::Bitcast { dst, src, .. }
            | Instruction::FloatUnary { dst, src, .. } => {
                *dst = map(dst);
                operand(src, map);
            }
            Instruction::IntBinary { dst, lhs, rhs, .. }
            | Instruction::FloatBinary { dst, lhs, rhs, .. }
            | Instruction::IntCmp { dst, lhs, rhs, .. }
            | Instruction::FloatCmp { dst, lhs, rhs, .. } => {
                *dst = map(dst);
                operand(lhs, map);
                operand(rhs, map);
            }
            Instruction::Select {
                dst,
                cond,
                true_val,
                false_val,
                ..
            } => {
                *dst = map(dst);
                *cond = map(cond);
                operand(true_val, map);
                operand(false_val, map);
            }
            Instruction::Load { dst, addr, .. } => {
                *dst = map(dst);
                address(addr, map);
            }
            Instruction::Store { src, addr, .. } => {
                operand(src, map);
                address(addr, map);
            }
            Instruction::StackAddr { dst, slot } => {
                *dst = map(dst);
                if let Register::Virtual(fresh) = map(&Register::Virtual(*slot)) {
                    *slot = fresh;
                }
            }
            Instruction::SymbolAddress { dst, .. } => {
                *dst = map(dst);
            }
            Instruction::Lea { dst, base, .. } => {
                *dst = map(dst);
                *base = map(base);
            }
            Instruction::VectorOp { dst, operands, .. } => {
                *dst = map(dst);
                for value in operands {
                    operand(value, map);
                }
            }
            Instruction::Br { cond, .. } | Instruction::Switch { value: cond, .. } => {
                *cond = map(cond)
            }
            Instruction::Call { args, ret, .. } => {
                for value in args {
                    operand(value, map);
                }
                if let Some(dst) = ret {
                    *dst = map(dst);
                }
            }
            Instruction::TailCall { args, .. } => {
                for value in args {
                    operand(value, map);
                }
            }
            Instruction::Ret { value } => {
                if let Some(value) = value {
                    operand(value, map);
                }
            }
            Instruction::Jmp { .. }
            | Instruction::Unreachable
            | Instruction::SafePoint
            | Instruction::StackMap { .. }
            | Instruction::PatchPoint { .. }
            | Instruction::Comment { .. } => {}
            #[cfg(feature = "nightly")]
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, Immediate, IntBinOp, MirType, ScalarType};
    use crate::{mir::codegen::from_ir, parser::parse_module};

    #[test]
    fn noinline_callees_are_never_inlined() {
        let ir = parse_module(
            "@noinline fn @helper() -> i64 {\nentry:\nret.i64 7\n}\nfn @main() -> i64 {\nentry:\n%v = call @helper()\nret.i64 %v\n}",
        )
        .expect("parse noinline caller");
        let mut module = from_ir(&ir, "noinline caller").expect("lower noinline caller");
        assert!(module.get_function("helper").expect("callee").sig.no_inline);
        let count = ModuleInlining::new()
            .inline_functions(&mut module)
            .expect("run inlining");
        assert_eq!(count, 0);
    }

    #[test]
    fn external_callees_are_never_inlined() {
        let ir = parse_module(
            "@extern fn @helper() -> i64 {\nentry:\nret.i64 7\n}\nfn @main() -> i64 {\nentry:\n%v = call @helper()\nret.i64 %v\n}",
        )
        .expect("parse extern caller");
        let mut module = from_ir(&ir, "extern caller").expect("lower extern caller");
        assert!(module.is_external("helper"));
        let count = ModuleInlining::new()
            .inline_functions(&mut module)
            .expect("run inlining");
        assert_eq!(count, 0);
        let calls = module
            .get_function("main")
            .expect("caller")
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|instruction| {
                matches!(instruction, Instruction::Call { name, .. } if name == "helper")
            })
            .count();
        assert_eq!(calls, 1);
    }

    #[test]
    fn inlining_keeps_values_out_of_stack_storage() {
        let ir = parse_module(
            "fn @main() -> void {\nentry:\n%p = alloc.stack i64\ncall @leaf(%p)\nret.void\n}\nfn @leaf(ptr %p) -> i64 {\nentry:\n%x = add.i64 1, 2\nstore.i64 %p, %x\nret.i64 %x\n}",
        ).expect("parse stack caller");
        let mut module = from_ir(&ir, "stack caller").expect("lower stack caller");
        let count = ModuleInlining::new()
            .inline_functions(&mut module)
            .expect("inline leaf");
        assert!(count > 0);
        let caller = module.get_function("main").expect("caller");
        let instructions: Vec<_> = caller
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .collect();
        let slots: Vec<_> = instructions
            .iter()
            .filter_map(|instruction| {
                if let Instruction::StackAddr { slot, .. } = instruction {
                    Some(Register::Virtual(*slot))
                } else {
                    None
                }
            })
            .collect();
        assert!(instructions.iter().all(|instruction| {
            instruction
                .def_reg()
                .is_none_or(|register| !slots.contains(register))
        }));
        caller.validate().expect("inlined caller verifies");
    }

    #[test]
    fn repeated_stack_callees_keep_distinct_storage_and_reservations() {
        let ir = parse_module(
            "fn @leaf(i64 %x) -> i64 {
entry:
%p = alloc.stack [8 x i64]
store.i64 %p, %x
%r = load.i64 %p
ret.i64 %r
}
fn @main() -> i64 {
entry:
%p = alloc.stack [8 x i64]
%a = call @leaf(11)
%b = call @leaf(22)
%c = add.i64 %a, %b
ret.i64 %c
}",
        )
        .unwrap();
        let mut module = from_ir(&ir, "stack").unwrap();
        assert_eq!(
            ModuleInlining::new().inline_functions(&mut module).unwrap(),
            2
        );
        let caller = &module.functions["main"];
        assert_eq!(caller.stack_reservations.len(), 3);
        let slots: HashSet<_> = caller
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| {
                if let Instruction::StackAddr { slot, .. } = i {
                    Some(*slot)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(slots.len(), 3);
        assert!(
            caller
                .stack_reservations
                .keys()
                .all(|slot| slots.contains(slot))
        );
        assert!(
            caller.blocks.iter().flat_map(|b| &b.instructions).all(|i| {
                !matches!(i.def_reg(), Some(Register::Virtual(v)) if slots.contains(v))
            })
        );
        caller.validate().unwrap();
    }

    #[test]
    fn direct_and_mutual_recursion_remain_calls() {
        for source in [
            "fn @a() -> void {
entry:
call @a()
ret.void
}",
            "fn @a() -> void {
entry:
call @b()
ret.void
}
fn @b() -> void {
entry:
call @a()
ret.void
}",
        ] {
            let ir = parse_module(source).unwrap();
            let mut module = from_ir(&ir, "recursion").unwrap();
            let original = module.clone();
            assert_eq!(
                ModuleInlining::new().inline_functions(&mut module).unwrap(),
                0
            );
            assert_eq!(module, original);
        }
    }

    #[test]
    fn test_inline_multi_block() {
        let mut module = Module::new("test_module");

        // Callee: 2 blocks
        // entry:
        //   v0 = add p0, 1
        //   jmp exit
        // exit:
        //   ret v0
        let callee = FunctionBuilder::new("callee")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()), // p0 is v0 (param 0)
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        module.add_function(callee);

        // Caller:
        // entry:
        //   v1 = call callee(10)
        //   ret v1
        let caller = FunctionBuilder::new("caller")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Call {
                name: "callee".to_string(),
                args: vec![Operand::Immediate(Immediate::I64(10))],
                ret: Some(VirtualReg::gpr(1).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        module.add_function(caller);

        let inline_pass = ModuleInlining::new();
        let count = inline_pass
            .inline_functions(&mut module)
            .expect("Inlining failed");

        assert!(count > 0, "Should have inlined 1 function");

        let caller = module.functions.get("caller").unwrap();
        // Multi-Block inlining splits entry -> Entry, Split. And inserts CalleeEntry, CalleeExit.
        // Total 4 blocks expected.
        assert!(
            caller.blocks.len() >= 3,
            "Expected at least 3 blocks after inlining, got {}",
            caller.blocks.len()
        );

        // Verify Call is gone
        let has_call = caller.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|i| matches!(i, Instruction::Call { .. }))
        });
        assert!(!has_call, "Call instruction should be removed");
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod execution_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{
        env,
        error::Error,
        fs,
        process::{self, Command},
    };

    #[test]
    fn inlining_preserves_native_and_mir_assembly_outputs() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "gcc"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-inlining-{}", process::id()));
        fs::create_dir_all(&directory)?;
        for case in [
            "nested_calls",
            "functions",
            "test_u8",
            "stack_operations",
            "type_conversions",
            "edge_cases_bug_test",
        ] {
            let source = format!("testcases/{case}.lamina");
            let expected = fs::read_to_string(format!("testcases/{case}.expected"))?;
            for level in ["1", "2", "3"] {
                for allocator in ["incremental", "linear", "graph"] {
                    for mir in [false, true] {
                        let executable =
                            directory.join(format!("{case}-{level}-{allocator}-{mir}"));
                        let mut flags = vec!["--opt-level", level, "--regalloc", allocator];
                        if mir {
                            flags.extend(["--emit-mir-asm", "--target", "x86_64_linux"]);
                        }
                        compile(&source, &executable, &flags)?;
                        if mir {
                            let assembled = Command::new("as")
                                .arg("--64")
                                .arg(executable.with_extension("s"))
                                .arg("-o")
                                .arg(executable.with_extension("o"))
                                .output()?;
                            assert!(
                                assembled.status.success(),
                                "{}",
                                String::from_utf8_lossy(&assembled.stderr)
                            );
                            let linked = Command::new("gcc")
                                .arg("-no-pie")
                                .arg(executable.with_extension("o"))
                                .arg("-o")
                                .arg(&executable)
                                .output()?;
                            assert!(
                                linked.status.success(),
                                "{}",
                                String::from_utf8_lossy(&linked.stderr)
                            );
                        }
                        let output = Command::new(&executable).output()?;
                        assert_eq!(
                            String::from_utf8_lossy(&output.stdout).trim_end_matches('\n'),
                            expected.trim_end_matches('\n'),
                            "{case} O{level} {allocator} mir={mir}"
                        );
                    }
                }
            }
        }
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
