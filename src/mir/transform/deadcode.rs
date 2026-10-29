//! Dead code elimination transform for MIR.

use crate::mir::{
    Block, Function, Instruction, Module, Register,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_platform::PointerWidth;
use std::collections::{HashMap, HashSet};

/// Statistics about dead code elimination
#[derive(Debug, Default)]
pub struct DeadCodeStats {
    /// Number of instructions removed
    pub instructions_removed: usize,
    /// Number of registers that became dead
    pub registers_freed: usize,
}

/// Dead Code Elimination transform
#[derive(Default)]
pub struct DeadCodeElimination;

impl Transform for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead_code_elimination"
    }

    fn description(&self) -> &'static str {
        "Removes instructions that define registers which are never used"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::DeadCodeElimination
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
            .map(|stats| stats.instructions_removed > 0)
    }
}

impl DeadCodeElimination {
    /// Apply dead code elimination to a function
    pub fn apply_internal(&self, func: &mut Function) -> Result<DeadCodeStats, TransformError> {
        let mut stats = DeadCodeStats::default();
        let pointer = func.pointer_width;

        // 1. Compute liveness analysis (inter-block)
        let live_out_map = self.compute_liveness(func)?;

        // 2. Remove dead instructions using liveness info
        for block in &mut func.blocks {
            // Start with registers live at the end of the block
            let mut live_regs = live_out_map.get(&block.label).cloned().unwrap_or_default();

            let removed = self.remove_dead_instructions_in_block(block, &mut live_regs, pointer);
            stats.instructions_removed += removed;
        }

        Ok(stats)
    }

    /// Compute liveness analysis (live-out sets for each block)
    fn compute_liveness(
        &self,
        func: &Function,
    ) -> Result<HashMap<String, HashSet<Register>>, TransformError> {
        let pointer = func.pointer_width;
        let mut live_in: HashMap<String, HashSet<Register>> = HashMap::new();
        let mut live_out: HashMap<String, HashSet<Register>> = HashMap::new();

        for block in &func.blocks {
            live_in.insert(block.label.clone(), HashSet::new());
            live_out.insert(block.label.clone(), HashSet::new());
        }

        let mut changed = true;

        while changed {
            changed = false;

            // Process blocks in reverse order (heuristic for faster convergence)
            for block in func.blocks.iter().rev() {
                let label = &block.label;

                // 1. Calculate LiveOut = Union(LiveIn(successors))
                let mut current_live_out = HashSet::new();
                if let Some(terminator) = block.instructions.last() {
                    match &terminator.instr {
                        Instruction::Jmp { target } => {
                            if let Some(succ_live_in) = live_in.get(target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                        }
                        Instruction::Br {
                            true_target,
                            false_target,
                            ..
                        } => {
                            if let Some(succ_live_in) = live_in.get(true_target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                            if let Some(succ_live_in) = live_in.get(false_target) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                        }
                        Instruction::Switch { cases, default, .. } => {
                            if let Some(succ_live_in) = live_in.get(default) {
                                current_live_out.extend(succ_live_in.iter().cloned());
                            }
                            for (_, case_target) in cases {
                                if let Some(succ_live_in) = live_in.get(case_target) {
                                    current_live_out.extend(succ_live_in.iter().cloned());
                                }
                            }
                        }
                        _ => {} // Return or others have no successors within function
                    }
                }

                // Safe: live_out is initialized for all blocks at function start
                // If this fails, it indicates a bug in the initialization logic
                let prev_live_out = live_out.get(label).ok_or(TransformError::InvalidState(
                    "block missing from live_out map",
                ))?;
                if current_live_out != *prev_live_out {
                    live_out.insert(label.clone(), current_live_out.clone());
                    changed = true;
                }

                // 2. Calculate LiveIn = Use U (LiveOut - Def)
                let mut current_live_in = current_live_out.clone();
                for instr in block.instructions.iter().rev() {
                    if self.is_dead_instruction(instr, &current_live_in, pointer) {
                        continue;
                    }
                    if let Some(def) = instr.def_reg() {
                        current_live_in.remove(def);
                    }
                    for use_reg in instr.use_regs() {
                        current_live_in.insert(use_reg.clone());
                    }
                }

                // Safe: live_in is initialized for all blocks at function start
                // If this fails, it indicates a bug in the initialization logic
                let prev_live_in = live_in.get(label).ok_or(TransformError::InvalidState(
                    "block missing from live_in map",
                ))?;
                if current_live_in != *prev_live_in {
                    live_in.insert(label.clone(), current_live_in);
                    changed = true;
                }
            }
        }

        Ok(live_out)
    }

    /// Remove dead instructions within a single basic block
    fn remove_dead_instructions_in_block(
        &self,
        block: &mut Block,
        live_regs: &mut HashSet<Register>,
        pointer: PointerWidth,
    ) -> usize {
        let mut removed_count = 0;
        let mut instructions_to_keep = Vec::new();

        // We collect instructions to keep, then reverse them back
        for instr in block.instructions.iter().rev() {
            if self.is_dead_instruction(instr, live_regs, pointer) {
                removed_count += 1;
                // Don't add to keep list
            } else {
                instructions_to_keep.push(instr.clone());

                // Update liveness for the kept instruction
                if let Some(def_reg) = instr.def_reg() {
                    live_regs.remove(def_reg);
                }
                for use_reg in instr.use_regs() {
                    live_regs.insert(use_reg.clone());
                }
            }
        }

        if removed_count > 0 {
            instructions_to_keep.reverse();
            block.instructions = instructions_to_keep;
        }

        removed_count
    }

    /// Check if an instruction is dead and can be safely removed
    fn is_dead_instruction(
        &self,
        instr: &Instruction,
        live_regs: &HashSet<Register>,
        pointer: PointerWidth,
    ) -> bool {
        // Never remove terminators
        if instr.is_terminator() {
            return false;
        }

        if let Some(def_reg) = instr.def_reg() {
            // If the defined register is not live, instruction is dead
            if !live_regs.contains(def_reg) {
                // Additional check: ensure instruction has no side effects
                return self.has_no_side_effects(instr, pointer);
            }
        } else {
            // Instructions that don't define a register are usually for side effects (e.g. stores)
            // But some might be useless?
            // Assuming has_no_side_effects check covers it.
            return self.has_no_side_effects(instr, pointer);
        }

        false
    }

    /// Check if an instruction has no side effects and can be safely removed
    fn has_no_side_effects(&self, instr: &Instruction, pointer: PointerWidth) -> bool {
        if instr.may_trap_integer(pointer) {
            return false;
        }
        match instr {
            // Pure arithmetic operations - safe to remove
            Instruction::Copy { .. }
            | Instruction::IntBinary { .. }
            | Instruction::FloatBinary { .. }
            | Instruction::FloatUnary { .. }
            | Instruction::IntUnary { .. }
            | Instruction::IntRotate { .. }
            | Instruction::IntCmp { .. }
            | Instruction::IntOverflow { .. }
            | Instruction::FloatCmp { .. }
            | Instruction::Select { .. }
            | Instruction::VectorOp { .. }
            | Instruction::IntConvert { .. }
            | Instruction::IntToFloat { .. }
            | Instruction::FloatToInt { .. }
            | Instruction::FloatConvert { .. }
            | Instruction::Bitcast { .. }
            | Instruction::StackAddr { .. }
            | Instruction::SymbolAddress { .. }
            | Instruction::ThreadLocalAddr { .. }
                | Instruction::Lea { .. } => true,

            // SIMD register-only ops are also pure (nightly only)
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { .. }
            | Instruction::SimdUnary { .. }
            | Instruction::SimdTernary { .. }
            | Instruction::SimdShuffle { .. }
            | Instruction::SimdExtract { .. }
            | Instruction::SimdInsert { .. } => true,

            // Instructions with side effects - never remove
            Instruction::Load { .. } // Reads memory (could fault)
            | Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::CallIndirect { .. }
            | Instruction::TailCall { .. }
            | Instruction::Ret { .. }
            | Instruction::Jmp { .. }
            | Instruction::Br { .. }
            | Instruction::Switch { .. }
            | Instruction::Unreachable
            | Instruction::SafePoint
            | Instruction::StackMap { .. }
            | Instruction::PatchPoint { .. }
            | Instruction::Comment { .. } => false,

            // Atomic operations and fences have side effects
            Instruction::AtomicLoad { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::AtomicBinary { .. }
            | Instruction::AtomicCompareExchange { .. }
            | Instruction::Fence { .. } => false,

            Instruction::InlineAsm { .. } => false,

            // Variadic argument access writes through its va_list
            Instruction::VaStart { .. }
            | Instruction::VaArg { .. }
            | Instruction::VaEnd { .. }
            | Instruction::VaCopy { .. } => false,

            // SIMD memory operations have side effects (nightly only)
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad { .. } | Instruction::SimdStore { .. } => false,
        }
    }
}

/// Ends each block at the first call to a `@noreturn` function, so the code after it is
/// unreachable. Returns the number of blocks changed.
pub fn truncate_after_noreturn_calls(module: &mut Module) -> usize {
    let noreturn: HashSet<String> = module
        .functions
        .iter()
        .filter(|(_, func)| func.sig.noreturn)
        .map(|(name, _)| name.clone())
        .collect();
    if noreturn.is_empty() {
        return 0;
    }

    let mut changed = 0;
    for func in module.functions.values_mut() {
        for block in &mut func.blocks {
            let Some(index) = block.instructions.iter().position(
                |inst| matches!(&inst.instr, Instruction::Call { name, .. } if noreturn.contains(name)),
            ) else {
                continue;
            };
            let already_ends = block.instructions.len() == index + 2
                && matches!(
                    &block.instructions.last().map(|i| &i.instr),
                    Some(Instruction::Unreachable)
                );
            if already_ends {
                continue;
            }
            block.instructions.truncate(index + 1);
            block.instructions.push(Instruction::Unreachable.into());
            changed += 1;
        }
    }
    changed
}

/// Deletes calls to `@pure` or `@const` functions whose result is never read. Repeats
/// until a round removes nothing, since removing a call can leave its argument unread.
/// Returns the number of calls removed.
pub fn remove_dead_pure_calls(module: &mut Module) -> usize {
    let pure: HashSet<String> = module
        .functions
        .iter()
        .filter(|(_, func)| func.sig.pure)
        .map(|(name, _)| name.clone())
        .collect();
    if pure.is_empty() {
        return 0;
    }

    let mut total = 0;
    loop {
        let mut round = 0;
        for func in module.functions.values_mut() {
            let mut uses: HashMap<Register, usize> = HashMap::new();
            for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
                for reg in inst.use_regs() {
                    *uses.entry(reg.clone()).or_insert(0) += 1;
                }
            }
            for block in &mut func.blocks {
                let before = block.instructions.len();
                block.instructions.retain(|inst| match &inst.instr {
                    Instruction::Call { name, ret, .. } if pure.contains(name) => match ret {
                        Some(reg) => uses.get(reg).is_some_and(|&n| n > 0),
                        None => false,
                    },
                    _ => true,
                });
                round += before - block.instructions.len();
            }
        }
        if round == 0 {
            break;
        }
        total += round;
    }
    total
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        AddressMode, FunctionBuilder, Immediate, IntBinOp, MemoryAttrs, MirType, Operand,
        ScalarType, VirtualReg, transform::test_utils::get_block,
    };
    use lamina_mir::Located;

    fn defines(block: &Block, reg: VirtualReg) -> bool {
        let reg: Register = reg.into();
        block.instructions.iter().any(|i| i.def_reg() == Some(&reg))
    }

    fn ends_with_ret(block: &Block) -> bool {
        matches!(
            &block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Ret { .. })
        )
    }

    fn has_int_binary(block: &Block) -> bool {
        block
            .instructions
            .iter()
            .any(|i| matches!(&i.instr, Instruction::IntBinary { .. }))
    }

    #[test]
    fn test_dead_code_elimination_basic() {
        // Create a function with dead instructions
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Dead instruction: v1 = v0 + 42 (v1 never used)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(42)),
            })
            // Live instruction: v2 = v0 + 10 (v2 used in ret)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination;

        let changed = dce.apply(&mut func).expect("DCE should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");
        assert!(!defines(entry, VirtualReg::gpr(1)));
        assert!(defines(entry, VirtualReg::gpr(2)));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_empty_function() {
        // Test with empty function - should not panic
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No changes expected
    }

    #[test]
    fn test_dce_single_block_all_live() {
        // All instructions are live - nothing should be removed
        let mut func = FunctionBuilder::new("all_live")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
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

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");
        assert!(!changed);
        assert!(defines(&func.blocks[0], VirtualReg::gpr(1)));
        assert!(ends_with_ret(&func.blocks[0]));
    }

    #[test]
    fn test_dce_preserves_terminators() {
        // Terminators must never be removed even if they define no live registers
        let mut func = FunctionBuilder::new("terminators")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());

        // Jmp and Ret should still exist
        let entry = get_block(&func, "entry");
        assert!(matches!(
            &entry.instructions.last().map(|i| &i.instr),
            Some(Instruction::Jmp { .. })
        ));
    }

    #[test]
    fn test_dce_preserves_side_effects() {
        // Store and Call have side effects - must not be removed
        let mut func = FunctionBuilder::new("side_effects")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Store {
                ty: MirType::Scalar(ScalarType::I64),
                src: Operand::Immediate(Immediate::I64(42)),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Call {
                name: "lamina.writebyte".to_string(),
                args: vec![Operand::Immediate(Immediate::I64(1))],
                ret: None,
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");

        assert!(!changed);
        let entry = &func.blocks[0];
        assert!(matches!(
            &entry.instructions[0].instr,
            Instruction::Store { .. }
        ));
        assert!(matches!(
            &entry.instructions[1].instr,
            Instruction::Call { .. }
        ));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_chain_of_dead_instructions() {
        // Chain of dead instructions: v1 = f(v0), v2 = f(v1), v3 = f(v2)
        // None used in return
        let mut func = FunctionBuilder::new("dead_chain")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(3)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(99))),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");
        assert!(changed);
        let entry = &func.blocks[0];
        assert!(!has_int_binary(entry));
        assert!(!defines(entry, VirtualReg::gpr(1)));
        assert!(!defines(entry, VirtualReg::gpr(2)));
        assert!(!defines(entry, VirtualReg::gpr(3)));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_multi_block_liveness() {
        // Value defined in entry, used in exit block
        let mut func = FunctionBuilder::new("multi_block")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");

        // v1 is used in exit block, so entry's add should remain
        assert!(!changed);
        let entry = get_block(&func, "entry");
        assert!(defines(entry, VirtualReg::gpr(1)));
        assert!(matches!(
            &entry.instructions.last().map(|i| &i.instr),
            Some(Instruction::Jmp { .. })
        ));
    }

    #[test]
    fn test_dce_loop_back_edge_liveness() {
        // Test liveness across loop back-edge
        // v0 defined in entry, used in loop, loop branches back
        let mut func = FunctionBuilder::new("loop_liveness")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());

        // Loop body should be preserved (v0 is used)
        let loop_block = get_block(&func, "loop");
        assert!(defines(loop_block, VirtualReg::gpr(0)));
        assert!(matches!(
            &loop_block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Br { .. })
        ));
    }

    #[test]
    fn test_dce_does_not_infinite_loop() {
        // Stress test: large function should not cause infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Add 500 dead instructions
        for i in 0..500 {
            func.blocks[0].instructions.insert(
                0,
                Located::from(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i + 1).into(),
                    lhs: Operand::Immediate(Immediate::I64(i as i64)),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                }),
            );
        }
        func.blocks[0]
            .instructions
            .push(Located::from(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            }));

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());
        assert!(!has_int_binary(&func.blocks[0]));
        assert!(ends_with_ret(&func.blocks[0]));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod annotation_tests {
    use super::*;
    use crate::{
        mir::{codegen::from_ir, transform::TransformPipeline},
        mir_codegen::x86_64::generate_mir_x86_64,
        parser::parse_module,
    };
    use lamina_platform::TargetOperatingSystem;

    fn lower(source: &str) -> Module {
        let ir = parse_module(source).expect("parse");
        from_ir(&ir, "annotated").expect("lower")
    }

    fn calls_in(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(&i.instr, Instruction::Call { .. }))
            .count()
    }

    const NORETURN_CALLER: &str = "fn @stop() -> void {\nentry:\n ret.void\n}\nfn @main() -> i64 {\nentry:\n call @stop()\n %x = add.i64 1, 2\n ret.i64 %x\n}";

    #[test]
    fn code_after_noreturn_call_becomes_unreachable() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        assert_eq!(truncate_after_noreturn_calls(&mut module), 1);
        let block = &module.get_function("main").expect("main").blocks[0];
        assert_eq!(block.instructions.len(), 2, "{:?}", block.instructions);
        assert!(matches!(
            &block.instructions[0].instr,
            Instruction::Call { .. }
        ));
        assert!(matches!(
            &block.instructions[1].instr,
            Instruction::Unreachable
        ));
    }

    #[test]
    fn calls_to_returning_functions_keep_following_code() {
        let mut module = lower(NORETURN_CALLER);
        assert_eq!(truncate_after_noreturn_calls(&mut module), 0);
        let block = &module.get_function("main").expect("main").blocks[0];
        assert!(
            !block
                .instructions
                .iter()
                .any(|i| matches!(&i.instr, Instruction::Unreachable))
        );
        assert!(matches!(
            &block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Ret { .. })
        ));
    }

    #[test]
    fn noreturn_truncation_survives_the_standard_pipeline() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        let main = module.get_function("main").expect("main");
        let instructions: Vec<&Instruction> = main
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .map(|i| &i.instr)
            .collect();
        assert!(
            instructions
                .iter()
                .any(|i| matches!(i, Instruction::Call { .. }))
        );
        assert!(matches!(
            instructions.last(),
            Some(Instruction::Unreachable)
        ));
        assert!(
            !instructions
                .iter()
                .any(|i| matches!(i, Instruction::IntBinary { .. }))
        );
    }

    #[test]
    fn noreturn_call_emits_trap_after_the_call_on_x86_64() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        let mut output = Vec::new();
        generate_mir_x86_64(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("emit x86_64");
        let text = String::from_utf8(output).expect("utf8 assembly");
        let call = text.find("call stop").expect(&text);
        assert!(text[call..].contains("ud2"), "{text}");
    }

    #[test]
    fn noreturn_call_compiles_on_every_native_backend() {
        use crate::error::LaminaError;
        use crate::mir::Module;
        use crate::mir_codegen::{
            arm::aarch64::generate_mir_aarch64, powerpc::powerpc64::generate_mir_ppc64,
            riscv::generate_mir_riscv,
        };
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        type Emitter = fn(&Module, &mut Vec<u8>, TargetOperatingSystem) -> Result<(), LaminaError>;
        let emitters: [(&str, Emitter); 4] = [
            ("x86_64", generate_mir_x86_64::<Vec<u8>>),
            ("aarch64", |module, writer, os| {
                generate_mir_aarch64(module, writer, os, PointerWidth::Bits64)
            }),
            ("riscv", generate_mir_riscv::<Vec<u8>>),
            ("ppc64le", generate_mir_ppc64::<Vec<u8>>),
        ];
        for (name, emit) in emitters {
            let mut output = Vec::new();
            emit(&module, &mut output, TargetOperatingSystem::Linux)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    fn pure_source(annotation: &str, used: bool) -> String {
        let tail = if used {
            "%v = call @k()\n ret.i64 %v"
        } else {
            "call @k()\n ret.i64 1"
        };
        format!(
            "{annotation}fn @k() -> i64 {{\nentry:\n ret.i64 4\n}}\nfn @main() -> i64 {{\nentry:\n {tail}\n}}"
        )
    }

    #[test]
    fn unused_pure_call_is_removed() {
        let mut module = lower(&pure_source("@pure ", false));
        assert_eq!(remove_dead_pure_calls(&mut module), 1);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 0);
    }

    #[test]
    fn unused_call_to_unannotated_function_is_kept() {
        let mut module = lower(&pure_source("", false));
        assert_eq!(remove_dead_pure_calls(&mut module), 0);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 1);
    }

    #[test]
    fn used_pure_call_is_kept() {
        let mut module = lower(&pure_source("@pure ", true));
        assert_eq!(remove_dead_pure_calls(&mut module), 0);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 1);
    }
}
