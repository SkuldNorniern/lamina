//! Loop optimization transforms for MIR.

use crate::mir::{
    Block, Function, Instruction, Register,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, calculate_dominators,
    },
};
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, hash_map::Entry},
};

/// Loop invariant code motion that moves loop-invariant computations outside loops.
#[derive(Default)]
pub struct LoopInvariantCodeMotion;

impl Transform for LoopInvariantCodeMotion {
    fn name(&self) -> &'static str {
        "loop_invariant_code_motion"
    }

    fn description(&self) -> &'static str {
        "Moves loop-invariant computations outside loops"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests_licm {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn loop_discovery_follows_block_order() {
        let mut function = FunctionBuilder::new("nested")
            .block("entry")
            .instr(Instruction::Jmp {
                target: "outer".into(),
            })
            .block("outer")
            .instr(Instruction::Jmp {
                target: "inner".into(),
            })
            .block("inner")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "inner".into(),
                false_target: "latch".into(),
            })
            .block("latch")
            .instr(Instruction::Jmp {
                target: "outer".into(),
            })
            .build();
        for _ in 0..32 {
            let headers: Vec<_> = LoopInvariantCodeMotion
                .find_loops(&function)
                .into_iter()
                .map(|info| info.header)
                .collect();
            assert_eq!(headers, ["outer", "inner"]);
        }
        function.blocks.swap(1, 2);
        let headers: Vec<_> = LoopInvariantCodeMotion
            .find_loops(&function)
            .into_iter()
            .map(|info| info.header)
            .collect();
        assert_eq!(headers, ["inner", "outer"]);
    }

    #[test]
    fn test_licm_basic() {
        // Create a loop with invariant calculation
        // entry:
        //   jmp loop
        // loop:
        //   v0 = add v0, 1  (induction)
        //   v1 = add 5, 10  (invariant)
        //   v2 = lt v0, v1
        //   br v2, loop, exit
        // exit:
        //   ret

        let mut func = FunctionBuilder::new("test_licm")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            }) // init v0
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
            // Invariant: 5 + 10
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(5)),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(1).into()),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("LICM failed");

        assert!(changed);

        // Loop block should no longer have the 5+10 addition
        let loop_block = func.get_block("loop").expect("loop block exists");
        let has_invariant = loop_block.instructions.iter().any(|i| {
            if let Instruction::IntBinary { lhs, rhs, .. } = i {
                matches!(lhs, Operand::Immediate(Immediate::I64(5)))
                    && matches!(rhs, Operand::Immediate(Immediate::I64(10)))
            } else {
                false
            }
        });
        assert!(
            !has_invariant,
            "Invariant instruction should be moved out of loop"
        );

        let pre_header = func.get_block("loop_pre").expect("pre-header created");
        let has_moved = pre_header.instructions.iter().any(|i| {
            if let Instruction::IntBinary { lhs, rhs, .. } = i {
                matches!(lhs, Operand::Immediate(Immediate::I64(5)))
                    && matches!(rhs, Operand::Immediate(Immediate::I64(10)))
            } else {
                false
            }
        });
        assert!(has_moved, "Invariant instruction should be in pre-header");
    }
}

impl LoopInvariantCodeMotion {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        let max_loops = 10;
        let max_iterations_per_loop = 5;
        let headers: Vec<String> = self
            .find_loops(func)
            .into_iter()
            .take(max_loops)
            .map(|loop_info| loop_info.header)
            .collect();

        for header in headers {
            let mut iterations = 0;
            while iterations < max_iterations_per_loop {
                let Some(loop_info) = self
                    .find_loops(func)
                    .into_iter()
                    .find(|candidate| candidate.header == header)
                else {
                    break;
                };
                if loop_info.blocks.len() > 50 || !self.optimize_loop(func, &loop_info)? {
                    break;
                }
                changed = true;
                iterations += 1;
            }
        }

        Ok(changed)
    }

    pub(super) fn find_loops(&self, func: &Function) -> Vec<LoopInfo> {
        let mut loops = Vec::new();

        let dominators = calculate_dominators(func);

        for block in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } = instr
                {
                    // Check for back edges: target dominates source
                    if self.is_back_edge(&dominators, &block.label, true_target)
                        && let Some(loop_info) =
                            self.analyze_loop(func, true_target, &block.label, &dominators)
                    {
                        loops.push(loop_info);
                    }
                    if self.is_back_edge(&dominators, &block.label, false_target)
                        && let Some(loop_info) =
                            self.analyze_loop(func, false_target, &block.label, &dominators)
                    {
                        loops.push(loop_info);
                    }
                }
                if let Instruction::Jmp { target } = instr
                    && self.is_back_edge(&dominators, &block.label, target)
                    && let Some(loop_info) =
                        self.analyze_loop(func, target, &block.label, &dominators)
                {
                    loops.push(loop_info);
                }
            }
        }

        // Merge loops with the same header to handle multiple backedges
        let mut merged_loops: HashMap<String, HashSet<String>> = HashMap::new();

        for loop_info in loops {
            match merged_loops.entry(loop_info.header) {
                Entry::Vacant(e) => {
                    e.insert(loop_info.blocks);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().extend(loop_info.blocks);
                }
            }
        }

        func.blocks
            .iter()
            .filter_map(|block| {
                merged_loops.remove(&block.label).map(|blocks| LoopInfo {
                    header: block.label.clone(),
                    blocks,
                })
            })
            .collect()
    }

    fn is_back_edge(
        &self,
        dominators: &HashMap<String, HashSet<String>>,
        from: &str,
        to: &str,
    ) -> bool {
        // A back edge exists if the target (header) dominates the source
        if let Some(doms) = dominators.get(from) {
            doms.contains(to)
        } else {
            false
        }
    }

    fn analyze_loop(
        &self,
        func: &Function,
        header: &str,
        back_edge_source: &str,
        _dominators: &HashMap<String, HashSet<String>>,
    ) -> Option<LoopInfo> {
        // Natural loop identification:
        // A natural loop consists of the header, the back edge source,
        // and all nodes that can reach the back edge without going through the header.
        // (Optimized: we can traverse backwards from back_edge_source stopping at header)
        let mut loop_blocks = HashSet::new();
        let mut to_visit = Vec::new();
        let mut visited = HashSet::new();
        const MAX_LOOP_ANALYSIS_ITERATIONS: usize = 1000;
        let mut iterations = 0;

        // First, collect all blocks that can reach the back edge source
        // Initialize queue with back edge source
        if header != back_edge_source {
            to_visit.push(back_edge_source.to_string());
        }
        loop_blocks.insert(header.to_string());
        loop_blocks.insert(back_edge_source.to_string());

        while let Some(block_label) = to_visit.pop() {
            if iterations >= MAX_LOOP_ANALYSIS_ITERATIONS {
                return None;
            }
            iterations += 1;

            if visited.contains(&block_label) {
                continue;
            }
            visited.insert(block_label.clone());
            loop_blocks.insert(block_label.clone());

            // Add predecessors that are not the header
            for block in &func.blocks {
                if self.has_edge_to(func, &block.label, &block_label) && block.label != header {
                    to_visit.push(block.label.clone());
                }
            }
        }

        // Add the header if not already included
        loop_blocks.insert(header.to_string());

        if !loop_blocks.contains(header) || !loop_blocks.contains(back_edge_source) {
            return None;
        }

        if loop_blocks.len() > 50 {
            return None;
        }

        Some(LoopInfo {
            header: header.to_string(),
            blocks: loop_blocks,
        })
    }

    fn has_edge_to(&self, func: &Function, from: &str, to: &str) -> bool {
        func.get_block(from)
            .is_some_and(|b| b.successors().contains(&to))
    }

    fn optimize_loop(
        &self,
        func: &mut Function,
        loop_info: &LoopInfo,
    ) -> Result<bool, TransformError> {
        let mut changed = false;

        let invariant_instrs = self.find_invariant_instructions(func, loop_info)?;

        if !invariant_instrs.is_empty() {
            self.move_invariant_instructions(func, loop_info, &invariant_instrs)?;
            changed = true;
        }

        Ok(changed)
    }

    fn find_invariant_instructions(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
    ) -> Result<Vec<(usize, usize)>, TransformError> {
        let mut invariant = Vec::new();
        let defs_in_loop = self.collect_defs_in_loop(func, loop_info);

        // Count how many times each register is defined inside the loop.
        // If a register is defined more than once (e.g. accumulator %sum, induction
        // var %j), it cannot be safely hoisted — one of those assignments is a
        // loop-carried update that must execute every iteration.
        let mut def_counts: HashMap<Register, usize> = HashMap::new();
        for block in &func.blocks {
            if !loop_info.blocks.contains(&block.label) {
                continue;
            }
            for instr in &block.instructions {
                if let Some(def) = instr.def_reg() {
                    *def_counts.entry(def.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut total_defs: HashMap<Register, usize> = HashMap::new();
        let mut use_sites: HashMap<Register, Vec<(usize, usize)>> = HashMap::new();
        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                if let Some(def) = instr.def_reg() {
                    *total_defs.entry(def.clone()).or_insert(0) += 1;
                }
                for reg in instr.use_regs() {
                    use_sites
                        .entry(reg.clone())
                        .or_default()
                        .push((block_idx, instr_idx));
                }
            }
        }

        let max_invariant_instructions = 20;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            if !loop_info.blocks.contains(&block.label) {
                continue;
            }

            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                // Skip if this register is defined more than once in the loop.
                if let Some(def) = instr.def_reg()
                    && def_counts.get(def).copied().unwrap_or(0) > 1
                {
                    continue;
                }
                if let Some(def) = instr.def_reg() {
                    let single_def = total_defs.get(def) == Some(&1);
                    let reads_after_def = use_sites.get(def).is_none_or(|sites| {
                        sites.iter().all(|&(b, i)| b == block_idx && i > instr_idx)
                    });
                    if !single_def || !reads_after_def {
                        continue;
                    }
                }
                if self.is_invariant_instruction(func, loop_info, &defs_in_loop, instr) {
                    invariant.push((block_idx, instr_idx));

                    if invariant.len() >= max_invariant_instructions {
                        break;
                    }
                }
            }

            if invariant.len() >= max_invariant_instructions {
                break;
            }
        }

        Ok(invariant)
    }

    fn is_invariant_instruction(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
        defs_in_loop: &HashSet<Register>,
        instr: &Instruction,
    ) -> bool {
        // An instruction is invariant if:
        // 1. It defines a register
        // 2. All its operands are either constants or defined outside the loop
        // 3. It's not a side-effecting instruction

        if let Some(_def_reg) = instr.def_reg() {
            // Check if all operands are invariant
            let operands_invariant = instr
                .use_regs()
                .iter()
                .all(|reg| self.is_invariant_register(func, loop_info, defs_in_loop, reg));

            // Check if instruction has no side effects
            let no_side_effects = !self.has_side_effects(instr);

            operands_invariant && no_side_effects
        } else {
            false
        }
    }

    fn is_invariant_register(
        &self,
        _func: &Function,
        _loop_info: &LoopInfo,
        defs_in_loop: &HashSet<Register>,
        reg: &Register,
    ) -> bool {
        match reg {
            Register::Physical(_) => false,
            Register::Virtual(_) => !defs_in_loop.contains(reg),
        }
    }

    fn has_side_effects(&self, instr: &Instruction) -> bool {
        instr.may_trap_integer()
            || matches!(
                instr,
                Instruction::Load { .. }
                    | Instruction::Store { .. }
                    | Instruction::Call { .. }
                    | Instruction::Ret { .. }
            )
    }

    fn move_invariant_instructions(
        &self,
        func: &mut Function,
        loop_info: &LoopInfo,
        invariant_instrs: &[(usize, usize)],
    ) -> Result<(), TransformError> {
        if invariant_instrs.is_empty() {
            return Ok(());
        }

        let header_idx = func
            .blocks
            .iter()
            .position(|b| b.label == loop_info.header)
            .ok_or(TransformError::InvalidState("loop header not found"))?;

        // Generate unique pre-header label to avoid conflicts
        let mut pre_header_label = format!("{}_pre", loop_info.header);
        let mut counter = 0;
        while func.blocks.iter().any(|b| b.label == pre_header_label) {
            counter += 1;
            pre_header_label = format!("{}_pre_{}", loop_info.header, counter);
        }

        let mut pre_header_block = Block {
            label: pre_header_label.clone(),
            instructions: Vec::new(),
        };

        let mut sorted_invariant = invariant_instrs.to_vec();
        sorted_invariant.sort_by_key(|t| Reverse(t.1));
        sorted_invariant.sort_by_key(|t| Reverse(t.0));

        sorted_invariant.dedup();
        let mut instructions_to_move = Vec::new();
        for &(block_idx, instr_idx) in &sorted_invariant {
            if block_idx >= func.blocks.len() {
                continue;
            }
            let block = &func.blocks[block_idx];
            if instr_idx >= block.instructions.len() {
                continue;
            }
            let instr = block.instructions[instr_idx].clone();
            instructions_to_move.push(instr);
        }

        // Move instructions to the pre-header block
        for instr in instructions_to_move {
            pre_header_block.instructions.push(instr);
        }

        // Add jump to the original header
        pre_header_block.instructions.push(Instruction::Jmp {
            target: loop_info.header.clone(),
        });

        // Insert pre-header before the original header
        func.blocks.insert(header_idx, pre_header_block);

        // Remove original instructions from loop blocks
        // Since we sorted by (block_idx, instr_idx) descending, we can remove them safely
        for &(block_idx, instr_idx) in &sorted_invariant {
            // Adjust for the inserted pre-header block
            let adjusted_block_idx = if block_idx >= header_idx {
                block_idx + 1
            } else {
                block_idx
            };
            if adjusted_block_idx < func.blocks.len()
                && instr_idx < func.blocks[adjusted_block_idx].instructions.len()
            {
                func.blocks[adjusted_block_idx]
                    .instructions
                    .remove(instr_idx);
            }
        }

        // Redirect non-loop incoming edges to go through the pre-header.
        // Back-edges from inside the loop are skipped.
        for (block_idx, block) in func.blocks.iter_mut().enumerate() {
            if block_idx == header_idx || loop_info.blocks.contains(&block.label) {
                continue;
            }

            for instr in &mut block.instructions {
                match instr {
                    Instruction::Jmp { target } if *target == loop_info.header => {
                        *target = pre_header_label.clone();
                    }
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        if *true_target == loop_info.header {
                            *true_target = pre_header_label.clone();
                        }
                        if *false_target == loop_info.header {
                            *false_target = pre_header_label.clone();
                        }
                    }
                    _ => {}
                }
            }
        }

        // If the loop header was the function entry point, update it to the pre-header
        if func.entry == loop_info.header {
            func.entry = pre_header_label;
        }

        Ok(())
    }

    fn collect_defs_in_loop(&self, func: &Function, loop_info: &LoopInfo) -> HashSet<Register> {
        let mut defs = HashSet::new();
        for block in &func.blocks {
            if !loop_info.blocks.contains(&block.label) {
                continue;
            }
            for instr in &block.instructions {
                if let Some(def) = instr.def_reg() {
                    defs.insert(def.clone());
                }
            }
        }
        defs
    }
}

#[derive(Debug)]
pub(super) struct LoopInfo {
    pub(super) header: String,
    pub(super) blocks: HashSet<String>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::mir::transform::LoopUnrolling;
    use crate::mir::{
        AddressMode, FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MemoryAttrs, MirType, Operand,
        ScalarType, VirtualReg,
    };

    #[test]
    fn test_licm_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let result = licm.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No changes expected
    }

    #[test]
    fn test_licm_no_loops() {
        // Function with no loops - nothing to optimize
        let mut func = FunctionBuilder::new("no_loop")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_licm_keeps_register_read_before_its_definition() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut func = FunctionBuilder::new("read_before_def")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(add(1, imm(5), imm(0)))
            .instr(add(2, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(3).into(),
                lhs: reg(0),
                rhs: imm(3),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(3).into(),
                true_target: "body".to_string(),
                false_target: "exit".to_string(),
            })
            .block("body")
            .instr(add(2, reg(2), reg(1)))
            .instr(add(1, imm(2), imm(3)))
            .instr(add(0, reg(0), imm(1)))
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(reg(2)),
            })
            .build();

        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");

        let defines_v1_constant = |instr: &Instruction| {
            matches!(
                instr,
                Instruction::IntBinary {
                    dst: Register::Virtual(v),
                    lhs: Operand::Immediate(Immediate::I64(2)),
                    ..
                } if *v == VirtualReg::gpr(1)
            )
        };
        let hoisted = func
            .get_block("loop_pre")
            .is_some_and(|pre| pre.instructions.iter().any(defines_v1_constant));
        assert!(!hoisted);
        let body = func.get_block("body").expect("body block exists");
        assert!(body.instructions.iter().any(defines_v1_constant));
    }

    #[test]
    fn test_licm_self_loop_does_not_capture_predecessors() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut func = FunctionBuilder::new("self_loop_predecessor")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(9).into(),
                lhs: reg(0),
                rhs: imm(1),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(9).into(),
                true_target: "then".to_string(),
                false_target: "hdr".to_string(),
            })
            .block("then")
            .instr(add(5, imm(7), imm(0)))
            .instr(add(6, reg(5), imm(1)))
            .instr(Instruction::Jmp {
                target: "hdr".to_string(),
            })
            .block("hdr")
            .instr(add(1, reg(1), imm(1)))
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(2).into(),
                lhs: reg(1),
                rhs: imm(4),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "hdr".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(imm(0)),
            })
            .build();

        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");

        let then_block = func.get_block("then").expect("then block exists");
        assert!(then_block.instructions.iter().any(|i| matches!(
            i,
            Instruction::IntBinary {
                dst: Register::Virtual(v),
                ..
            } if *v == VirtualReg::gpr(5)
        )));
    }

    #[test]
    fn test_licm_nested_loops_keep_inner_hoisted_values_in_outer_scope() {
        use crate::runtime::executor::interpret_mir_function;

        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let less = |dst: u32, lhs: u32, bound: i64| Instruction::IntCmp {
            op: IntCmpOp::SLt,
            ty: MirType::Scalar(ScalarType::I1),
            dst: VirtualReg::gpr(dst).into(),
            lhs: Operand::Register(VirtualReg::gpr(lhs).into()),
            rhs: Operand::Immediate(Immediate::I64(bound)),
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut inner_body = FunctionBuilder::new("nested_hoist")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(add(1, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "outer".to_string(),
            })
            .block("inner")
            .instr(less(15, 12, 3))
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(15).into(),
                true_target: "inner_body".to_string(),
                false_target: "outer_latch".to_string(),
            })
            .block("inner_body")
            .instr(add(2, imm(1), imm(0)));
        for (dst, src) in [(3, 2), (4, 3), (6, 4), (7, 6), (9, 7), (10, 9), (11, 10)] {
            inner_body = inner_body.instr(add(dst, reg(src), imm(1)));
        }
        let mut func = inner_body
            .instr(add(1, reg(1), reg(11)))
            .instr(add(12, reg(12), imm(1)))
            .instr(Instruction::Jmp {
                target: "inner".to_string(),
            })
            .block("outer")
            .instr(less(14, 0, 2))
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(14).into(),
                true_target: "outer_body".to_string(),
                false_target: "exit".to_string(),
            })
            .block("outer_body")
            .instr(add(12, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "inner".to_string(),
            })
            .block("outer_latch")
            .instr(add(0, reg(0), imm(1)))
            .instr(Instruction::Jmp {
                target: "outer".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(reg(1)),
            })
            .build();

        let expected = interpret_mir_function(&func, &[]).expect("interpret before LICM");
        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");
        let actual = interpret_mir_function(&func, &[]).ok().flatten();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_licm_side_effect_not_moved() {
        // Store inside loop should NOT be moved out
        let mut func = FunctionBuilder::new("side_effect")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
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
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("should succeed");
        // Store has side effects - should not be moved
        assert!(!changed);
    }

    #[test]
    fn test_loop_unrolling_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let unroll = LoopUnrolling;
        let result = unroll.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_loop_unrolling_no_loops() {
        let mut func = FunctionBuilder::new("no_loop")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let unroll = LoopUnrolling;
        let changed = unroll.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_loop_unrolling_large_bound_not_unrolled() {
        // Loop with bound > 32 should not be unrolled
        let mut func = FunctionBuilder::new("large_bound")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
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
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(100)), // > 32
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let unroll = LoopUnrolling;
        let changed = unroll.apply(&mut func).expect("should succeed");
        // Loop with large bound should not be unrolled
        assert!(!changed);
        // Original loop should still exist
        assert!(func.get_block("loop").is_some());
    }

    #[test]
    fn test_licm_stress_no_infinite_loop() {
        // Create function with nested-looking structure but no actual loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Add many blocks in linear fashion
        for i in 0..50 {
            let label = format!("block{}", i);
            let next_label = if i < 49 {
                format!("block{}", i + 1)
            } else {
                "exit".to_string()
            };

            let mut block = Block::new(&label);
            block.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(i).into(),
                lhs: Operand::Immediate(Immediate::I64(i as i64)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            });
            block.push(Instruction::Jmp { target: next_label });
            func.add_block(block);
        }

        let mut exit_block = Block::new("exit");
        exit_block.push(Instruction::Ret { value: None });
        func.add_block(exit_block);

        // Update entry to jump to block0
        func.blocks[0].instructions.push(Instruction::Jmp {
            target: "block0".to_string(),
        });

        let licm = LoopInvariantCodeMotion;
        let result = licm.apply(&mut func);
        assert!(result.is_ok());
    }

    #[test]
    fn test_loop_unrolling_stress_no_infinite_loop() {
        // Many simple blocks should not cause infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        // Add 100 independent blocks
        for i in 0..100 {
            let label = format!("dead_block_{}", i);
            let mut block = Block::new(&label);
            block.push(Instruction::Ret { value: None });
            func.add_block(block);
        }

        let unroll = LoopUnrolling;
        let result = unroll.apply(&mut func);
        assert!(result.is_ok());
    }
}
