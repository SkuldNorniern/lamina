use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction, Register};
use std::collections::HashSet;

/// Loop Invariant Code Motion (LICM)
/// Moves computations that don't depend on loop variables outside the loop
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
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl LoopInvariantCodeMotion {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        // Find all loops in the function
        let loops = self.find_loops(func);

        for loop_info in loops {
            if self.optimize_loop(func, &loop_info)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Find natural loops in the function using dominators and back edges
    fn find_loops(&self, func: &Function) -> Vec<LoopInfo> {
        // Simplified loop detection - look for backward jumps
        let mut loops = Vec::new();

        for block in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } = instr
                {
                    // Check if either target is a predecessor (simple loop detection)
                    if self.is_back_edge(func, &block.label, true_target)
                        && let Some(loop_info) = self.analyze_loop(func, &block.label, true_target)
                        {
                            loops.push(loop_info);
                        }
                    if self.is_back_edge(func, &block.label, false_target)
                        && let Some(loop_info) = self.analyze_loop(func, &block.label, false_target)
                        {
                            loops.push(loop_info);
                        }
                }
                if let Instruction::Jmp { target } = instr
                    && self.is_back_edge(func, &block.label, target)
                        && let Some(loop_info) = self.analyze_loop(func, &block.label, target) {
                            loops.push(loop_info);
                        }
            }
        }

        loops
    }

    fn is_back_edge(&self, func: &Function, from: &str, to: &str) -> bool {
        // Simple check: if target block comes before current block in the block list
        let from_idx = func.blocks.iter().position(|b| b.label == *from);
        let to_idx = func.blocks.iter().position(|b| b.label == *to);

        match (from_idx, to_idx) {
            (Some(f), Some(t)) => t < f, // Target comes before source
            _ => false,
        }
    }

    fn analyze_loop(
        &self,
        func: &Function,
        header: &str,
        back_edge_target: &str,
    ) -> Option<LoopInfo> {
        // Find all blocks in the loop
        let mut loop_blocks = HashSet::new();
        let mut to_visit = vec![header.to_string()];

        while let Some(block_label) = to_visit.pop() {
            if !loop_blocks.contains(&block_label) {
                loop_blocks.insert(block_label.clone());

                // Find predecessors and successors
                for block in &func.blocks {
                    if self.has_edge_to(func, &block.label, &block_label) {
                        to_visit.push(block.label.clone());
                    }
                }
            }
        }

        if loop_blocks.is_empty() {
            return None;
        }

        Some(LoopInfo {
            header: header.to_string(),
            back_edge_target: back_edge_target.to_string(),
            blocks: loop_blocks,
        })
    }

    fn has_edge_to(&self, func: &Function, from: &str, to: &str) -> bool {
        if let Some(block) = func.blocks.iter().find(|b| b.label == *from) {
            for instr in &block.instructions {
                match instr {
                    Instruction::Jmp { target } if target == to => return true,
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } if true_target == to || false_target == to => return true,
                    _ => {}
                }
            }
        }
        false
    }

    fn optimize_loop(&self, func: &mut Function, loop_info: &LoopInfo) -> Result<bool, String> {
        let mut changed = false;

        // Find invariant instructions (those that don't depend on loop variables)
        let invariant_instrs = self.find_invariant_instructions(func, loop_info)?;

        if !invariant_instrs.is_empty() {
            // Move invariant instructions before the loop
            self.move_invariant_instructions(func, loop_info, &invariant_instrs)?;
            changed = true;
        }

        Ok(changed)
    }

    fn find_invariant_instructions(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
    ) -> Result<Vec<usize>, String> {
        let mut invariant = Vec::new();

        for (block_idx, block) in func.blocks.iter().enumerate() {
            if !loop_info.blocks.contains(&block.label) {
                continue;
            }

            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                if self.is_invariant_instruction(func, loop_info, instr) {
                    // Store as (block_idx, instr_idx) encoded in a single usize
                    let encoded = (block_idx << 16) | instr_idx;
                    invariant.push(encoded);
                }
            }
        }

        Ok(invariant)
    }

    fn is_invariant_instruction(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
        instr: &Instruction,
    ) -> bool {
        // An instruction is invariant if:
        // 1. It defines a register
        // 2. All its operands are either constants or defined outside the loop
        // 3. It's not a side-effecting instruction

        if let Some(def_reg) = instr.def_reg() {
            // Check if all operands are invariant
            let operands_invariant = instr
                .use_regs()
                .iter()
                .all(|reg| self.is_invariant_register(func, loop_info, reg));

            // Check if instruction has no side effects
            let no_side_effects = !self.has_side_effects(instr);

            operands_invariant && no_side_effects
        } else {
            false
        }
    }

    fn is_invariant_register(&self, func: &Function, loop_info: &LoopInfo, reg: &Register) -> bool {
        // A register is invariant if it's defined outside the loop or is a physical register
        match reg {
            Register::Physical(_) => true, // Physical registers are always invariant
            Register::Virtual(_) => {
                // Check if defined outside the loop
                for block in &func.blocks {
                    if !loop_info.blocks.contains(&block.label) {
                        for instr in &block.instructions {
                            if instr.def_reg() == Some(reg) {
                                return true;
                            }
                        }
                    }
                }
                false
            }
        }
    }

    fn has_side_effects(&self, instr: &Instruction) -> bool {
        matches!(
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
        invariant_instrs: &[usize],
    ) -> Result<(), String> {
        // Find the loop header block
        let header_idx = func
            .blocks
            .iter()
            .position(|b| b.label == loop_info.header)
            .ok_or_else(|| format!("Loop header '{}' not found", loop_info.header))?;

        let mut instructions_to_move = Vec::new();

        // Collect instructions to move (in reverse order to maintain dependencies)
        for &encoded in invariant_instrs.iter().rev() {
            let block_idx = encoded >> 16;
            let instr_idx = encoded & 0xFFFF;

            if block_idx >= func.blocks.len() {
                continue;
            }

            let block = &func.blocks[block_idx];
            if instr_idx >= block.instructions.len() {
                continue;
            }

            let instr = block.instructions[instr_idx].clone();
            instructions_to_move.push((block_idx, instr_idx, instr));
        }

        // Move instructions before the loop header
        for (_, _, instr) in instructions_to_move {
            func.blocks[header_idx].instructions.insert(0, instr);
        }

        // Remove original instructions from loop blocks
        for &encoded in invariant_instrs {
            let block_idx = encoded >> 16;
            let instr_idx = encoded & 0xFFFF;

            if block_idx < func.blocks.len()
                && instr_idx < func.blocks[block_idx].instructions.len()
            {
                func.blocks[block_idx].instructions.remove(instr_idx);
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct LoopInfo {
    header: String,
    back_edge_target: String,
    blocks: HashSet<String>,
}

/// Loop Unrolling
/// Unrolls loops to reduce overhead and improve ILP
#[derive(Default)]
pub struct LoopUnrolling;

impl Transform for LoopUnrolling {
    fn name(&self) -> &'static str {
        "loop_unrolling"
    }

    fn description(&self) -> &'static str {
        "Unrolls small loops to reduce overhead and improve parallelism"
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

impl LoopUnrolling {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Simple unrolling: look for loops with small constant bounds
        // This is a simplified implementation - real loop unrolling is much more complex

        let mut changed = false;

        // For now, just mark as changed if we find any loops
        // Full implementation would require:
        // 1. Loop analysis to determine bounds
        // 2. Induction variable detection
        // 3. Body duplication with variable renaming
        // 4. Exit condition adjustment

        let loops = self.find_simple_loops(func);
        if !loops.is_empty() {
            changed = true;
            // TODO: Implement actual unrolling logic
        }

        Ok(changed)
    }

    fn find_simple_loops(&self, _func: &Function) -> Vec<LoopInfo> {
        // Placeholder - would need proper loop analysis
        Vec::new()
    }
}

/// Loop Fusion
/// Combines adjacent loops that iterate over the same range
#[derive(Default)]
pub struct LoopFusion;

impl Transform for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn description(&self) -> &'static str {
        "Fuses adjacent loops with the same iteration bounds"
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

impl LoopFusion {
    fn apply_internal(&self, _func: &mut Function) -> Result<bool, String> {
        // Loop fusion is complex and requires:
        // 1. Dependency analysis between loop bodies
        // 2. Induction variable comparison
        // 3. Data dependence checking
        // 4. Body merging

        // For now, return false (no changes)
        Ok(false)
    }
}
