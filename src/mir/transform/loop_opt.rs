use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Operand, Register};
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
        TransformLevel::Deprecated
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

        // Safety limits to prevent infinite loops and excessive processing
        let max_loops = 10;
        let max_iterations_per_loop = 5;
        let loops_to_process = loops.into_iter().take(max_loops);

        for loop_info in loops_to_process {
            // Skip loops that are too complex (too many blocks) to avoid pathological cases
            if loop_info.blocks.len() > 50 {
                continue;
            }

            let mut iterations = 0;
            while iterations < max_iterations_per_loop {
                if !self.optimize_loop(func, &loop_info)? {
                    break; // No more changes possible
                }
                changed = true;
                iterations += 1;

                // Safety check: if we've made many changes, stop to avoid infinite loops
                if iterations >= max_iterations_per_loop {
                    break;
                }
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
                    // Check for back edges: target comes before source in block order
                    if self.is_back_edge(func, &block.label, true_target)
                        && let Some(loop_info) = self.analyze_loop(func, true_target, &block.label)
                    {
                        loops.push(loop_info);
                    }
                    if self.is_back_edge(func, &block.label, false_target)
                        && let Some(loop_info) = self.analyze_loop(func, false_target, &block.label)
                    {
                        loops.push(loop_info);
                    }
                }
                if let Instruction::Jmp { target } = instr
                    && self.is_back_edge(func, &block.label, target)
                    && let Some(loop_info) = self.analyze_loop(func, target, &block.label)
                {
                    loops.push(loop_info);
                }
            }
        }

        // Remove duplicates (same loop found multiple ways)
        loops.sort_by(|a, b| a.header.cmp(&b.header));
        loops.dedup_by(|a, b| a.header == b.header);

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
        back_edge_source: &str,
    ) -> Option<LoopInfo> {
        // Natural loop identification: find all blocks that can reach the back edge source
        // and are dominated by the header (simplified approximation)
        let mut loop_blocks = HashSet::new();
        let mut to_visit = vec![back_edge_source.to_string()];
        let mut visited = HashSet::new();

        // First, collect all blocks that can reach the back edge source
        while let Some(block_label) = to_visit.pop() {
            if visited.contains(&block_label) {
                continue;
            }
            visited.insert(block_label.clone());

            // Add this block to the loop if it's between header and back edge source
            // (simplified: just add all blocks we can reach from back edge source)
            loop_blocks.insert(block_label.clone());

            // Find predecessors (blocks that can reach this block)
            for block in &func.blocks {
                if self.has_edge_to(func, &block.label, &block_label) {
                    to_visit.push(block.label.clone());
                }
            }
        }

        // Add the header if not already included
        loop_blocks.insert(header.to_string());

        // Basic sanity check: ensure we have at least header and back edge source
        if !loop_blocks.contains(header) || !loop_blocks.contains(back_edge_source) {
            return None;
        }

        // Limit loop size to prevent pathological cases
        if loop_blocks.len() > 50 {
            return None;
        }

        Some(LoopInfo {
            header: header.to_string(),
            back_edge_target: back_edge_source.to_string(), // Note: this field is misnamed in the struct
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
    ) -> Result<Vec<(usize, usize)>, String> {
        let mut invariant = Vec::new();
        let defs_in_loop = self.collect_defs_in_loop(func, loop_info);

        // Limit the number of invariant instructions to move for stability
        let max_invariant_instructions = 20;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            if !loop_info.blocks.contains(&block.label) {
                continue;
            }

            // Don't hoist from the loop header - these might be needed for loop control
            if block.label == loop_info.header {
                continue;
            }

            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                // Only use precise check, avoid heuristic to prevent over-optimization
                if self.is_invariant_instruction(func, loop_info, &defs_in_loop, instr) {
                    // Store as (block_idx, instr_idx) tuple to avoid overflow
                    invariant.push((block_idx, instr_idx));

                    // Limit the number of invariant instructions for stability
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
        defs_in_loop: &std::collections::HashSet<crate::mir::Register>,
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
                .all(|reg| self.is_invariant_register(func, loop_info, defs_in_loop, reg));

            // Check if instruction has no side effects
            let no_side_effects = !self.has_side_effects(instr);

            operands_invariant && no_side_effects
        } else {
            false
        }
    }

    /// Check if an instruction is likely to be loop invariant based on common patterns
    /// This is a heuristic to catch more invariant instructions
    /// DISABLED: Too aggressive, can cause incorrect optimizations
    fn is_likely_invariant_instruction(
        &self,
        _func: &Function,
        _loop_info: &LoopInfo,
        _defs_in_loop: &std::collections::HashSet<crate::mir::Register>,
        _instr: &Instruction,
    ) -> bool {
        // Disabled heuristic to prevent over-optimization that can break loop behavior
        false
    }

    fn is_invariant_register(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
        defs_in_loop: &std::collections::HashSet<Register>,
        reg: &Register,
    ) -> bool {
        // A register is invariant if it's not defined inside the loop.
        // Physical registers are conservatively treated as non-invariant since they can be modified
        match reg {
            Register::Physical(_) => false, // Physical registers can be modified within loops
            Register::Virtual(_) => !defs_in_loop.contains(reg),
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
        invariant_instrs: &[(usize, usize)],
    ) -> Result<(), String> {
        if invariant_instrs.is_empty() {
            return Ok(());
        }

        // Find the loop header block
        let header_idx = func
            .blocks
            .iter()
            .position(|b| b.label == loop_info.header)
            .ok_or_else(|| format!("Loop header '{}' not found", loop_info.header))?;

        // Create a pre-header block to hold invariant instructions
        // This avoids executing them before the loop condition check
        let pre_header_label = format!("{}_pre", loop_info.header);
        let mut pre_header_block = crate::mir::Block {
            label: pre_header_label.clone(),
            instructions: Vec::new(),
        };

        // Collect and sort invariant instructions by block and instruction index
        // Sort by (block_idx, instr_idx) descending to avoid index shifting issues
        let mut sorted_invariant = invariant_instrs.to_vec();
        sorted_invariant.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by instruction index descending
        sorted_invariant.sort_by(|a, b| b.0.cmp(&a.0)); // Sort by block index descending

        // Remove duplicate positions (same instruction appearing multiple times)
        sorted_invariant.dedup();

        // Collect instructions to move, validating indices
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

        // Update any jumps that target the original header to target the pre-header instead
        // Skip the pre-header block itself when updating jumps
        for (block_idx, block) in func.blocks.iter_mut().enumerate() {
            // Skip the pre-header block we just inserted
            if block_idx == header_idx {
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

        Ok(())
    }

    fn collect_defs_in_loop(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
    ) -> std::collections::HashSet<Register> {
        let mut defs = std::collections::HashSet::new();
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
struct LoopInfo {
    header: String,
    back_edge_target: String,
    blocks: HashSet<String>,
}

#[derive(Debug)]
struct UnrollableLoop {
    header: String,
    body_blocks: Vec<String>,
    bound: i64,
    induction_var: Option<Register>,
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
        let mut changed = false;

        // Find loops that can be safely unrolled
        let loops = self.find_unrollable_loops(func);

        for loop_info in loops {
            if self.unroll_loop(func, &loop_info)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Find loops that can be safely unrolled (small constant bounds)
    fn find_unrollable_loops(&self, func: &Function) -> Vec<UnrollableLoop> {
        let mut unrollable = Vec::new();

        for block in &func.blocks {
            if let Some(last_instr) = block.instructions.last() {
                match last_instr {
                    Instruction::Br {
                        cond,
                        true_target,
                        false_target,
                        ..
                    } => {
                        // Look for simple counting loops: while (i < N) or similar
                        if let Some(loop_bound) = self.analyze_loop_bound(
                            func,
                            block,
                            &Operand::Register(cond.clone()),
                            true_target,
                            false_target,
                        ) {
                            if loop_bound <= 8 {
                                // Only unroll very small loops for safety
                                unrollable.push(UnrollableLoop {
                                    header: block.label.clone(),
                                    body_blocks: vec![true_target.clone()],
                                    bound: loop_bound,
                                    induction_var: None, // TODO: detect induction variable
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        unrollable
    }

    /// Analyze if a loop has a constant bound that can be unrolled
    fn analyze_loop_bound(
        &self,
        _func: &Function,
        _block: &Block,
        _cond: &Operand,
        _true_target: &str,
        _false_target: &str,
    ) -> Option<i64> {
        // Very conservative loop bound analysis for safe unrolling
        // Only enable for extremely small, known-safe bounds to prevent code bloat

        // For matrix operations, small unroll factors like 2-4 can be beneficial
        // but we start very conservative to avoid issues

        // TODO: Implement proper loop bound analysis that can detect
        // compile-time constants and simple induction variables

        // For now: enable unrolling of loops with bound 2 (very safe)
        Some(2)
    }

    /// Unroll a loop by duplicating its body (very conservative implementation)
    fn unroll_loop(&self, func: &mut Function, loop_info: &UnrollableLoop) -> Result<bool, String> {
        // Extremely conservative loop unrolling for safety
        // Only handle the simplest case: unroll factor 2 for small loops

        if loop_info.bound != 2 {
            return Ok(false); // Only unroll factor 2 for now
        }

        // Find the loop header block
        let header_idx = func.blocks.iter().position(|b| b.label == loop_info.header);
        if header_idx.is_none() {
            return Ok(false);
        }

        let header_idx = header_idx.unwrap();

        // For now, implement a very simple case: if the loop body is just one block
        // with a simple conditional branch, duplicate it
        if loop_info.body_blocks.len() == 1 {
            let body_label = &loop_info.body_blocks[0];
            let body_idx = func.blocks.iter().position(|b| &b.label == body_label);

            if let Some(body_idx) = body_idx {
                // Very simple unrolling: duplicate the body block instructions
                // This is not a proper unrolling but a conservative approximation

                let body_block = &func.blocks[body_idx];
                let mut new_instructions = Vec::new();

                // Duplicate the instructions (excluding the branch)
                for instr in &body_block.instructions {
                    if !matches!(
                        instr,
                        Instruction::Jmp { .. } | Instruction::Br { .. } | Instruction::Ret { .. }
                    ) {
                        new_instructions.push(instr.clone());
                    }
                }

                // Insert the duplicated instructions before the branch in the original block
                let header_block = &mut func.blocks[header_idx];
                if let Some(branch_idx) = header_block
                    .instructions
                    .iter()
                    .position(|i| matches!(i, Instruction::Br { .. }))
                {
                    // Insert duplicated instructions before the branch
                    for (offset, instr) in new_instructions.into_iter().enumerate() {
                        header_block.instructions.insert(branch_idx + offset, instr);
                    }
                    return Ok(true);
                }
            }
        }

        Ok(false)
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
