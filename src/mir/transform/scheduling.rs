//! Instruction scheduling transform for MIR.

use crate::mir::{
    Block, Function, Instruction, IntBinOp, Register,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::{
    cmp::{Ordering, max},
    collections::{BinaryHeap, HashMap, HashSet},
    mem,
};

/// Instruction scheduling that reorders instructions to improve ILP.
#[derive(Default)]
pub struct InstructionScheduling;

impl Transform for InstructionScheduling {
    fn name(&self) -> &'static str {
        "instruction_scheduling"
    }

    fn description(&self) -> &'static str {
        "Reorders instructions to improve instruction-level parallelism"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl InstructionScheduling {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 2_000;

        if func.blocks.len() > MAX_BLOCKS {
            return Ok(false); // Skip optimization on huge functions for safety
        }

        for block in &mut func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                continue;
            }
            if self.schedule_block(block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn schedule_block(&self, block: &mut Block) -> bool {
        if block.instructions.len() < 3 {
            return false;
        }

        // 1. Build Dependency Graph
        let graph = self.build_dependency_graph(&block.instructions);

        // 2. Calculate Priorities (Critical Path)
        let priorities = self.calculate_priorities(&graph, &block.instructions);

        // 3. List Scheduling
        let scheduled_indices = self.list_schedule(&graph, &priorities, block.instructions.len());

        // 4. Reorder Instructions
        if self.is_order_changed(&scheduled_indices) {
            let old_instructions = mem::take(&mut block.instructions);
            let mut new_instructions = Vec::with_capacity(old_instructions.len());
            for &idx in &scheduled_indices {
                new_instructions.push(old_instructions[idx].clone());
            }
            block.instructions = new_instructions;
            true
        } else {
            false
        }
    }

    fn build_dependency_graph(&self, instructions: &[Instruction]) -> DependencyGraph {
        let mut graph = DependencyGraph::new(instructions.len());
        let mut reg_defs: HashMap<Register, usize> = HashMap::new();
        let mut reg_readers: HashMap<Register, Vec<usize>> = HashMap::new();

        // Track memory dependencies
        // Conservative approach:
        // - Loads depend on previous Stores (RAW)
        // - Stores depend on previous Stores (WAW) and Loads (WAR)
        // For simplicity in this pass, we chain all memory ops to preserve relative order.
        let mut last_memory_op: Option<usize> = None;

        for (idx, instr) in instructions.iter().enumerate() {
            // 1. Data Dependencies (Register RAW)
            for use_reg in instr.use_regs() {
                if let Some(&def_idx) = reg_defs.get(use_reg) {
                    graph.add_edge(def_idx, idx);
                }
                reg_readers.entry(use_reg.clone()).or_default().push(idx);
            }

            // 2. Register Output Dependencies (WAW) - to ensure we don't reorder defs to same reg
            // (Though SSA should prevent this, MIR might not be strict SSA here)
            if let Some(def_reg) = instr.def_reg() {
                if let Some(&prev_def_idx) = reg_defs.get(def_reg) {
                    graph.add_edge(prev_def_idx, idx);
                }
                if let Some(readers) = reg_readers.remove(def_reg) {
                    for reader in readers {
                        if reader != idx {
                            graph.add_edge(reader, idx);
                        }
                    }
                }
                reg_defs.insert(def_reg.clone(), idx);
            }

            // 3. Memory Dependencies
            if self.is_memory_op(instr) || instr.may_trap_integer() {
                if let Some(prev_mem_idx) = last_memory_op {
                    graph.add_edge(prev_mem_idx, idx);
                }
                last_memory_op = Some(idx);
            }

            // 4. Barrier Dependencies (Call, Ret, volatile)
            // Calls are memory barriers and have side effects. Dependencies chain through them.
            if matches!(
                instr,
                Instruction::Call { .. } | Instruction::Ret { .. } | Instruction::Switch { .. }
            ) {
                // Make this instruction depend on EVERYTHING before it?
                // Or effectively act as a barrier.
                // Ideally, we chain it with the memory chain.
                if let Some(prev_mem_idx) = last_memory_op {
                    graph.add_edge(prev_mem_idx, idx);
                }
                last_memory_op = Some(idx);
            }
        }

        if let [
            ..,
            Instruction::IntCmp { dst, .. } | Instruction::FloatCmp { dst, .. },
            Instruction::Br { cond, .. },
        ] = instructions
            && dst == cond
        {
            let comparison_index = instructions.len() - 2;
            for index in 0..comparison_index {
                graph.add_edge(index, comparison_index);
            }
        }

        // 5. Control Dependencies
        // The terminator (Branch/Jmp/Ret) must depend on everything that affects it's condition
        // or effectively be last.
        // We ensure terminators are last by giving them implicit dependence on all roots?
        // Actually, terminators use registers, so RAW covers condition.
        // But we must ensure no instruction is moved AFTER the terminator.
        // Since terminator is usually last, and we only schedule valid instructions,
        // we just ensure the terminator index is constrained.
        // By construction, terminators shouldn't have successors in the block.
        // And we simply must ensure all instructions are scheduled.

        // Special case: Make sure the terminator (last instruction) depends on side-effecting ops?
        // Simpler: Just make the last instruction depend on the last memory op.
        if let Some(last_inst_idx) = instructions.len().checked_sub(1)
            && let Some(prev_mem_idx) = last_memory_op
            && prev_mem_idx != last_inst_idx
        {
            graph.add_edge(prev_mem_idx, last_inst_idx);
        }

        graph
    }

    fn calculate_priorities(
        &self,
        graph: &DependencyGraph,
        instructions: &[Instruction],
    ) -> HashMap<usize, usize> {
        let mut priorities = HashMap::new();
        // Calculate latency-weighted depth from sinks up
        // We need a topological sort or just simple recursion with memoization.
        // Since it's a DAG, memoization works.

        let mut visited = HashSet::new();
        for i in 0..instructions.len() {
            self.compute_depth(i, graph, instructions, &mut priorities, &mut visited);
        }
        priorities
    }

    fn compute_depth(
        &self,
        node: usize,
        graph: &DependencyGraph,
        instructions: &[Instruction],
        priorities: &mut HashMap<usize, usize>,
        visited: &mut HashSet<usize>,
    ) -> usize {
        if let Some(&p) = priorities.get(&node) {
            return p;
        }

        if visited.contains(&node) {
            return 0; // Cycle detected (shouldn't happen in DAG)
        }
        visited.insert(node);

        let latency = self.get_latency(&instructions[node]);
        let mut max_succ_depth = 0;

        if let Some(succs) = graph.edges.get(&node) {
            for &succ in succs {
                max_succ_depth = max(
                    max_succ_depth,
                    self.compute_depth(succ, graph, instructions, priorities, visited),
                );
            }
        }

        let depth = latency + max_succ_depth;
        visited.remove(&node);
        priorities.insert(node, depth);
        depth
    }

    fn get_latency(&self, instr: &Instruction) -> usize {
        match instr {
            Instruction::Load { .. } | Instruction::FloatBinary { .. } => 3,
            Instruction::IntBinary {
                op: IntBinOp::SDiv | IntBinOp::UDiv,
                ..
            } => 4,
            Instruction::IntBinary {
                op: IntBinOp::Mul, ..
            } => 2,
            Instruction::Call { .. } => 5,
            _ => 1,
        }
    }

    fn list_schedule(
        &self,
        graph: &DependencyGraph,
        priorities: &HashMap<usize, usize>,
        num_instrs: usize,
    ) -> Vec<usize> {
        let mut scheduled = Vec::with_capacity(num_instrs);
        let mut in_degree = graph.in_degree.clone();

        // Priority Queue stores (priority, index).
        // BinaryHeap is max-heap, so higher priority (depth) comes first.
        let mut ready_queue = BinaryHeap::new();

        for (i, &degree) in in_degree.iter().enumerate().take(num_instrs) {
            if degree == 0 {
                ready_queue.push(ScheduledItem {
                    priority: *priorities.get(&i).unwrap_or(&0),
                    index: i,
                });
            }
        }

        while let Some(item) = ready_queue.pop() {
            let u = item.index;
            scheduled.push(u);

            if let Some(succs) = graph.edges.get(&u) {
                for &v in succs {
                    if let Some(degree) = in_degree.get_mut(v) {
                        *degree -= 1;
                        if *degree == 0 {
                            ready_queue.push(ScheduledItem {
                                priority: *priorities.get(&v).unwrap_or(&0),
                                index: v,
                            });
                        }
                    }
                }
            }
        }

        scheduled
    }

    fn is_order_changed(&self, indices: &[usize]) -> bool {
        for (i, &idx) in indices.iter().enumerate() {
            if i != idx {
                return true;
            }
        }
        false
    }

    fn is_memory_op(&self, instr: &Instruction) -> bool {
        #[cfg(feature = "nightly")]
        {
            // Atomics and fences carry ordering, so they belong on the memory
            // chain; without them they can drift across ordinary accesses.
            if matches!(
                instr,
                Instruction::AtomicLoad { .. }
                    | Instruction::AtomicStore { .. }
                    | Instruction::AtomicBinary { .. }
                    | Instruction::AtomicCompareExchange { .. }
                    | Instruction::Fence { .. }
            ) {
                return true;
            }
        }
        matches!(instr, Instruction::Load { .. } | Instruction::Store { .. })
    }
}

struct DependencyGraph {
    edges: HashMap<usize, Vec<usize>>, // Adjacency list
    in_degree: Vec<usize>,
}

impl DependencyGraph {
    fn new(size: usize) -> Self {
        Self {
            edges: HashMap::new(),
            in_degree: vec![0; size],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.entry(from).or_default().push(to);
        self.in_degree[to] += 1;
    }
}

#[derive(Eq, PartialEq)]
struct ScheduledItem {
    priority: usize,
    index: usize,
}

impl Ord for ScheduledItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        // Break ties with original index (lower index first) to preserve stability
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for ScheduledItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        AddressMode, FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MemoryAttrs, MirType, Operand,
        ScalarType, VirtualReg,
    };

    #[test]
    fn terminal_comparisons_stay_adjacent_to_branches() {
        let ty = MirType::Scalar(ScalarType::I64);
        let mut block = Block::new("body");
        let a: Register = VirtualReg::gpr(0).into();
        let b: Register = VirtualReg::gpr(1).into();
        let counter: Register = VirtualReg::gpr(2).into();
        let flag: Register = VirtualReg::gpr(3).into();
        for _ in 0..8 {
            block.instructions.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty,
                dst: a.clone(),
                lhs: a.clone().into(),
                rhs: b.clone().into(),
            });
        }
        block.instructions.push(Instruction::IntBinary {
            op: IntBinOp::Sub,
            ty,
            dst: counter.clone(),
            lhs: counter.clone().into(),
            rhs: Operand::imm_i64(8),
        });
        block.instructions.push(Instruction::IntCmp {
            op: IntCmpOp::Ne,
            ty,
            dst: flag.clone(),
            lhs: counter.into(),
            rhs: Operand::imm_i64(0),
        });
        block.instructions.push(Instruction::Br {
            cond: flag,
            true_target: "body".into(),
            false_target: "exit".into(),
        });
        let tail = block.instructions[block.instructions.len() - 2..].to_vec();
        InstructionScheduling.schedule_block(&mut block);
        assert_eq!(block.instructions[block.instructions.len() - 2..], tail);
    }

    #[test]
    fn test_scheduling_keeps_reads_before_redefinition() {
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
        let mut func = FunctionBuilder::new("read_before_redefine")
            .returns(i64_ty)
            .block("entry")
            .instr(add(2, imm(40), imm(0)))
            .instr(add(8, imm(1), imm(0)))
            .instr(add(8, reg(8), reg(2)))
            .instr(add(2, reg(2), imm(100)))
            .instr(add(11, reg(2), imm(1)))
            .instr(add(12, reg(11), reg(8)))
            .instr(Instruction::Ret {
                value: Some(reg(12)),
            })
            .build();

        InstructionScheduling
            .apply(&mut func)
            .expect("Scheduling failed");

        let instrs = &func.blocks[0].instructions;
        let read_pos = instrs
            .iter()
            .position(|i| {
                matches!(i, Instruction::IntBinary { dst: Register::Virtual(v), lhs: Operand::Register(_), .. } if v.id == 8)
            })
            .unwrap();
        let redefine_pos = instrs
            .iter()
            .position(|i| {
                matches!(i, Instruction::IntBinary { dst: Register::Virtual(v), rhs: Operand::Immediate(Immediate::I64(100)), .. } if v.id == 2)
            })
            .unwrap();
        assert!(read_pos < redefine_pos);
    }

    #[test]
    fn test_scheduling_latency_hiding() {
        // Create a sequence:
        // 0: Load r1
        // 1: Add r2 = r1 + 1 (Specific dependency on 0)
        // 2: Add r3 = 5 + 5 (Independent)
        // 3: Add r4 = 6 + 6 (Independent)
        //
        // Scheduler should move 2 and 3 between 0 and 1 to hide Load latency.
        // Expected: 0, 2, 3, 1 (or 0, 3, 2, 1)

        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // 0: Load (High latency)
            .instr(Instruction::Load {
                dst: VirtualReg::gpr(1).into(),
                ty: MirType::Scalar(ScalarType::I64),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(10).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            // 1: Dependent Add
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            // 2: Independent Add
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Immediate(Immediate::I64(5)),
                rhs: Operand::Immediate(Immediate::I64(5)),
            })
            // 3: Independent Add
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(),
                lhs: Operand::Immediate(Immediate::I64(6)),
                rhs: Operand::Immediate(Immediate::I64(6)),
            })
            .build();

        let mut func = func;
        let pass = InstructionScheduling;
        let _changed = pass.apply(&mut func).expect("Scheduling failed");

        // Verify all instructions are still present (scheduler is correct)
        let block = &func.blocks[0];
        let instrs = &block.instructions;
        assert_eq!(instrs.len(), 4, "All instructions should be present");

        // Find indices
        let load_idx = instrs
            .iter()
            .position(|i| matches!(i, Instruction::Load { .. }))
            .unwrap();
        let r2_def_idx = instrs
            .iter()
            .position(|i| {
                if let Instruction::IntBinary { dst, .. } = i
                    && let Register::Virtual(vreg) = dst
                {
                    return vreg.id == 2;
                }
                false
            })
            .unwrap();

        // The dependent Add (r2) MUST come after the Load (r1)
        assert!(
            r2_def_idx > load_idx,
            "Dependent instruction must come after its source"
        );
    }
}
