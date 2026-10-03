use lamina_mir::{Function, Instruction, Register, VirtualReg};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

/// Opaque handle that allows dynamic dispatch over register allocators without
/// leaking architecture-specific physical register types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysRegHandle {
    /// Physical registers represented by their canonical assembly name.
    Named(&'static str),
}

impl PhysRegHandle {
    /// Retrieve the assembly name if the handle stores one.
    pub fn as_named(self) -> Option<&'static str> {
        match self {
            PhysRegHandle::Named(name) => Some(name),
        }
    }
}

/// Conversion helpers that allow a concrete physical register type to be
/// converted to/from a [`PhysRegHandle`] for dynamic dispatch.
pub trait PhysRegConvertible: Copy + Eq {
    /// Convert the register into an opaque handle.
    fn into_handle(self) -> PhysRegHandle;

    /// Try to rebuild the register from an opaque handle.
    fn from_handle(handle: PhysRegHandle) -> Option<Self>
    where
        Self: Sized;
}

impl PhysRegConvertible for &'static str {
    fn into_handle(self) -> PhysRegHandle {
        PhysRegHandle::Named(self)
    }

    fn from_handle(handle: PhysRegHandle) -> Option<Self> {
        match handle {
            PhysRegHandle::Named(name) => Some(name),
        }
    }
}

/// Target-facing interface for MIR register allocation (per-function/incremental).
///
/// The trait stays purposefully small: code generators typically need a
/// lightweight scratch register pool, a stable mapping from virtual to
/// physical registers, and explicit hooks to reserve or release physical
/// registers that are pre-coloured by the ABI. Architecture backends can build
/// richer policies on top of this contract without forcing every target to
/// adopt the same strategy.
pub trait LocalRegisterAllocator {
    /// Architecture-specific physical register handle.
    type PhysReg: PhysRegConvertible;

    /// Acquire a short-lived scratch register. Returns `None` when the dedicated
    /// pool is exhausted so the caller may spill or choose an alternate path.
    fn alloc_scratch(&mut self) -> Option<Self::PhysReg>;

    /// Release a scratch register obtained through [`LocalRegisterAllocator::alloc_scratch`].
    fn free_scratch(&mut self, phys: Self::PhysReg);

    /// Look up the physical register currently assigned to the virtual
    /// register, when available.
    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg>;

    /// Ensure that the virtual register has a permanent mapping. Implementers
    /// can reject unsupported register classes by returning `None`, signalling
    /// that the caller should spill.
    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg>;

    /// Resolve the backing physical register for an arbitrary MIR register
    /// (virtual or physical).
    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg>;

    /// Mark a physical register as occupied, removing it from the allocator's
    /// free pool if necessary.
    fn occupy(&mut self, phys: Self::PhysReg);

    /// Release a previously occupied physical register back to the pool.
    fn release(&mut self, phys: Self::PhysReg);

    /// Test whether the allocator currently treats the physical register as
    /// occupied.
    fn is_occupied(&self, phys: Self::PhysReg) -> bool;
}

/// Backward-compatible alias for [`LocalRegisterAllocator`].
pub use LocalRegisterAllocator as RegisterAllocator;

/// Object-safe wrapper around [`LocalRegisterAllocator`] permitting dynamic dispatch
/// via `dyn LocalRegisterAllocatorDyn`.
pub trait LocalRegisterAllocatorDyn {
    fn alloc_scratch_dyn(&mut self) -> Option<PhysRegHandle>;
    fn free_scratch_dyn(&mut self, phys: PhysRegHandle);
    fn get_mapping_dyn(&self, vreg: &VirtualReg) -> Option<PhysRegHandle>;
    fn ensure_mapping_dyn(&mut self, vreg: VirtualReg) -> Option<PhysRegHandle>;
    fn mapped_for_register_dyn(&self, reg: &Register) -> Option<PhysRegHandle>;
    fn occupy_dyn(&mut self, phys: PhysRegHandle);
    fn release_dyn(&mut self, phys: PhysRegHandle);
    fn is_occupied_dyn(&self, phys: PhysRegHandle) -> bool;
}

/// Backward-compatible alias for [`LocalRegisterAllocatorDyn`].
pub use LocalRegisterAllocatorDyn as RegisterAllocatorDyn;

/// Result of allocating a virtual register: either a physical register or a
/// stack spill slot (byte offset from the frame base pointer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocation<R: Copy> {
    Register(R),
    /// Byte offset from the frame base (negative = below the saved frame pointer).
    Spill(i32),
}

/// Live interval: the range `[start, end]` of instruction indices (0-based,
/// counting sequentially across all blocks in program order) during which
/// `vreg` must be kept alive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveInterval {
    pub vreg: VirtualReg,
    /// Index of the instruction that first defines `vreg`.
    pub start: usize,
    /// Index of the last instruction that uses `vreg`.
    pub end: usize,
}

/// Classic linear scan register allocator (Poletto & Sarkar, 1999).
///
/// Usage:
/// ```ignore
/// let intervals = LinearScanAllocator::compute_intervals(&function);
/// let free_regs = vec!["r12", "r13", "r14", "r15", "rbx"];
/// let alloc_map = LinearScanAllocator::allocate(&intervals, &free_regs);
/// ```
pub struct LinearScanAllocator;

impl LinearScanAllocator {
    /// Compute interval hulls from instruction operands and control-flow liveness.
    pub fn compute_intervals(function: &Function) -> Vec<LiveInterval> {
        let mut intervals: HashMap<VirtualReg, LiveInterval> = HashMap::new();
        let mut positions = Vec::new();
        let mut uses = Vec::new();
        let mut defs = Vec::new();
        let mut pos = 0;
        let record = |intervals: &mut HashMap<VirtualReg, LiveInterval>, vreg, pos| {
            let interval = intervals.entry(vreg).or_insert(LiveInterval {
                vreg,
                start: pos,
                end: pos,
            });
            interval.start = interval.start.min(pos);
            interval.end = interval.end.max(pos);
        };
        for param in &function.sig.params {
            if let Register::Virtual(vreg) = param.reg {
                record(&mut intervals, vreg, 0);
            }
        }
        for block in &function.blocks {
            let start = pos;
            let mut block_uses = HashSet::new();
            let mut block_defs = HashSet::new();
            for instr in &block.instructions {
                let mut used: Vec<_> = instr
                    .use_regs()
                    .into_iter()
                    .filter_map(|reg| {
                        if let Register::Virtual(vreg) = reg {
                            Some(*vreg)
                        } else {
                            None
                        }
                    })
                    .collect();
                if let Instruction::StackAddr { slot, .. } = instr {
                    used.push(*slot);
                }
                for vreg in used {
                    record(&mut intervals, vreg, pos);
                    if !block_defs.contains(&vreg) {
                        block_uses.insert(vreg);
                    }
                }
                if let Some(Register::Virtual(vreg)) = instr.def_reg() {
                    record(&mut intervals, *vreg, pos);
                    block_defs.insert(*vreg);
                }
                pos += 1;
            }
            positions.push((start, pos.saturating_sub(1).max(start)));
            uses.push(block_uses);
            defs.push(block_defs);
        }
        let labels: HashMap<_, _> = function
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.label.as_str(), index))
            .collect();
        let mut live_in = vec![HashSet::new(); function.blocks.len()];
        let mut live_out = live_in.clone();
        loop {
            let mut changed = false;
            for (index, block) in function.blocks.iter().enumerate().rev() {
                let out: HashSet<_> = block
                    .successors()
                    .iter()
                    .filter_map(|label| labels.get(label))
                    .flat_map(|successor| live_in[*successor].iter().copied())
                    .collect();
                let mut input = uses[index].clone();
                input.extend(out.difference(&defs[index]).copied());
                changed |= input != live_in[index] || out != live_out[index];
                live_in[index] = input;
                live_out[index] = out;
            }
            if !changed {
                break;
            }
        }
        for (index, (start, end)) in positions.into_iter().enumerate() {
            for vreg in &live_in[index] {
                record(&mut intervals, *vreg, start);
            }
            for vreg in &live_out[index] {
                record(&mut intervals, *vreg, end);
            }
        }
        let mut result: Vec<_> = intervals.into_values().collect();
        result.sort_by_key(|interval| (interval.start, interval.vreg));
        result
    }

    /// Run the linear scan allocation algorithm.
    ///
    /// `intervals` must be sorted by `start` (as returned by `compute_intervals`).
    /// `available_regs` is the pool of physical registers to draw from (any `Copy + Eq`
    /// type works — for x86_64 pass `&["r12", "r13", ...]`).
    ///
    /// Returns a map from each `VirtualReg` to either a physical register or a
    /// spill slot (negative byte offset from frame base).
    pub fn allocate<R: Copy + Eq>(
        intervals: &[LiveInterval],
        available_regs: &[R],
    ) -> HashMap<VirtualReg, Allocation<R>> {
        let mut result: HashMap<VirtualReg, Allocation<R>> = HashMap::new();
        // Active intervals: sorted by end point (ascending).
        let mut active: Vec<(&LiveInterval, R)> = Vec::new();
        let mut free: Vec<R> = available_regs.to_vec();
        let mut next_spill: i32 = -8; // first spill at [rbp - 8]

        for interval in intervals {
            // Expire intervals that ended before this one started.
            let current_start = interval.start;
            let mut freed: Vec<R> = Vec::new();
            active.retain(|(ai, reg)| {
                if ai.end < current_start {
                    freed.push(*reg);
                    false
                } else {
                    true
                }
            });
            free.extend(freed);

            if let Some(reg) = free.pop() {
                // Assign a free register.
                result.insert(interval.vreg, Allocation::Register(reg));
                // Insert into active, keeping it sorted by end point.
                let pos = active
                    .binary_search_by_key(&interval.end, |(ai, _)| ai.end)
                    .unwrap_or_else(|i| i);
                active.insert(pos, (interval, reg));
            } else {
                // Spill the interval with the longest remaining lifetime (last in
                // `active` since it's sorted by end point).
                match active.last().cloned() {
                    Some((spill_interval, spill_reg)) if spill_interval.end > interval.end => {
                        // Spill the existing active interval; give its register to `interval`.
                        result.insert(spill_interval.vreg, Allocation::Spill(next_spill));
                        next_spill -= 8;
                        active.pop();
                        result.insert(interval.vreg, Allocation::Register(spill_reg));
                        let pos = active
                            .binary_search_by_key(&interval.end, |(ai, _)| ai.end)
                            .unwrap_or_else(|i| i);
                        active.insert(pos, (interval, spill_reg));
                    }
                    _ => {
                        // Spill the current interval.
                        result.insert(interval.vreg, Allocation::Spill(next_spill));
                        next_spill -= 8;
                    }
                }
            }
        }

        result
    }
}

/// Interference check for live intervals: overlap on the global instruction index line.
#[inline]
pub fn intervals_interfere(a: &LiveInterval, b: &LiveInterval) -> bool {
    a.start <= b.end && b.start <= a.end
}

/// Greedy graph-coloring register allocator (interference graph from live intervals).
///
/// Intervals that overlap cannot share a register. Nodes are colored in **descending
/// degree** (Welsh–Powell order) to reduce spill pressure compared to arbitrary order.
/// When no color is available among `available_regs`, the interval is spilled using the
/// same slot layout as [`LinearScanAllocator::allocate`].
pub struct GraphColorAllocator;

impl GraphColorAllocator {
    /// Color `intervals` using at most `available_regs.len()` registers.
    ///
    /// `intervals` may be in any order; internally sorted by interference degree.
    pub fn allocate<R: Copy + Eq + Hash>(
        intervals: &[LiveInterval],
        available_regs: &[R],
    ) -> HashMap<VirtualReg, Allocation<R>> {
        if intervals.is_empty() {
            return HashMap::new();
        }

        let mut order: Vec<usize> = (0..intervals.len()).collect();
        order.sort_by(|&i, &j| {
            let deg_i = intervals
                .iter()
                .enumerate()
                .filter(|(k, other)| *k != i && intervals_interfere(&intervals[i], other))
                .count();
            let deg_j = intervals
                .iter()
                .enumerate()
                .filter(|(k, other)| *k != j && intervals_interfere(&intervals[j], other))
                .count();
            deg_j.cmp(&deg_i).then_with(|| i.cmp(&j))
        });

        let mut result: HashMap<VirtualReg, Allocation<R>> = HashMap::new();
        let mut next_spill: i32 = -8;

        for idx in order {
            let interval = &intervals[idx];
            let mut blocked: HashSet<R> = HashSet::new();
            for (j, other) in intervals.iter().enumerate() {
                if j == idx || !intervals_interfere(interval, other) {
                    continue;
                }
                if let Some(Allocation::Register(r)) = result.get(&other.vreg) {
                    blocked.insert(*r);
                }
            }

            let mut picked: Option<R> = None;
            for reg in available_regs {
                if !blocked.contains(reg) {
                    picked = Some(*reg);
                    break;
                }
            }

            match picked {
                Some(r) => {
                    result.insert(interval.vreg, Allocation::Register(r));
                }
                None => {
                    result.insert(interval.vreg, Allocation::Spill(next_spill));
                    next_spill -= 8;
                }
            }
        }

        result
    }
}

impl<T> LocalRegisterAllocatorDyn for T
where
    T: LocalRegisterAllocator,
{
    fn alloc_scratch_dyn(&mut self) -> Option<PhysRegHandle> {
        self.alloc_scratch().map(PhysRegConvertible::into_handle)
    }

    fn free_scratch_dyn(&mut self, phys: PhysRegHandle) {
        if let Some(reg) = <T::PhysReg as PhysRegConvertible>::from_handle(phys) {
            self.free_scratch(reg);
        } else {
            debug_assert!(false, "failed to decode physical register handle");
        }
    }

    fn get_mapping_dyn(&self, vreg: &VirtualReg) -> Option<PhysRegHandle> {
        self.get_mapping(vreg).map(PhysRegConvertible::into_handle)
    }

    fn ensure_mapping_dyn(&mut self, vreg: VirtualReg) -> Option<PhysRegHandle> {
        self.ensure_mapping(vreg)
            .map(PhysRegConvertible::into_handle)
    }

    fn mapped_for_register_dyn(&self, reg: &Register) -> Option<PhysRegHandle> {
        self.mapped_for_register(reg)
            .map(PhysRegConvertible::into_handle)
    }

    fn occupy_dyn(&mut self, phys: PhysRegHandle) {
        if let Some(reg) = <T::PhysReg as PhysRegConvertible>::from_handle(phys) {
            self.occupy(reg);
        } else {
            debug_assert!(false, "failed to decode physical register handle");
        }
    }

    fn release_dyn(&mut self, phys: PhysRegHandle) {
        if let Some(reg) = <T::PhysReg as PhysRegConvertible>::from_handle(phys) {
            self.release(reg);
        } else {
            debug_assert!(false, "failed to decode physical register handle");
        }
    }

    fn is_occupied_dyn(&self, phys: PhysRegHandle) -> bool {
        if let Some(reg) = <T::PhysReg as PhysRegConvertible>::from_handle(phys) {
            self.is_occupied(reg)
        } else {
            debug_assert!(false, "failed to decode physical register handle");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lamina_mir::{
        Block, FunctionBuilder, Immediate, Instruction, IntBinOp, MirType, Operand, Register,
        ScalarType, VirtualReg,
    };

    fn make_add_function() -> Function {
        // fn add(v0: i64, v1: i64) -> i64 {
        // entry:
        //   v2 = add v0, v1
        //   ret v2
        // }
        let v0 = Register::Virtual(VirtualReg::gpr(0));
        let v1 = Register::Virtual(VirtualReg::gpr(1));
        let v2 = Register::Virtual(VirtualReg::gpr(2));
        let i64_ty = MirType::Scalar(ScalarType::I64);

        FunctionBuilder::new("add")
            .param(v0.clone(), i64_ty)
            .param(v1.clone(), i64_ty)
            .returns(i64_ty)
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64_ty,
                dst: v2.clone(),
                lhs: Operand::Register(v0),
                rhs: Operand::Register(v1),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(v2)),
            })
            .build()
    }

    #[test]
    fn test_compute_intervals_basic() {
        let func = make_add_function();
        let intervals = LinearScanAllocator::compute_intervals(&func);

        // v0, v1 are params (start=0), v2 is defined at instruction 0
        assert!(!intervals.is_empty());
        // All three vregs should appear
        let vreg_ids: Vec<u32> = intervals.iter().map(|i| i.vreg.id).collect();
        assert!(vreg_ids.contains(&0));
        assert!(vreg_ids.contains(&1));
        assert!(vreg_ids.contains(&2));
    }

    #[test]
    fn test_compute_intervals_sorted_by_start() {
        let func = make_add_function();
        let intervals = LinearScanAllocator::compute_intervals(&func);
        let starts: Vec<usize> = intervals.iter().map(|i| i.start).collect();
        let mut sorted = starts.clone();
        sorted.sort_unstable();
        assert_eq!(starts, sorted, "intervals should be sorted by start");
    }

    #[test]
    fn copy_contributes_source_use_and_destination_def() {
        let f64_ty = MirType::Scalar(ScalarType::F64);
        let src = Register::Virtual(VirtualReg::fpr(0));
        let dst = Register::Virtual(VirtualReg::fpr(1));
        let func = FunctionBuilder::new("copy")
            .param(src.clone(), f64_ty)
            .returns(f64_ty)
            .block("entry")
            .instr(Instruction::Copy {
                ty: f64_ty,
                dst: dst.clone(),
                src: Operand::Register(src),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        let intervals = LinearScanAllocator::compute_intervals(&func);
        let ids: Vec<u32> = intervals.iter().map(|interval| interval.vreg.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
    }

    #[test]
    fn intervals_cover_values_live_across_back_edges() {
        let invariant = VirtualReg::gpr(0);
        let temporary = VirtualReg::gpr(1);
        let mut function = FunctionBuilder::new("loop")
            .param(Register::Virtual(invariant), MirType::i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: "header".into(),
            })
            .build();
        let mut header = Block::new("header");
        header.push(Instruction::Br {
            cond: Register::Virtual(invariant),
            true_target: "body".into(),
            false_target: "exit".into(),
        });
        function.add_block(header);
        let mut exit = Block::new("exit");
        exit.push(Instruction::Ret { value: None });
        function.add_block(exit);
        let mut body = Block::new("body");
        body.push(Instruction::Copy {
            ty: MirType::i64(),
            dst: Register::Virtual(temporary),
            src: Operand::Immediate(Immediate::I64(42)),
        });
        body.push(Instruction::Jmp {
            target: "header".into(),
        });
        function.add_block(body);
        let intervals = LinearScanAllocator::compute_intervals(&function);
        let live = intervals
            .iter()
            .find(|interval| interval.vreg == invariant)
            .expect("invariant");
        assert_eq!((live.start, live.end), (0, 4));
        for plan in [
            LinearScanAllocator::allocate(&intervals, &["r12"]),
            GraphColorAllocator::allocate(&intervals, &["r12"]),
        ] {
            assert!(
                matches!(plan[&invariant], Allocation::Spill(_))
                    || matches!(plan[&temporary], Allocation::Spill(_))
            );
        }
    }

    #[test]
    fn stack_address_slots_participate_in_intervals() {
        let slot = VirtualReg::gpr(0);
        let pointer = VirtualReg::gpr(1);
        let function = FunctionBuilder::new("address")
            .block("entry")
            .instr(Instruction::StackAddr {
                dst: Register::Virtual(pointer),
                slot,
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(pointer))),
            })
            .build();
        let intervals = LinearScanAllocator::compute_intervals(&function);
        assert_eq!(intervals.len(), 2);
        assert!(intervals.iter().any(|interval| interval.vreg == slot));
    }

    #[test]
    fn test_allocate_fits_in_registers() {
        let func = make_add_function();
        let intervals = LinearScanAllocator::compute_intervals(&func);
        let regs = ["r12", "r13", "r14", "r15"];
        let alloc = LinearScanAllocator::allocate(&intervals, &regs);

        // With 3 vregs and 4 registers, no spills should occur.
        for interval in &intervals {
            let a = alloc
                .get(&interval.vreg)
                .expect("every vreg should be allocated");
            assert!(
                matches!(a, Allocation::Register(_)),
                "vreg {:?} should be in a register, got {:?}",
                interval.vreg,
                a
            );
        }
    }

    #[test]
    fn test_allocate_spills_when_registers_exhausted() {
        // Build a function with more vregs than available registers.
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut func = FunctionBuilder::new("spill_test").returns(i64_ty).build();
        let mut block = Block::new("entry");

        // 8 vregs all live simultaneously: v0+v1 → v2, v3+v4 → v5, v6+v7 → v8
        for i in 0u32..8 {
            let vi = Register::Virtual(VirtualReg::gpr(i));
            let vj = Register::Virtual(VirtualReg::gpr(i + 1));
            let vd = Register::Virtual(VirtualReg::gpr(i + 2));
            block.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64_ty,
                dst: vd,
                lhs: Operand::Register(vi),
                rhs: Operand::Register(vj),
            });
        }
        block.push(Instruction::Ret { value: None });
        func.add_block(block);

        let intervals = LinearScanAllocator::compute_intervals(&func);
        let regs = ["r12", "r13"]; // only 2 registers
        let alloc = LinearScanAllocator::allocate(&intervals, &regs);

        // Some vregs must be spilled since we have more live vregs than registers.
        let has_spill = alloc.values().any(|a| matches!(a, Allocation::Spill(_)));
        assert!(has_spill, "expected spills when registers are exhausted");

        // All spill offsets should be multiples of 8 and negative.
        for a in alloc.values() {
            if let Allocation::Spill(offset) = a {
                assert!(*offset < 0, "spill offset should be negative");
                assert_eq!(offset % 8, 0, "spill offset should be 8-byte aligned");
            }
        }
    }

    #[test]
    fn graph_color_fits_three_vregs_without_spill() {
        let func = make_add_function();
        let intervals = LinearScanAllocator::compute_intervals(&func);
        let regs = ["r12", "r13", "r14", "r15"];
        let gc = GraphColorAllocator::allocate(&intervals, &regs);
        for interval in &intervals {
            let a = gc.get(&interval.vreg).expect("allocated");
            assert!(
                matches!(a, Allocation::Register(_)),
                "graph color should keep simple add in registers, got {:?}",
                a
            );
        }
    }

    #[test]
    fn graph_color_spills_when_register_count_exhausted() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut func = FunctionBuilder::new("spill_gc").returns(i64_ty).build();
        let mut block = Block::new("entry");
        for i in 0u32..8 {
            let vi = Register::Virtual(VirtualReg::gpr(i));
            let vj = Register::Virtual(VirtualReg::gpr(i + 1));
            let vd = Register::Virtual(VirtualReg::gpr(i + 2));
            block.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64_ty,
                dst: vd,
                lhs: Operand::Register(vi),
                rhs: Operand::Register(vj),
            });
        }
        block.push(Instruction::Ret { value: None });
        func.add_block(block);

        let intervals = LinearScanAllocator::compute_intervals(&func);
        let regs = ["r12", "r13"];
        let gc = GraphColorAllocator::allocate(&intervals, &regs);
        let has_spill = gc.values().any(|a| matches!(a, Allocation::Spill(_)));
        assert!(
            has_spill,
            "graph color should spill when k=2 and pressure is high"
        );
    }
}
