use crate::mir::{Register, VirtualReg};

/// Target-facing interface for MIR register allocation.
///
/// The trait stays purposefully small: code generators typically need a
/// lightweight scratch register pool, a stable mapping from virtual to
/// physical registers, and explicit hooks to reserve or release physical
/// registers that are pre-coloured by the ABI. Architecture backends can build
/// richer policies on top of this contract without forcing every target to
/// adopt the same strategy.
pub trait RegisterAllocator {
    /// Architecture-specific physical register handle.
    type PhysReg: Copy + Eq;

    /// Acquire a short-lived scratch register. Returns `None` when the dedicated
    /// pool is exhausted so the caller may spill or choose an alternate path.
    fn alloc_scratch(&mut self) -> Option<Self::PhysReg>;

    /// Release a scratch register obtained through [`alloc_scratch`].
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
