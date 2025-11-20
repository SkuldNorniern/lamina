use crate::mir::{Register, VirtualReg};

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
    type PhysReg: PhysRegConvertible;

    /// Acquire a short-lived scratch register. Returns `None` when the dedicated
    /// pool is exhausted so the caller may spill or choose an alternate path.
    fn alloc_scratch(&mut self) -> Option<Self::PhysReg>;

    /// Release a scratch register obtained through [`RegisterAllocator::alloc_scratch`].
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

/// Object-safe wrapper around [`RegisterAllocator`] permitting dynamic dispatch
/// via `dyn RegisterAllocatorDyn`.
pub trait RegisterAllocatorDyn {
    fn alloc_scratch_dyn(&mut self) -> Option<PhysRegHandle>;
    fn free_scratch_dyn(&mut self, phys: PhysRegHandle);
    fn get_mapping_dyn(&self, vreg: &VirtualReg) -> Option<PhysRegHandle>;
    fn ensure_mapping_dyn(&mut self, vreg: VirtualReg) -> Option<PhysRegHandle>;
    fn mapped_for_register_dyn(&self, reg: &Register) -> Option<PhysRegHandle>;
    fn occupy_dyn(&mut self, phys: PhysRegHandle);
    fn release_dyn(&mut self, phys: PhysRegHandle);
    fn is_occupied_dyn(&self, phys: PhysRegHandle) -> bool;
}

impl<T> RegisterAllocatorDyn for T
where
    T: RegisterAllocator,
{
    fn alloc_scratch_dyn(&mut self) -> Option<PhysRegHandle> {
        self.alloc_scratch().map(|reg| reg.into_handle())
    }

    fn free_scratch_dyn(&mut self, phys: PhysRegHandle) {
        if let Some(reg) = <T::PhysReg as PhysRegConvertible>::from_handle(phys) {
            self.free_scratch(reg);
        } else {
            debug_assert!(false, "failed to decode physical register handle");
        }
    }

    fn get_mapping_dyn(&self, vreg: &VirtualReg) -> Option<PhysRegHandle> {
        self.get_mapping(vreg).map(|reg| reg.into_handle())
    }

    fn ensure_mapping_dyn(&mut self, vreg: VirtualReg) -> Option<PhysRegHandle> {
        self.ensure_mapping(vreg).map(|reg| reg.into_handle())
    }

    fn mapped_for_register_dyn(&self, reg: &Register) -> Option<PhysRegHandle> {
        self.mapped_for_register(reg).map(|r| r.into_handle())
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
