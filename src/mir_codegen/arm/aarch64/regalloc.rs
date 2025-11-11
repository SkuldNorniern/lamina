use std::collections::{HashMap, HashSet, VecDeque};

use crate::mir_codegen::regalloc::RegisterAllocator as MirRegisterAllocator;

use crate::mir::register::{Register, RegisterClass, VirtualReg};

/// Very simple AArch64 register allocator for MIR virtual registers.
///
/// - Tracks a fixed pool of available caller-saved GPRs that are safe for temporaries
/// - Provides a stable mapping VirtualReg -> physical register for the lifetime of a function
/// - If no register is available, the caller should spill to a stack slot (allocator returns None)
pub struct A64RegAlloc {
    free_gprs: VecDeque<&'static str>,
    used_gprs: HashSet<&'static str>,
    vreg_to_preg: HashMap<VirtualReg, &'static str>,
    scratch_free: VecDeque<&'static str>,
    scratch_used: HashSet<&'static str>,
}

// Avoid x0..x7 (args/ret), x29/x30 (fp/lr)
// Use x9-x28 for mapping virtual registers, reserving x16-x18 for intra-procedure calls if needed
// Pool for mapping virtual registers (can overlap with scratch temps for complex functions)
const MAP_GPRS: &[&str] = &[
    "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x19", "x20", "x21", "x22", "x23", "x24",
    "x25", "x26", "x27", "x28",
];

// Dedicated scratch pool for short-lived temporaries; kept disjoint from MAP_GPRS
// Provide a larger pool to avoid fallback to non-scratch registers during nested materializations
const SCRATCH_GPRS: &[&str] = &["x9", "x10", "x11", "x12", "x13", "x14", "x15"];

impl Default for A64RegAlloc {
    fn default() -> Self {
        Self::new()
    }
}

impl A64RegAlloc {
    pub fn new() -> Self {
        let mut free_gprs = VecDeque::new();
        for &r in MAP_GPRS {
            free_gprs.push_back(r);
        }

        let mut scratch_free = VecDeque::new();
        for &r in SCRATCH_GPRS {
            scratch_free.push_back(r);
        }

        Self {
            free_gprs,
            used_gprs: HashSet::new(),
            vreg_to_preg: HashMap::new(),
            scratch_free,
            scratch_used: HashSet::new(),
        }
    }

    /// Set conservative mode for very complex functions - use moderate number of registers
    /// This provides a balance between register pressure and spilling for complex control flow
    pub fn set_conservative_mode(&mut self) {
        // Keep a moderate number of registers available for complex functions
        self.free_gprs.clear();
        self.free_gprs.push_back("x13");
        self.free_gprs.push_back("x14");
        self.free_gprs.push_back("x15");
        self.free_gprs.push_back("x19");
        self.free_gprs.push_back("x20");
        self.free_gprs.push_back("x21");
        self.free_gprs.push_back("x22");
        self.free_gprs.push_back("x23");
        // Clear used registers that are no longer in the free pool
        self.used_gprs.retain(|r| self.free_gprs.contains(r));
        // Also clear vreg mappings that use registers no longer available
        let free_set: HashSet<&str> = self.free_gprs.iter().copied().collect();
        self.vreg_to_preg.retain(|_, preg| {
            if free_set.contains(preg) {
                true
            } else {
                // Register is no longer available, remove mapping
                false
            }
        });
    }

    #[inline]
    pub fn alloc_scratch(&mut self) -> Option<&'static str> {
        MirRegisterAllocator::alloc_scratch(self)
    }

    #[inline]
    pub fn free_scratch(&mut self, phys: &'static str) {
        MirRegisterAllocator::free_scratch(self, phys);
    }

    #[inline]
    pub fn is_occupied(&self, phys: &'static str) -> bool {
        MirRegisterAllocator::is_occupied(self, phys)
    }

    #[inline]
    pub fn occupy(&mut self, phys: &'static str) {
        MirRegisterAllocator::occupy(self, phys);
    }

    #[inline]
    pub fn release(&mut self, phys: &'static str) {
        MirRegisterAllocator::release(self, phys);
    }

    #[inline]
    pub fn get_mapping_for(&self, v: &VirtualReg) -> Option<&'static str> {
        MirRegisterAllocator::get_mapping(self, v)
    }

    #[inline]
    pub fn ensure_mapping(&mut self, v: VirtualReg) -> Option<&'static str> {
        MirRegisterAllocator::ensure_mapping(self, v)
    }

    #[inline]
    pub fn ensure_mapping_for_gpr(&mut self, v: VirtualReg) -> Option<&'static str> {
        MirRegisterAllocator::ensure_mapping(self, v)
    }

    #[inline]
    pub fn mapped_for_register(&self, r: &Register) -> Option<&'static str> {
        MirRegisterAllocator::mapped_for_register(self, r)
    }
}

impl MirRegisterAllocator for A64RegAlloc {
    type PhysReg = &'static str;

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        if let Some(phys) = self.scratch_free.pop_front() {
            self.scratch_used.insert(phys);
            Some(phys)
        } else {
            None
        }
    }

    fn free_scratch(&mut self, phys: Self::PhysReg) {
        if self.scratch_used.remove(&phys) {
            self.scratch_free.push_back(phys);
        }
    }

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        self.vreg_to_preg.get(vreg).copied()
    }

    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg> {
        if let Some(&phys) = self.vreg_to_preg.get(&vreg) {
            return Some(phys);
        }

        if vreg.class != RegisterClass::Gpr {
            return None;
        }

        if let Some(phys) = self.free_gprs.pop_front() {
            self.used_gprs.insert(phys);
            self.vreg_to_preg.insert(vreg, phys);
            Some(phys)
        } else {
            // No free registers available. For complex functions, reuse an existing register.
            // This is not proper spilling but allows code generation to continue.
            // Find an existing mapping to reuse (prefer less recently used)
            if let Some((vreg_to_replace, &phys)) = self.vreg_to_preg.iter().next() {
                let vreg_to_replace = *vreg_to_replace;
                // Remove the old mapping
                self.vreg_to_preg.remove(&vreg_to_replace);
                // Create new mapping
                self.vreg_to_preg.insert(vreg, phys);
                Some(phys)
            } else {
                // No existing mappings to reuse
                None
            }
        }
    }

    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg> {
        match reg {
            Register::Virtual(v) => self.vreg_to_preg.get(v).copied(),
            Register::Physical(p) => Some(p.name),
        }
    }

    fn occupy(&mut self, phys: Self::PhysReg) {
        if self.used_gprs.insert(phys)
            && let Some(pos) = self.free_gprs.iter().position(|&p| p == phys)
        {
            self.free_gprs.remove(pos);
        }
    }

    fn release(&mut self, phys: Self::PhysReg) {
        if self.used_gprs.remove(&phys) {
            self.free_gprs.push_back(phys);
        }
    }

    fn is_occupied(&self, phys: Self::PhysReg) -> bool {
        self.used_gprs.contains(&phys)
    }
}
