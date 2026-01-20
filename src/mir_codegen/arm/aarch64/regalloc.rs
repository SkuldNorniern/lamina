use std::collections::{HashMap, HashSet, VecDeque};

use crate::mir::{Register, RegisterClass, VirtualReg};
use crate::mir_codegen::regalloc::RegisterAllocator as MirRegisterAllocator;

// AArch64 register allocator for MIR virtual registers.

/// AArch64 register allocator for MIR virtual registers.
///
/// Tracks a fixed pool of available caller-saved GPRs and provides
/// stable mappings from virtual registers to physical registers.
/// Returns None when no register is available (caller should spill to stack).
pub struct A64RegAlloc {
    free_gprs: VecDeque<&'static str>,
    used_gprs: HashSet<&'static str>,
    vreg_to_preg: HashMap<VirtualReg, &'static str>,
    scratch_free: VecDeque<&'static str>,
    scratch_used: HashSet<&'static str>,
}

// Only use caller-saved registers (x9-x15) until we implement callee-saved save/restore
// x19-x28 are callee-saved and must be saved/restored in prologue/epilogue if used
const MAP_GPRS: &[&str] = &["x9", "x10", "x11", "x12", "x13", "x14", "x15"];

const SCRATCH_GPRS: &[&str] = &["x9", "x10", "x11", "x12", "x13", "x14", "x15"];

impl Default for A64RegAlloc {
    fn default() -> Self {
        Self::new()
    }
}

impl A64RegAlloc {
    /// Creates a new register allocator with default register pools.
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

    /// Sets conservative mode for complex functions, using fewer registers to reduce pressure.
    pub fn set_conservative_mode(&mut self) {
        self.free_gprs.clear();
        self.free_gprs.push_back("x13");
        self.free_gprs.push_back("x14");
        self.free_gprs.push_back("x15");
        // Removed x19-x23 - these are callee-saved and need prologue/epilogue save/restore
        self.used_gprs.retain(|r| self.free_gprs.contains(r));
        let free_set: HashSet<&str> = self.free_gprs.iter().copied().collect();
        self.vreg_to_preg.retain(|_, preg| free_set.contains(preg));
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
        } else if let Some((vreg_to_replace, &phys)) = self.vreg_to_preg.iter().next() {
            let vreg_to_replace = *vreg_to_replace;
            self.vreg_to_preg.remove(&vreg_to_replace);
            self.vreg_to_preg.insert(vreg, phys);
            Some(phys)
        } else {
            None
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
