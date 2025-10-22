use std::collections::{HashMap, HashSet, VecDeque};

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

// Avoid x0..x7 (args/ret), x9..x12 (temporaries used by emitter), x29/x30 (fp/lr)
// Provide a pool that is unlikely to conflict with ABI conventions for leaf codegen
// Pool for mapping virtual registers (never used for scratch temps)
const MAP_GPRS: &[&str] = &[
    "x13", "x14", "x15",
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
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

    /// Allocate a scratch physical GPR not tied to any virtual register mapping.
    pub fn alloc_scratch(&mut self) -> Option<&'static str> {
        if let Some(phys) = self.scratch_free.pop_front() {
            self.scratch_used.insert(phys);
            return Some(phys);
        }
        None
    }

    /// Free a previously allocated scratch register.
    pub fn free_scratch(&mut self, phys: &'static str) {
        if self.scratch_used.remove(phys) {
            self.scratch_free.push_back(phys);
        }
    }

    /// Returns true if the physical register is currently occupied by any mapping.
    pub fn is_occupied(&self, phys: &str) -> bool {
        self.used_gprs.contains(phys)
    }

    /// Mark a physical register as occupied (removed from free list) if present.
    pub fn occupy(&mut self, phys: &'static str) {
        if !self.used_gprs.contains(phys) {
            self.used_gprs.insert(phys);
            if let Some(pos) = self.free_gprs.iter().position(|&p| p == phys) {
                self.free_gprs.remove(pos);
            }
        }
    }

    /// Release a physical register back to the free list (not used in the simple stable mapping).
    pub fn release(&mut self, phys: &'static str) {
        if self.used_gprs.remove(phys) {
            self.free_gprs.push_back(phys);
        }
    }

    /// Get an existing mapping for a virtual register if any.
    pub fn get_mapping_for(&self, v: &VirtualReg) -> Option<&'static str> {
        self.vreg_to_preg.get(v).copied()
    }

    /// Ensure a mapping for a GPR-class virtual register; returns the assigned physical register,
    /// or None if no registers remain (caller should spill to stack in that case).
    pub fn ensure_mapping_for_gpr(&mut self, v: VirtualReg) -> Option<&'static str> {
        if let Some(&p) = self.vreg_to_preg.get(&v) {
            return Some(p);
        }
        if v.class != RegisterClass::Gpr {
            return None;
        }
        if let Some(phys) = self.free_gprs.pop_front() {
            self.used_gprs.insert(phys);
            self.vreg_to_preg.insert(v, phys);
            Some(phys)
        } else {
            None
        }
    }

    /// Get the mapped physical register for any Register (virtual or physical).
    pub fn mapped_for_register(&self, r: &Register) -> Option<&'static str> {
        match r {
            Register::Virtual(v) => self.get_mapping_for(v),
            Register::Physical(p) => Some(p.name),
        }
    }
}


