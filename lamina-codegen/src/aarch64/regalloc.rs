use std::collections::{HashMap, HashSet, VecDeque};

use crate::regalloc::{Allocation, LocalRegisterAllocator as MirRegisterAllocator};
use lamina_mir::{Register, RegisterClass, VirtualReg};

// AArch64 register allocator for MIR virtual registers.

/// AArch64 register allocator for MIR virtual registers.
///
/// Tracks a fixed pool of available caller-saved GPRs and gives
/// stable mappings from virtual registers to physical registers.
/// Returns None when no register is available (caller should spill to stack).
pub struct A64RegAlloc {
    free_gprs: VecDeque<&'static str>,
    used_gprs: HashSet<&'static str>,
    global_mappings: HashMap<VirtualReg, &'static str>,
    vreg_to_preg: HashMap<VirtualReg, &'static str>,
    scratch_free: VecDeque<&'static str>,
    scratch_used: HashSet<&'static str>,
}

const MAP_GPRS: &[&str] = &["x13", "x14", "x15"];

/// Callee-saved registers a global plan may draw on, once the prologue saves them.
///
/// x16 and x17 are the emitter's own scratch, x18 is the platform register on Darwin
/// and Windows, and x19 and x20 are the address emitter's last-resort temporaries.
const CALLEE_SAVED_GPRS: &[&str] = &["x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"];

const SCRATCH_GPRS: &[&str] = &["x9", "x10", "x11", "x12"];

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
            global_mappings: HashMap::new(),
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

    /// GPR candidates for global allocation. `callee_saved` widens the pool with
    /// x21-x28, which the prologue then has to save and the epilogue restore.
    pub fn gpr_pool_for_global_allocation(callee_saved: bool) -> Vec<&'static str> {
        let mut pool = MAP_GPRS.to_vec();
        if callee_saved {
            pool.extend_from_slice(CALLEE_SAVED_GPRS);
        }
        pool
    }

    /// The callee-saved registers a plan hands out, in register-number order.
    pub fn callee_saved_in_plan(
        plan: &HashMap<VirtualReg, Allocation<&'static str>>,
    ) -> Vec<&'static str> {
        CALLEE_SAVED_GPRS
            .iter()
            .copied()
            .filter(|reg| {
                plan.iter().any(|(vreg, alloc)| {
                    vreg.class == RegisterClass::Gpr
                        && matches!(alloc, Allocation::Register(phys) if phys == reg)
                })
            })
            .collect()
    }

    pub fn from_global_plan(plan: &HashMap<VirtualReg, Allocation<&'static str>>) -> Self {
        let mut s = Self::new();
        s.free_gprs.clear();
        for (&vreg, alloc) in plan {
            if vreg.class != RegisterClass::Gpr {
                continue;
            }
            if let Allocation::Register(phys) = alloc
                && (MAP_GPRS.contains(phys) || CALLEE_SAVED_GPRS.contains(phys))
            {
                s.global_mappings.insert(vreg, *phys);
                s.used_gprs.insert(*phys);
            }
        }
        s
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
        if let Some(&phys) = self.global_mappings.get(vreg) {
            return Some(phys);
        }
        self.vreg_to_preg.get(vreg).copied()
    }

    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg> {
        if let Some(phys) = MirRegisterAllocator::get_mapping(self, &vreg) {
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
            Register::Virtual(v) => MirRegisterAllocator::get_mapping(self, v),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_plan_mappings_survive_and_are_shared() {
        let first = VirtualReg::gpr(0);
        let second = VirtualReg::gpr(1);
        let third = VirtualReg::gpr(2);
        let plan = HashMap::from([
            (first, Allocation::Register("x21")),
            (second, Allocation::Register("x21")),
            (third, Allocation::Spill(-8)),
        ]);
        let mut allocator = A64RegAlloc::from_global_plan(&plan);
        assert_eq!(allocator.get_mapping_for(&first), Some("x21"));
        assert_eq!(allocator.get_mapping_for(&second), Some("x21"));
        assert_eq!(allocator.get_mapping_for(&third), None);
        assert_eq!(allocator.ensure_mapping(VirtualReg::gpr(9)), None);
        assert_eq!(allocator.get_mapping_for(&first), Some("x21"));
    }

    #[test]
    fn callee_saved_pool_is_opt_in() {
        let narrow = A64RegAlloc::gpr_pool_for_global_allocation(false);
        let wide = A64RegAlloc::gpr_pool_for_global_allocation(true);
        assert_eq!(narrow, vec!["x13", "x14", "x15"]);
        assert!(wide.starts_with(&["x13", "x14", "x15"]));
        assert_eq!(wide.len(), 11);
        for reserved in ["x16", "x17", "x18", "x19", "x20", "x29", "x30"] {
            assert!(
                !wide.contains(&reserved),
                "{reserved} must stay unallocated"
            );
        }
        for scratch in SCRATCH_GPRS {
            assert!(!wide.contains(scratch));
        }
    }

    #[test]
    fn callee_saved_in_plan_reports_register_order() {
        let plan = HashMap::from([
            (VirtualReg::gpr(0), Allocation::Register("x24")),
            (VirtualReg::gpr(1), Allocation::Register("x21")),
            (VirtualReg::gpr(2), Allocation::Register("x24")),
            (VirtualReg::gpr(3), Allocation::Register("x13")),
            (VirtualReg::gpr(4), Allocation::Spill(-16)),
        ]);
        assert_eq!(A64RegAlloc::callee_saved_in_plan(&plan), vec!["x21", "x24"]);
    }

    #[test]
    fn local_allocation_stays_on_caller_saved_registers() {
        let mut allocator = A64RegAlloc::new();
        let mut seen = Vec::new();
        for index in 0..3 {
            seen.push(allocator.ensure_mapping(VirtualReg::gpr(index)).unwrap());
        }
        assert_eq!(seen, vec!["x13", "x14", "x15"]);
    }
}
