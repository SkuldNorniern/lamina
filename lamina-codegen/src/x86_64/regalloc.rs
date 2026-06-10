//! x86_64 register allocator with platform-aware register selection.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::regalloc::{Allocation, LocalRegisterAllocator as MirRegisterAllocator};
use lamina_mir::{Register, RegisterClass, VirtualReg};
use lamina_platform::TargetOperatingSystem;

/// x86_64 register allocator supporting System V AMD64 and Microsoft x64 ABIs.
///
/// Uses platform-appropriate GPR pools for stable virtual-to-physical register mappings
/// and separate scratch pools for short-lived temporaries.
pub struct X64RegAlloc {
    target_os: TargetOperatingSystem,
    free_gprs: VecDeque<&'static str>,
    used_gprs: HashSet<&'static str>,
    vreg_to_preg: HashMap<VirtualReg, &'static str>,
    scratch_free: VecDeque<&'static str>,
    scratch_used: HashSet<&'static str>,
}

const SYSV_MAP_GPRS: &[&str] = &["r12", "r13", "r14", "r15", "rbx"];
const SYSV_LEAF_MAP_GPRS: &[&str] = &["r12", "r13", "r14", "r15", "rbx", "r10", "r11"];
const SYSV_SCRATCH_GPRS: &[&str] = &["r10", "r11"]; // keep as-is for non-leaf
const SYSV_LEAF_SCRATCH_GPRS: &[&str] = &["rcx", "rdx"]; // for leaf functions

const WIN64_MAP_GPRS: &[&str] = &["rbx", "rsi", "rdi", "r12", "r13", "r14", "r15"];
const WIN64_LEAF_MAP_GPRS: &[&str] = &[
    "rbx", "rsi", "rdi", "r12", "r13", "r14", "r15", "r10", "r11",
];
const WIN64_SCRATCH_GPRS: &[&str] = &["r10", "r11"];
const WIN64_LEAF_SCRATCH_GPRS: &[&str] = &["rcx", "rdx"];

impl Default for X64RegAlloc {
    fn default() -> Self {
        Self::new_default()
    }
}

impl X64RegAlloc {
    /// Creates a new register allocator for the specified target OS.
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self::with_target_os(target_os)
    }

    /// Creates a new register allocator with default target OS (Linux).
    pub fn new_default() -> Self {
        Self::with_target_os(TargetOperatingSystem::Linux)
    }

    fn with_target_os(target_os: TargetOperatingSystem) -> Self {
        let (map_gprs, scratch_gprs) = match target_os {
            TargetOperatingSystem::Windows => (WIN64_MAP_GPRS, WIN64_SCRATCH_GPRS),
            _ => (SYSV_MAP_GPRS, SYSV_SCRATCH_GPRS),
        };

        let mut free_gprs = VecDeque::new();
        for &r in map_gprs {
            free_gprs.push_back(r);
        }

        let mut scratch_free = VecDeque::new();
        for &r in scratch_gprs {
            scratch_free.push_back(r);
        }

        Self {
            target_os,
            free_gprs,
            used_gprs: HashSet::new(),
            vreg_to_preg: HashMap::new(),
            scratch_free,
            scratch_used: HashSet::new(),
        }
    }

    /// Creates a new register allocator for leaf functions (no calls), expanding the available
    /// register pool by including r10/r11 as mappable GPRs since they won't be clobbered.
    pub fn new_leaf(target_os: TargetOperatingSystem) -> Self {
        let (map_gprs, scratch_gprs) = match target_os {
            TargetOperatingSystem::Windows => (WIN64_LEAF_MAP_GPRS, WIN64_LEAF_SCRATCH_GPRS),
            _ => (SYSV_LEAF_MAP_GPRS, SYSV_LEAF_SCRATCH_GPRS),
        };
        let mut free_gprs = VecDeque::new();
        for &r in map_gprs {
            free_gprs.push_back(r);
        }
        let mut scratch_free = VecDeque::new();
        for &r in scratch_gprs {
            scratch_free.push_back(r);
        }
        Self {
            target_os,
            free_gprs,
            used_gprs: HashSet::new(),
            vreg_to_preg: HashMap::new(),
            scratch_free,
            scratch_used: HashSet::new(),
        }
    }

    /// GPRs available for global allocation, in the same order as the initial `free_gprs` deque.
    pub fn gpr_pool_for_global_allocation(
        target_os: TargetOperatingSystem,
        leaf: bool,
    ) -> Vec<&'static str> {
        let map_gprs = match (target_os, leaf) {
            (TargetOperatingSystem::Windows, true) => WIN64_LEAF_MAP_GPRS,
            (TargetOperatingSystem::Windows, false) => WIN64_MAP_GPRS,
            (_, true) => SYSV_LEAF_MAP_GPRS,
            (_, false) => SYSV_MAP_GPRS,
        };
        map_gprs.to_vec()
    }

    /// Rebuild allocator state from a precomputed GPR plan (global linear scan or graph coloring).
    pub fn from_global_plan(
        target_os: TargetOperatingSystem,
        leaf: bool,
        plan: &HashMap<VirtualReg, Allocation<&'static str>>,
    ) -> Self {
        let mut s = if leaf {
            Self::new_leaf(target_os)
        } else {
            Self::new(target_os)
        };
        for (&vreg, alloc) in plan {
            if vreg.class != RegisterClass::Gpr {
                continue;
            }
            if let Allocation::Register(phys) = alloc
                && s.vreg_to_preg.insert(vreg, *phys).is_none()
            {
                s.used_gprs.insert(*phys);
                if let Some(pos) = s.free_gprs.iter().position(|&p| p == *phys) {
                    let _ = s.free_gprs.remove(pos);
                }
            }
        }
        s
    }

    /// Sets conservative mode for complex functions, using fewer registers to reduce pressure.
    pub fn set_conservative_mode(&mut self) {
        self.free_gprs.clear();

        match self.target_os {
            TargetOperatingSystem::Windows => {
                self.free_gprs.push_back("rbx");
                self.free_gprs.push_back("rsi");
                self.free_gprs.push_back("r12");
                self.free_gprs.push_back("r13");
                self.free_gprs.push_back("r14");
            }
            _ => {
                self.free_gprs.push_back("r12");
                self.free_gprs.push_back("r13");
                self.free_gprs.push_back("r14");
                self.free_gprs.push_back("rbx");
            }
        }

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

impl MirRegisterAllocator for X64RegAlloc {
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

#[cfg(test)]
mod tests {
    use super::*;
    use lamina_platform::TargetOperatingSystem;

    #[test]
    fn new_linux_has_sysv_pool() {
        let ra = X64RegAlloc::new(TargetOperatingSystem::Linux);
        let pool = X64RegAlloc::gpr_pool_for_global_allocation(TargetOperatingSystem::Linux, false);
        assert_eq!(ra.free_gprs.len(), pool.len());
        for r in &pool {
            assert!(ra.free_gprs.contains(r));
        }
    }

    #[test]
    fn new_windows_has_win64_pool() {
        let ra = X64RegAlloc::new(TargetOperatingSystem::Windows);
        let pool =
            X64RegAlloc::gpr_pool_for_global_allocation(TargetOperatingSystem::Windows, false);
        assert_eq!(ra.free_gprs.len(), pool.len());
    }

    #[test]
    fn ensure_mapping_returns_same_reg_for_same_vreg() {
        let mut ra = X64RegAlloc::new_default();
        let vreg = VirtualReg::gpr(0);
        let first = ra.ensure_mapping(vreg);
        let second = ra.ensure_mapping(vreg);
        assert!(first.is_some());
        assert_eq!(first, second);
    }

    #[test]
    fn alloc_and_free_scratch_cycle() {
        let mut ra = X64RegAlloc::new_default();
        let scratch = ra.alloc_scratch();
        assert!(
            scratch.is_some(),
            "scratch should be available on fresh allocator"
        );
        let phys = scratch.unwrap();
        assert!(ra.scratch_used.contains(phys));
        ra.free_scratch(phys);
        assert!(!ra.scratch_used.contains(phys));
        assert!(ra.scratch_free.contains(&phys));
    }

    #[test]
    fn conservative_mode_shrinks_pool() {
        let mut ra = X64RegAlloc::new_default();
        let full_size = ra.free_gprs.len();
        ra.set_conservative_mode();
        assert!(ra.free_gprs.len() < full_size);
    }

    #[test]
    fn leaf_mode_expands_pool_vs_non_leaf() {
        let non_leaf =
            X64RegAlloc::gpr_pool_for_global_allocation(TargetOperatingSystem::Linux, false);
        let leaf = X64RegAlloc::gpr_pool_for_global_allocation(TargetOperatingSystem::Linux, true);
        assert!(leaf.len() > non_leaf.len());
    }

    #[test]
    fn is_occupied_tracks_occupy_and_release() {
        let mut ra = X64RegAlloc::new_default();
        let phys = "rbx";
        assert!(!ra.is_occupied(phys));
        ra.occupy(phys);
        assert!(ra.is_occupied(phys));
        ra.release(phys);
        assert!(!ra.is_occupied(phys));
    }

    #[test]
    fn non_gpr_vreg_returns_none() {
        use lamina_mir::RegisterClass;
        let mut ra = X64RegAlloc::new_default();
        let fp_vreg = VirtualReg {
            id: 99,
            class: RegisterClass::Fpr,
        };
        assert!(ra.ensure_mapping(fp_vreg).is_none());
    }
}
