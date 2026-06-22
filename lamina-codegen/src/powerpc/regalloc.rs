//! Simple PowerPC64 register allocator.
//!
//! Maps virtual registers to caller-saved GPRs in order.
//! When exhausted, the caller should spill to a stack slot instead.

use crate::regalloc::{Allocation, LocalRegisterAllocator as MirRegisterAllocator};
use lamina_mir::{Register, RegisterClass, VirtualReg};
use lamina_platform::TargetOperatingSystem;
use std::collections::HashMap;

/// Caller-saved general-purpose registers available for allocation.
///
/// We exclude r1 (sp), r2 (TOC), r13 (thread pointer on Linux).
/// We leave r3/r4 available since they are the primary return/arg pair
/// but start our scratch pool at r5 to allow arg passing to use r3-r10.
const AVAILABLE_GPRS: &[&str] = &["5", "6", "7", "8", "9", "10", "11", "12"];

pub struct Ppc64RegAlloc {
    #[allow(dead_code)]
    target_os: TargetOperatingSystem,
    allocated_gprs: HashMap<&'static str, VirtualReg>,
    stack_slots: HashMap<VirtualReg, i32>,
    next_stack_slot: i32,
}

impl Default for Ppc64RegAlloc {
    fn default() -> Self {
        Self::new(TargetOperatingSystem::Linux)
    }
}

impl Ppc64RegAlloc {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            target_os,
            allocated_gprs: HashMap::new(),
            stack_slots: HashMap::new(),
            next_stack_slot: -8,
        }
    }

    pub fn get_mapping_for(&self, vreg: &VirtualReg) -> Option<&'static str> {
        for (reg, allocated_vreg) in &self.allocated_gprs {
            if allocated_vreg == vreg {
                return Some(*reg);
            }
        }
        None
    }

    pub fn gpr_pool_for_global_allocation() -> Vec<&'static str> {
        AVAILABLE_GPRS.to_vec()
    }

    pub fn from_global_plan(
        target_os: TargetOperatingSystem,
        plan: &HashMap<VirtualReg, Allocation<&'static str>>,
    ) -> Self {
        let mut s = Self::new(target_os);
        for (&vreg, alloc) in plan {
            if vreg.class != RegisterClass::Gpr {
                continue;
            }
            if let Allocation::Register(phys) = alloc
                && AVAILABLE_GPRS.contains(phys)
            {
                s.allocated_gprs.insert(*phys, vreg);
            }
        }
        s
    }
}

impl MirRegisterAllocator for Ppc64RegAlloc {
    type PhysReg = &'static str;

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        for &reg in AVAILABLE_GPRS {
            if !self.allocated_gprs.contains_key(reg) {
                return Some(reg);
            }
        }
        AVAILABLE_GPRS.first().copied()
    }

    fn free_scratch(&mut self, phys: Self::PhysReg) {
        self.allocated_gprs.remove(phys);
    }

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        self.get_mapping_for(vreg)
    }

    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg> {
        if vreg.class != RegisterClass::Gpr {
            return None;
        }
        if let Some(phys) = self.get_mapping(&vreg) {
            return Some(phys);
        }
        if let Some(phys) = self.alloc_scratch() {
            self.allocated_gprs.insert(phys, vreg);
            return Some(phys);
        }
        let slot = self.next_stack_slot;
        self.stack_slots.insert(vreg, slot);
        self.next_stack_slot -= 8;
        None
    }

    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg> {
        match reg {
            Register::Virtual(v) => self.get_mapping(v),
            Register::Physical(p) => Some(p.name),
        }
    }

    fn occupy(&mut self, _phys: Self::PhysReg) {}

    fn release(&mut self, phys: Self::PhysReg) {
        self.allocated_gprs.remove(phys);
    }

    fn is_occupied(&self, phys: Self::PhysReg) -> bool {
        self.allocated_gprs.contains_key(phys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_gpr_sequential() {
        let mut ra = Ppc64RegAlloc::new(TargetOperatingSystem::Linux);
        let v0 = VirtualReg::gpr(0);
        let v1 = VirtualReg::gpr(1);
        let r0 = ra.ensure_mapping(v0).unwrap();
        let r1 = ra.ensure_mapping(v1).unwrap();
        assert_ne!(r0, r1);
    }

    #[test]
    fn test_allocate_same_vreg_stable() {
        let mut ra = Ppc64RegAlloc::new(TargetOperatingSystem::Linux);
        let v = VirtualReg::gpr(0);
        let r1 = ra.ensure_mapping(v);
        let r2 = ra.ensure_mapping(v);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_alloc_scratch_not_none() {
        let mut ra = Ppc64RegAlloc::new(TargetOperatingSystem::Linux);
        let s = ra.alloc_scratch();
        assert!(s.is_some());
    }

    #[test]
    fn test_release_allows_reuse() {
        let mut ra = Ppc64RegAlloc::new(TargetOperatingSystem::Linux);
        let s = ra.alloc_scratch().unwrap();
        ra.free_scratch(s);
        let s2 = ra.alloc_scratch();
        assert!(s2.is_some());
    }
}
