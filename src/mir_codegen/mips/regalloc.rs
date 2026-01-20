use std::collections::{HashMap, HashSet, VecDeque};

use crate::mir::register::{Register, RegisterClass, VirtualReg};
use crate::mir_codegen::regalloc::RegisterAllocator as MirRegisterAllocator;

/// Minimal MIPS register allocator used by the MIR code generator.
///
/// Strategy mirrors other backends:
/// - A small pool of mapping GPRs provides stable VirtualReg -> physical names
/// - A disjoint scratch pool is leased for short-lived temporaries
/// - No spilling is performed here; callers handle stack slots if `None` is returned
pub struct MipsRegAlloc {
    free_gprs: VecDeque<&'static str>,
    used_gprs: HashSet<&'static str>,
    vreg_to_preg: HashMap<VirtualReg, &'static str>,
    scratch_free: VecDeque<&'static str>,
    scratch_used: HashSet<&'static str>,
}

// Conservative mapping pool (mostly callee-saved under MIPS ABI): s0-s7, s8
// Avoid $zero, $at, arg regs ($a0-$a3), ret regs ($v0-$v1), $gp, $sp, $fp, $ra
const MAP_GPRS: &[&str] = &[
    "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7", "$s8",
];

// Dedicated scratch pool for ephemeral use during emission.
// Keep disjoint from MAP_GPRS. t0-t9 are caller-saved temporaries on MIPS.
const SCRATCH_GPRS: &[&str] = &[
    "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7", "$t8", "$t9",
];

impl Default for MipsRegAlloc {
    fn default() -> Self {
        Self::new()
    }
}

impl MipsRegAlloc {
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
    pub fn mapped_for_register(&self, r: &Register) -> Option<&'static str> {
        MirRegisterAllocator::mapped_for_register(self, r)
    }
}

impl MirRegisterAllocator for MipsRegAlloc {
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
        if self.used_gprs.insert(phys) {
            if let Some(pos) = self.free_gprs.iter().position(|&p| p == phys) {
                self.free_gprs.remove(pos);
            }
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
