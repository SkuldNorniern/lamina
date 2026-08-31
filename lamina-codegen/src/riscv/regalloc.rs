use std::collections::HashMap;

use crate::regalloc::{Allocation, LocalRegisterAllocator as MirRegisterAllocator};
use crate::riscv::frame::RiscVFrame;
use crate::riscv::target::{RiscVTarget, Xlen};
use lamina_mir::{Register, RegisterClass, VirtualReg};
use lamina_platform::TargetOperatingSystem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiscVRegisterConvention {
    CallerSaved,
    CalleeSaved,
    Reserved,
}

/// RISC-V register allocator with platform-aware register selection.
///
/// RISC-V has 32 general-purpose registers (x0-x31):
/// - x0: zero (hardwired zero)
/// - x1: ra (return address)
/// - x2: sp (stack pointer)
/// - x3: gp (global pointer)
/// - x4: tp (thread pointer)
/// - x5-x7, x28-x31: temporaries
/// - x8: fp/s0 (frame pointer/saved register)
/// - x9-x15: s1-s7 (saved registers)
/// - x16-x27: a0-a7 (argument registers), t0-t6 (temporaries)
///
pub struct RiscVRegAlloc {
    target: RiscVTarget,
    #[allow(dead_code)]
    target_os: TargetOperatingSystem,
    available_gprs: Vec<&'static str>,
    allocated_gprs: HashMap<&'static str, VirtualReg>,
    stack_slots: HashMap<VirtualReg, i32>,
    next_stack_slot: i32,
}

impl Default for RiscVRegAlloc {
    fn default() -> Self {
        Self::new(TargetOperatingSystem::Linux)
    }
}

impl RiscVRegAlloc {
    pub const CALLER_SAVED_REGISTERS: &'static [&'static str] = &[
        "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    ];
    pub const CALLEE_SAVED_REGISTERS: &'static [&'static str] = &[
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    ];
    pub const RESERVED_REGISTERS: &'static [&'static str] = &["zero", "ra", "sp", "gp", "tp"];

    /// Registers this allocator may hand out, by ABI name.
    ///
    /// ABI names, not `x` numbers, because the emitter writes `a0`, `a1` and `t0`
    /// directly. While this list said `x10` and the emitter said `a0`, they were the
    /// same register under two spellings and nothing could see the clash.
    ///
    /// Excluded and why:
    ///   a0-a7 (x10-x17)  arguments and return values, and the emitter names a0/a1
    ///   t0     (x5)      the emitter's own scratch
    ///   s0/fp  (x8)      the frame pointer
    ///   ra, sp, gp, tp, zero     reserved
    const AVAILABLE_REGISTERS: &'static [&'static str] = &[
        "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
        "s10", "s11",
    ];

    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self::with_target(target_os, RiscVTarget::general(Xlen::Rv64))
    }

    /// The ISA this allocator is emitting for. Held here because every helper that needs
    /// the word size already receives the allocator.
    pub fn target(&self) -> RiscVTarget {
        self.target
    }

    pub fn register_convention(reg: &str) -> Option<RiscVRegisterConvention> {
        if Self::CALLER_SAVED_REGISTERS.contains(&reg) {
            Some(RiscVRegisterConvention::CallerSaved)
        } else if Self::CALLEE_SAVED_REGISTERS.contains(&reg) || reg == "fp" {
            Some(RiscVRegisterConvention::CalleeSaved)
        } else if Self::RESERVED_REGISTERS.contains(&reg) {
            Some(RiscVRegisterConvention::Reserved)
        } else {
            None
        }
    }

    pub fn caller_saved_mappings(
        &self,
        live_vregs: &[VirtualReg],
    ) -> Vec<(VirtualReg, &'static str)> {
        let mut mappings: Vec<_> = live_vregs
            .iter()
            .filter_map(|vreg| {
                let physical = self.get_mapping(vreg)?;
                (Self::register_convention(physical) == Some(RiscVRegisterConvention::CallerSaved))
                    .then_some((*vreg, physical))
            })
            .collect();
        mappings.sort_unstable_by_key(|(_, physical)| *physical);
        mappings
    }

    pub fn used_callee_saved_registers(&self) -> Vec<&'static str> {
        let mut registers: Vec<_> = self
            .allocated_gprs
            .keys()
            .copied()
            .filter(|reg| {
                Self::register_convention(reg) == Some(RiscVRegisterConvention::CalleeSaved)
            })
            .collect();
        registers.sort_unstable_by_key(|reg| {
            Self::CALLEE_SAVED_REGISTERS
                .iter()
                .position(|candidate| candidate == reg)
        });
        registers
    }

    pub fn with_target(target_os: TargetOperatingSystem, target: RiscVTarget) -> Self {
        Self {
            target,
            target_os,
            available_gprs: Self::AVAILABLE_REGISTERS.to_vec(),
            allocated_gprs: HashMap::new(),
            stack_slots: HashMap::new(),
            next_stack_slot: RiscVFrame::calculate_stack_offset(0, target),
        }
    }

    /// Set conservative mode (limit to fewer registers)
    pub fn set_conservative_mode(&mut self) {
        self.available_gprs = vec!["t1", "t2"];
    }

    /// Get stack slot for a virtual register
    pub fn get_stack_slot(&self, vreg: &VirtualReg) -> Option<i32> {
        self.stack_slots.get(vreg).copied()
    }

    pub fn gpr_pool_for_global_allocation() -> Vec<&'static str> {
        Self::AVAILABLE_REGISTERS.to_vec()
    }

    pub fn from_global_plan(
        target_os: TargetOperatingSystem,
        target: RiscVTarget,
        plan: &HashMap<VirtualReg, Allocation<&'static str>>,
        stack_slots: &HashMap<VirtualReg, i32>,
    ) -> Self {
        let mut s = Self::with_target(target_os, target);
        let mut min_spill = 0i32;
        for (&vreg, alloc) in plan {
            if vreg.class != RegisterClass::Gpr {
                continue;
            }
            match alloc {
                Allocation::Register(phys) => {
                    if s.available_gprs.contains(phys) {
                        s.allocated_gprs.insert(*phys, vreg);
                    }
                }
                Allocation::Spill(_) => {
                    if let Some(off) = stack_slots.get(&vreg) {
                        s.stack_slots.insert(vreg, *off);
                        if *off < min_spill {
                            min_spill = *off;
                        }
                    }
                }
            }
        }
        let word_bytes = target.word_bytes();
        s.next_stack_slot = if min_spill == 0 {
            RiscVFrame::calculate_stack_offset(0, target)
        } else {
            min_spill - word_bytes
        };
        s
    }
}

impl MirRegisterAllocator for RiscVRegAlloc {
    type PhysReg = &'static str;

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        self.available_gprs
            .iter()
            .copied()
            .find(|reg| !self.allocated_gprs.contains_key(reg))
    }

    fn free_scratch(&mut self, phys: Self::PhysReg) {
        self.allocated_gprs.remove(phys);
    }

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        for (reg, allocated_vreg) in &self.allocated_gprs {
            if allocated_vreg == vreg {
                return Some(*reg);
            }
        }
        None
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

        let stack_slot = self.next_stack_slot;
        self.stack_slots.insert(vreg, stack_slot);
        self.next_stack_slot -= self.target.word_bytes();
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

    /// Registers the RISC-V emitter writes by name, plus those the ABI reserves.
    const OFF_LIMITS: &[&str] = &[
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "ra", "sp", "gp", "tp", "zero", "fp",
    ];

    #[test]
    fn pool_avoids_registers_the_emitter_uses_by_name() {
        for reg in RiscVRegAlloc::AVAILABLE_REGISTERS {
            assert!(
                !OFF_LIMITS.contains(reg),
                "{reg} is in the pool but the emitter or the ABI owns it"
            );
        }
        let mut conservative = RiscVRegAlloc::default();
        conservative.set_conservative_mode();
        for reg in &conservative.available_gprs {
            assert!(!OFF_LIMITS.contains(reg), "{reg} in conservative pool");
        }
    }

    #[test]
    fn pool_is_written_in_abi_names() {
        for reg in RiscVRegAlloc::AVAILABLE_REGISTERS {
            assert!(
                !reg.starts_with('x'),
                "{reg} is an x-number; the emitter uses ABI names"
            );
        }
    }

    #[test]
    fn abi_register_file_is_fully_classified() {
        let mut classified: Vec<&str> = Vec::new();
        classified.extend(RiscVRegAlloc::CALLER_SAVED_REGISTERS.iter().copied());
        classified.extend(RiscVRegAlloc::CALLEE_SAVED_REGISTERS.iter().copied());
        classified.extend(RiscVRegAlloc::RESERVED_REGISTERS.iter().copied());
        classified.sort_unstable();
        classified.dedup();
        assert_eq!(classified.len(), 32);
        assert_eq!(
            RiscVRegAlloc::register_convention("t3"),
            Some(RiscVRegisterConvention::CallerSaved)
        );
        assert_eq!(
            RiscVRegAlloc::register_convention("s7"),
            Some(RiscVRegisterConvention::CalleeSaved)
        );
        assert_eq!(
            RiscVRegAlloc::register_convention("ra"),
            Some(RiscVRegisterConvention::Reserved)
        );
    }

    #[test]
    fn global_pool_has_caller_and_callee_saved_registers() {
        assert!(RiscVRegAlloc::AVAILABLE_REGISTERS.iter().any(|reg| {
            RiscVRegAlloc::register_convention(reg) == Some(RiscVRegisterConvention::CallerSaved)
        }));
        assert!(RiscVRegAlloc::AVAILABLE_REGISTERS.iter().any(|reg| {
            RiscVRegAlloc::register_convention(reg) == Some(RiscVRegisterConvention::CalleeSaved)
        }));
    }

    #[test]
    fn rv32_spills_use_word_slots_below_the_saved_pair() {
        let target = RiscVTarget::general(Xlen::Rv32);
        let mut allocator = RiscVRegAlloc::with_target(TargetOperatingSystem::Linux, target);
        for id in 0..RiscVRegAlloc::AVAILABLE_REGISTERS.len() {
            assert!(
                allocator
                    .ensure_mapping(VirtualReg::gpr(id as u32))
                    .is_some()
            );
        }
        let first_spill = VirtualReg::gpr(RiscVRegAlloc::AVAILABLE_REGISTERS.len() as u32);
        let second_spill = VirtualReg::gpr(first_spill.id + 1);
        assert!(allocator.ensure_mapping(first_spill).is_none());
        assert!(allocator.ensure_mapping(second_spill).is_none());
        assert_eq!(allocator.get_stack_slot(&first_spill), Some(-20));
        assert_eq!(allocator.get_stack_slot(&second_spill), Some(-24));
    }

    #[test]
    fn exhausted_pool_reports_none_rather_than_a_live_register() {
        let mut alloc = RiscVRegAlloc::default();
        let total = alloc.available_gprs.len();
        let mut handed_out = Vec::new();
        for i in 0..total {
            let phys = alloc.alloc_scratch().expect("pool should still have room");
            assert!(!handed_out.contains(&phys), "{phys} handed out twice");
            alloc.allocated_gprs.insert(phys, VirtualReg::gpr(i as u32));
            handed_out.push(phys);
        }
        assert_eq!(alloc.alloc_scratch(), None);
    }
}
