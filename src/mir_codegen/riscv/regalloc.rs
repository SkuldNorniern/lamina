use crate::mir::register::{Register, RegisterClass, VirtualReg};
use crate::mir_codegen::regalloc::RegisterAllocator as MirRegisterAllocator;
use crate::target::TargetOperatingSystem;

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
/// For simplicity, we'll use a subset of these registers.
pub struct RiscVRegAlloc {
    target_os: TargetOperatingSystem,
    // Available general-purpose registers for allocation
    available_gprs: Vec<&'static str>,
    // Currently allocated registers
    allocated_gprs: std::collections::HashMap<&'static str, VirtualReg>,
    // Stack slot assignments for spilled registers
    stack_slots: std::collections::HashMap<VirtualReg, i32>,
    next_stack_slot: i32,
}

impl Default for RiscVRegAlloc {
    fn default() -> Self {
        Self::new(TargetOperatingSystem::Linux)
    }
}

impl RiscVRegAlloc {
    // RISC-V general-purpose registers available for allocation
    // We exclude: x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp), x8 (fp)
    const AVAILABLE_REGISTERS: &'static [&'static str] = &[
        "x5", "x6", "x7", // t0-t2
        "x9", "x10", "x11", "x12", "x13", "x14", "x15", // s1-s7
        "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", // a0-a7
        "x24", "x25", "x26", "x27", // t3-t6
        "x28", "x29", "x30", "x31", // t3-t6 (duplicate for simplicity)
    ];

    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            target_os,
            available_gprs: Self::AVAILABLE_REGISTERS.to_vec(),
            allocated_gprs: std::collections::HashMap::new(),
            stack_slots: std::collections::HashMap::new(),
            next_stack_slot: -8, // Start below the saved registers
        }
    }

    /// Set conservative mode (limit to fewer registers)
    pub fn set_conservative_mode(&mut self) {
        // Use only a subset of registers for conservative allocation
        self.available_gprs = vec![
            "x9", "x10", "x11", "x12", "x13", "x14", "x15", // s1-s7
            "x16", "x17", "x18", "x19", // a0-a3
        ];
    }

    /// Get stack slot for a virtual register
    pub fn get_stack_slot(&self, vreg: &VirtualReg) -> Option<i32> {
        self.stack_slots.get(vreg).copied()
    }
}

impl MirRegisterAllocator for RiscVRegAlloc {
    type PhysReg = &'static str;

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        // Try to allocate a register first
        for &reg in &self.available_gprs {
            if !self.allocated_gprs.contains_key(reg) {
                return Some(reg);
            }
        }
        // If no registers available, return the first one (will cause spilling)
        self.available_gprs.first().copied()
    }

    fn free_scratch(&mut self, phys: Self::PhysReg) {
        self.allocated_gprs.remove(phys);
    }

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        // Check if we have a direct register mapping
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

        // Check if already mapped
        if let Some(phys) = self.get_mapping(&vreg) {
            return Some(phys);
        }

        // Try to allocate a new register
        if let Some(phys) = self.alloc_scratch() {
            self.allocated_gprs.insert(phys, vreg);
            return Some(phys);
        }

        // No registers available, assign to stack
        let stack_slot = self.next_stack_slot;
        self.stack_slots.insert(vreg, stack_slot);
        self.next_stack_slot -= 8; // 8 bytes per slot
        None // Return None to indicate stack allocation
    }

    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg> {
        match reg {
            Register::Virtual(v) => self.get_mapping(v),
            Register::Physical(p) => Some(p.name),
        }
    }

    fn occupy(&mut self, _phys: Self::PhysReg) {
        // Mark register as occupied (though we don't use this in our simple allocator)
    }

    fn release(&mut self, phys: Self::PhysReg) {
        self.allocated_gprs.remove(phys);
    }

    fn is_occupied(&self, phys: Self::PhysReg) -> bool {
        self.allocated_gprs.contains_key(phys)
    }
}

// RegisterAllocatorDyn is automatically implemented via the blanket impl in regalloc.rs
