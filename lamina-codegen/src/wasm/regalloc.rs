use lamina_mir::register::{Register, RegisterClass, VirtualReg};
use crate::regalloc::{
    PhysRegConvertible, PhysRegHandle, RegisterAllocator as MirRegisterAllocator,
};

/// WASM "register allocator" for stack-based virtual machine.
///
/// WASM doesn't use physical registers like x86_64, but we still need to track
/// virtual registers for stack operations. This is a simplified allocator
/// that maps virtual registers to stack positions.
pub struct WasmRegAlloc {
    vreg_to_stack: std::collections::HashMap<VirtualReg, u32>,
    next_stack_slot: u32,
}

impl Default for WasmRegAlloc {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmRegAlloc {
    pub fn new() -> Self {
        Self {
            vreg_to_stack: std::collections::HashMap::new(),
            next_stack_slot: 0,
        }
    }

    /// Get the stack position for a virtual register
    pub fn get_stack_position(&self, vreg: &VirtualReg) -> Option<u32> {
        self.vreg_to_stack.get(vreg).copied()
    }

    /// Allocate a stack position for a virtual register
    pub fn allocate_stack(&mut self, vreg: VirtualReg) -> u32 {
        if let Some(pos) = self.vreg_to_stack.get(&vreg) {
            *pos
        } else {
            let pos = self.next_stack_slot;
            self.vreg_to_stack.insert(vreg, pos);
            self.next_stack_slot += 1;
            pos
        }
    }
}

impl MirRegisterAllocator for WasmRegAlloc {
    type PhysReg = u32; // Stack position as "physical register"

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        // For WASM, we can always allocate more stack space
        let pos = self.next_stack_slot;
        self.next_stack_slot += 1;
        Some(pos)
    }

    fn free_scratch(&mut self, _phys: Self::PhysReg) {
        // WASM stack is managed differently, no explicit freeing needed
    }

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        self.vreg_to_stack.get(vreg).copied()
    }

    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg> {
        if vreg.class != RegisterClass::Gpr {
            return None;
        }
        Some(self.allocate_stack(vreg))
    }

    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg> {
        match reg {
            Register::Virtual(v) => self.vreg_to_stack.get(v).copied(),
            Register::Physical(_) => None, // WASM doesn't have physical registers
        }
    }

    fn occupy(&mut self, _phys: Self::PhysReg) {
        // No-op for WASM
    }

    fn release(&mut self, _phys: Self::PhysReg) {
        // No-op for WASM
    }

    fn is_occupied(&self, _phys: Self::PhysReg) -> bool {
        false // Stack positions are always available
    }
}

// RegisterAllocatorDyn is automatically implemented via the blanket impl in regalloc.rs

// Implement PhysRegConvertible for u32 (stack position)
impl PhysRegConvertible for u32 {
    fn into_handle(self) -> PhysRegHandle {
        PhysRegHandle::Named(format!("{}", self).leak())
    }

    fn from_handle(handle: PhysRegHandle) -> Option<Self> {
        match handle {
            PhysRegHandle::Named(name) => name.parse().ok(),
        }
    }
}
