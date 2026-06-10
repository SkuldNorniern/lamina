use std::collections::HashMap;

use crate::regalloc::{
    LocalRegisterAllocator as MirRegisterAllocator, PhysRegConvertible, PhysRegHandle,
};
use lamina_mir::{Register, RegisterClass, VirtualReg};

/// Newtype over a WASM stack slot index, used as the "physical register" for WASM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WasmStackSlot(pub u32);

impl PhysRegConvertible for WasmStackSlot {
    fn into_handle(self) -> PhysRegHandle {
        PhysRegHandle::Named(format!("wasm_slot_{}", self.0).leak())
    }

    fn from_handle(handle: PhysRegHandle) -> Option<Self> {
        match handle {
            PhysRegHandle::Named(name) => name
                .strip_prefix("wasm_slot_")
                .and_then(|n| n.parse().ok())
                .map(WasmStackSlot),
        }
    }
}

/// WASM "register allocator" for stack-based virtual machine.
pub struct WasmRegAlloc {
    vreg_to_stack: HashMap<VirtualReg, u32>,
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
            vreg_to_stack: HashMap::new(),
            next_stack_slot: 0,
        }
    }

    pub fn get_stack_position(&self, vreg: &VirtualReg) -> Option<u32> {
        self.vreg_to_stack.get(vreg).copied()
    }

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
    type PhysReg = WasmStackSlot;

    fn alloc_scratch(&mut self) -> Option<Self::PhysReg> {
        let pos = self.next_stack_slot;
        self.next_stack_slot += 1;
        Some(WasmStackSlot(pos))
    }

    fn free_scratch(&mut self, _phys: Self::PhysReg) {}

    fn get_mapping(&self, vreg: &VirtualReg) -> Option<Self::PhysReg> {
        self.vreg_to_stack.get(vreg).copied().map(WasmStackSlot)
    }

    fn ensure_mapping(&mut self, vreg: VirtualReg) -> Option<Self::PhysReg> {
        if vreg.class != RegisterClass::Gpr {
            return None;
        }
        Some(WasmStackSlot(self.allocate_stack(vreg)))
    }

    fn mapped_for_register(&self, reg: &Register) -> Option<Self::PhysReg> {
        match reg {
            Register::Virtual(v) => self.vreg_to_stack.get(v).copied().map(WasmStackSlot),
            Register::Physical(_) => None,
        }
    }

    fn occupy(&mut self, _phys: Self::PhysReg) {}

    fn release(&mut self, _phys: Self::PhysReg) {}

    fn is_occupied(&self, _phys: Self::PhysReg) -> bool {
        false
    }
}
