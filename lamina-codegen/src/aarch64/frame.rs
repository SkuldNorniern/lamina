//! Stack frame layout for AArch64 code generation.
//!
//! This module provides frame mapping for virtual registers to stack slots.

use lamina_mir::{Function, Register};
use std::collections::{HashMap, HashSet};

/// Maps virtual registers to stack slot offsets.
pub struct FrameMap {
    pub slots: HashMap<Register, i32>,
    pub frame_size: i32,
    /// Registers written by some instruction. A register that is only ever
    /// read is a stack-allocation placeholder: its slot *is* the storage, so
    /// its address is the value. Anything else holds a value that must be
    /// loaded before use.
    defined: HashSet<Register>,
}

impl FrameMap {
    /// Creates a frame map from a function, assigning stack slots to all virtual registers.
    pub fn from_function(f: &Function) -> Self {
        let mut regs: HashSet<Register> = HashSet::new();
        let mut defined: HashSet<Register> = HashSet::new();
        for p in &f.sig.params {
            regs.insert(p.reg.clone());
        }
        for b in &f.blocks {
            for ins in &b.instructions {
                if let Some(d) = ins.def_reg() {
                    regs.insert(d.clone());
                    defined.insert(d.clone());
                }
                for u in ins.use_regs() {
                    regs.insert(u.clone());
                }
            }
        }

        let mut reg_vec: Vec<Register> = regs.into_iter().collect();
        reg_vec.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

        let mut slots = HashMap::new();
        let mut offset: i32 = -8;
        for r in reg_vec {
            if let Register::Virtual(v) = &r {
                // A register with a reservation addresses the lowest byte of a
                // multi-slot block, so it sits at the deepest slot of that block.
                let extra = f.stack_reservations.get(v).copied().unwrap_or(0) as i32;
                offset -= 8 * extra;
                slots.insert(r.clone(), offset);
                offset -= 8;
            }
        }
        let mut frame_size = -offset - 8;
        if frame_size < 0 {
            frame_size = 0;
        }
        frame_size = (frame_size + 15) & !15;
        Self {
            slots,
            frame_size,
            defined,
        }
    }

    pub fn recompute_frame_size_from_slots(&mut self) {
        let mut min_off = 0i32;
        for &o in self.slots.values() {
            if o < min_off {
                min_off = o;
            }
        }
        // The deepest slot starts at `min_off`, so the frame must span all of
        // it; `-min_off - 8` left that slot below sp.
        let mut frame_size = (-min_off).max(0);
        frame_size = (frame_size + 15) & !15;
        self.frame_size = frame_size;
    }

    /// True when the register holds a value rather than being a stack
    /// allocation placeholder whose slot address is the value.
    pub fn holds_value(&self, r: &Register) -> bool {
        self.defined.contains(r)
    }

    /// Returns the stack slot offset for a register, if it has one.
    pub fn slot_of(&self, r: &Register) -> Option<i32> {
        self.slots.get(r).copied()
    }
}
