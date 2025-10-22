use std::collections::{HashMap, HashSet};
use crate::mir::{Function, Register};

/// Simple frame map that assigns each virtual register a stack slot.
pub struct FrameMap {
    pub slots: HashMap<Register, i32>,
    pub frame_size: i32,
}

impl FrameMap {
    pub fn from_function(f: &Function) -> Self {
        let mut regs: HashSet<Register> = HashSet::new();
        for p in &f.sig.params {
            regs.insert(p.reg.clone());
        }
        for b in &f.blocks {
            for ins in &b.instructions {
                if let Some(d) = ins.def_reg() {
                    regs.insert(d.clone());
                }
                for u in ins.use_regs() {
                    regs.insert(u.clone());
                }
            }
        }

        let mut slots = HashMap::new();
        let mut offset: i32 = -8;
        for r in regs.into_iter() {
            if matches!(r, Register::Virtual(_)) {
                slots.insert(r, offset);
                offset -= 8;
            }
        }
        let mut frame_size = -offset - 8;
        if frame_size < 0 { frame_size = 0; }
        frame_size = (frame_size + 15) & !15;
        Self { slots, frame_size }
    }

    pub fn slot_of(&self, r: &Register) -> Option<i32> {
        self.slots.get(r).copied()
    }
}


