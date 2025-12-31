//! Register management for LUMIR.
//!
//! This module defines the register representation used in LUMIR, including
//! virtual registers (pre-allocation) and physical registers (post-allocation).
//!
//! ## Register Classes
//!
//! - **GPR**: General Purpose Registers for integers and pointers
//! - **FPR**: Floating Point Registers for scalar floating-point operations
//! - **Vec**: Vector Registers for SIMD operations
//!
//! ## Register Allocation
//!
//! Virtual registers are assigned sequentially during IR-to-MIR conversion.
//! Physical registers are assigned during the register allocation phase,
//! which maps virtual registers to target-specific physical registers.
use std::fmt;

/// Register class determines which physical registers can be used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// General Purpose Register (integers, pointers)
    Gpr,
    /// Floating Point Register (scalar floats)
    Fpr,
    /// Vector Register (SIMD operations)
    Vec,
}

/// Virtual register (pre-register allocation)
/// on Target Codegen, this will be mapped to a physical register
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualReg {
    pub id: u32,
    pub class: RegisterClass,
}

impl VirtualReg {
    pub fn new(id: u32, class: RegisterClass) -> Self {
        Self { id, class }
    }

    pub fn gpr(id: u32) -> Self {
        Self::new(id, RegisterClass::Gpr)
    }

    pub fn fpr(id: u32) -> Self {
        Self::new(id, RegisterClass::Fpr)
    }

    pub fn vec(id: u32) -> Self {
        Self::new(id, RegisterClass::Vec)
    }
}

impl fmt::Display for VirtualReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.id)
    }
}

/// Physical register (from __asm__ or any direct register allocation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalReg {
    pub name: &'static str,
    pub class: RegisterClass,
}

impl PhysicalReg {
    pub const fn new(name: &'static str, class: RegisterClass) -> Self {
        Self { name, class }
    }
}

impl fmt::Display for PhysicalReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Register (either virtual or physical)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Register {
    Virtual(VirtualReg),
    Physical(PhysicalReg),
}

impl Register {
    pub fn class(&self) -> RegisterClass {
        match self {
            Register::Virtual(v) => v.class,
            Register::Physical(p) => p.class,
        }
    }

    pub fn is_virtual(&self) -> bool {
        matches!(self, Register::Virtual(_))
    }

    pub fn is_physical(&self) -> bool {
        matches!(self, Register::Physical(_))
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::Virtual(v) => write!(f, "{}", v),
            Register::Physical(p) => write!(f, "{}", p),
        }
    }
}

impl From<VirtualReg> for Register {
    fn from(v: VirtualReg) -> Self {
        Register::Virtual(v)
    }
}

impl From<PhysicalReg> for Register {
    fn from(p: PhysicalReg) -> Self {
        Register::Physical(p)
    }
}

/// Allocator for virtual registers
#[derive(Default)]
pub struct VirtualRegAllocator {
    next_id: u32,
}

impl VirtualRegAllocator {
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    pub fn allocate(&mut self, class: RegisterClass) -> VirtualReg {
        let id = self.next_id;
        self.next_id += 1;
        VirtualReg::new(id, class)
    }

    pub fn allocate_gpr(&mut self) -> VirtualReg {
        self.allocate(RegisterClass::Gpr)
    }

    pub fn allocate_fpr(&mut self) -> VirtualReg {
        self.allocate(RegisterClass::Fpr)
    }

    pub fn allocate_vec(&mut self) -> VirtualReg {
        self.allocate(RegisterClass::Vec)
    }

    pub fn reset(&mut self) {
        self.next_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_reg_creation() {
        let v0 = VirtualReg::gpr(0);
        assert_eq!(v0.id, 0);
        assert_eq!(v0.class, RegisterClass::Gpr);
        assert_eq!(v0.to_string(), "v0");
    }

    #[test]
    fn test_virtual_reg_allocator() {
        let mut allocator = VirtualRegAllocator::new();

        let v0 = allocator.allocate_gpr();
        let v1 = allocator.allocate_fpr();
        let v2 = allocator.allocate_vec();

        assert_eq!(v0.id, 0);
        assert_eq!(v1.id, 1);
        assert_eq!(v2.id, 2);

        assert_eq!(v0.class, RegisterClass::Gpr);
        assert_eq!(v1.class, RegisterClass::Fpr);
        assert_eq!(v2.class, RegisterClass::Vec);
    }

    #[test]
    fn test_physical_reg() {
        let rax = PhysicalReg::new("rax", RegisterClass::Gpr);
        assert_eq!(rax.to_string(), "rax");
    }

    #[test]
    fn test_register_enum() {
        let v0 = Register::Virtual(VirtualReg::gpr(0));
        let rax = Register::Physical(PhysicalReg::new("rax", RegisterClass::Gpr));

        assert!(v0.is_virtual());
        assert!(!v0.is_physical());
        assert!(rax.is_physical());
        assert!(!rax.is_virtual());
    }
}
