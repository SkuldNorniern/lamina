//! WASM codegen state.

use std::collections::HashMap;

use crate::{Identifier, Literal, Value};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegType {
    Global,
    Local,
    Generic,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register<'a> {
    pub size: u64,
    pub address: u64,
    pub name: Identifier<'a>,
    pub reg_type: RegType,
}

impl Ord for Register<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size.cmp(&other.size)
    }
}

impl PartialOrd for Register<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GlobalRef {
    Wasm(u64),
    Memory(u64),
}

pub struct CodegenState<'a> {
    pub output_memory: Vec<u8>,
    pub out_expressions: Vec<super::generate::ModuleExpression<'a>>,
    globals: HashMap<&'a str, GlobalRef>,
    global_values: HashMap<&'a str, Option<Value<'a>>>,
    global_next: u64,
    mem_registers: Vec<Register<'a>>,
}

impl<'a> Default for CodegenState<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CodegenState<'a> {
    pub fn new() -> Self {
        Self {
            output_memory: vec![0; 64 * 1024], // each page is 64 KB
            out_expressions: vec![],
            globals: HashMap::new(),
            global_values: HashMap::new(),
            global_next: 0,
            mem_registers: Vec::new(),
        }
    }
    pub fn add_wasm_global(
        &mut self,
        name: &'a str,
        ty: super::generate::NumericType,
        val: Option<Value<'a>>,
    ) {
        self.out_expressions
            .push(super::generate::ModuleExpression::Global { name: None, ty });
        self.globals.insert(name, GlobalRef::Wasm(self.global_next));
        self.global_values.insert(name, val);
        self.global_next += 1;
    }
    pub fn get_global(&self, name: &'a str) -> Result<GlobalRef, crate::LaminaError> {
        self.globals.get(&name).copied().ok_or_else(|| {
            crate::LaminaError::ValidationError(format!(
                "Global '{}' not found in WASM codegen state",
                name
            ))
        })
    }
    pub fn get_global_value(
        &self,
        name: &'a str,
    ) -> Result<&Option<Value<'a>>, crate::LaminaError> {
        self.global_values.get(&name).ok_or_else(|| {
            crate::LaminaError::ValidationError(format!(
                "Global value '{}' not found in WASM codegen state",
                name
            ))
        })
    }
    fn set_memory_contents_for_value(
        &mut self,
        address: usize,
        val: Value<'a>,
    ) -> Result<(), crate::LaminaError> {
        let ptr_addr = address;
        match val {
            Value::Constant(lit) => match lit {
                Literal::Bool(v) => self.output_memory[ptr_addr] = if v { 1 } else { 0 },
                Literal::Char(v) => self.output_memory[ptr_addr..ptr_addr + 4]
                    .clone_from_slice((v as u32).to_le_bytes().as_slice()),
                Literal::I8(v) => self.output_memory[ptr_addr] = v as u8,
                Literal::U8(v) => self.output_memory[ptr_addr] = v,
                Literal::I16(v) => self.output_memory[ptr_addr..ptr_addr + 2]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
                Literal::U16(v) => self.output_memory[ptr_addr..ptr_addr + 2]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
                Literal::I32(v) => self.output_memory[ptr_addr..ptr_addr + 4]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
                Literal::U32(v) => self.output_memory[ptr_addr..ptr_addr + 4]
                    .clone_from_slice(v.to_le_bytes().as_slice()),

                Literal::String(v) => {
                    self.output_memory[ptr_addr..ptr_addr + v.len()].clone_from_slice(v.as_bytes())
                }

                Literal::I64(v) => self.output_memory[ptr_addr..ptr_addr + 8]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
                Literal::U64(v) => self.output_memory[ptr_addr..ptr_addr + 8]
                    .clone_from_slice(v.to_le_bytes().as_slice()),

                Literal::F32(v) => self.output_memory[ptr_addr..ptr_addr + 4]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
                Literal::F64(v) => self.output_memory[ptr_addr..ptr_addr + 8]
                    .clone_from_slice(v.to_le_bytes().as_slice()),
            },
            Value::Global(id) => {
                let from = self
                    .mem_registers
                    .iter()
                    .find(|v| v.name == id)
                    .ok_or_else(|| {
                        crate::LaminaError::ValidationError(format!(
                            "Memory register '{}' not found in WASM codegen state",
                            id
                        ))
                    })?;

                let from_bytes = self.output_memory
                    [(from.address as usize)..(from.address + from.size) as usize]
                    .to_vec();

                self.output_memory[ptr_addr..ptr_addr + from.size as usize]
                    .clone_from_slice(from_bytes.as_slice());
            }
            Value::Variable(_) => {
                return Err(crate::LaminaError::ValidationError(
                    "Variable values cannot be stored in memory directly".to_string(),
                ));
            }
        }
        Ok(())
    }
    pub fn add_memory_register(
        &mut self,
        global: Register<'a>,
        val: Option<Value<'a>>,
    ) -> &mut Self {
        if global.address > self.output_memory.len() as u64 {
            self.output_memory.append(&mut vec![0; 64 * 1024]);
        }
        self.mem_registers.push(global);
        self.mem_registers.sort();
        self.globals
            .insert(global.name, GlobalRef::Memory(global.address));
        self.global_values.insert(global.name, val);
        self
    }
    /// Find the most optimal address for something of the provided size.
    ///
    /// This will search for the smallest opening that matches the provided specifications.
    ///
    /// NOTES: ALIGN *MUST* BE A POWER OF TWO!
    pub fn get_next_address(&self, size: u64, align: u64) -> u64 {
        assert!(align.is_power_of_two());

        let mut options = Vec::new();
        for i in 1..self.mem_registers.len() - 1 {
            let globals = (self.mem_registers[i - 1], self.mem_registers[i]);
            let mut base = globals.0.address + globals.0.size;

            base = (base + align - 1) & !(align - 1);

            let diff = globals.1.address as i64 - base as i64;

            if diff > size as i64 {
                options.push((diff, base));
            } else if diff == size as i64 {
                // can't be any better then a perfect fit :P
                return base;
            }
        }
        if options.is_empty() {
            return self
                .mem_registers
                .last()
                .map(|v| v.address + v.size)
                .unwrap_or(0);
        }
        options.sort_by(|v1, v2| v1.0.cmp(&v2.0));
        options[0].1
    }

    pub fn has_any_mem_regs(&self) -> bool {
        !self.mem_registers.is_empty()
    }
}
