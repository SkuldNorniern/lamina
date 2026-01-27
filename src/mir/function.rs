//! Function representation in LUMIR.
//!
//! Structures for representing functions in LUMIR,
//! including function signatures, parameters, and basic blocks.
use super::block::Block;
use super::register::Register;
use super::types::MirType;
use std::fmt;

/// Function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// Virtual register assigned to this parameter
    pub reg: Register,

    /// Type of the parameter
    pub ty: MirType,
}

impl Parameter {
    pub fn new(reg: Register, ty: MirType) -> Self {
        Self { reg, ty }
    }
}

/// Function signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Function name
    pub name: String,

    /// Parameters (in order)
    pub params: Vec<Parameter>,

    /// Return type (None for void)
    pub ret_ty: Option<MirType>,
}

impl Signature {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            ret_ty: None,
        }
    }

    pub fn with_params(mut self, params: Vec<Parameter>) -> Self {
        self.params = params;
        self
    }

    pub fn with_return(mut self, ty: MirType) -> Self {
        self.ret_ty = Some(ty);
        self
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    pub fn is_void(&self) -> bool {
        self.ret_ty.is_none()
    }
}

/// LUMIR function
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// Function signature
    pub sig: Signature,

    /// Basic blocks (ordered map for deterministic iteration)
    pub blocks: Vec<Block>,

    /// Entry block label
    pub entry: String,
}

impl Function {
    /// Create a new function with the given signature
    pub fn new(sig: Signature) -> Self {
        Self {
            sig,
            blocks: Vec::new(),
            entry: "entry".to_string(),
        }
    }

    /// Set the entry block label
    pub fn with_entry(mut self, entry: impl Into<String>) -> Self {
        self.entry = entry.into();
        self
    }

    /// Add a basic block to this function
    pub fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    /// Get a basic block by label
    pub fn get_block(&self, label: &str) -> Option<&Block> {
        self.blocks.iter().find(|b| b.label == label)
    }

    /// Get a mutable reference to a basic block by label
    pub fn get_block_mut(&mut self, label: &str) -> Option<&mut Block> {
        self.blocks.iter_mut().find(|b| b.label == label)
    }

    /// Get the entry block
    pub fn entry_block(&self) -> Option<&Block> {
        self.get_block(&self.entry)
    }

    /// Get a mutable reference to the entry block
    pub fn entry_block_mut(&mut self) -> Option<&mut Block> {
        let entry = self.entry.clone();
        self.get_block_mut(&entry)
    }

    /// Get all block labels
    pub fn block_labels(&self) -> Vec<&str> {
        self.blocks.iter().map(|b| b.label.as_str()).collect()
    }

    /// Total number of instructions across all blocks
    pub fn instruction_count(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }

    /// Check if this function is well-formed
    pub fn validate(&self) -> Result<(), String> {
        // Check that entry block exists
        if self.entry_block().is_none() {
            return Err(format!("Entry block '{}' not found", self.entry));
        }

        // Check that all blocks have unique labels
        let mut seen_labels = std::collections::HashSet::new();
        for block in &self.blocks {
            if !seen_labels.insert(&block.label) {
                return Err(format!("Duplicate block label: {}", block.label));
            }
        }

        // Check that all blocks have terminators
        for block in &self.blocks {
            if !block.has_terminator() {
                return Err(format!("Block '{}' has no terminator", block.label));
            }
        }

        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Signature header
        write!(f, "fn {}(", self.sig.name)?;
        for (i, p) in self.sig.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} {}", p.reg, p.ty)?;
        }
        write!(f, ")")?;
        if let Some(ret) = &self.sig.ret_ty {
            write!(f, " -> {}", ret)?;
        }
        writeln!(f, " {{")?;

        // Emit blocks in order
        for block in &self.blocks {
            writeln!(f, "{}", block)?;
        }

        write!(f, "}}")
    }
}

/// Builder for constructing LUMIR functions
pub struct FunctionBuilder {
    function: Function,
    current_block: Option<String>,
}

impl FunctionBuilder {
    /// Create a new function builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            function: Function::new(Signature::new(name)),
            current_block: None,
        }
    }

    /// Add a parameter to the function
    pub fn param(mut self, reg: Register, ty: MirType) -> Self {
        self.function.sig.params.push(Parameter::new(reg, ty));
        self
    }

    /// Set the return type
    pub fn returns(mut self, ty: MirType) -> Self {
        self.function.sig.ret_ty = Some(ty);
        self
    }

    /// Create a new basic block and make it the current block
    pub fn block(mut self, label: impl Into<String>) -> Self {
        let label = label.into();
        self.function.add_block(Block::new(label.clone()));
        self.current_block = Some(label);
        self
    }

    /// Add an instruction to the current block
    pub fn instr(mut self, instr: super::instruction::Instruction) -> Self {
        if let Some(ref label) = self.current_block
            && let Some(block) = self.function.get_block_mut(label)
        {
            block.push(instr);
        }
        self
    }

    /// Build the function
    pub fn build(self) -> Function {
        self.function
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::instruction::{Immediate, Instruction, IntBinOp, Operand};
    use crate::mir::register::VirtualReg;
    use crate::mir::types::ScalarType;

    #[test]
    fn test_signature_creation() {
        let sig = Signature::new("test_func").with_return(MirType::Scalar(ScalarType::I64));

        assert_eq!(sig.name, "test_func");
        assert!(sig.ret_ty.is_some());
        assert_eq!(sig.param_count(), 0);
    }

    #[test]
    fn test_function_builder() {
        let func = FunctionBuilder::new("add")
            .param(
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I64),
            )
            .param(
                Register::Virtual(VirtualReg::gpr(1)),
                MirType::Scalar(ScalarType::I64),
            )
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(2)),
                lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(0))),
                rhs: Operand::Register(Register::Virtual(VirtualReg::gpr(1))),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(2)))),
            })
            .build();

        assert_eq!(func.sig.name, "add");
        assert_eq!(func.sig.param_count(), 2);
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.instruction_count(), 2);
    }

    #[test]
    fn test_function_validation() {
        let mut func = Function::new(Signature::new("test"));

        // Missing entry block
        assert!(func.validate().is_err());

        // Add entry block with terminator
        let mut entry = Block::new("entry");
        entry.push(Instruction::Ret { value: None });
        func.add_block(entry);

        assert!(func.validate().is_ok());
    }
}
