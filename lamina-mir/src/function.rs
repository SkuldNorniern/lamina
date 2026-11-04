//! Function representation in LUMIR.
//!
//! This module defines the structures for representing functions in LUMIR,
//! including function signatures, parameters, and basic blocks.
use lamina_platform::{PointerWidth, SourceLoc};

use crate::{
    block::Block,
    instruction::{Instruction, Operand},
    register::Register,
    register::VirtualReg,
    types::{Aggregate, MirType, Signedness},
    verify::verify_function,
};
use std::collections::HashMap;
use std::fmt;

/// Function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// Virtual register assigned to this parameter
    pub reg: Register,

    /// Type of the parameter
    pub ty: MirType,

    /// Signedness of an integer parameter
    pub signedness: Signedness,

    /// Set when the parameter is an aggregate passed by value. `reg` then names the
    /// frame slot holding the callee's copy, and `ty` is the pointer to it.
    pub aggregate: Option<Aggregate>,

    /// The name the parameter had in the source, without its sigil. Debug info only;
    /// `None` where the frontend supplied none.
    pub name: Option<String>,

    /// Set by a `byval` attribute the frontend stated and lowering verified against its
    /// own classification. A backend that does not place this parameter as a by-value
    /// stack copy must refuse rather than place it some other way.
    pub byval_claimed: bool,
}

impl Parameter {
    /// Creates a signed parameter that is not an aggregate; chain `with_signedness` or
    /// `with_aggregate` for others.
    pub fn new(reg: Register, ty: MirType) -> Self {
        Self {
            reg,
            ty,
            signedness: Signedness::Signed,
            aggregate: None,
            name: None,
            byval_claimed: false,
        }
    }

    pub fn with_signedness(mut self, signedness: Signedness) -> Self {
        self.signedness = signedness;
        self
    }

    pub fn with_aggregate(mut self, aggregate: Aggregate) -> Self {
        self.aggregate = Some(aggregate);
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_byval_claim(mut self, claimed: bool) -> Self {
        self.byval_claimed = claimed;
        self
    }
}

/// Launch limits for a GPU kernel. GPU codegen rejects a zero workgroup dimension, a
/// workgroup larger than 1024 threads or than `max_threads`, and a `max_threads` outside
/// 1..=1024.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelMetadata {
    pub required_workgroup_size: Option<[u32; 3]>,
    pub max_threads: Option<u32>,
}

/// ELF-style symbol visibility requested by a function annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolVisibility {
    Internal,
    Private,
    Hidden,
    Protected,
}

/// Section a function is placed in, so the linker groups it with other cold or hot code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodePlacement {
    Cold,
    Hot,
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

    /// Signedness of an integer return value
    pub ret_signedness: Signedness,

    /// Set when the function returns an aggregate by value. `ret_ty` is then the pointer
    /// to it, and a call's result register names the caller's slot for the copy.
    pub ret_aggregate: Option<Aggregate>,

    /// Set by a `sret` marker the frontend stated. A backend that does not return this
    /// function's result through a hidden pointer must refuse rather than return it
    /// some other way.
    pub sret_claimed: bool,

    pub kernel: Option<KernelMetadata>,

    /// Set by `@noinline`; the inliner must leave calls to this function alone.
    pub no_inline: bool,

    /// Set by `@inline`; the inliner accepts a larger body than its default size limit.
    pub inline_hint: bool,

    /// Set by `@noreturn`; no code after a call to this function is reachable.
    pub noreturn: bool,

    /// Set by `@pure` or `@const`; a call whose result is unused can be deleted.
    pub pure: bool,

    /// Set by `@internal`, `@private`, `@hidden` or `@protected`.
    pub visibility: Option<SymbolVisibility>,

    /// Set by `@cold` or `@hot`.
    pub placement: Option<CodePlacement>,

    /// Set by `@weak`; the symbol is emitted as a weak definition.
    pub weak: bool,

    /// Set by `@section("name")`; the function is emitted into that ELF section.
    pub section: Option<String>,

    /// Set by `@align(n)`; the function entry is aligned to `n` bytes.
    pub align: Option<u32>,

    /// Set by `@naked`; the function is emitted with no prologue or epilogue.
    pub naked: bool,

    /// Set by a trailing `...` on an `@extern` declaration. A call to one follows the
    /// variadic side of the platform convention.
    pub variadic: bool,

    /// Where the function was declared in the source. Debug info only.
    pub decl: Option<SourceLoc>,
}

impl Signature {
    /// Creates a void, parameterless, non-kernel signature that the inliner may inline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            ret_ty: None,
            ret_signedness: Signedness::Signed,
            ret_aggregate: None,
            sret_claimed: false,
            kernel: None,
            no_inline: false,
            inline_hint: false,
            noreturn: false,
            pure: false,
            visibility: None,
            placement: None,
            weak: false,
            section: None,
            align: None,
            naked: false,
            variadic: false,
            decl: None,
        }
    }

    /// Marks the function as a GPU kernel, which GPU codegen requires to be void with a
    /// single block.
    pub fn with_kernel(mut self, metadata: KernelMetadata) -> Self {
        self.kernel = Some(metadata);
        self
    }

    pub fn with_params(mut self, params: Vec<Parameter>) -> Self {
        self.params = params;
        self
    }

    pub fn with_return(mut self, ty: MirType) -> Self {
        self.ret_ty = Some(ty);
        self
    }

    pub fn with_return_signedness(mut self, signedness: Signedness) -> Self {
        self.ret_signedness = signedness;
        self
    }

    pub fn with_return_aggregate(mut self, aggregate: Aggregate) -> Self {
        self.ret_aggregate = Some(aggregate);
        self
    }

    pub fn with_decl(mut self, decl: Option<SourceLoc>) -> Self {
        self.decl = decl;
        self
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// True when there is no return value. An aggregate return is not void, since `ret_ty`
    /// then holds the pointer.
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

    /// Extra 8-byte stack slots a virtual register needs beyond its own, for
    /// stack allocations larger than one slot. The register addresses the
    /// lowest byte of the reservation.
    pub stack_reservations: HashMap<VirtualReg, usize>,

    /// Required alignment in bytes for a virtual register's stack slot, set by
    /// `stackalloc ... align n` when n exceeds the 8-byte slot size.
    pub stack_alignments: HashMap<VirtualReg, usize>,

    /// Registers that hold the address of a legalized integer pair a call passes
    /// where the callee's declaration gives no type for it: past the named
    /// parameters of a variadic callee. The aggregate is what the callee reads
    /// there. Nothing else tells such an address from a pointer argument, so a
    /// backend without this passes the address where the value belongs.
    pub variadic_wide_arguments: HashMap<VirtualReg, Aggregate>,

    /// Registers a `va_arg` of a legalized integer pair defines. The register names a
    /// two-word stack reservation the backend writes the value read out of the
    /// variadic area into, not a register that comes to hold the value. A backend
    /// without this reads one word where two are owed.
    pub variadic_wide_results: HashMap<VirtualReg, Aggregate>,

    /// Width of data pointers on the target this function is lowered for. Sizes of
    /// pointer-typed values derive from it.
    pub pointer_width: PointerWidth,

    /// Whether integer registers on the target this function is lowered for are 64 bits.
    /// Sets the register type of an I/O intrinsic's result, which is an i64 wherever
    /// registers are 64 bits.
    pub int64_regs: bool,

    /// The source name of each virtual register the frontend named, without its sigil.
    /// Debug info only; a register a pass invented is absent.
    pub value_names: HashMap<VirtualReg, String>,
}

impl Function {
    /// Create a new function with the given signature. Pointers are 64-bit until
    /// [`Function::with_pointer_width`] says otherwise.
    pub fn new(sig: Signature) -> Self {
        Self {
            sig,
            blocks: Vec::new(),
            entry: "entry".to_string(),
            stack_reservations: HashMap::new(),
            stack_alignments: HashMap::new(),
            variadic_wide_arguments: HashMap::new(),
            variadic_wide_results: HashMap::new(),
            pointer_width: PointerWidth::Bits64,
            int64_regs: true,
            value_names: HashMap::new(),
        }
    }

    pub fn with_pointer_width(mut self, pointer_width: PointerWidth) -> Self {
        self.pointer_width = pointer_width;
        self
    }

    pub fn with_int64_regs(mut self, int64_regs: bool) -> Self {
        self.int64_regs = int64_regs;
        self
    }

    /// The aggregate a call argument is passed by value as when the callee's
    /// declaration gives no type for it, or `None` for an ordinary argument.
    pub fn variadic_wide_argument(&self, arg: &Operand) -> Option<Aggregate> {
        match arg {
            Operand::Register(Register::Virtual(vreg)) => {
                self.variadic_wide_arguments.get(vreg).copied()
            }
            _ => None,
        }
    }

    /// The aggregate a `va_arg` writes into the stack reservation `dst` names, or
    /// `None` when `dst` is an ordinary destination register.
    pub fn variadic_wide_result(&self, dst: &Register) -> Option<Aggregate> {
        match dst {
            Register::Virtual(vreg) => self.variadic_wide_results.get(vreg).copied(),
            _ => None,
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
        self.blocks.iter().map(Block::len).sum()
    }

    /// Check if this function is well-formed
    pub fn validate(&self) -> Result<(), String> {
        verify_function(self).map_err(|errors| errors.join("\n"))
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
            write!(f, " -> {ret}")?;
        }
        writeln!(f, " {{")?;

        // Emit blocks in order
        for block in &self.blocks {
            writeln!(f, "{block}")?;
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
    pub fn instr(mut self, instr: Instruction) -> Self {
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
    use crate::{
        instruction::{Instruction, IntBinOp, Operand},
        register::VirtualReg,
        types::ScalarType,
    };

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

        // No blocks at all is a declaration, which carries only its signature.
        assert!(func.validate().is_ok());

        // A block that the entry does not name leaves the entry dangling.
        let mut other = Block::new("other");
        other.push(Instruction::Ret { value: None });
        func.add_block(other);
        assert!(func.validate().is_err());

        // Add entry block with terminator
        let mut entry = Block::new("entry");
        entry.push(Instruction::Ret { value: None });
        func.add_block(entry);

        assert!(func.validate().is_ok());
    }
}
