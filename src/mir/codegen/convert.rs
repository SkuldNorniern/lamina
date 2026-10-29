//! IR-to-MIR conversion implementation.
//!
//! Main conversion logic that transforms high-level
//! Lamina IR into low-level LUMIR. The conversion process includes:
//!
//! - Function signature conversion
//! - Instruction lowering
//! - Variable binding and register assignment
//! - Control flow graph construction
//! - Type mapping and validation

use crate::mir::parallel::insert_edge_moves;

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    mem,
    ops::{Deref, DerefMut},
};

use crate::{
    ir::{
        Module as IRModule,
        function::{Function as IRFunction, FunctionAnnotation},
        instruction::{
            AllocType, BinaryOp as IRBin, CmpOp as IRCmp, Instruction as IRInst, UnaryOp as IRUnary,
        },
        module::GlobalInit,
        types::{
            Literal as IRLit, PrimitiveType as IRPrim, Type as IRType, Value as IRVal,
            align_of_type, resolve_type_names, sizeof_type, struct_field_byte_offset,
        },
    },
    mir::{
        AddressMode, AsmOperand, Block, CodePlacement, FloatBinOp, FloatCmpOp, FloatUnOp, Function,
        Global, Immediate, Instruction, IntBinOp, IntCmpOp, IntRotOp, IntUnOp, MemoryAttrs,
        MirType, Module, Operand, Parameter, Register, RegisterClass, ScalarType, Signature,
        SymbolVisibility, VirtualReg, VirtualRegAllocator,
        codegen::{
            error::FromIRError,
            mapping::{classify_aggregate, is_aggregate, map_ir_prim, map_ir_type, signedness_of},
        },
        io_intrinsic_result_type,
        verify::{PhiVerification, verify_before_phi_lowering},
    },
};

#[cfg(feature = "nightly")]
use crate::ir::ModuleAnnotation;
#[cfg(feature = "nightly")]
use crate::ir::instruction::SimdOp as IRSimdOp;
use crate::ir::instruction::{
    AsmOperand as IRAsmOperand, AtomicBinOp as IRAtomicBinOp, MemoryOrdering as IRMemOrd,
};
#[cfg(feature = "nightly")]
use crate::mir::RelocationModel;
#[cfg(feature = "nightly")]
use crate::mir::SimdOp;
use crate::mir::{AtomicBinOp, MemoryOrdering};
use lamina_codegen::abi::RuntimeSymbol;

mod wide;

fn ir_ordering_to_mir(ordering: &IRMemOrd) -> MemoryOrdering {
    match ordering {
        IRMemOrd::Relaxed => MemoryOrdering::Relaxed,
        IRMemOrd::Acquire => MemoryOrdering::Acquire,
        IRMemOrd::Release => MemoryOrdering::Release,
        IRMemOrd::AcqRel => MemoryOrdering::AcqRel,
        IRMemOrd::SeqCst => MemoryOrdering::SeqCst,
    }
}

fn float_binary(
    op: FloatBinOp,
    ty: MirType,
    dst: Register,
    lhs: Operand,
    rhs: Operand,
) -> Instruction {
    Instruction::FloatBinary {
        op,
        ty,
        dst,
        lhs,
        rhs,
    }
}

/// Map an IR binary op to its MIR opcode, choosing signed vs. unsigned variants
/// based on the operand type. Float-applicable ops on float types are handled
/// separately by the caller before this function is reached.
fn lower_int_binop(op: &IRBin, ty: IRPrim) -> Option<IntBinOp> {
    let unsigned = matches!(ty, IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64);
    Some(match op {
        IRBin::Add => IntBinOp::Add,
        IRBin::Sub => IntBinOp::Sub,
        IRBin::Mul => IntBinOp::Mul,
        IRBin::Div => {
            if unsigned {
                IntBinOp::UDiv
            } else {
                IntBinOp::SDiv
            }
        }
        IRBin::Rem => {
            if unsigned {
                IntBinOp::URem
            } else {
                IntBinOp::SRem
            }
        }
        IRBin::And => IntBinOp::And,
        IRBin::Or => IntBinOp::Or,
        IRBin::Xor => IntBinOp::Xor,
        IRBin::Shl => IntBinOp::Shl,
        // Arithmetic shift for signed types, logical shift for unsigned types.
        IRBin::Shr => {
            if unsigned {
                IntBinOp::LShr
            } else {
                IntBinOp::AShr
            }
        }
        IRBin::Rotl | IRBin::Rotr => return None,
    })
}

fn lower_int_unop(op: &IRUnary) -> Option<IntUnOp> {
    match op {
        IRUnary::Popcount => Some(IntUnOp::Popcount),
        IRUnary::Ctlz => Some(IntUnOp::Ctlz),
        IRUnary::Cttz => Some(IntUnOp::Cttz),
        IRUnary::Bswap => Some(IntUnOp::Bswap),
        IRUnary::Neg | IRUnary::Not | IRUnary::Sqrt => None,
    }
}

fn is_integer_prim(ty: IRPrim) -> bool {
    matches!(
        ty,
        IRPrim::I8
            | IRPrim::I16
            | IRPrim::I32
            | IRPrim::I64
            | IRPrim::U8
            | IRPrim::U16
            | IRPrim::U32
            | IRPrim::U64
    )
}

/// Bindings from IR value names to MIR registers, together with the register class each
/// name needs. The class comes from the IR type of the defining instruction, so a value
/// first seen as an operand still gets a register of the right class.
#[derive(Default)]
struct VarBindings<'a> {
    regs: HashMap<&'a str, Register>,
    classes: HashMap<&'a str, RegisterClass>,
    /// The two 64-bit words of each 128-bit value, low word first.
    wide: HashMap<&'a str, [Register; 2]>,
    /// Compiler-rt routines this function calls, to be declared in the module.
    libcalls: BTreeMap<String, wide::LibCall>,
}

impl<'a> Deref for VarBindings<'a> {
    type Target = HashMap<&'a str, Register>;
    fn deref(&self) -> &Self::Target {
        &self.regs
    }
}

impl<'a> DerefMut for VarBindings<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.regs
    }
}

impl<'a> VarBindings<'a> {
    fn class_of(&self, name: &str) -> RegisterClass {
        self.classes
            .get(name)
            .copied()
            .unwrap_or(RegisterClass::Gpr)
    }

    fn fresh(&self, name: &str, vreg_alloc: &mut VirtualRegAllocator) -> Register {
        Register::Virtual(allocate_in_class(self.class_of(name), vreg_alloc))
    }
}

fn allocate_in_class(class: RegisterClass, vreg_alloc: &mut VirtualRegAllocator) -> VirtualReg {
    match class {
        RegisterClass::Fpr => vreg_alloc.allocate_fpr(),
        RegisterClass::Vec => vreg_alloc.allocate_vec(),
        _ => vreg_alloc.allocate_gpr(),
    }
}

/// Resolve an IR value to a MIR operand, allocating a fresh binding on first encounter.
fn resolve_operand<'a>(
    v: &IRVal<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut VarBindings<'a>,
    globals: &mut GlobalAddresses<'a>,
) -> Result<Operand, FromIRError> {
    match v {
        IRVal::Variable(id) => {
            if var_to_reg.wide.contains_key(id) {
                return Err(FromIRError::UnsupportedInstruction);
            }
            if let Some(r) = var_to_reg.get(id) {
                Ok(Operand::Register(r.clone()))
            } else {
                let r = var_to_reg.fresh(id, vreg_alloc);
                var_to_reg.insert(*id, r.clone());
                Ok(Operand::Register(r))
            }
        }
        IRVal::Constant(lit) => literal_to_immediate(lit).map(Operand::Immediate),
        IRVal::Global(name) => Ok(Operand::Register(globals.register(name, vreg_alloc)?)),
    }
}

/// Look up `result` in the register map; if absent, allocate a fresh register of the
/// class its IR type calls for and bind it.
fn resolve_or_alloc<'a>(
    result: &'a str,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut VarBindings<'a>,
) -> Register {
    if let Some(existing) = var_to_reg.get(result) {
        existing.clone()
    } else {
        let fresh = var_to_reg.fresh(result, vreg_alloc);
        var_to_reg.insert(result, fresh.clone());
        fresh
    }
}

/// Bit-width for bitcast: F32 = 32 bits, F64 = 64 bits (valid bitcast widths).
fn int_bits_for_bitcast(ty: &IRPrim) -> u32 {
    match ty {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Bool => 8,
        IRPrim::I16 | IRPrim::U16 => 16,
        IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 32,
        IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 64,
        IRPrim::I128 | IRPrim::U128 => 128,
    }
}

fn normalize_lowered_literal(operand: &mut Operand, ty: MirType) {
    operand.normalize_integer_immediate(ty);
    if ty == MirType::f32()
        && let Operand::Immediate(Immediate::F64(value)) = operand
    {
        *operand = Operand::imm_f32(*value as f32);
    }
}

fn declared_align(align: Option<u32>) -> usize {
    align.map_or(1, |align| align as usize)
}

/// Lowers an IR module to MIR, named `name`.
///
/// The result is target independent: an aggregate crossing a function boundary keeps
/// its layout in an [`Aggregate`](crate::mir::types::Aggregate) marker and each backend
/// decides where the words go. Phis survive, so the caller lowers them before handing
/// the module to a backend. Anything MIR cannot express comes back as a
/// [`FromIRError`] naming the value.
pub fn from_ir(ir: &IRModule<'_>, name: &str) -> Result<Module, FromIRError> {
    let mut mir_module = Module::new(name);
    #[cfg(feature = "nightly")]
    {
        mir_module.relocation = relocation_model(&ir.annotations);
    }

    for (global_name, decl) in &ir.global_declarations {
        let resolved = resolve_type_names(&decl.ty, &ir.type_declarations)
            .ok_or(FromIRError::UnsupportedType)?;
        let mut global = if is_aggregate(&resolved) {
            let size = sizeof_type(&resolved).ok_or(FromIRError::UnsupportedType)? as usize;
            let align = (align_of_type(&resolved) as usize).max(declared_align(decl.align));
            let bytes =
                aggregate_global_bytes(global_name, &resolved, size, decl.initializer.as_ref())?;
            let mut global = Global::new(*global_name, MirType::i8()).with_layout(size, align);
            if let Some(bytes) = bytes {
                global = global.with_initializer(bytes);
            }
            global
        } else {
            if wide::is_wide_type(&resolved) {
                return Err(FromIRError::UnsupportedGlobal(format!(
                    "@{global_name} is a 128-bit integer; globals of i128 and u128 are not lowered yet"
                )));
            }
            let ty = map_ir_type(&resolved)?;
            let initializer = global_initializer(global_name, ty, decl.initializer.as_ref())?;
            let mut global = Global::new(*global_name, ty);
            global.align = global.align.max(declared_align(decl.align));
            if let Some(bytes) = initializer {
                global = global.with_initializer(bytes);
            }
            global
        };
        if decl.exported {
            global = global.exported();
        }
        mir_module.globals.insert(global_name.to_string(), global);
    }

    let mut libcalls = BTreeMap::new();
    for (func_name, ir_func) in &ir.functions {
        // Check if function is external before converting
        if ir_func.annotations.contains(&FunctionAnnotation::Extern) {
            mir_module.mark_external(*func_name);
        }
        if ir_func.annotations.contains(&FunctionAnnotation::Export) {
            mir_module.mark_exported(*func_name);
        }

        let mir_func = convert_function(func_name, ir_func, ir, &mut libcalls)?;
        mir_module.add_function(mir_func);
    }
    for (name, libcall) in libcalls {
        if !mir_module.functions.contains_key(&name) {
            mir_module.mark_external(name.clone());
            mir_module.add_function(libcall.declaration());
        }
    }

    mir_module
        .validate_in_pipeline()
        .map_err(FromIRError::InvalidMir)?;
    Ok(mir_module)
}

#[cfg(feature = "nightly")]
fn relocation_model(annotations: &[ModuleAnnotation]) -> RelocationModel {
    annotations
        .last()
        .map_or(RelocationModel::Static, |annotation| match annotation {
            ModuleAnnotation::PositionIndependentCode => RelocationModel::Pic,
            ModuleAnnotation::PositionIndependentExecutable => RelocationModel::Pie,
        })
}

/// What lowering knows about the values in a function beyond the MIR types: the
/// type of each register.
struct ValueFacts<'t, 'a> {
    types: &'t HashMap<Register, MirType>,
    /// Named parameter count of each variadic callee, which says where its default
    /// argument promotions begin.
    variadic_fixed: &'t HashMap<&'a str, usize>,
    module: &'t IRModule<'a>,
}

fn conflicting_definitions(name: &str) -> FromIRError {
    FromIRError::ConflictingDefinitions(name.to_string())
}

/// Where each element of a global's initializer list lands.
enum Slots {
    Strided { stride: usize, count: usize },
    Fixed(Vec<(usize, usize)>),
}

impl Slots {
    fn len(&self) -> usize {
        match self {
            Slots::Strided { count, .. } => *count,
            Slots::Fixed(slots) => slots.len(),
        }
    }

    fn get(&self, index: usize) -> Option<(usize, usize)> {
        match self {
            Slots::Strided { stride, count } if index < *count => Some((index * stride, *stride)),
            Slots::Strided { .. } => None,
            Slots::Fixed(slots) => slots.get(index).copied(),
        }
    }
}

/// Bytes for an array global written as an element list. Elements left out are zero.
fn element_list_bytes(
    name: &str,
    ty: &IRType<'_>,
    size: usize,
    elements: &[IRLit<'_>],
) -> Result<Vec<u8>, FromIRError> {
    // (offset, width) of every slot the list can fill, in order. An array keeps its
    // stride instead of one entry per element, so a long array costs nothing here.
    let slots: Slots = match ty {
        IRType::Array { element_type, size } => Slots::Strided {
            stride: sizeof_type(element_type).ok_or(FromIRError::UnsupportedType)? as usize,
            count: usize::try_from(*size).map_err(|_| FromIRError::UnsupportedType)?,
        },
        IRType::Struct(fields) => Slots::Fixed(
            fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let at = struct_field_byte_offset(fields, index)
                        .ok_or(FromIRError::UnsupportedType)? as usize;
                    let width =
                        sizeof_type(&field.ty).ok_or(FromIRError::UnsupportedType)? as usize;
                    Ok((at, width))
                })
                .collect::<Result<Vec<_>, FromIRError>>()?,
        ),
        _ => {
            return Err(FromIRError::UnsupportedGlobal(format!(
                "@{name} takes an element list only as an array or struct"
            )));
        }
    };
    if elements.len() > slots.len() {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} holds {} elements, too few for the {} given",
            slots.len(),
            elements.len()
        )));
    }
    // Only the filled prefix is written out; the tail stays implicit zero, which keeps
    // a sparsely initialized array global off the heap.
    let filled = elements
        .len()
        .checked_sub(1)
        .and_then(|last| slots.get(last))
        .map_or(0, |(at, width)| at + width)
        .min(size);
    let mut bytes = vec![0u8; filled];
    for (index, literal) in elements.iter().enumerate() {
        let (at, width) = slots.get(index).ok_or(FromIRError::UnsupportedType)?;
        let encoded = literal_bytes(name, literal)?;
        if encoded.len() != width {
            return Err(FromIRError::UnsupportedGlobal(format!(
                "@{name} element {index} is {} bytes, not the {width} its slot takes",
                encoded.len()
            )));
        }
        bytes[at..at + width].copy_from_slice(&encoded);
    }
    Ok(bytes)
}

/// Bytes for an aggregate global. Only a string literal in a byte array carries data
/// today; every other aggregate starts zeroed.
fn aggregate_global_bytes(
    name: &str,
    ty: &IRType<'_>,
    size: usize,
    value: Option<&GlobalInit<'_>>,
) -> Result<Option<Vec<u8>>, FromIRError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let GlobalInit::List(elements) = value {
        return element_list_bytes(name, ty, size, elements).map(Some);
    }
    let GlobalInit::Value(IRVal::Constant(IRLit::String(text))) = value else {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} needs a string literal or an element list; other aggregate initializers are not supported yet"
        )));
    };
    let element_is_byte = matches!(
        ty,
        IRType::Array { element_type, .. }
            if matches!(
                element_type.as_ref(),
                IRType::Primitive(IRPrim::I8 | IRPrim::U8 | IRPrim::Bool)
            )
    );
    if !element_is_byte {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} takes a string literal only in a byte array"
        )));
    }
    let mut bytes = string_literal_bytes(name, text)?;
    if bytes.len() >= size {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} holds {size} bytes, too few for \"{text}\" and its terminator"
        )));
    }
    bytes.resize(size, 0);
    Ok(Some(bytes))
}

/// Bytes a string literal stands for, escapes resolved. The parser keeps the literal
/// as it was written so it can borrow it; the bytes are decided here.
fn string_literal_bytes(name: &str, text: &str) -> Result<Vec<u8>, FromIRError> {
    let mut bytes = Vec::with_capacity(text.len());
    let mut rest = text.chars();
    while let Some(c) = rest.next() {
        if c != '\\' {
            let mut buffer = [0u8; 4];
            bytes.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
            continue;
        }
        let escape = rest.next().ok_or_else(|| {
            FromIRError::UnsupportedGlobal(format!("@{name} ends in a lone backslash"))
        })?;
        let byte = match escape {
            'n' => b'\n',
            't' => b'\t',
            'r' => b'\r',
            '0' => 0,
            '\\' => b'\\',
            '"' => b'"',
            '\'' => b'\'',
            'x' => {
                let digits: String = rest.by_ref().take(2).collect();
                u8::from_str_radix(&digits, 16).map_err(|_| {
                    FromIRError::UnsupportedGlobal(format!(
                        "@{name} has \\x{digits}, which is not two hex digits"
                    ))
                })?
            }
            other => {
                return Err(FromIRError::UnsupportedGlobal(format!(
                    "@{name} has \\{other}, which is not an escape"
                )));
            }
        };
        bytes.push(byte);
    }
    Ok(bytes)
}

fn global_initializer(
    name: &str,
    ty: MirType,
    value: Option<&GlobalInit<'_>>,
) -> Result<Option<Vec<u8>>, FromIRError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let GlobalInit::Value(IRVal::Constant(literal)) = value else {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} needs a literal initializer"
        )));
    };
    let bytes = literal_bytes(name, literal)?;
    if bytes.len() != ty.size_bytes() {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} initializer width does not match its type {ty}"
        )));
    }
    Ok(Some(bytes))
}

/// Little-endian bytes of a literal, at the width its own type takes.
fn literal_bytes(name: &str, literal: &IRLit<'_>) -> Result<Vec<u8>, FromIRError> {
    let bytes = match literal {
        IRLit::I8(v) => v.to_le_bytes().to_vec(),
        IRLit::I16(v) => v.to_le_bytes().to_vec(),
        IRLit::I32(v) => v.to_le_bytes().to_vec(),
        IRLit::I64(v) => v.to_le_bytes().to_vec(),
        IRLit::U8(v) => v.to_le_bytes().to_vec(),
        IRLit::U16(v) => v.to_le_bytes().to_vec(),
        IRLit::U32(v) => v.to_le_bytes().to_vec(),
        IRLit::U64(v) => v.to_le_bytes().to_vec(),
        IRLit::I128(v) => v.to_le_bytes().to_vec(),
        IRLit::U128(v) => v.to_le_bytes().to_vec(),
        IRLit::F32(v) => v.to_le_bytes().to_vec(),
        IRLit::F64(v) => v.to_le_bytes().to_vec(),
        IRLit::Bool(b) => vec![u8::from(*b)],
        IRLit::String(_) => {
            return Err(FromIRError::UnsupportedGlobal(format!(
                "@{name} initializer must be an integer, float or bool literal"
            )));
        }
    };
    Ok(bytes)
}

/// Address registers for referenced globals. Each global gets one register,
/// defined by a `SymbolAddress` at the top of the entry block.
struct GlobalAddresses<'a> {
    declared: HashSet<&'a str>,
    regs: BTreeMap<&'a str, Register>,
}

impl<'a> GlobalAddresses<'a> {
    fn new(module: &IRModule<'a>) -> Self {
        Self {
            declared: module.global_declarations.keys().copied().collect(),
            regs: BTreeMap::new(),
        }
    }

    fn register(
        &mut self,
        name: &'a str,
        vreg_alloc: &mut VirtualRegAllocator,
    ) -> Result<Register, FromIRError> {
        if !self.declared.contains(name) {
            return Err(FromIRError::UnknownVariable);
        }
        Ok(self
            .regs
            .entry(name)
            .or_insert_with(|| Register::Virtual(vreg_alloc.allocate_gpr()))
            .clone())
    }

    fn prelude(&self) -> Vec<Instruction> {
        self.regs
            .iter()
            .map(|(name, dst)| Instruction::SymbolAddress {
                dst: dst.clone(),
                symbol: (*name).to_string(),
            })
            .collect()
    }
}

fn convert_function<'a>(
    name: &'a str,
    f: &IRFunction<'a>,
    module: &IRModule<'a>,
    libcalls: &mut BTreeMap<String, wide::LibCall>,
) -> Result<Function, FromIRError> {
    if !f.signature.variadic
        && f.basic_blocks
            .values()
            .flat_map(|block| &block.instructions)
            .any(|instr| {
                matches!(
                    instr,
                    IRInst::VaStart { .. }
                        | IRInst::VaArg { .. }
                        | IRInst::VaEnd { .. }
                        | IRInst::VaCopy { .. }
                )
            })
    {
        return Err(FromIRError::VariadicArgument(format!(
            "va_* instruction in @{name}, which is not variadic"
        )));
    }
    let mut vreg_alloc = VirtualRegAllocator::new();
    let mut mir_sig = Signature::new(name);
    mir_sig.no_inline = f.annotations.contains(&FunctionAnnotation::NoInline);
    mir_sig.inline_hint = f.annotations.contains(&FunctionAnnotation::Inline);
    mir_sig.noreturn = f.annotations.contains(&FunctionAnnotation::NoReturn);
    mir_sig.pure = f
        .annotations
        .iter()
        .any(|a| matches!(a, FunctionAnnotation::Pure | FunctionAnnotation::Const));
    mir_sig.visibility = f.annotations.iter().find_map(|a| match a {
        FunctionAnnotation::Internal => Some(SymbolVisibility::Internal),
        FunctionAnnotation::Private => Some(SymbolVisibility::Private),
        FunctionAnnotation::Hidden => Some(SymbolVisibility::Hidden),
        FunctionAnnotation::Protected => Some(SymbolVisibility::Protected),
        _ => None,
    });
    mir_sig.placement = f.annotations.iter().find_map(|a| match a {
        FunctionAnnotation::Cold => Some(CodePlacement::Cold),
        FunctionAnnotation::Hot => Some(CodePlacement::Hot),
        _ => None,
    });
    mir_sig.weak = f.annotations.contains(&FunctionAnnotation::Weak);
    // A weak definition may be overridden at link time, so calls must reach the symbol.
    mir_sig.no_inline |= mir_sig.weak;
    mir_sig.section = f.annotations.iter().find_map(|a| match a {
        FunctionAnnotation::Section(name) => Some(name.clone()),
        _ => None,
    });
    mir_sig.align = f.annotations.iter().find_map(|a| match a {
        FunctionAnnotation::Align(alignment) => Some(*alignment),
        _ => None,
    });
    mir_sig.variadic = f.signature.variadic;
    if f.annotations.contains(&FunctionAnnotation::Kernel) {
        mir_sig.kernel = Some(Default::default());
    }

    match &f.signature.return_type {
        IRType::Void => {}
        other => {
            let resolved = resolve_type_names(other, &module.type_declarations)
                .ok_or(FromIRError::UnsupportedType)?;
            if let Some(aggregate) = classify_aggregate(&resolved, &format!("return of @{name}"))? {
                mir_sig = mir_sig
                    .with_return(MirType::ptr())
                    .with_return_aggregate(aggregate);
            } else {
                let ty = map_ir_type(other)?;
                mir_sig = mir_sig
                    .with_return(ty)
                    .with_return_signedness(signedness_of(other));
            }
        }
    }

    let mut var_to_reg = VarBindings {
        classes: value_register_classes(f, module)?,
        ..Default::default()
    };
    let mut register_types = HashMap::new();
    let mut reservations: HashMap<VirtualReg, usize> = HashMap::new();
    let mut alignments: HashMap<VirtualReg, usize> = HashMap::new();
    let mut globals = GlobalAddresses::new(module);

    let mut aggregate_prelude = Vec::new();
    for param in &f.signature.params {
        let resolved = resolve_type_names(&param.ty, &module.type_declarations)
            .ok_or(FromIRError::UnsupportedType)?;
        if let Some(aggregate) =
            classify_aggregate(&resolved, &format!("%{} of @{name}", param.name))?
        {
            let slot = vreg_alloc.allocate_gpr();
            let words = aggregate.words();
            if words > 1 {
                reservations.insert(slot, words - 1);
            }
            let addr = Register::Virtual(vreg_alloc.allocate_gpr());
            register_types.insert(addr.clone(), MirType::ptr());
            mir_sig.params.push(
                Parameter::new(Register::Virtual(slot), MirType::ptr()).with_aggregate(aggregate),
            );
            aggregate_prelude.push(Instruction::StackAddr {
                dst: addr.clone(),
                slot,
            });
            if wide::is_wide_type(&resolved) {
                let pair = wide::load_wide_parameter(addr, &mut vreg_alloc, &mut aggregate_prelude);
                var_to_reg.wide.insert(param.name, pair);
            } else {
                var_to_reg.insert(param.name, addr);
            }
            continue;
        }
        let class = reg_class_for_type(&param.ty);
        let v = match class {
            RegisterClass::Fpr => Register::Virtual(vreg_alloc.allocate_fpr()),
            RegisterClass::Vec => Register::Virtual(vreg_alloc.allocate_vec()),
            _ => Register::Virtual(vreg_alloc.allocate_gpr()),
        };
        let mir_ty = map_ir_type(&param.ty)?;
        register_types.insert(v.clone(), mir_ty);
        mir_sig
            .params
            .push(Parameter::new(v.clone(), mir_ty).with_signedness(signedness_of(&param.ty)));
        var_to_reg.insert(param.name, v);
    }

    if f.basic_blocks.is_empty() && f.annotations.contains(&FunctionAnnotation::Extern) {
        // A declaration carries no body; the signature is what callers need.
        return Ok(Function::new(mir_sig));
    }

    let variadic_fixed: HashMap<&'a str, usize> = module
        .functions
        .iter()
        .filter(|(_, callee)| callee.signature.variadic)
        .map(|(name, callee)| (*name, callee.signature.params.len()))
        .collect();

    let mut mir_func = Function::new(mir_sig).with_entry(f.entry_block);

    let mut names: HashSet<&str> = f.signature.params.iter().map(|p| p.name).collect();
    let mut reassigned_names = HashSet::new();
    for block in f.basic_blocks.values() {
        for instruction in &block.instructions {
            if let Some(name) = instruction.defined_var()
                && !names.insert(name)
            {
                reassigned_names.insert(name);
            }
        }
    }

    // BFS traversal starting from entry block, guard with visited to avoid cycles
    let mut visited: HashSet<&'a str> = HashSet::new();
    let mut worklist: VecDeque<&'a str> = VecDeque::new();

    // Ensure entry exists
    if !f.basic_blocks.contains_key(f.entry_block) {
        return Err(FromIRError::MissingEntryBlock);
    }
    worklist.push_back(f.entry_block);
    visited.insert(f.entry_block);

    while let Some(label) = worklist.pop_front() {
        let ir_block = &f.basic_blocks[label];
        let mut mir_block = Block::new(label);

        // Registers already defined in this block. A scan of the emitted instructions per
        // instruction made conversion quadratic in the size of a single-block function.
        let mut defined_in_block: HashSet<Register> = HashSet::new();
        for instr in &ir_block.instructions {
            let mut resolved;
            let instr = if matches!(instr, IRInst::Alloc { .. } | IRInst::GetElemPtr { .. }) {
                resolved = instr.clone();
                let ty = match &mut resolved {
                    IRInst::Alloc { allocated_ty, .. } => allocated_ty,
                    IRInst::GetElemPtr { element_type, .. } => element_type,
                    _ => unreachable!(),
                };
                *ty = resolve_type_names(ty, &module.type_declarations)
                    .ok_or(FromIRError::UnsupportedType)?;
                &resolved
            } else {
                instr
            };
            let mut mir_instrs = convert_instruction(
                instr,
                &mut vreg_alloc,
                &mut var_to_reg,
                &mut reservations,
                &mut alignments,
                &ValueFacts {
                    types: &register_types,
                    variadic_fixed: &variadic_fixed,
                    module,
                },
                &mut globals,
            )
            .map_err(|error| match error {
                FromIRError::UnsupportedInstruction => FromIRError::UnsupportedInstructionIn {
                    instruction: instr.to_string(),
                    block: label.to_string(),
                    function: f.name.to_string(),
                },
                other => other,
            })?;
            if f.basic_blocks.len() == 1
                && let IRInst::Binary { result, .. } | IRInst::WriteByte { result, .. } = instr
                && !var_to_reg.wide.contains_key(result)
                && let Some(last) = mir_instrs.last_mut()
                && let Some(old_dst) = last.def_reg().cloned()
                && defined_in_block.contains(&old_dst)
            {
                let fresh = Register::Virtual(match &old_dst {
                    Register::Virtual(vreg) if vreg.class == RegisterClass::Fpr => {
                        vreg_alloc.allocate_fpr()
                    }
                    _ => vreg_alloc.allocate_gpr(),
                });
                match last {
                    Instruction::IntBinary { dst, .. }
                    | Instruction::IntRotate { dst, .. }
                    | Instruction::FloatBinary { dst, .. } => *dst = fresh.clone(),
                    Instruction::Call { ret, .. } => *ret = Some(fresh.clone()),
                    _ => return Err(FromIRError::UnsupportedInstruction),
                }
                var_to_reg.insert(result, fresh);
            }
            for mut mir_instr in mir_instrs {
                let mut result_address = None;
                if let Instruction::Call {
                    name, ret: Some(_), ..
                } = &mir_instr
                    && let Some(callee) = module.functions.get(name.as_str())
                    && let Some(resolved) =
                        resolve_type_names(&callee.signature.return_type, &module.type_declarations)
                    && let Some(aggregate) =
                        classify_aggregate(&resolved, &format!("return of @{name}"))?
                {
                    let slot = vreg_alloc.allocate_gpr();
                    let words = aggregate.words();
                    if words > 1 {
                        reservations.insert(slot, words - 1);
                    }
                    let Instruction::Call { ret, .. } = &mut mir_instr else {
                        return Err(FromIRError::InvalidIR);
                    };
                    let address = ret
                        .replace(Register::Virtual(slot))
                        .ok_or(FromIRError::InvalidIR)?;
                    register_types.insert(address.clone(), MirType::ptr());
                    result_address = Some(Instruction::StackAddr { dst: address, slot });
                }
                if let Instruction::Call { name, args, .. } = &mut mir_instr
                    && let Some(callee) = module.functions.get(name.as_str())
                {
                    for (argument, parameter) in args.iter_mut().zip(&callee.signature.params) {
                        if is_aggregate(&parameter.ty)
                            || wide::is_wide_type(&parameter.ty)
                            || resolve_type_names(&parameter.ty, &module.type_declarations)
                                .is_some_and(|ty| is_aggregate(&ty) || wide::is_wide_type(&ty))
                        {
                            continue;
                        }
                        let ty = map_ir_type(&parameter.ty)?;
                        normalize_lowered_literal(argument, ty);
                    }
                }
                mir_instr.normalize_integer_immediates(mir_func.sig.ret_ty);
                if let Some(dst) = mir_instr.def_reg() {
                    let ty = match &mir_instr {
                        Instruction::IntConvert { to, .. }
                        | Instruction::Bitcast { to, .. }
                        | Instruction::FloatConvert { to, .. }
                        | Instruction::IntToFloat { to, .. }
                        | Instruction::FloatToInt { to, .. } => Some(*to),
                        Instruction::Copy { ty, .. }
                        | Instruction::IntBinary { ty, .. }
                        | Instruction::FloatBinary { ty, .. }
                        | Instruction::FloatUnary { ty, .. }
                        | Instruction::IntUnary { ty, .. }
                        | Instruction::IntRotate { ty, .. }
                        | Instruction::Select { ty, .. }
                        | Instruction::Load { ty, .. }
                        | Instruction::AtomicLoad { ty, .. }
                        | Instruction::AtomicBinary { ty, .. }
                        | Instruction::VaArg { ty, .. } => Some(*ty),
                        Instruction::IntCmp { .. } | Instruction::FloatCmp { .. } => {
                            Some(MirType::i1())
                        }
                        Instruction::CallIndirect { ret_ty, .. } => *ret_ty,
                        Instruction::Call { name, .. } => module
                            .functions
                            .get(name.as_str())
                            .and_then(|callee| map_ir_type(&callee.signature.return_type).ok())
                            .or_else(|| io_intrinsic_result_type(name)),
                        Instruction::Lea { .. } | Instruction::StackAddr { .. } => {
                            Some(MirType::ptr())
                        }
                        _ => None,
                    };
                    if let Some(ty) = ty {
                        register_types.insert(dst.clone(), ty);
                    }
                }
                if let Some(dst) = mir_instr.def_reg() {
                    defined_in_block.insert(dst.clone());
                }
                mir_block.push(mir_instr);
                if let Some(address) = result_address {
                    if let Some(dst) = address.def_reg() {
                        defined_in_block.insert(dst.clone());
                    }
                    mir_block.push(address);
                }
            }
        }

        // After converting this block, enqueue IR successors (prevents borrow/move issues)
        if let Some(term) = ir_block.instructions.last() {
            match term {
                IRInst::Br {
                    true_label,
                    false_label,
                    ..
                } => {
                    if !visited.contains(true_label) && f.basic_blocks.contains_key(true_label) {
                        visited.insert(true_label);
                        worklist.push_back(true_label);
                    }
                    if !visited.contains(false_label) && f.basic_blocks.contains_key(false_label) {
                        visited.insert(false_label);
                        worklist.push_back(false_label);
                    }
                }
                IRInst::Jmp { target_label } => {
                    if !visited.contains(target_label) && f.basic_blocks.contains_key(target_label)
                    {
                        visited.insert(target_label);
                        worklist.push_back(target_label);
                    }
                }
                IRInst::Switch { default, cases, .. } => {
                    if !visited.contains(default) && f.basic_blocks.contains_key(default) {
                        visited.insert(default);
                        worklist.push_back(default);
                    }
                    for (_, case_label) in cases {
                        let case_str: &str = case_label;
                        if !visited.contains(case_str) && f.basic_blocks.contains_key(case_str) {
                            visited.insert(case_str);
                            worklist.push_back(case_str);
                        }
                    }
                }
                _ => {}
            }
        }

        mir_func.add_block(mir_block);
    }

    {
        let mut phis = Vec::new();
        let mut edge_moves: HashMap<(String, String), Vec<Instruction>> = HashMap::new();

        for (succ_label, ir_block) in &f.basic_blocks {
            for instr in &ir_block.instructions {
                if let IRInst::Phi {
                    result,
                    ty,
                    incoming,
                } = instr
                    && wide::is_wide_type(ty)
                    && let Some(pair) = var_to_reg.wide.get(result).cloned()
                {
                    let mut halves: Vec<PhiVerification> = pair
                        .iter()
                        .map(|dst| PhiVerification {
                            block: (*succ_label).to_string(),
                            dst: dst.clone(),
                            ty: MirType::i64(),
                            incoming: Vec::new(),
                        })
                        .collect();
                    for (value, pred_label) in incoming {
                        if !visited.contains(pred_label) {
                            continue;
                        }
                        let (lo, hi) = wide::phi_words(
                            value,
                            &mut vreg_alloc,
                            &mut var_to_reg,
                            &mut globals,
                            &mut reservations,
                            module,
                        )?;
                        for ((half, dst), src) in halves.iter_mut().zip(&pair).zip([lo, hi]) {
                            half.incoming.push(((*pred_label).to_string(), src.clone()));
                            edge_moves
                                .entry(((*pred_label).to_string(), (*succ_label).to_string()))
                                .or_default()
                                .push(Instruction::Copy {
                                    ty: MirType::i64(),
                                    dst: dst.clone(),
                                    src,
                                });
                        }
                    }
                    phis.extend(halves);
                    continue;
                }
                if let IRInst::Phi {
                    result,
                    ty,
                    incoming,
                } = instr
                {
                    if let Some(dst_reg) = var_to_reg.get(result).cloned() {
                        let mir_ty = map_ir_type(ty)?;
                        let mut phi = PhiVerification {
                            block: (*succ_label).to_string(),
                            dst: dst_reg.clone(),
                            ty: mir_ty,
                            incoming: Vec::new(),
                        };
                        for (val, pred_label) in incoming {
                            if !visited.contains(pred_label) {
                                continue;
                            }
                            let src_op = resolve_operand(
                                val,
                                &mut vreg_alloc,
                                &mut var_to_reg,
                                &mut globals,
                            )?;
                            let mut mov_like = Instruction::Copy {
                                ty: mir_ty,
                                dst: dst_reg.clone(),
                                src: src_op,
                            };
                            mov_like.normalize_integer_immediates(None);
                            if let Instruction::Copy { src, .. } = &mov_like {
                                phi.incoming.push(((*pred_label).to_string(), src.clone()));
                            }
                            edge_moves
                                .entry(((*pred_label).to_string(), (*succ_label).to_string()))
                                .or_default()
                                .push(mov_like);
                        }
                        phis.push(phi);
                    } else {
                        // Bind destination conservatively to avoid cascading unknowns
                        let new_dst = Register::Virtual(vreg_alloc.allocate_gpr());
                        var_to_reg.insert(*result, new_dst);
                    }
                }
            }
        }

        let mut prelude = aggregate_prelude;
        prelude.extend(globals.prelude());
        if !prelude.is_empty()
            && let Some(entry) = mir_func.get_block_mut(f.entry_block)
        {
            entry.instructions.splice(0..0, prelude);
        }

        verify_before_phi_lowering(
            &mir_func,
            &phis,
            &reassigned_names
                .iter()
                .filter_map(|name| var_to_reg.get(name).cloned())
                .collect(),
        )
        .map_err(FromIRError::InvalidMir)?;

        insert_edge_moves(
            &mut mir_func,
            edge_moves.into_iter().collect(),
            &mut vreg_alloc,
        );
    }

    mir_func.stack_reservations = reservations;
    mir_func.stack_alignments = alignments;
    libcalls.extend(mem::take(&mut var_to_reg.libcalls));

    Ok(mir_func)
}

fn convert_instruction<'a>(
    instr: &IRInst<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut VarBindings<'a>,
    reservations: &mut HashMap<VirtualReg, usize>,
    alignments: &mut HashMap<VirtualReg, usize>,
    facts: &ValueFacts<'_, 'a>,
    globals: &mut GlobalAddresses<'a>,
) -> Result<Vec<Instruction>, FromIRError> {
    let mut wide_cx = wide::Cx {
        vregs: &mut *vreg_alloc,
        vars: &mut *var_to_reg,
        reservations: &mut *reservations,
        globals: &mut *globals,
        module: facts.module,
        out: Vec::new(),
    };
    if wide::lower_instruction(instr, &mut wide_cx)? {
        return Ok(wide_cx.out);
    }
    let register_types = facts.types;
    fn ir_address_mode_to_mir<'b>(
        ptr: &IRVal<'b>,
        vreg_alloc: &mut VirtualRegAllocator,
        var_to_reg: &mut VarBindings<'b>,
        globals: &mut GlobalAddresses<'b>,
    ) -> Result<AddressMode, FromIRError> {
        let base = match ptr {
            IRVal::Variable(id) => {
                if let Some(r) = var_to_reg.get(id) {
                    r.clone()
                } else {
                    let r = Register::Virtual(vreg_alloc.allocate_gpr());
                    var_to_reg.insert(*id, r.clone());
                    r
                }
            }
            IRVal::Global(name) => globals.register(name, vreg_alloc)?,
            IRVal::Constant(_) => return Err(FromIRError::UnsupportedInstruction),
        };
        Ok(AddressMode::BaseOffset {
            base,
            offset: 0,
            address_space: Default::default(),
        })
    }

    match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            // Reuse existing binding for result if it exists; otherwise allocate a new one
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = match *ty {
                    IRPrim::F32 | IRPrim::F64 => Register::Virtual(vreg_alloc.allocate_fpr()),
                    _ => Register::Virtual(vreg_alloc.allocate_gpr()),
                };
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            let mir_ty = map_ir_prim(*ty)?;
            // Read operands using the PREVIOUS bindings to avoid using the new dest mapping
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;

            let mir = match (op, *ty) {
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FAdd, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FSub, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FMul, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FDiv, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Rotl | IRBin::Rotr, IRPrim::F32 | IRPrim::F64) => {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                (IRBin::Rotl, _) => Instruction::IntRotate {
                    op: IntRotOp::Rotl,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Rotr, _) => Instruction::IntRotate {
                    op: IntRotOp::Rotr,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (op, prim_ty) => match lower_int_binop(op, prim_ty) {
                    Some(op) => Instruction::IntBinary {
                        op,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    },
                    None => return Err(FromIRError::UnsupportedInstruction),
                },
            };
            Ok(vec![mir])
        }
        // SSA merge: create a binding for the phi result so subsequent uses resolve.
        // Semantics are not materialized here; a later SSA elimination pass should lower this.
        IRInst::Phi { result, .. } => {
            resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![])
        }
        IRInst::Unary {
            op,
            result,
            ty,
            value,
        } => {
            let mir_ty = map_ir_prim(*ty)?;
            let float = mir_ty.is_float();
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, mir_ty);
            let instruction = match (op, float) {
                (IRUnary::Sqrt, true) => Instruction::FloatUnary {
                    op: FloatUnOp::FSqrt,
                    ty: mir_ty,
                    dst,
                    src,
                },
                (IRUnary::Neg, true) => Instruction::FloatUnary {
                    op: FloatUnOp::FNeg,
                    ty: mir_ty,
                    dst,
                    src,
                },
                (IRUnary::Neg, false) => Instruction::IntBinary {
                    op: IntBinOp::Sub,
                    ty: mir_ty,
                    dst,
                    lhs: Operand::Immediate(zero_immediate(mir_ty)),
                    rhs: src,
                },
                (IRUnary::Not, false) => Instruction::IntBinary {
                    op: IntBinOp::Xor,
                    ty: mir_ty,
                    dst,
                    lhs: src,
                    rhs: Operand::Immediate(all_ones_immediate(mir_ty)),
                },
                (IRUnary::Popcount | IRUnary::Ctlz | IRUnary::Cttz | IRUnary::Bswap, true) => {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                (IRUnary::Popcount | IRUnary::Ctlz | IRUnary::Cttz | IRUnary::Bswap, false)
                    if !is_integer_prim(*ty) =>
                {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                (IRUnary::Popcount | IRUnary::Ctlz | IRUnary::Cttz | IRUnary::Bswap, false) => {
                    let op = lower_int_unop(op).ok_or(FromIRError::UnsupportedInstruction)?;
                    Instruction::IntUnary {
                        op,
                        ty: mir_ty,
                        dst,
                        src,
                    }
                }
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![instruction])
        }
        IRInst::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;
            let mir = match (*ty, op) {
                (IRPrim::F32 | IRPrim::F64, IRCmp::Eq) => Instruction::FloatCmp {
                    op: FloatCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ne) => Instruction::FloatCmp {
                    op: FloatCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Lt) => Instruction::FloatCmp {
                    op: FloatCmpOp::Lt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Le) => Instruction::FloatCmp {
                    op: FloatCmpOp::Le,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Gt) => Instruction::FloatCmp {
                    op: FloatCmpOp::Gt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ge) => Instruction::FloatCmp {
                    op: FloatCmpOp::Ge,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Eq) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::Eq,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ne) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::Ne,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Lt) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::ULt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Le) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::ULe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Gt) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::UGt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ge) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::UGe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (_, IRCmp::Eq) => Instruction::IntCmp {
                    op: IntCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ne) => Instruction::IntCmp {
                    op: IntCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Lt) => Instruction::IntCmp {
                    op: IntCmpOp::SLt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Le) => Instruction::IntCmp {
                    op: IntCmpOp::SLe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Gt) => Instruction::IntCmp {
                    op: IntCmpOp::SGt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ge) => Instruction::IntCmp {
                    op: IntCmpOp::SGe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
            };
            Ok(vec![mir])
        }
        IRInst::Br {
            condition,
            true_label,
            false_label,
        } => {
            let cond_reg = match condition {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::Br {
                cond: cond_reg,
                true_target: (*true_label).to_string(),
                false_target: (*false_label).to_string(),
            }])
        }
        IRInst::Switch {
            ty: _,
            value,
            default,
            cases,
        } => {
            // Lower IR switch to MIR switch using a dedicated MIR instruction.
            // The MIR lowering and backends treat the value as an integer in a register
            // and cases as i64 immediates.
            let value_reg = match value {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };

            let mut mir_cases = Vec::with_capacity(cases.len());

            for (lit, label) in cases {
                let key: i64 = match lit {
                    IRLit::I8(v) => *v as i64,
                    IRLit::I16(v) => *v as i64,
                    IRLit::I32(v) => *v as i64,
                    IRLit::I64(v) => *v,
                    IRLit::U8(v) => *v as i64,
                    IRLit::U16(v) => *v as i64,
                    IRLit::U32(v) => *v as i64,
                    IRLit::U64(v) => *v as i64,
                    IRLit::I128(_) | IRLit::U128(_) => {
                        return Err(FromIRError::UnsupportedType);
                    }
                    IRLit::Bool(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
                    IRLit::F32(_) | IRLit::F64(_) | IRLit::String(_) => {
                        return Err(FromIRError::UnsupportedType);
                    }
                };
                mir_cases.push((key, (*label).to_string()));
            }

            Ok(vec![Instruction::Switch {
                value: value_reg,
                cases: mir_cases,
                default: (*default).to_string(),
            }])
        }
        IRInst::Jmp { target_label } => Ok(vec![Instruction::Jmp {
            target: (*target_label).to_string(),
        }]),
        IRInst::Unreachable => Ok(vec![Instruction::Unreachable]),
        IRInst::Ret { ty, value } => {
            if matches!(ty, IRType::Void) {
                return Ok(vec![Instruction::Ret { value: None }]);
            }
            let op = match value {
                Some(v) => Some(resolve_operand(v, vreg_alloc, var_to_reg, globals)?),
                None => None,
            };
            Ok(vec![Instruction::Ret { value: op }])
        }
        IRInst::Load {
            result,
            ty,
            ptr,
            volatile,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                IRVal::Global(name) => globals.register(name, vreg_alloc)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let addr = AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            };
            Ok(vec![Instruction::Load {
                ty: mir_ty,
                dst,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: *volatile,
                },
            }])
        }
        IRInst::Store {
            ty,
            ptr,
            value,
            volatile,
        } => {
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                IRVal::Global(name) => globals.register(name, vreg_alloc)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, mir_ty);
            let addr = AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            };
            Ok(vec![Instruction::Store {
                ty: mir_ty,
                src,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: *volatile,
                },
            }])
        }
        IRInst::Call {
            result,
            func_name,
            args,
        } => {
            let mut mir_args = Vec::new();
            for a in args {
                mir_args.push(resolve_operand(a, vreg_alloc, var_to_reg, globals)?);
            }
            // C widens a variadic f32 to double before the call, and every ABI's
            // variadic rules are written for the widened value.
            let fixed = facts
                .variadic_fixed
                .get(func_name)
                .copied()
                .unwrap_or(mir_args.len());
            let mut instructions = Vec::new();
            for arg in mir_args.iter_mut().skip(fixed) {
                let Operand::Register(reg) = arg else {
                    continue;
                };
                if facts.types.get(reg) != Some(&MirType::f32()) {
                    continue;
                }
                let dst = Register::Virtual(vreg_alloc.allocate_fpr());
                instructions.push(Instruction::FloatConvert {
                    from: MirType::f32(),
                    to: MirType::f64(),
                    dst: dst.clone(),
                    src: arg.clone(),
                });
                *arg = Operand::Register(dst);
            }
            let ret = result.map(|res| resolve_or_alloc(res, vreg_alloc, var_to_reg));
            instructions.push(Instruction::Call {
                name: (*func_name).to_string(),
                args: mir_args,
                ret,
            });
            Ok(instructions)
        }
        IRInst::FuncAddr { result, func_name } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::SymbolAddress {
                dst,
                symbol: (*func_name).to_string(),
            }])
        }
        IRInst::CallIndirect {
            result,
            target,
            args,
            variadic_fixed,
            return_type,
        } => {
            let target = resolve_operand(target, vreg_alloc, var_to_reg, globals)?;
            let mut mir_args = Vec::new();
            let mut arg_types = Vec::new();
            let mut instructions = Vec::new();
            let fixed = variadic_fixed.unwrap_or(args.len());
            for (index, (arg_ty, argument)) in args.iter().enumerate() {
                let mut mir_ty = map_ir_type(arg_ty)?;
                let mut operand = resolve_operand(argument, vreg_alloc, var_to_reg, globals)?;
                normalize_lowered_literal(&mut operand, mir_ty);
                if index >= fixed && mir_ty == MirType::f32() {
                    let dst = Register::Virtual(vreg_alloc.allocate_fpr());
                    instructions.push(Instruction::FloatConvert {
                        from: MirType::f32(),
                        to: MirType::f64(),
                        dst: dst.clone(),
                        src: operand,
                    });
                    operand = Operand::Register(dst);
                    mir_ty = MirType::f64();
                }
                arg_types.push(mir_ty);
                mir_args.push(operand);
            }
            let ret_ty = match return_type {
                IRType::Void => None,
                other => Some(map_ir_type(other)?),
            };
            let ret = result.map(|name| resolve_or_alloc(name, vreg_alloc, var_to_reg));
            instructions.push(Instruction::CallIndirect {
                target,
                arg_types,
                args: mir_args,
                variadic_fixed: *variadic_fixed,
                ret,
                ret_ty,
            });
            Ok(instructions)
        }
        IRInst::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        }
        | IRInst::Trunc {
            result,
            source_type,
            target_type,
            value,
        }
        | IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let signed = matches!(instr, IRInst::SignExtend { .. });
            let widening = !matches!(instr, IRInst::Trunc { .. });
            let name = if signed {
                "sext"
            } else if widening {
                "zext"
            } else {
                "trunc"
            };
            if from.is_float() || to.is_float() {
                return Err(FromIRError::InvalidConversion(format!(
                    "{name} relates two integers; use fptoint or inttofp to cross between integers and floats"
                )));
            }
            let bits = |ty: MirType| {
                if ty == MirType::i1() {
                    1
                } else {
                    ty.size_bytes() * 8
                }
            };
            if widening && bits(to) < bits(from) {
                return Err(FromIRError::InvalidConversion(format!(
                    "{name} from {source_type} to {target_type} narrows; use trunc"
                )));
            }
            if !widening && bits(to) > bits(from) {
                return Err(FromIRError::InvalidConversion(format!(
                    "trunc from {source_type} to {target_type} widens; use zext or sext"
                )));
            }
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            if from == to {
                return Ok(vec![Instruction::Copy { ty: to, dst, src }]);
            }
            Ok(vec![Instruction::IntConvert {
                from,
                to,
                signed,
                dst,
                src,
            }])
        }
        IRInst::IntToFloat {
            result,
            signed,
            source_type,
            target_type,
            value,
        } => {
            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let dst = match var_to_reg.get(result) {
                Some(existing) => existing.clone(),
                None => {
                    let fresh = Register::Virtual(vreg_alloc.allocate_fpr());
                    var_to_reg.insert(result, fresh.clone());
                    fresh
                }
            };
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, from);
            Ok(vec![Instruction::IntToFloat {
                from,
                to,
                signed: *signed,
                dst,
                src,
            }])
        }
        IRInst::FloatToInt {
            result,
            signed,
            source_type,
            target_type,
            value,
        } => {
            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, from);
            Ok(vec![Instruction::FloatToInt {
                from,
                to,
                signed: *signed,
                dst,
                src,
            }])
        }
        IRInst::FloatConvert {
            result,
            source_type,
            target_type,
            value,
        } => {
            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            if from == to || !from.is_float() || !to.is_float() {
                return Err(FromIRError::InvalidConversion(format!(
                    "fpcast from {source_type} to {target_type} converts between the float types only"
                )));
            }
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, from);
            Ok(vec![Instruction::FloatConvert { from, to, dst, src }])
        }
        IRInst::Bitcast {
            result,
            source_type,
            target_type,
            value,
        } => {
            if int_bits_for_bitcast(source_type) != int_bits_for_bitcast(target_type) {
                return Err(FromIRError::UnsupportedType);
            }

            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut src, from);
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Bitcast { from, to, dst, src }])
        }

        IRInst::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);

            let mir_ty = map_ir_type(ty)?;

            // Condition must be a variable bound to a MIR register.
            let cond_reg = match cond {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };

            let mut tv = resolve_operand(true_val, vreg_alloc, var_to_reg, globals)?;
            let mut fv = resolve_operand(false_val, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut tv, mir_ty);
            normalize_lowered_literal(&mut fv, mir_ty);

            Ok(vec![Instruction::Select {
                dst,
                ty: mir_ty,
                cond: cond_reg,
                true_val: tv,
                false_val: fv,
            }])
        }
        IRInst::PtrToInt {
            result,
            ptr_value,
            target_type,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(ptr_value, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::IntConvert {
                from: MirType::ptr(),
                to: map_ir_prim(*target_type)?,
                signed: false,
                dst,
                src: val_op,
            }])
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type: _,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(int_value, vreg_alloc, var_to_reg, globals)?;
            let from = operand_type(&val_op, register_types);
            Ok(vec![Instruction::IntConvert {
                from,
                to: MirType::ptr(),
                signed: false,
                dst,
                src: val_op,
            }])
        }
        IRInst::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
            field_byte_offset,
        } => {
            let base = pointer_base(struct_ptr, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            // Without the struct type the text syntax leaves no layout to read, so
            // the fallback is one 8-byte slot per field.
            let offset = match field_byte_offset {
                Some(bytes) => {
                    i64::try_from(*bytes).map_err(|_| FromIRError::UnsupportedInstruction)?
                }
                None => i64::try_from(*field_index)
                    .ok()
                    .and_then(|index: i64| index.checked_mul(8))
                    .ok_or(FromIRError::UnsupportedInstruction)?,
            };
            Ok(displace_pointer(dst, base, offset, vreg_alloc))
        }
        IRInst::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        } => {
            // Constant-index GEP -> single LEA; variable index currently unsupported in MIR single-op model
            let base = pointer_base(array_ptr, vreg_alloc, var_to_reg, globals)?;
            let elem_size = sizeof_type(element_type)
                .and_then(|size| i64::try_from(size).ok())
                .ok_or(FromIRError::UnsupportedInstruction)?;
            let idx_const = match index {
                IRVal::Constant(lit) => match lit {
                    IRLit::I64(v) => Some(*v),
                    IRLit::I32(v) => Some(*v as i64),
                    IRLit::U64(v) => Some(*v as i64),
                    IRLit::U32(v) => Some(*v as i64),
                    _ => None,
                },
                _ => None,
            };
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);

            if let Some(iv) = idx_const {
                let offset = iv
                    .checked_mul(elem_size)
                    .ok_or(FromIRError::UnsupportedInstruction)?;
                Ok(displace_pointer(dst, base, offset, vreg_alloc))
            } else {
                let index_op = resolve_operand(index, vreg_alloc, var_to_reg, globals)?;
                let from = operand_type(&index_op, register_types);
                let wide_index = Register::Virtual(vreg_alloc.allocate_gpr());
                let scaled = Register::Virtual(vreg_alloc.allocate_gpr());
                let address = Register::Virtual(vreg_alloc.allocate_gpr());
                let sum = Register::Virtual(vreg_alloc.allocate_gpr());
                Ok(vec![
                    Instruction::IntConvert {
                        from,
                        to: MirType::i64(),
                        signed: from.size_bytes() < 8,
                        dst: wide_index.clone(),
                        src: index_op,
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Mul,
                        ty: MirType::i64(),
                        dst: scaled.clone(),
                        lhs: Operand::Register(wide_index),
                        rhs: Operand::Immediate(Immediate::I64(elem_size)),
                    },
                    Instruction::IntConvert {
                        from: MirType::ptr(),
                        to: MirType::i64(),
                        signed: false,
                        dst: address.clone(),
                        src: Operand::Register(base),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: MirType::i64(),
                        dst: sum.clone(),
                        lhs: Operand::Register(address),
                        rhs: Operand::Register(scaled),
                    },
                    Instruction::IntConvert {
                        from: MirType::i64(),
                        to: MirType::ptr(),
                        signed: false,
                        dst,
                        src: Operand::Register(sum),
                    },
                ])
            }
        }
        IRInst::Write {
            buffer,
            size,
            result,
        } => {
            let buf = resolve_operand(buffer, vreg_alloc, var_to_reg, globals)?;
            let sz = resolve_operand(size, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "lamina.write".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        IRInst::WriteByte { value, result } => {
            let v = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "lamina.writebyte".to_string(),
                args: vec![v],
                ret: Some(dst),
            }])
        }
        IRInst::ReadByte { result } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "lamina.readbyte".to_string(),
                args: vec![],
                ret: Some(dst),
            }])
        }
        IRInst::Read {
            buffer,
            size,
            result,
        } => {
            let buf = resolve_operand(buffer, vreg_alloc, var_to_reg, globals)?;
            let sz = resolve_operand(size, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "lamina.read".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        IRInst::VaStart { ap } => Ok(vec![Instruction::VaStart {
            ap: resolve_operand(ap, vreg_alloc, var_to_reg, globals)?,
        }]),
        IRInst::VaArg { result, ty, ap } => {
            if matches!(ty, IRPrim::Bool | IRPrim::F32) {
                return Err(FromIRError::VariadicArgument(format!(
                    "va_arg.{ty} is not supported; C passes a bool or float as an int or double"
                )));
            }
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let ty = map_ir_prim(*ty)?;
            let ap = resolve_operand(ap, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::VaArg { ty, dst, ap }])
        }
        IRInst::VaEnd { ap } => Ok(vec![Instruction::VaEnd {
            ap: resolve_operand(ap, vreg_alloc, var_to_reg, globals)?,
        }]),
        IRInst::VaCopy { dst, src } => Ok(vec![Instruction::VaCopy {
            dst: resolve_operand(dst, vreg_alloc, var_to_reg, globals)?,
            src: resolve_operand(src, vreg_alloc, var_to_reg, globals)?,
        }]),
        IRInst::Alloc {
            result,
            alloc_type,
            allocated_ty,
            align,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);

            let size = sizeof_type(allocated_ty).ok_or(FromIRError::UnsupportedType)?;
            let size = i64::try_from(size).map_err(|_| FromIRError::UnsupportedType)?;
            match alloc_type {
                AllocType::Stack => {
                    let storage_vreg = vreg_alloc.allocate_gpr();
                    let slots_needed = usize::try_from((size as u64).div_ceil(8))
                        .map_err(|_| FromIRError::UnsupportedType)?;
                    if slots_needed > 1 {
                        reservations.insert(storage_vreg, slots_needed - 1);
                    }
                    if let Some(align) = *align {
                        if align > 16 {
                            return Err(FromIRError::UnsupportedAlignment(format!(
                                "stackalloc align {align} for %{result}: frames are only 16-byte aligned and frame realignment is not implemented yet"
                            )));
                        }
                        if align > 8 {
                            alignments.insert(storage_vreg, align as usize);
                        }
                    }
                    Ok(vec![Instruction::StackAddr {
                        dst,
                        slot: storage_vreg,
                    }])
                }
            }
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdBinary {
            op,
            result,
            vector_type,
            lhs,
            rhs,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_vec());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_type(vector_type)?;
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;
            let mir_op = match op {
                IRSimdOp::Add => SimdOp::Add,
                IRSimdOp::Sub => SimdOp::Sub,
                IRSimdOp::Mul => SimdOp::Mul,
                IRSimdOp::Div => SimdOp::Div,
                IRSimdOp::Min => SimdOp::Min,
                IRSimdOp::Max => SimdOp::Max,
                IRSimdOp::Abs => SimdOp::Abs,
                IRSimdOp::Neg => SimdOp::Neg,
                IRSimdOp::Sqrt => SimdOp::Sqrt,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::SimdBinary {
                op: mir_op,
                ty: mir_ty,
                dst,
                lhs: lhs_op,
                rhs: rhs_op,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdUnary {
            op,
            result,
            vector_type,
            operand,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_vec());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_type(vector_type)?;
            let src_op = resolve_operand(operand, vreg_alloc, var_to_reg, globals)?;
            let mir_op = match op {
                IRSimdOp::Abs => SimdOp::Abs,
                IRSimdOp::Neg => SimdOp::Neg,
                IRSimdOp::Sqrt => SimdOp::Sqrt,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::SimdUnary {
                op: mir_op,
                ty: mir_ty,
                dst,
                src: src_op,
            }])
        }
        IRInst::AtomicLoad {
            result,
            ty,
            ptr,
            ordering,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::AtomicLoad {
                ty: mir_ty,
                dst,
                addr,
                ordering: ir_ordering_to_mir(ordering),
            }])
        }
        IRInst::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        } => {
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg, globals)?;
            let val_op = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::AtomicStore {
                ty: mir_ty,
                src: val_op,
                addr,
                ordering: ir_ordering_to_mir(ordering),
            }])
        }
        IRInst::AtomicBinary {
            op,
            result,
            ty,
            ptr,
            value,
            ordering,
        } => {
            let mir_op = match op {
                IRAtomicBinOp::Add => AtomicBinOp::Add,
                IRAtomicBinOp::Sub => AtomicBinOp::Sub,
                IRAtomicBinOp::And => AtomicBinOp::And,
                IRAtomicBinOp::Or => AtomicBinOp::Or,
                IRAtomicBinOp::Xor => AtomicBinOp::Xor,
                IRAtomicBinOp::Exchange => AtomicBinOp::Exchange,
                IRAtomicBinOp::CompareExchange => return Err(FromIRError::UnsupportedInstruction),
            };
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg, globals)?;
            let val_op = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::AtomicBinary {
                op: mir_op,
                ty: mir_ty,
                dst,
                addr,
                value: val_op,
                ordering: ir_ordering_to_mir(ordering),
            }])
        }
        IRInst::Fence { ordering } => Ok(vec![Instruction::Fence {
            ordering: ir_ordering_to_mir(ordering),
        }]),
        IRInst::InlineAsm {
            result,
            template,
            operands,
            clobbers,
        } => {
            let mut mir_operands = Vec::with_capacity(operands.len());
            for operand in operands {
                mir_operands.push(match operand {
                    IRAsmOperand::In { location, value } => AsmOperand::Input {
                        location: (*location).to_owned(),
                        value: resolve_operand(value, vreg_alloc, var_to_reg, globals)?,
                    },
                    IRAsmOperand::Out { location } => AsmOperand::Output {
                        location: (*location).to_owned(),
                    },
                });
            }
            let result = match result {
                Some((name, ty)) => Some((
                    resolve_or_alloc(name, vreg_alloc, var_to_reg),
                    map_ir_prim(*ty)?,
                )),
                None => None,
            };
            Ok(vec![Instruction::InlineAsm {
                template: (*template).to_owned(),
                operands: mir_operands,
                clobbers: clobbers
                    .iter()
                    .filter(|name| !matches!(**name, "memory" | "flags"))
                    .map(|name| (*name).to_owned())
                    .collect(),
                clobber_memory: clobbers.contains(&"memory"),
                result,
            }])
        }
        IRInst::AtomicCompareExchange {
            result,
            ty,
            ptr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg, globals)?;
            let expected = resolve_operand(expected, vreg_alloc, var_to_reg, globals)?;
            let desired = resolve_operand(desired, vreg_alloc, var_to_reg, globals)?;
            Ok(vec![Instruction::AtomicCompareExchange {
                ty: mir_ty,
                dst,
                addr,
                expected,
                desired,
                success_ordering: ir_ordering_to_mir(success_ordering),
                failure_ordering: ir_ordering_to_mir(failure_ordering),
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdTernary { .. }
        | IRInst::SimdShuffle { .. }
        | IRInst::SimdExtract { .. }
        | IRInst::SimdInsert { .. }
        | IRInst::SimdLoad { .. }
        | IRInst::SimdStore { .. } => Err(FromIRError::UnsupportedInstruction),
    }
}

fn literal_to_immediate(l: &IRLit<'_>) -> Result<Immediate, FromIRError> {
    let imm = match l {
        IRLit::I8(v) => Immediate::I8(*v),
        IRLit::I16(v) => Immediate::I16(*v),
        IRLit::I32(v) => Immediate::I32(*v),
        IRLit::I64(v) => Immediate::I64(*v),
        IRLit::U8(v) => Immediate::I8(*v as i8),
        IRLit::U16(v) => Immediate::I16(*v as i16),
        IRLit::U32(v) => Immediate::I32(*v as i32),
        IRLit::U64(v) => Immediate::I64(*v as i64),
        IRLit::F32(v) => Immediate::F32(*v),
        IRLit::F64(v) => Immediate::F64(*v),
        IRLit::Bool(b) => Immediate::I8(if *b { 1 } else { 0 }),
        IRLit::I128(_) | IRLit::U128(_) | IRLit::String(_) => {
            return Err(FromIRError::UnsupportedInstruction);
        }
    };
    Ok(imm)
}

/// Adds a constant byte offset to a pointer. `Lea` carries a signed 32-bit
/// displacement, so a wider offset goes through an integer add instead.
fn displace_pointer(
    dst: Register,
    base: Register,
    offset: i64,
    vreg_alloc: &mut VirtualRegAllocator,
) -> Vec<Instruction> {
    if let Ok(offset) = i32::try_from(offset) {
        return vec![Instruction::Lea { dst, base, offset }];
    }
    let address = Register::Virtual(vreg_alloc.allocate_gpr());
    let sum = Register::Virtual(vreg_alloc.allocate_gpr());
    vec![
        Instruction::IntConvert {
            from: MirType::ptr(),
            to: MirType::i64(),
            signed: false,
            dst: address.clone(),
            src: Operand::Register(base),
        },
        Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::i64(),
            dst: sum.clone(),
            lhs: Operand::Register(address),
            rhs: Operand::Immediate(Immediate::I64(offset)),
        },
        Instruction::IntConvert {
            from: MirType::i64(),
            to: MirType::ptr(),
            signed: false,
            dst,
            src: Operand::Register(sum),
        },
    ]
}

/// Register holding the address a field or element offset applies to. A global name
/// resolves to its symbol address, the same register `load` and `store` use.
fn pointer_base<'a>(
    ptr: &IRVal<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &VarBindings<'a>,
    globals: &mut GlobalAddresses<'a>,
) -> Result<Register, FromIRError> {
    match ptr {
        IRVal::Variable(id) => var_to_reg
            .get(id)
            .cloned()
            .ok_or(FromIRError::UnknownVariable),
        IRVal::Global(name) => globals.register(name, vreg_alloc),
        IRVal::Constant(_) => Err(FromIRError::UnsupportedInstruction),
    }
}

fn zero_immediate(ty: MirType) -> Immediate {
    match ty.size_bytes() {
        1 => Immediate::I8(0),
        2 => Immediate::I16(0),
        4 => Immediate::I32(0),
        _ => Immediate::I64(0),
    }
}

fn all_ones_immediate(ty: MirType) -> Immediate {
    match ty.size_bytes() {
        1 => Immediate::I8(-1),
        2 => Immediate::I16(-1),
        4 => Immediate::I32(-1),
        _ => Immediate::I64(-1),
    }
}

fn operand_type(operand: &Operand, types: &HashMap<Register, MirType>) -> MirType {
    match operand {
        Operand::Register(register) => types.get(register).copied().unwrap_or(MirType::i64()),
        Operand::Immediate(Immediate::I8(_)) => MirType::Scalar(ScalarType::I8),
        Operand::Immediate(Immediate::I16(_)) => MirType::Scalar(ScalarType::I16),
        Operand::Immediate(Immediate::I32(_)) => MirType::i32(),
        Operand::Immediate(Immediate::F32(_)) => MirType::f32(),
        Operand::Immediate(Immediate::F64(_)) => MirType::f64(),
        _ => MirType::i64(),
    }
}

fn reg_class_for_prim(ty: IRPrim) -> RegisterClass {
    match ty {
        IRPrim::F32 | IRPrim::F64 => RegisterClass::Fpr,
        _ => RegisterClass::Gpr,
    }
}

/// Register class of every value a function defines, keyed by IR name. Lowering reads
/// an operand before the instruction defining it in the single pass over a block, so the
/// class cannot be deduced from the binding that already exists.
fn value_register_classes<'a>(
    f: &IRFunction<'a>,
    module: &IRModule<'a>,
) -> Result<HashMap<&'a str, RegisterClass>, FromIRError> {
    let mut classes: HashMap<&'a str, RegisterClass> = HashMap::new();
    let class_of_type = |ty: &IRType<'a>| {
        let resolved = resolve_type_names(ty, &module.type_declarations);
        reg_class_for_type(resolved.as_ref().unwrap_or(ty))
    };
    for param in &f.signature.params {
        insert_register_class(&mut classes, param.name, class_of_type(&param.ty))?;
    }
    for block in f.basic_blocks.values() {
        for instruction in &block.instructions {
            let (name, class) = match instruction {
                IRInst::Binary { result, ty, .. } | IRInst::Unary { result, ty, .. } => {
                    (*result, reg_class_for_prim(*ty))
                }
                IRInst::Select { result, ty, .. }
                | IRInst::Load { result, ty, .. }
                | IRInst::AtomicLoad { result, ty, .. }
                | IRInst::AtomicBinary { result, ty, .. }
                | IRInst::Phi { result, ty, .. } => (*result, class_of_type(ty)),
                IRInst::VaArg { result, ty, .. } => (*result, reg_class_for_prim(*ty)),
                IRInst::Bitcast {
                    result,
                    target_type,
                    ..
                } => (*result, reg_class_for_prim(*target_type)),
                IRInst::IntToFloat { result, .. } => (*result, RegisterClass::Fpr),
                IRInst::FloatConvert { result, .. } => (*result, RegisterClass::Fpr),
                IRInst::CallIndirect {
                    result: Some(result),
                    return_type,
                    ..
                } => (*result, class_of_type(return_type)),
                IRInst::Call {
                    result: Some(result),
                    func_name,
                    ..
                } => (
                    *result,
                    module
                        .functions
                        .get(func_name)
                        .map(|callee| class_of_type(&callee.signature.return_type))
                        .unwrap_or(RegisterClass::Gpr),
                ),
                other => match other.defined_var() {
                    Some(name) => (name, RegisterClass::Gpr),
                    None => continue,
                },
            };
            insert_register_class(&mut classes, name, class)?;
        }
    }
    Ok(classes)
}

fn insert_register_class<'a>(
    classes: &mut HashMap<&'a str, RegisterClass>,
    name: &'a str,
    class: RegisterClass,
) -> Result<(), FromIRError> {
    if *classes.entry(name).or_insert(class) != class {
        return Err(conflicting_definitions(name));
    }
    Ok(())
}

fn reg_class_for_type(ty: &IRType<'_>) -> RegisterClass {
    match ty {
        IRType::Primitive(IRPrim::F32) | IRType::Primitive(IRPrim::F64) => RegisterClass::Fpr,
        #[cfg(feature = "nightly")]
        IRType::Vector { .. } => RegisterClass::Vec,
        _ => RegisterClass::Gpr,
    }
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
