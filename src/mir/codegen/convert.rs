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

use lamina_mir::Located;
use lamina_platform::{Endianness, PointerWidth, TargetArchitecture, TargetDataLayout};

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
            AllocType, BinaryOp as IRBin, CmpOp as IRCmp, Instruction as IRInst,
            OverflowOp as IROverflow, UnaryOp as IRUnary,
        },
        module::GlobalInit,
        types::{
            Literal as IRLit, PrimitiveType as IRPrim, Type as IRType, Value as IRVal,
            VectorShiftForm, align_of_type, classify_vector_shift, resolve_type_names, sizeof_type,
            struct_field_byte_offset,
        },
        value_types,
    },
    mir::{
        AddressMode, AsmOperand, Block, CodePlacement, FloatBinOp, FloatCmpOp, FloatUnOp, Function,
        Global, GlobalRelocation, Immediate, Instruction, IntBinOp, IntCmpOp, IntOverflowOp,
        IntRotOp, IntUnOp, MemoryAttrs, MirType, Module, Operand, Parameter, Register,
        RegisterClass, ScalarType, Signature, SymbolVisibility, TlsModel as MirTlsModel,
        VectorLane, VectorOp, VectorType, VirtualReg, VirtualRegAllocator,
        codegen::{
            error::FromIRError,
            mapping::{
                classify_aggregate, is_aggregate, map_ir_prim, map_ir_type, map_ir_vector,
                signedness_of, wide_aggregate,
            },
        },
        io_intrinsic_result_type,
    },
};

#[cfg(feature = "nightly")]
use crate::ir::ModuleAnnotation;
use crate::ir::instruction::{AsmOperand as IRAsmOperand, AtomicBinOp as IRAtomicBinOp};
use crate::mir::AtomicBinOp;
#[cfg(feature = "nightly")]
use crate::mir::RelocationModel;
use lamina_codegen::abi::RuntimeSymbol;

mod wide;

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

/// The alignment attributed to an address nothing is known about. Above every natural
/// alignment, so an unknown base never lowers the alignment of an access through it.
const UNCONSTRAINED_ALIGN: u8 = 128;

/// The alignment of `base + offset`, given the alignment of `base`.
fn displaced_align(base: u8, offset: u64) -> u8 {
    if offset == 0 {
        return base;
    }
    let step = 1u64 << offset.trailing_zeros();
    base.min(u8::try_from(step).unwrap_or(UNCONSTRAINED_ALIGN))
}

/// The alignment of every pointer a function derives from an address it can reason
/// about, keyed by the IR name that holds it.
///
/// A name missing from the map is a pointer whose alignment the lowering cannot bound:
/// a parameter, a loaded pointer, a call result. Those keep [`UNCONSTRAINED_ALIGN`], the
/// same assumption C makes of a pointer it is handed.
type PointerAligns<'a> = HashMap<&'a str, u8>;

/// Power of two every value of an index is known to be a multiple of, by IR name.
/// An index missing from the map is a multiple of one and nothing more.
type IndexSteps<'a> = HashMap<&'a str, u64>;

/// Ceiling on a tracked step, past any alignment a target asks for.
const MAX_STEP: u64 = 1 << 32;

/// The largest power of two, up to [`MAX_STEP`], that divides `value`. Zero is a
/// multiple of everything.
fn step_of(value: u128) -> u64 {
    if value == 0 {
        return MAX_STEP;
    }
    let zeros = value.trailing_zeros();
    if zeros >= MAX_STEP.trailing_zeros() {
        MAX_STEP
    } else {
        1u64 << zeros
    }
}

fn literal_step(literal: &IRLit<'_>) -> Option<u64> {
    let value = match literal {
        IRLit::I8(v) => i64::from(*v),
        IRLit::I16(v) => i64::from(*v),
        IRLit::I32(v) => i64::from(*v),
        IRLit::I64(v) => *v,
        IRLit::U8(v) => i64::from(*v),
        IRLit::U16(v) => i64::from(*v),
        IRLit::U32(v) => i64::from(*v),
        IRLit::U64(v) => *v as i64,
        _ => return None,
    };
    Some(step_of(u128::from(value.unsigned_abs())))
}

fn index_step<'a>(value: &IRVal<'a>, steps: &IndexSteps<'a>) -> u64 {
    match value {
        IRVal::Variable(id) => steps.get(id).copied().unwrap_or(1),
        IRVal::Constant(literal) => literal_step(literal).unwrap_or(1),
        _ => 1,
    }
}

/// Records what each integer value is known to be a multiple of, so a variable array
/// index scaled by the element size does not look worse aligned than it is.
///
/// Wrapping never adds a low one bit, so each rule holds for the wrapped result too.
fn record_index_step<'a>(instr: &IRInst<'a>, steps: &mut IndexSteps<'a>) {
    let (result, step) = match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let (a, b) = (index_step(lhs, steps), index_step(rhs, steps));
            let step = match op {
                IRBin::Add | IRBin::Sub | IRBin::Or | IRBin::Xor => a.min(b),
                IRBin::Mul => step_of(u128::from(a) * u128::from(b)),
                IRBin::And => a.max(b),
                IRBin::Shl => match literal_shift(rhs, *ty) {
                    Some(by) => step_of(u128::from(a).checked_shl(by).unwrap_or(0)),
                    None => 1,
                },
                _ => 1,
            };
            (*result, step)
        }
        IRInst::ZeroExtend { result, value, .. }
        | IRInst::SignExtend { result, value, .. }
        | IRInst::Trunc { result, value, .. }
        | IRInst::Bitcast { result, value, .. }
        | IRInst::Copy { result, value, .. } => (*result, index_step(value, steps)),
        _ => return,
    };
    if step > 1 {
        steps.insert(result, step);
    }
}

/// A shift count written as a literal, reduced the way the shift itself reduces it.
fn literal_shift(rhs: &IRVal<'_>, ty: IRPrim) -> Option<u32> {
    let IRVal::Constant(literal) = rhs else {
        return None;
    };
    let count = match literal {
        IRLit::I8(v) => i64::from(*v),
        IRLit::I16(v) => i64::from(*v),
        IRLit::I32(v) => i64::from(*v),
        IRLit::I64(v) => *v,
        IRLit::U8(v) => i64::from(*v),
        IRLit::U16(v) => i64::from(*v),
        IRLit::U32(v) => i64::from(*v),
        IRLit::U64(v) => *v as i64,
        _ => return None,
    };
    let bits = int_bits_for_bitcast(&ty);
    u32::try_from(count).ok().map(|count| count & (bits - 1))
}

/// The alignment of the storage a global names.
fn global_align(module: &IRModule<'_>, name: &str, layout: TargetDataLayout) -> u8 {
    let Some(decl) = module.global_declarations.get(name) else {
        return UNCONSTRAINED_ALIGN;
    };
    let Some(resolved) = resolve_type_names(&decl.ty, &module.type_declarations) else {
        return UNCONSTRAINED_ALIGN;
    };
    let align = (align_of_type(&resolved, layout) as usize).max(declared_align(decl.align));
    u8::try_from(align).unwrap_or(UNCONSTRAINED_ALIGN)
}

fn pointer_align<'a>(
    value: &IRVal<'a>,
    known: &PointerAligns<'a>,
    module: &IRModule<'a>,
    layout: TargetDataLayout,
) -> u8 {
    match value {
        IRVal::Variable(id) => known.get(id).copied().unwrap_or(UNCONSTRAINED_ALIGN),
        IRVal::Global(name) => global_align(module, name, layout),
        _ => UNCONSTRAINED_ALIGN,
    }
}

/// Records the alignment of each pointer `f` derives, so a load or store through one
/// states the alignment of the address rather than of the type it reads.
///
/// Walks the blocks in the order the conversion walks them. The IR is in SSA form with
/// no pointer-valued phi, so a definition is always seen before its uses.
fn pointer_alignments<'a>(
    f: &IRFunction<'a>,
    module: &IRModule<'a>,
    layout: TargetDataLayout,
) -> PointerAligns<'a> {
    let mut known: PointerAligns<'a> = HashMap::new();
    let mut steps: IndexSteps<'a> = HashMap::new();
    let mut visited: HashSet<&'a str> = HashSet::new();
    let mut worklist: VecDeque<&'a str> = VecDeque::new();
    if !f.basic_blocks.contains_key(f.entry_block) {
        return known;
    }
    worklist.push_back(f.entry_block);
    visited.insert(f.entry_block);
    while let Some(label) = worklist.pop_front() {
        let block = &f.basic_blocks[label];
        for instr in &block.instructions {
            record_index_step(instr, &mut steps);
            match instr {
                IRInst::GetFieldPtr {
                    result,
                    struct_ptr,
                    struct_type,
                    field_index,
                } => {
                    let Some(resolved) = resolve_type_names(struct_type, &module.type_declarations)
                    else {
                        continue;
                    };
                    let IRType::Struct { fields, packed } = &resolved else {
                        continue;
                    };
                    let Some(offset) =
                        struct_field_byte_offset(fields, *packed, *field_index, layout)
                    else {
                        continue;
                    };
                    let base = pointer_align(struct_ptr, &known, module, layout);
                    known.insert(result, displaced_align(base, offset));
                }
                IRInst::GetElemPtr {
                    result,
                    array_ptr,
                    index,
                    element_type,
                } => {
                    let Some(resolved) =
                        resolve_type_names(element_type, &module.type_declarations)
                    else {
                        continue;
                    };
                    let Some(size) = sizeof_type(&resolved, layout) else {
                        continue;
                    };
                    let base = pointer_align(array_ptr, &known, module, layout);
                    let step = step_of(u128::from(index_step(index, &steps)) * u128::from(size));
                    known.insert(result, displaced_align(base, step));
                }
                IRInst::Copy { result, value, .. } => {
                    let align = pointer_align(value, &known, module, layout);
                    if align != UNCONSTRAINED_ALIGN {
                        known.insert(result, align);
                    }
                }
                _ => {}
            }
        }
        for target in block.instructions.iter().flat_map(successors) {
            if visited.insert(target) {
                worklist.push_back(target);
            }
        }
    }
    known
}

fn successors<'a>(instr: &IRInst<'a>) -> Vec<&'a str> {
    match instr {
        IRInst::Jmp { target_label } => vec![*target_label],
        IRInst::Br {
            true_label,
            false_label,
            ..
        } => vec![*true_label, *false_label],
        IRInst::Switch { cases, default, .. } => {
            let mut labels: Vec<&'a str> = cases.iter().map(|(_, label)| *label).collect();
            labels.push(*default);
            labels
        }
        _ => Vec::new(),
    }
}

/// The alignment a load or store is carried out at: the one the instruction states, or
/// the alignment the lowering can prove of the address, capped at the natural alignment
/// of the accessed type.
///
/// The cap keeps a derived alignment reporting what the access needs rather than what
/// the address happens to exceed, so only an address that is genuinely short of the
/// natural alignment — a field of a packed struct, say — comes out below it.
fn access_alignment(
    stated: Option<u32>,
    ty: MirType,
    address: u8,
    layout: TargetDataLayout,
    what: &str,
) -> Result<u8, FromIRError> {
    let natural = ty.alignment(layout.pointer_width) as u8;
    let Some(stated) = stated else {
        return Ok(natural.min(address));
    };
    u8::try_from(stated).map_err(|_| {
        FromIRError::UnsupportedAlignment(format!(
            "{what} align {stated} exceeds the 128-byte maximum a memory access can state"
        ))
    })
}

fn normalize_lowered_literal(operand: &mut Operand, ty: MirType) {
    operand.normalize_integer_immediate(ty);
    if ty == MirType::Scalar(ScalarType::F32)
        && let Operand::Immediate(Immediate::F64(value)) = operand
    {
        *operand = Operand::imm_f32(*value as f32);
    }
}

fn declared_align(align: Option<u32>) -> usize {
    align.map_or(1, |align| align as usize)
}

/// Lowers an IR module to MIR, named `name`, against the x86_64 data layout.
///
/// Pointer width, endianness and register width come from that layout, so the result
/// suits no other target. Every other target goes through [`from_ir_for_target`].
///
/// An aggregate crossing a function boundary keeps its layout in an
/// [`Aggregate`](crate::mir::types::Aggregate) marker and each backend decides where
/// the words go. Anything MIR cannot express comes back as a [`FromIRError`] naming
/// the value.
pub fn from_ir_x86_64(ir: &IRModule<'_>, name: &str) -> Result<Module, FromIRError> {
    from_ir_for_target(ir, name, TargetArchitecture::X86_64.data_layout())
}

pub fn from_ir_for_target(
    ir: &IRModule<'_>,
    name: &str,
    layout: TargetDataLayout,
) -> Result<Module, FromIRError> {
    let int64_regs = layout.has_64bit_int_registers();
    let endianness = layout.endianness;
    let mut mir_module = Module::new(name);
    #[cfg(feature = "nightly")]
    {
        mir_module.relocation = relocation_model(&ir.annotations);
    }

    let mut global_declarations: Vec<_> = ir.global_declarations.iter().collect();
    global_declarations.sort_unstable_by_key(|(name, _)| **name);
    for (global_name, decl) in global_declarations {
        let resolved = resolve_type_names(&decl.ty, &ir.type_declarations)
            .ok_or(FromIRError::UnsupportedType)?;
        let mut global = if is_aggregate(&resolved) {
            let size = sizeof_type(&resolved, layout).ok_or(FromIRError::UnsupportedType)? as usize;
            let align = (align_of_type(&resolved, layout) as usize).max(declared_align(decl.align));
            let bytes = aggregate_global_bytes(
                global_name,
                &resolved,
                size,
                decl.initializer.as_ref(),
                layout,
                endianness,
            )?;
            let mut global = Global::new(
                *global_name,
                MirType::Scalar(ScalarType::I8),
                layout.pointer_width,
            )
            .with_layout(size, align);
            if let Some(bytes) = bytes {
                global = global.with_initializer(bytes);
            }
            global
        } else {
            if matches!(resolved, IRType::Primitive(IRPrim::I128 | IRPrim::U128)) {
                return Err(FromIRError::UnsupportedGlobal(format!(
                    "@{global_name} is a 128-bit integer; globals of i128 and u128 are not lowered yet"
                )));
            }
            if wide::is_wide_type(&resolved, int64_regs) {
                // Two words, addressed as bytes: the scalar type would name a register
                // the target does not have.
                let size = wide_aggregate(int64_regs).size;
                let align = size.max(declared_align(decl.align));
                let bytes =
                    wide_global_bytes(global_name, decl.initializer.as_ref(), size, endianness)?;
                let mut global = Global::new(
                    *global_name,
                    MirType::Scalar(ScalarType::I8),
                    layout.pointer_width,
                )
                .with_layout(size, align);
                if let Some(bytes) = bytes {
                    global = global.with_initializer(bytes);
                }
                if decl.exported {
                    global = global.exported();
                }
                if let Some(model) = decl.thread_local {
                    global = global.thread_local(model);
                }
                mir_module.globals.insert(global_name.to_string(), global);
                continue;
            }
            let ty = map_ir_type(&resolved)?;
            let initializer = global_initializer(
                global_name,
                ty,
                decl.initializer.as_ref(),
                layout,
                endianness,
            )?;
            for hole in &initializer.relocations {
                let known = ir.functions.contains_key(hole.symbol.as_str())
                    || ir.global_declarations.contains_key(hole.symbol.as_str());
                if !known {
                    return Err(FromIRError::UnsupportedGlobal(format!(
                        "@{global_name} holds the address of {}, which this module neither defines nor declares",
                        hole.symbol
                    )));
                }
            }
            let mut global = Global::new(*global_name, ty, layout.pointer_width);
            global.align = global.align.max(declared_align(decl.align));
            if let Some(bytes) = initializer.bytes {
                global = global.with_initializer(bytes);
            }
            global.relocations = initializer.relocations;
            global
        };
        if decl.exported {
            global = global.exported();
        }
        if let Some(model) = decl.thread_local {
            global = global.thread_local(model);
        }
        mir_module.globals.insert(global_name.to_string(), global);
    }

    let mut libcalls = BTreeMap::new();
    let mut functions: Vec<_> = ir.functions.iter().collect();
    functions.sort_unstable_by_key(|(name, _)| **name);
    for (func_name, ir_func) in functions {
        // Check if function is external before converting
        if ir_func.annotations.contains(&FunctionAnnotation::Extern) {
            mir_module.mark_external(*func_name);
        }
        if ir_func.annotations.contains(&FunctionAnnotation::Export) {
            mir_module.mark_exported(*func_name);
        }

        let mir_func = convert_function(
            func_name,
            ir_func,
            ir,
            &mut libcalls,
            layout,
            int64_regs,
            endianness,
        )?
        .with_pointer_width(layout.pointer_width)
        .with_int64_regs(int64_regs);
        mir_module.add_function(mir_func);
    }
    for (name, libcall) in libcalls {
        if !mir_module.functions.contains_key(&name) {
            mir_module.mark_external(name.clone());
            mir_module.add_function(
                libcall
                    .declaration(int64_regs)
                    .with_pointer_width(layout.pointer_width)
                    .with_int64_regs(int64_regs),
            );
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
    /// The IR type of each named value, which says whether a vector shift's count is
    /// one count for every lane or one per lane.
    ir_types: &'t HashMap<&'a str, IRType<'a>>,
    /// Named parameter count of each variadic callee, which says where its default
    /// argument promotions begin.
    variadic_fixed: &'t HashMap<&'a str, usize>,
    module: &'t IRModule<'a>,
    /// Alignment of each pointer the function derives, by the IR name holding it.
    pointer_aligns: &'t PointerAligns<'a>,
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
    layout: TargetDataLayout,
    endianness: Endianness,
) -> Result<Vec<u8>, FromIRError> {
    // (offset, width) of every slot the list can fill, in order. An array keeps its
    // stride instead of one entry per element, so a long array costs nothing here.
    let slots: Slots = match ty {
        IRType::Array { element_type, size } => Slots::Strided {
            stride: sizeof_type(element_type, layout).ok_or(FromIRError::UnsupportedType)? as usize,
            count: usize::try_from(*size).map_err(|_| FromIRError::UnsupportedType)?,
        },
        IRType::Struct { fields, packed } => Slots::Fixed(
            fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let at = struct_field_byte_offset(fields, *packed, index, layout)
                        .ok_or(FromIRError::UnsupportedType)? as usize;
                    let width = sizeof_type(&field.ty, layout)
                        .ok_or(FromIRError::UnsupportedType)?
                        as usize;
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
        let encoded = literal_bytes(name, literal, endianness)?;
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
    layout: TargetDataLayout,
    endianness: Endianness,
) -> Result<Option<Vec<u8>>, FromIRError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let GlobalInit::List(elements) = value {
        return element_list_bytes(name, ty, size, elements, layout, endianness).map(Some);
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

/// Bytes of a global that takes two registers, at its full width in memory.
fn wide_global_bytes(
    name: &str,
    value: Option<&GlobalInit<'_>>,
    size: usize,
    endianness: Endianness,
) -> Result<Option<Vec<u8>>, FromIRError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let GlobalInit::Value(IRVal::Constant(literal)) = value else {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} needs a literal initializer"
        )));
    };
    let mut bytes = literal_bytes(name, literal, endianness)?;
    if bytes.len() > size {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} initializer is {} bytes, wider than the {size} its type takes",
            bytes.len()
        )));
    }
    let sign = if literal_is_negative(literal) {
        0xff
    } else {
        0
    };
    // The widening bytes are the high ones, which come last only in little-endian order.
    match endianness {
        Endianness::Little => bytes.resize(size, sign),
        Endianness::Big => {
            let mut widened = vec![sign; size.saturating_sub(bytes.len())];
            widened.append(&mut bytes);
            bytes = widened;
        }
    }
    Ok(Some(bytes))
}

fn literal_is_negative(literal: &IRLit<'_>) -> bool {
    match literal {
        IRLit::I8(v) => *v < 0,
        IRLit::I16(v) => *v < 0,
        IRLit::I32(v) => *v < 0,
        IRLit::I64(v) => *v < 0,
        IRLit::I128(v) => *v < 0,
        _ => false,
    }
}

/// A global's bytes, and the symbol addresses those bytes cannot hold.
struct InitializedGlobal {
    bytes: Option<Vec<u8>>,
    relocations: Vec<GlobalRelocation>,
}

fn global_initializer(
    name: &str,
    ty: MirType,
    value: Option<&GlobalInit<'_>>,
    layout: TargetDataLayout,
    endianness: Endianness,
) -> Result<InitializedGlobal, FromIRError> {
    let Some(value) = value else {
        return Ok(InitializedGlobal {
            bytes: None,
            relocations: Vec::new(),
        });
    };
    let literal = match value {
        GlobalInit::Value(IRVal::Constant(literal)) => literal,
        GlobalInit::Function(symbol) => {
            let size = layout.pointer_width.bytes();
            if ty.size_bytes(layout.pointer_width) != size {
                return Err(FromIRError::UnsupportedGlobal(format!(
                    "@{name} holds the address of @{symbol}, so its type must be ptr, not {ty}"
                )));
            }
            return Ok(InitializedGlobal {
                bytes: Some(vec![0; size]),
                relocations: vec![GlobalRelocation {
                    offset: 0,
                    symbol: symbol.to_string(),
                    addend: 0,
                    size,
                }],
            });
        }
        GlobalInit::Value(IRVal::Global(symbol)) => {
            let size = layout.pointer_width.bytes();
            if ty.size_bytes(layout.pointer_width) != size {
                return Err(FromIRError::UnsupportedGlobal(format!(
                    "@{name} holds the address of @{symbol}, so its type must be ptr, not {ty}"
                )));
            }
            return Ok(InitializedGlobal {
                bytes: Some(vec![0; size]),
                relocations: vec![GlobalRelocation {
                    offset: 0,
                    symbol: symbol.to_string(),
                    addend: 0,
                    size,
                }],
            });
        }
        _ => {
            return Err(FromIRError::UnsupportedGlobal(format!(
                "@{name} needs a literal initializer"
            )));
        }
    };
    let bytes = literal_bytes(name, literal, endianness)?;
    if bytes.len() != ty.size_bytes(layout.pointer_width) {
        return Err(FromIRError::UnsupportedGlobal(format!(
            "@{name} initializer width does not match its type {ty}"
        )));
    }
    Ok(InitializedGlobal {
        bytes: Some(bytes),
        relocations: Vec::new(),
    })
}

/// Bytes of a literal in the target's byte order, at the width its own type takes.
fn literal_bytes(
    name: &str,
    literal: &IRLit<'_>,
    endianness: Endianness,
) -> Result<Vec<u8>, FromIRError> {
    let mut bytes = match literal {
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
    if endianness == Endianness::Big {
        bytes.reverse();
    }
    Ok(bytes)
}

/// Address registers for referenced globals. Each global gets one register,
/// defined by a `SymbolAddress` at the top of the entry block.
struct GlobalAddresses<'a> {
    declared: HashSet<&'a str>,
    thread_local: BTreeMap<&'a str, MirTlsModel>,
    regs: BTreeMap<&'a str, Register>,
}

impl<'a> GlobalAddresses<'a> {
    fn new(module: &IRModule<'a>) -> Self {
        Self {
            declared: module.global_declarations.keys().copied().collect(),
            thread_local: module
                .global_declarations
                .iter()
                .filter_map(|(name, global)| Some((*name, global.thread_local?)))
                .collect(),
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
            .map(|(name, dst)| match self.thread_local.get(name) {
                Some(model) => Instruction::ThreadLocalAddr {
                    dst: dst.clone(),
                    symbol: (*name).to_string(),
                    model: *model,
                },
                None => Instruction::SymbolAddress {
                    dst: dst.clone(),
                    symbol: (*name).to_string(),
                },
            })
            .collect()
    }
}

fn convert_function<'a>(
    name: &'a str,
    f: &IRFunction<'a>,
    module: &IRModule<'a>,
    libcalls: &mut BTreeMap<String, wide::LibCall>,
    layout: TargetDataLayout,
    int64_regs: bool,
    endianness: Endianness,
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
        .any(|a| matches!(a, FunctionAnnotation::Pure));
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
    mir_sig.naked = f.annotations.contains(&FunctionAnnotation::Naked);
    // A naked body is exactly what the author wrote, so it must not be inlined.
    mir_sig.no_inline |= mir_sig.naked;
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

    let quad_at_boundary = |ty: &IRType<'a>| {
        resolve_type_names(ty, &module.type_declarations)
            .is_some_and(|resolved| wide::is_quad_type(&resolved, int64_regs))
    };
    if quad_at_boundary(&f.signature.return_type) {
        return Err(FromIRError::NotLegalizable(format!(
            "the return type of '@{name}' names a 128-bit integer on a 32-bit target: that takes four registers, and only pairs are split"
        )));
    }
    for param in &f.signature.params {
        if quad_at_boundary(&param.ty) {
            return Err(FromIRError::NotLegalizable(format!(
                "parameter '%{}' of '@{name}' names a 128-bit integer on a 32-bit target: that takes four registers, and only pairs are split",
                param.name
            )));
        }
    }

    match &f.signature.return_type {
        IRType::Void => {}
        other => {
            let resolved = resolve_type_names(other, &module.type_declarations)
                .ok_or(FromIRError::UnsupportedType)?;
            if let Some(aggregate) =
                classify_aggregate(&resolved, &format!("return of @{name}"), layout, int64_regs)?
            {
                mir_sig = mir_sig
                    .with_return(MirType::Scalar(ScalarType::Ptr))
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
    let ir_types = value_types(module, f);
    let mut reservations: HashMap<VirtualReg, usize> = HashMap::new();
    let mut alignments: HashMap<VirtualReg, usize> = HashMap::new();
    let mut globals = GlobalAddresses::new(module);
    let pointer_aligns = pointer_alignments(f, module, layout);

    let mut aggregate_prelude = Vec::new();
    for param in &f.signature.params {
        let resolved = resolve_type_names(&param.ty, &module.type_declarations)
            .ok_or(FromIRError::UnsupportedType)?;
        if let Some(aggregate) = classify_aggregate(
            &resolved,
            &format!("%{} of @{name}", param.name),
            layout,
            int64_regs,
        )? {
            let slot = vreg_alloc.allocate_gpr();
            let words = aggregate.words();
            if words > 1 {
                reservations.insert(slot, words - 1);
            }
            let addr = Register::Virtual(vreg_alloc.allocate_gpr());
            register_types.insert(addr.clone(), MirType::Scalar(ScalarType::Ptr));
            mir_sig.params.push(
                Parameter::new(Register::Virtual(slot), MirType::Scalar(ScalarType::Ptr))
                    .with_aggregate(aggregate),
            );
            aggregate_prelude.push(Instruction::StackAddr {
                dst: addr.clone(),
                slot,
            });
            if wide::is_wide_type(&resolved, int64_regs) {
                let pair = wide::load_wide_parameter(
                    addr,
                    &mut vreg_alloc,
                    &mut aggregate_prelude,
                    int64_regs,
                    endianness,
                );
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

    let mut mir_func = Function::new(mir_sig)
        .with_entry(f.entry_block)
        .with_pointer_width(layout.pointer_width)
        .with_int64_regs(int64_regs);

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
        for (ir_index, instr) in ir_block.instructions.iter().enumerate() {
            let loc = ir_block.loc_at(ir_index);
            let mut resolved;
            let instr = if matches!(
                instr,
                IRInst::Alloc { .. } | IRInst::GetElemPtr { .. } | IRInst::GetFieldPtr { .. }
            ) {
                resolved = instr.clone();
                let ty = match &mut resolved {
                    IRInst::Alloc { allocated_ty, .. } => allocated_ty,
                    IRInst::GetElemPtr { element_type, .. } => element_type,
                    IRInst::GetFieldPtr { struct_type, .. } => struct_type,
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
                    ir_types: &ir_types,
                    variadic_fixed: &variadic_fixed,
                    module,
                    pointer_aligns: &pointer_aligns,
                },
                &mut globals,
                layout,
                int64_regs,
                endianness,
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
                    && let Some(aggregate) = classify_aggregate(
                        &resolved,
                        &format!("return of @{name}"),
                        layout,
                        int64_regs,
                    )?
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
                    register_types.insert(address.clone(), MirType::Scalar(ScalarType::Ptr));
                    result_address = Some(Instruction::StackAddr { dst: address, slot });
                }
                if let Instruction::Call { name, args, .. } = &mut mir_instr
                    && let Some(callee) = module.functions.get(name.as_str())
                {
                    for (argument, parameter) in args.iter_mut().zip(&callee.signature.params) {
                        if is_aggregate(&parameter.ty)
                            || wide::is_wide_type(&parameter.ty, int64_regs)
                            || resolve_type_names(&parameter.ty, &module.type_declarations)
                                .is_some_and(|ty| {
                                    is_aggregate(&ty) || wide::is_wide_type(&ty, int64_regs)
                                })
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
                            Some(MirType::Scalar(ScalarType::I1))
                        }
                        Instruction::CallIndirect { ret_ty, .. } => *ret_ty,
                        Instruction::Call { name, .. } => module
                            .functions
                            .get(name.as_str())
                            .and_then(|callee| map_ir_type(&callee.signature.return_type).ok())
                            .or_else(|| io_intrinsic_result_type(name, int64_regs)),
                        Instruction::Lea { .. }
                        | Instruction::StackAddr { .. }
                        | Instruction::SymbolAddress { .. }
                        | Instruction::ThreadLocalAddr { .. } => {
                            Some(MirType::Scalar(ScalarType::Ptr))
                        }
                        Instruction::IntOverflow { .. } => Some(MirType::Scalar(ScalarType::I1)),
                        Instruction::AtomicCompareExchange { ty, .. }
                        | Instruction::InlineAsm {
                            result: Some((_, ty)),
                            ..
                        } => Some(*ty),
                        Instruction::VectorOp {
                            op: VectorOp::VExtractLane,
                            ty,
                            ..
                        } => Some(vector_lane_scalar(*ty)),
                        Instruction::VectorOp { ty, .. } => Some(*ty),
                        _ => None,
                    };
                    if let Some(ty) = ty {
                        register_types.insert(dst.clone(), ty);
                    }
                }
                if let Some(dst) = mir_instr.def_reg() {
                    defined_in_block.insert(dst.clone());
                }
                mir_block.push_at(mir_instr, loc);
                if let Some(address) = result_address {
                    if let Some(dst) = address.def_reg() {
                        defined_in_block.insert(dst.clone());
                    }
                    mir_block.push_at(address, loc);
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

    let mut prelude = aggregate_prelude;
    prelude.extend(globals.prelude());
    if !prelude.is_empty()
        && let Some(entry) = mir_func.get_block_mut(f.entry_block)
    {
        entry
            .instructions
            .splice(0..0, prelude.into_iter().map(Located::from));
    }

    mir_func.stack_reservations = reservations;
    mir_func.stack_alignments = alignments;
    libcalls.extend(mem::take(&mut var_to_reg.libcalls));

    Ok(mir_func)
}

#[allow(clippy::too_many_arguments)]
fn convert_instruction<'a>(
    instr: &IRInst<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut VarBindings<'a>,
    reservations: &mut HashMap<VirtualReg, usize>,
    alignments: &mut HashMap<VirtualReg, usize>,
    facts: &ValueFacts<'_, 'a>,
    globals: &mut GlobalAddresses<'a>,
    layout: TargetDataLayout,
    int64_regs: bool,
    endianness: Endianness,
) -> Result<Vec<Instruction>, FromIRError> {
    let mut wide_cx = wide::Cx {
        vregs: &mut *vreg_alloc,
        vars: &mut *var_to_reg,
        reservations: &mut *reservations,
        globals: &mut *globals,
        module: facts.module,
        layout,
        int64_regs,
        endianness,
        pointer_aligns: facts.pointer_aligns,
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
        IRInst::OverflowFlag {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let mir_ty = map_ir_prim(*ty)?;
            let mut lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
            let mut rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut lhs_op, mir_ty);
            normalize_lowered_literal(&mut rhs_op, mir_ty);
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::IntOverflow {
                op: match op {
                    IROverflow::Add => IntOverflowOp::Add,
                    IROverflow::Sub => IntOverflowOp::Sub,
                    IROverflow::Mul => IntOverflowOp::Mul,
                },
                ty: mir_ty,
                signed: !matches!(ty, IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64),
                dst,
                lhs: lhs_op,
                rhs: rhs_op,
            }])
        }
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
        IRInst::Copy { result, ty, value } => {
            let mir_ty = map_ir_type(ty)?;
            let src = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut mov = Instruction::Copy {
                ty: mir_ty,
                dst,
                src,
            };
            mov.normalize_integer_immediates(None);
            Ok(vec![mov])
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
                    lhs: Operand::Immediate(zero_immediate(mir_ty, layout.pointer_width)?),
                    rhs: src,
                },
                (IRUnary::Not, false) => Instruction::IntBinary {
                    op: IntBinOp::Xor,
                    ty: mir_ty,
                    dst,
                    lhs: src,
                    rhs: Operand::Immediate(all_ones_immediate(mir_ty, layout.pointer_width)?),
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
            align,
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
                    align: access_alignment(
                        *align,
                        mir_ty,
                        pointer_align(ptr, facts.pointer_aligns, facts.module, layout),
                        layout,
                        "load",
                    )?,
                    volatile: *volatile,
                    stated_align: align.is_some(),
                },
            }])
        }
        IRInst::Store {
            ty,
            ptr,
            value,
            volatile,
            align,
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
                    align: access_alignment(
                        *align,
                        mir_ty,
                        pointer_align(ptr, facts.pointer_aligns, facts.module, layout),
                        layout,
                        "store",
                    )?,
                    volatile: *volatile,
                    stated_align: align.is_some(),
                },
            }])
        }
        IRInst::Call {
            result,
            func_name,
            args,
        } => {
            let mut mir_args = Vec::new();
            for (_, a) in args {
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
                if facts.types.get(reg) != Some(&MirType::Scalar(ScalarType::F32)) {
                    continue;
                }
                let dst = Register::Virtual(vreg_alloc.allocate_fpr());
                instructions.push(Instruction::FloatConvert {
                    from: MirType::Scalar(ScalarType::F32),
                    to: MirType::Scalar(ScalarType::F64),
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
                if index >= fixed && mir_ty == MirType::Scalar(ScalarType::F32) {
                    let dst = Register::Virtual(vreg_alloc.allocate_fpr());
                    instructions.push(Instruction::FloatConvert {
                        from: MirType::Scalar(ScalarType::F32),
                        to: MirType::Scalar(ScalarType::F64),
                        dst: dst.clone(),
                        src: operand,
                    });
                    operand = Operand::Register(dst);
                    mir_ty = MirType::Scalar(ScalarType::F64);
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
                if ty == MirType::Scalar(ScalarType::I1) {
                    1
                } else {
                    ty.size_bytes(layout.pointer_width) * 8
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
                from: MirType::Scalar(ScalarType::Ptr),
                to: map_ir_prim(*target_type)?,
                signed: false,
                dst,
                src: val_op,
            }])
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type,
        } => {
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut val_op = resolve_operand(int_value, vreg_alloc, var_to_reg, globals)?;
            let from = map_ir_prim(*target_type)?;
            if matches!(val_op, Operand::Register(_)) {
                let actual = operand_type(&val_op, register_types)?;
                if actual != from {
                    return Err(FromIRError::InvalidConversion(format!(
                        "inttoptr.{target_type} applied to {int_value}, which is {actual}"
                    )));
                }
            } else {
                normalize_lowered_literal(&mut val_op, from);
            }
            Ok(vec![Instruction::IntConvert {
                from,
                to: MirType::Scalar(ScalarType::Ptr),
                signed: false,
                dst,
                src: val_op,
            }])
        }
        IRInst::GetFieldPtr {
            result,
            struct_ptr,
            struct_type,
            field_index,
        } => {
            let IRType::Struct { fields, packed } = struct_type else {
                return Err(FromIRError::UnsupportedType);
            };
            let base = pointer_base(struct_ptr, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let offset = struct_field_byte_offset(fields, *packed, *field_index, layout)
                .and_then(|bytes| i64::try_from(bytes).ok())
                .ok_or(FromIRError::UnsupportedType)?;
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
            let elem_size = sizeof_type(element_type, layout)
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
                let from = operand_type(&index_op, register_types)?;
                // Address arithmetic is done in the target's own word, so a 32-bit
                // target neither widens the index nor asks for a 64-bit add.
                let word = address_word(layout.pointer_width)?;
                let wide_index = Register::Virtual(vreg_alloc.allocate_gpr());
                let scaled = Register::Virtual(vreg_alloc.allocate_gpr());
                let address = Register::Virtual(vreg_alloc.allocate_gpr());
                let sum = Register::Virtual(vreg_alloc.allocate_gpr());
                Ok(vec![
                    Instruction::IntConvert {
                        from,
                        to: word,
                        signed: from.size_bytes(layout.pointer_width)
                            < word.size_bytes(layout.pointer_width),
                        dst: wide_index.clone(),
                        src: index_op,
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Mul,
                        ty: word,
                        dst: scaled.clone(),
                        lhs: Operand::Register(wide_index),
                        rhs: Operand::Immediate(Immediate::I64(elem_size)),
                    },
                    Instruction::IntConvert {
                        from: MirType::Scalar(ScalarType::Ptr),
                        to: word,
                        signed: false,
                        dst: address.clone(),
                        src: Operand::Register(base),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: word,
                        dst: sum.clone(),
                        lhs: Operand::Register(address),
                        rhs: Operand::Register(scaled),
                    },
                    Instruction::IntConvert {
                        from: word,
                        to: MirType::Scalar(ScalarType::Ptr),
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

            let size = sizeof_type(allocated_ty, layout).ok_or(
                if matches!(allocated_ty, IRType::VaList) {
                    FromIRError::NoVaListAbi
                } else {
                    FromIRError::UnsupportedType
                },
            )?;
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
        IRInst::VectorBinary {
            op,
            result,
            element_type,
            lanes,
            lhs,
            rhs,
        } => {
            let ty = map_ir_vector(*element_type, *lanes)?;
            let unsigned = matches!(
                element_type,
                IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64
            );
            if matches!(op, IRBin::Shl | IRBin::Shr) {
                let form = classify_vector_shift(*element_type, *lanes, rhs, facts.ir_types)
                    .map_err(|_| FromIRError::UnsupportedInstruction)?;
                let per_lane = form == VectorShiftForm::PerLane;
                let op = match (op, per_lane, unsigned) {
                    (IRBin::Shl, false, _) => VectorOp::VShl,
                    (IRBin::Shl, true, _) => VectorOp::VShlLanes,
                    (_, false, true) => VectorOp::VLShr,
                    (_, false, false) => VectorOp::VAShr,
                    (_, true, true) => VectorOp::VLShrLanes,
                    (_, true, false) => VectorOp::VAShrLanes,
                };
                let lhs = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
                let mut out = Vec::new();
                let count = if per_lane {
                    resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?
                } else {
                    uniform_shift_count(rhs, facts, vreg_alloc, var_to_reg, globals, &mut out)?
                };
                let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
                out.push(Instruction::VectorOp {
                    op,
                    ty,
                    dst,
                    operands: vec![lhs, count],
                });
                return Ok(out);
            }
            let op = match op {
                IRBin::Add => VectorOp::VAdd,
                IRBin::Sub => VectorOp::VSub,
                IRBin::Mul => VectorOp::VMul,
                IRBin::And => VectorOp::VAnd,
                IRBin::Or => VectorOp::VOr,
                IRBin::Xor => VectorOp::VXor,
                IRBin::Shl | IRBin::Shr | IRBin::Div | IRBin::Rem | IRBin::Rotl | IRBin::Rotr => {
                    return Err(FromIRError::UnsupportedInstruction);
                }
            };
            let lhs = resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?;
            let rhs = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::VectorOp {
                op,
                ty,
                dst,
                operands: vec![lhs, rhs],
            }])
        }
        IRInst::VectorSplat {
            result,
            element_type,
            lanes,
            value,
        } => {
            let ty = map_ir_vector(*element_type, *lanes)?;
            let mut value = resolve_operand(value, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut value, map_ir_prim(*element_type)?);
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::VectorOp {
                op: VectorOp::VSplat,
                ty,
                dst,
                operands: vec![value],
            }])
        }
        IRInst::VectorExtractLane {
            result,
            element_type,
            lanes,
            vector,
            lane,
        } => {
            let ty = map_ir_vector(*element_type, *lanes)?;
            let vector = resolve_operand(vector, vreg_alloc, var_to_reg, globals)?;
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::VectorOp {
                op: VectorOp::VExtractLane,
                ty,
                dst,
                operands: vec![vector, Operand::imm_i64(i64::from(*lane))],
            }])
        }
        IRInst::VectorInsertLane {
            result,
            element_type,
            lanes,
            vector,
            scalar,
            lane,
        } => {
            let ty = map_ir_vector(*element_type, *lanes)?;
            let vector = resolve_operand(vector, vreg_alloc, var_to_reg, globals)?;
            let mut scalar = resolve_operand(scalar, vreg_alloc, var_to_reg, globals)?;
            normalize_lowered_literal(&mut scalar, map_ir_prim(*element_type)?);
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::VectorOp {
                op: VectorOp::VInsertLane,
                ty,
                dst,
                operands: vec![vector, scalar, Operand::imm_i64(i64::from(*lane))],
            }])
        }
        IRInst::VectorShuffle {
            result,
            element_type,
            lanes,
            lhs,
            rhs,
            mask,
        } => {
            let ty = map_ir_vector(*element_type, *lanes)?;
            let scalar_ty = map_ir_prim(*element_type)?;
            let sources = [
                resolve_operand(lhs, vreg_alloc, var_to_reg, globals)?,
                resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?,
            ];
            let float = matches!(element_type, IRPrim::F32 | IRPrim::F64);
            let dst = resolve_or_alloc(result, vreg_alloc, var_to_reg);
            let mut zero = match element_type {
                IRPrim::F32 => Operand::Immediate(Immediate::F32(0.0)),
                IRPrim::F64 => Operand::Immediate(Immediate::F64(0.0)),
                _ => Operand::imm_i64(0),
            };
            normalize_lowered_literal(&mut zero, scalar_ty);
            let mut out = Vec::with_capacity(mask.len() * 2 + 1);
            let mut acc = if mask.is_empty() {
                dst.clone()
            } else {
                Register::Virtual(vreg_alloc.allocate_vec())
            };
            out.push(Instruction::VectorOp {
                op: VectorOp::VSplat,
                ty,
                dst: acc.clone(),
                operands: vec![zero],
            });
            for (lane, index) in mask.iter().enumerate() {
                let source = sources[usize::from(*index >= *lanes)].clone();
                let within = i64::from(index % (*lanes).max(1));
                let scalar = Register::Virtual(if float {
                    vreg_alloc.allocate_fpr()
                } else {
                    vreg_alloc.allocate_gpr()
                });
                out.push(Instruction::VectorOp {
                    op: VectorOp::VExtractLane,
                    ty,
                    dst: scalar.clone(),
                    operands: vec![source, Operand::imm_i64(within)],
                });
                let last = lane + 1 == mask.len();
                let next = if last {
                    dst.clone()
                } else {
                    Register::Virtual(vreg_alloc.allocate_vec())
                };
                out.push(Instruction::VectorOp {
                    op: VectorOp::VInsertLane,
                    ty,
                    dst: next.clone(),
                    operands: vec![
                        Operand::Register(acc),
                        Operand::Register(scalar),
                        Operand::imm_i64(lane as i64),
                    ],
                });
                acc = next;
            }
            Ok(out)
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
                ordering: *ordering,
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
                ordering: *ordering,
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
                ordering: *ordering,
            }])
        }
        IRInst::Fence { ordering } => Ok(vec![Instruction::Fence {
            ordering: *ordering,
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
                success_ordering: *success_ordering,
                failure_ordering: *failure_ordering,
            }])
        }
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

/// The integer an address is computed in on a target with this pointer width.
fn address_word(pointer: PointerWidth) -> Result<MirType, FromIRError> {
    match pointer {
        PointerWidth::Bits32 => Ok(MirType::Scalar(ScalarType::I32)),
        PointerWidth::Bits64 => Ok(MirType::Scalar(ScalarType::I64)),
        PointerWidth::Bits128 => Err(FromIRError::UnsupportedInstruction),
    }
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
            from: MirType::Scalar(ScalarType::Ptr),
            to: MirType::Scalar(ScalarType::I64),
            signed: false,
            dst: address.clone(),
            src: Operand::Register(base),
        },
        Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: sum.clone(),
            lhs: Operand::Register(address),
            rhs: Operand::Immediate(Immediate::I64(offset)),
        },
        Instruction::IntConvert {
            from: MirType::Scalar(ScalarType::I64),
            to: MirType::Scalar(ScalarType::Ptr),
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

fn zero_immediate(ty: MirType, pointer: PointerWidth) -> Result<Immediate, FromIRError> {
    match ty.size_bytes(pointer) {
        1 => Ok(Immediate::I8(0)),
        2 => Ok(Immediate::I16(0)),
        4 => Ok(Immediate::I32(0)),
        8 => Ok(Immediate::I64(0)),
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn all_ones_immediate(ty: MirType, pointer: PointerWidth) -> Result<Immediate, FromIRError> {
    match ty.size_bytes(pointer) {
        1 => Ok(Immediate::I8(-1)),
        2 => Ok(Immediate::I16(-1)),
        4 => Ok(Immediate::I32(-1)),
        8 => Ok(Immediate::I64(-1)),
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn vector_lane_scalar(ty: MirType) -> MirType {
    let MirType::Vector(VectorType::V128(lane) | VectorType::V256(lane)) = ty else {
        return ty;
    };
    MirType::Scalar(match lane {
        VectorLane::I8 => ScalarType::I8,
        VectorLane::I16 => ScalarType::I16,
        VectorLane::I32 => ScalarType::I32,
        VectorLane::I64 => ScalarType::I64,
        VectorLane::F32 => ScalarType::F32,
        VectorLane::F64 => ScalarType::F64,
    })
}

fn operand_type(
    operand: &Operand,
    types: &HashMap<Register, MirType>,
) -> Result<MirType, FromIRError> {
    Ok(match operand {
        Operand::Register(register) => types
            .get(register)
            .copied()
            .ok_or(FromIRError::UnknownVariable)?,
        Operand::Immediate(Immediate::I8(_)) => MirType::Scalar(ScalarType::I8),
        Operand::Immediate(Immediate::I16(_)) => MirType::Scalar(ScalarType::I16),
        Operand::Immediate(Immediate::I32(_)) => MirType::Scalar(ScalarType::I32),
        Operand::Immediate(Immediate::I64(_)) => MirType::Scalar(ScalarType::I64),
        Operand::Immediate(Immediate::F32(_)) => MirType::Scalar(ScalarType::F32),
        Operand::Immediate(Immediate::F64(_)) => MirType::Scalar(ScalarType::F64),
    })
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
/// The count of a one-count-for-every-lane shift, as a scalar `I64` operand. A narrower
/// count is widened without sign, so a count that was negative stays out of range.
fn uniform_shift_count<'a>(
    rhs: &IRVal<'a>,
    facts: &ValueFacts<'_, 'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut VarBindings<'a>,
    globals: &mut GlobalAddresses<'a>,
    out: &mut Vec<Instruction>,
) -> Result<Operand, FromIRError> {
    let mut operand = resolve_operand(rhs, vreg_alloc, var_to_reg, globals)?;
    let i64_ty = MirType::Scalar(ScalarType::I64);
    let IRVal::Variable(name) = rhs else {
        normalize_lowered_literal(&mut operand, i64_ty);
        return Ok(operand);
    };
    let Some(IRType::Primitive(prim)) = facts.ir_types.get(name) else {
        return Err(FromIRError::UnsupportedInstruction);
    };
    let from = map_ir_prim(*prim)?;
    if from == i64_ty {
        return Ok(operand);
    }
    let dst = Register::Virtual(vreg_alloc.allocate_gpr());
    out.push(Instruction::IntConvert {
        from,
        to: i64_ty,
        signed: false,
        dst: dst.clone(),
        src: operand,
    });
    Ok(Operand::reg(dst))
}

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
                | IRInst::Copy { result, ty, .. } => (*result, class_of_type(ty)),
                IRInst::VaArg { result, ty, .. } => (*result, reg_class_for_prim(*ty)),
                IRInst::VectorBinary { result, .. }
                | IRInst::VectorSplat { result, .. }
                | IRInst::VectorInsertLane { result, .. }
                | IRInst::VectorShuffle { result, .. } => (*result, RegisterClass::Vec),
                IRInst::VectorExtractLane {
                    result,
                    element_type,
                    ..
                } => (*result, reg_class_for_prim(*element_type)),
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
        IRType::Vector { .. } => RegisterClass::Vec,
        _ => RegisterClass::Gpr,
    }
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
