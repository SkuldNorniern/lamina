// C API for the IR builder.

use std::ffi::c_char;
use std::panic::AssertUnwindSafe;

use lamina_ir::instruction::{BinaryOp, CmpOp};
use lamina_ir::owned::{OwnedIRBuilder, OwnedParam, OwnedType, OwnedValue};
use lamina_ir::types::PrimitiveType;

use crate::error::{clear_error, set_error};
use crate::types::{LaminaBuilder, LaminaModule, LaminaType, LaminaValue};
use crate::{LaminaStatus, catch, cstr_to_str};

macro_rules! require_mut {
    ($ptr:expr, $name:literal) => {
        match unsafe { $ptr.as_mut() } {
            Some(v) => v,
            None => {
                set_error(concat!($name, " is null"));
                return LaminaStatus::ErrorInvalidArgument;
            }
        }
    };
}

macro_rules! require_ref {
    ($ptr:expr, $name:literal) => {
        match unsafe { $ptr.as_ref() } {
            Some(v) => v,
            None => {
                set_error(concat!($name, " is null"));
                return LaminaStatus::ErrorInvalidArgument;
            }
        }
    };
}

macro_rules! require_str {
    ($ptr:expr, $name:literal) => {
        match cstr_to_str($ptr) {
            Some(v) => v,
            None => {
                set_error(concat!($name, " is null or invalid UTF-8"));
                return LaminaStatus::ErrorInvalidArgument;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Builder lifecycle
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn lia_builder_create() -> *mut LaminaBuilder {
    match std::panic::catch_unwind(|| {
        Box::into_raw(Box::new(LaminaBuilder(OwnedIRBuilder::new())))
    }) {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_free(builder: *mut LaminaBuilder) {
    if !builder.is_null() {
        unsafe { drop(Box::from_raw(builder)) };
    }
}

// ---------------------------------------------------------------------------
// Type constructors
// ---------------------------------------------------------------------------

macro_rules! type_ctor {
    ($name:ident, $variant:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name() -> *mut LaminaType {
            Box::into_raw(Box::new(LaminaType($variant)))
        }
    };
}

type_ctor!(lia_type_void,  OwnedType::Void);
type_ctor!(lia_type_i8,    OwnedType::Primitive(PrimitiveType::I8));
type_ctor!(lia_type_i16,   OwnedType::Primitive(PrimitiveType::I16));
type_ctor!(lia_type_i32,   OwnedType::Primitive(PrimitiveType::I32));
type_ctor!(lia_type_i64,   OwnedType::Primitive(PrimitiveType::I64));
type_ctor!(lia_type_u8,    OwnedType::Primitive(PrimitiveType::U8));
type_ctor!(lia_type_u16,   OwnedType::Primitive(PrimitiveType::U16));
type_ctor!(lia_type_u32,   OwnedType::Primitive(PrimitiveType::U32));
type_ctor!(lia_type_u64,   OwnedType::Primitive(PrimitiveType::U64));
type_ctor!(lia_type_f32,   OwnedType::Primitive(PrimitiveType::F32));
type_ctor!(lia_type_f64,   OwnedType::Primitive(PrimitiveType::F64));
type_ctor!(lia_type_bool,  OwnedType::Primitive(PrimitiveType::Bool));
type_ctor!(lia_type_ptr,   OwnedType::Primitive(PrimitiveType::Ptr));

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_free(ty: *mut LaminaType) {
    if !ty.is_null() {
        unsafe { drop(Box::from_raw(ty)) };
    }
}

// ---------------------------------------------------------------------------
// Value constructors
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_var(name: *const c_char) -> *mut LaminaValue {
    match cstr_to_str(name) {
        Some(s) => Box::into_raw(Box::new(LaminaValue(OwnedValue::Variable(s.to_string())))),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_global(name: *const c_char) -> *mut LaminaValue {
    match cstr_to_str(name) {
        Some(s) => Box::into_raw(Box::new(LaminaValue(OwnedValue::Global(s.to_string())))),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lia_value_i8(v: i8) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::I8(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_i16(v: i16) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::I16(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_i32(v: i32) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::I32(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_i64(v: i64) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::I64(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_u8(v: u8) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::U8(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_u32(v: u32) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::U32(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_u64(v: u64) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::U64(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_f32(v: f32) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::F32(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_f64(v: f64) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::F64(v))))
}
#[unsafe(no_mangle)]
pub extern "C" fn lia_value_bool(v: bool) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::Bool(v))))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_string(s: *const c_char) -> *mut LaminaValue {
    match cstr_to_str(s) {
        Some(s) => Box::into_raw(Box::new(LaminaValue(OwnedValue::Str(s.to_string())))),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_free(val: *mut LaminaValue) {
    if !val.is_null() {
        unsafe { drop(Box::from_raw(val)) };
    }
}

// ---------------------------------------------------------------------------
// Function / block definition
// ---------------------------------------------------------------------------

/// A function parameter passed from C (borrowed — not heap-allocated).
#[repr(C)]
pub struct LaminaParam {
    pub name: *const c_char,
    pub ty: *const LaminaType,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_function(
    builder: *mut LaminaBuilder,
    name: *const c_char,
    params: *const LaminaParam,
    param_count: usize,
    return_type: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let name = require_str!(name, "name");
        let ret = require_ref!(return_type, "return_type").0.clone();
        let owned_params = match collect_params(params, param_count) {
            Ok(p) => p,
            Err(msg) => { set_error(msg); return LaminaStatus::ErrorInvalidArgument; }
        };
        b.0.function_with_params(name.to_string(), owned_params, ret);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_external_function(
    builder: *mut LaminaBuilder,
    name: *const c_char,
    params: *const LaminaParam,
    param_count: usize,
    return_type: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let name = require_str!(name, "name");
        let ret = require_ref!(return_type, "return_type").0.clone();
        let owned_params = match collect_params(params, param_count) {
            Ok(p) => p,
            Err(msg) => { set_error(msg); return LaminaStatus::ErrorInvalidArgument; }
        };
        b.0.external_function(name.to_string(), owned_params, ret);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_block(
    builder: *mut LaminaBuilder,
    name: *const c_char,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let name = require_str!(name, "name");
        b.0.block(name.to_string());
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_set_entry_block(
    builder: *mut LaminaBuilder,
    name: *const c_char,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let name = require_str!(name, "name");
        b.0.set_entry_block(name.to_string());
        clear_error();
        LaminaStatus::Ok
    }))
}

/// Finalises the builder and produces a module handle.
/// Builder remains valid and reusable after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_finish(
    builder: *const LaminaBuilder,
    module_out: *mut *mut LaminaModule,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_ref!(builder, "builder");
        if module_out.is_null() {
            set_error("module_out is null");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let ir_text = b.0.build_ir_text();
        *module_out = Box::into_raw(Box::new(LaminaModule(ir_text)));
        clear_error();
        LaminaStatus::Ok
    }))
}

// ---------------------------------------------------------------------------
// Op enums (C-facing; translated into Rust enums inside lamina-c)
// ---------------------------------------------------------------------------

#[repr(C)]
pub enum LaminaBinaryOp {
    Add = 0, Sub = 1, Mul = 2, Div = 3, Rem = 4,
    And = 5, Or = 6, Xor = 7, Shl = 8, Shr = 9,
}

impl LaminaBinaryOp {
    fn to_rust(self) -> BinaryOp {
        match self {
            LaminaBinaryOp::Add => BinaryOp::Add,
            LaminaBinaryOp::Sub => BinaryOp::Sub,
            LaminaBinaryOp::Mul => BinaryOp::Mul,
            LaminaBinaryOp::Div => BinaryOp::Div,
            LaminaBinaryOp::Rem => BinaryOp::Rem,
            LaminaBinaryOp::And => BinaryOp::And,
            LaminaBinaryOp::Or  => BinaryOp::Or,
            LaminaBinaryOp::Xor => BinaryOp::Xor,
            LaminaBinaryOp::Shl => BinaryOp::Shl,
            LaminaBinaryOp::Shr => BinaryOp::Shr,
        }
    }
}

#[repr(C)]
pub enum LaminaCmpOp {
    Eq = 0, Ne = 1, Gt = 2, Ge = 3, Lt = 4, Le = 5,
}

impl LaminaCmpOp {
    fn to_rust(self) -> CmpOp {
        match self {
            LaminaCmpOp::Eq => CmpOp::Eq,
            LaminaCmpOp::Ne => CmpOp::Ne,
            LaminaCmpOp::Gt => CmpOp::Gt,
            LaminaCmpOp::Ge => CmpOp::Ge,
            LaminaCmpOp::Lt => CmpOp::Lt,
            LaminaCmpOp::Le => CmpOp::Le,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction API (phase 1)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_binary(
    builder: *mut LaminaBuilder,
    op: LaminaBinaryOp,
    result: *const c_char,
    ty: *const LaminaType,
    lhs: *const LaminaValue,
    rhs: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let result = require_str!(result, "result");
        let ty_inner = require_ref!(ty, "ty");
        let lhs_inner = require_ref!(lhs, "lhs");
        let rhs_inner = require_ref!(rhs, "rhs");
        if let OwnedType::Primitive(prim) = ty_inner.0 {
            b.0.binary(op.to_rust(), result, prim, &lhs_inner.0, &rhs_inner.0);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("binary requires a primitive type");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_cmp(
    builder: *mut LaminaBuilder,
    op: LaminaCmpOp,
    result: *const c_char,
    ty: *const LaminaType,
    lhs: *const LaminaValue,
    rhs: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let result = require_str!(result, "result");
        let ty_inner = require_ref!(ty, "ty");
        let lhs_inner = require_ref!(lhs, "lhs");
        let rhs_inner = require_ref!(rhs, "rhs");
        if let OwnedType::Primitive(prim) = ty_inner.0 {
            b.0.cmp(op.to_rust(), result, prim, &lhs_inner.0, &rhs_inner.0);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("cmp requires a primitive type");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_branch(
    builder: *mut LaminaBuilder,
    condition: *const LaminaValue,
    true_label: *const c_char,
    false_label: *const c_char,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let cond = require_ref!(condition, "condition");
        let t = require_str!(true_label, "true_label");
        let f = require_str!(false_label, "false_label");
        b.0.branch(&cond.0, t.to_string(), f.to_string());
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_jump(
    builder: *mut LaminaBuilder,
    target: *const c_char,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let t = require_str!(target, "target");
        b.0.jump(t.to_string());
        clear_error();
        LaminaStatus::Ok
    }))
}

/// `result` may be NULL for void calls. `args` may be NULL when `arg_count` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_call(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    func_name: *const c_char,
    args: *const *const LaminaValue,
    arg_count: usize,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let fname = require_str!(func_name, "func_name");
        let res = if result.is_null() { None } else { cstr_to_str(result) };
        let owned_args = match collect_values(args, arg_count) {
            Ok(v) => v,
            Err(msg) => { set_error(msg); return LaminaStatus::ErrorInvalidArgument; }
        };
        b.0.call(res, fname.to_string(), &owned_args);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_phi(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
    values: *const *const LaminaValue,
    labels: *const *const c_char,
    count: usize,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let result = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        let mut incoming = Vec::with_capacity(count);
        for i in 0..count {
            let val_ptr = *values.add(i);
            let lbl_ptr = *labels.add(i);
            let val = match val_ptr.as_ref() {
                Some(v) => v,
                None => { set_error("phi value pointer is null"); return LaminaStatus::ErrorInvalidArgument; }
            };
            let lbl = match cstr_to_str(lbl_ptr) {
                Some(s) => s,
                None => { set_error("phi label pointer is null"); return LaminaStatus::ErrorInvalidArgument; }
            };
            incoming.push((val.0.clone(), lbl.to_string()));
        }
        b.0.phi(result, &t.0, &incoming);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_return(
    builder: *mut LaminaBuilder,
    ty: *const LaminaType,
    value: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let t = require_ref!(ty, "ty");
        let v = require_ref!(value, "value");
        b.0.ret(&t.0, &v.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_return_void(builder: *mut LaminaBuilder) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        b.0.ret_void();
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_alloc_stack(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        b.0.alloc_stack(r, &t.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_alloc_heap(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        b.0.alloc_heap(r, &t.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_load(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
    ptr: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        let p = require_ref!(ptr, "ptr");
        b.0.load(r, &t.0, &p.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_store(
    builder: *mut LaminaBuilder,
    ty: *const LaminaType,
    ptr: *const LaminaValue,
    value: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let t = require_ref!(ty, "ty");
        let p = require_ref!(ptr, "ptr");
        let v = require_ref!(value, "value");
        b.0.store(&t.0, &p.0, &v.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_dealloc(
    builder: *mut LaminaBuilder,
    ptr: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let p = require_ref!(ptr, "ptr");
        b.0.dealloc(&p.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_getelementptr(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    array_ptr: *const LaminaValue,
    index: *const LaminaValue,
    element_type: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let ap = require_ref!(array_ptr, "array_ptr");
        let idx = require_ref!(index, "index");
        let et = require_ref!(element_type, "element_type");
        if let OwnedType::Primitive(prim) = et.0 {
            b.0.getelementptr(r, &ap.0, &idx.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("getelementptr element_type must be primitive");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_struct_gep(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    struct_ptr: *const LaminaValue,
    field_index: usize,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let sp = require_ref!(struct_ptr, "struct_ptr");
        b.0.struct_gep(r, &sp.0, field_index);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_ptrtoint(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ptr_value: *const LaminaValue,
    target_type: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let p = require_ref!(ptr_value, "ptr_value");
        let tt = require_ref!(target_type, "target_type");
        if let OwnedType::Primitive(prim) = tt.0 {
            b.0.ptrtoint(r, &p.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("ptrtoint target_type must be primitive");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_inttoptr(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    int_value: *const LaminaValue,
    target_type: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let iv = require_ref!(int_value, "int_value");
        let tt = require_ref!(target_type, "target_type");
        if let OwnedType::Primitive(prim) = tt.0 {
            b.0.inttoptr(r, &iv.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("inttoptr target_type must be primitive");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

macro_rules! conv_inst {
    ($fn_name:ident, $method:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            builder: *mut LaminaBuilder,
            result: *const c_char,
            source_type: *const LaminaType,
            target_type: *const LaminaType,
            value: *const LaminaValue,
        ) -> LaminaStatus {
            catch(AssertUnwindSafe(|| unsafe {
                let b = require_mut!(builder, "builder");
                let r = require_str!(result, "result");
                let src = require_ref!(source_type, "source_type");
                let tgt = require_ref!(target_type, "target_type");
                let v = require_ref!(value, "value");
                match (&src.0, &tgt.0) {
                    (OwnedType::Primitive(s), OwnedType::Primitive(t)) => {
                        b.0.$method(r, *s, *t, &v.0);
                        clear_error();
                        LaminaStatus::Ok
                    }
                    _ => {
                        set_error(concat!(stringify!($fn_name), " source and target must be primitive"));
                        LaminaStatus::ErrorInvalidArgument
                    }
                }
            }))
        }
    };
}

conv_inst!(lia_builder_zext,    zext);
conv_inst!(lia_builder_sext,    sext);
conv_inst!(lia_builder_trunc,   trunc);
conv_inst!(lia_builder_bitcast, bitcast);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_select(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
    cond: *const LaminaValue,
    true_val: *const LaminaValue,
    false_val: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        let c = require_ref!(cond, "cond");
        let tv = require_ref!(true_val, "true_val");
        let fv = require_ref!(false_val, "false_val");
        b.0.select(r, &t.0, &c.0, &tv.0, &fv.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_write(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    buffer: *const LaminaValue,
    size: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let buf = require_ref!(buffer, "buffer");
        let sz = require_ref!(size, "size");
        b.0.write(r, &buf.0, &sz.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_read(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    buffer: *const LaminaValue,
    size: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let buf = require_ref!(buffer, "buffer");
        let sz = require_ref!(size, "size");
        b.0.read(r, &buf.0, &sz.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_write_byte(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    value: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let v = require_ref!(value, "value");
        b.0.write_byte(r, &v.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_read_byte(
    builder: *mut LaminaBuilder,
    result: *const c_char,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        b.0.read_byte(r);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_write_ptr(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ptr: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let p = require_ref!(ptr, "ptr");
        b.0.write_ptr(r, &p.0);
        clear_error();
        LaminaStatus::Ok
    }))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

unsafe fn collect_params(
    params: *const LaminaParam,
    count: usize,
) -> Result<Vec<OwnedParam>, &'static str> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return Ok(out);
    }
    if params.is_null() {
        return Err("params is null but param_count > 0");
    }
    for i in 0..count {
        let p = unsafe { &*params.add(i) };
        let name = cstr_to_str(p.name)
            .ok_or("parameter name is null or invalid UTF-8")?;
        let ty = unsafe { p.ty.as_ref() }
            .map(|t| t.0.clone())
            .ok_or("parameter type is null")?;
        out.push(OwnedParam { name: name.to_string(), ty });
    }
    Ok(out)
}

unsafe fn collect_values(
    vals: *const *const LaminaValue,
    count: usize,
) -> Result<Vec<OwnedValue>, &'static str> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return Ok(out);
    }
    if vals.is_null() {
        return Err("args is null but arg_count > 0");
    }
    for i in 0..count {
        let vp = unsafe { *vals.add(i) };
        match unsafe { vp.as_ref() } {
            Some(v) => out.push(v.0.clone()),
            None => return Err("arg pointer is null"),
        }
    }
    Ok(out)
}
