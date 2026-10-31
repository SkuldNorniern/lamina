// C API for the IR builder.

use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::null_mut;

use lamina_ir::instruction::{BinaryOp, CmpOp};
use lamina_ir::owned::{OwnedIRBuilder, OwnedParam, OwnedStructField, OwnedType, OwnedValue};
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
    catch_unwind(|| Box::into_raw(Box::new(LaminaBuilder(OwnedIRBuilder::new()))))
        .unwrap_or(null_mut())
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

type_ctor!(lia_type_void, OwnedType::Void);
type_ctor!(lia_type_i8, OwnedType::Primitive(PrimitiveType::I8));
type_ctor!(lia_type_i16, OwnedType::Primitive(PrimitiveType::I16));
type_ctor!(lia_type_i32, OwnedType::Primitive(PrimitiveType::I32));
type_ctor!(lia_type_i64, OwnedType::Primitive(PrimitiveType::I64));
type_ctor!(lia_type_u8, OwnedType::Primitive(PrimitiveType::U8));
type_ctor!(lia_type_u16, OwnedType::Primitive(PrimitiveType::U16));
type_ctor!(lia_type_u32, OwnedType::Primitive(PrimitiveType::U32));
type_ctor!(lia_type_u64, OwnedType::Primitive(PrimitiveType::U64));
type_ctor!(lia_type_f32, OwnedType::Primitive(PrimitiveType::F32));
type_ctor!(lia_type_f64, OwnedType::Primitive(PrimitiveType::F64));
type_ctor!(lia_type_bool, OwnedType::Primitive(PrimitiveType::Bool));
type_ctor!(lia_type_ptr, OwnedType::Primitive(PrimitiveType::Ptr));

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_free(ty: *mut LaminaType) {
    if !ty.is_null() {
        unsafe { drop(Box::from_raw(ty)) };
    }
}

/// Named type reference (`!TypeName`). Declare it with `lia_builder_declare_type` before use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_named(name: *const c_char) -> *mut LaminaType {
    match cstr_to_str(name) {
        Some(s) => Box::into_raw(Box::new(LaminaType(OwnedType::Named(s.to_string())))),
        None => null_mut(),
    }
}

/// Array type: `[size x element_type]`. Clones `element_type`; caller retains ownership.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_array(
    element_type: *const LaminaType,
    size: u64,
) -> *mut LaminaType {
    match unsafe { element_type.as_ref() } {
        Some(t) => Box::into_raw(Box::new(LaminaType(OwnedType::Array {
            element_type: Box::new(t.0.clone()),
            size,
        }))),
        None => null_mut(),
    }
}

/// Struct field descriptor passed from C.
#[repr(C)]
pub struct LaminaStructField {
    pub name: *const c_char,
    pub ty: *const LaminaType,
}

/// Struct type with named fields. Clones all field types.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_struct(
    fields: *const LaminaStructField,
    count: usize,
) -> *mut LaminaType {
    match unsafe { owned_struct_fields(fields, count) } {
        Some(owned_fields) => Box::into_raw(Box::new(LaminaType(OwnedType::Struct(owned_fields)))),
        None => null_mut(),
    }
}

/// Packed struct type with named fields, laid out without padding. Clones all field types.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_type_packed_struct(
    fields: *const LaminaStructField,
    count: usize,
) -> *mut LaminaType {
    match unsafe { owned_struct_fields(fields, count) } {
        Some(owned_fields) => {
            Box::into_raw(Box::new(LaminaType(OwnedType::PackedStruct(owned_fields))))
        }
        None => null_mut(),
    }
}

unsafe fn owned_struct_fields(
    fields: *const LaminaStructField,
    count: usize,
) -> Option<Vec<OwnedStructField>> {
    if count > 0 && fields.is_null() {
        return None;
    }
    let mut owned_fields = Vec::with_capacity(count);
    for i in 0..count {
        let f = unsafe { &*fields.add(i) };
        let name = cstr_to_str(f.name)?.to_string();
        let ty = unsafe { f.ty.as_ref() }?.0.clone();
        owned_fields.push(OwnedStructField { name, ty });
    }
    Some(owned_fields)
}

// ---------------------------------------------------------------------------
// Value constructors
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_var(name: *const c_char) -> *mut LaminaValue {
    match cstr_to_str(name) {
        Some(s) => Box::into_raw(Box::new(LaminaValue(OwnedValue::Variable(s.to_string())))),
        None => null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_value_global(name: *const c_char) -> *mut LaminaValue {
    match cstr_to_str(name) {
        Some(s) => Box::into_raw(Box::new(LaminaValue(OwnedValue::Global(s.to_string())))),
        None => null_mut(),
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
pub extern "C" fn lia_value_u16(v: u16) -> *mut LaminaValue {
    Box::into_raw(Box::new(LaminaValue(OwnedValue::U16(v))))
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
        None => null_mut(),
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
            Err(msg) => {
                set_error(msg);
                return LaminaStatus::ErrorInvalidArgument;
            }
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
            Err(msg) => {
                set_error(msg);
                return LaminaStatus::ErrorInvalidArgument;
            }
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

/// Declares a named type (`type !name = ty`) for `lia_type_named` references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_declare_type(
    builder: *mut LaminaBuilder,
    name: *const c_char,
    ty: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let name = require_str!(name, "name");
        let t = require_ref!(ty, "ty");
        b.0.declare_type(name.to_string(), t.0.clone());
        clear_error();
        LaminaStatus::Ok
    }))
}

// ---------------------------------------------------------------------------
// Op enums (C-facing; translated into Rust enums inside lamina-c)
// ---------------------------------------------------------------------------

#[repr(C)]
pub enum LaminaBinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4,
    And = 5,
    Or = 6,
    Xor = 7,
    Shl = 8,
    Shr = 9,
}

impl LaminaBinaryOp {
    fn into_rust(self) -> BinaryOp {
        match self {
            LaminaBinaryOp::Add => BinaryOp::Add,
            LaminaBinaryOp::Sub => BinaryOp::Sub,
            LaminaBinaryOp::Mul => BinaryOp::Mul,
            LaminaBinaryOp::Div => BinaryOp::Div,
            LaminaBinaryOp::Rem => BinaryOp::Rem,
            LaminaBinaryOp::And => BinaryOp::And,
            LaminaBinaryOp::Or => BinaryOp::Or,
            LaminaBinaryOp::Xor => BinaryOp::Xor,
            LaminaBinaryOp::Shl => BinaryOp::Shl,
            LaminaBinaryOp::Shr => BinaryOp::Shr,
        }
    }
}

#[repr(C)]
pub enum LaminaCmpOp {
    Eq = 0,
    Ne = 1,
    Gt = 2,
    Ge = 3,
    Lt = 4,
    Le = 5,
}

impl LaminaCmpOp {
    fn into_rust(self) -> CmpOp {
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
            b.0.binary(op.into_rust(), result, prim, &lhs_inner.0, &rhs_inner.0);
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
            b.0.cmp(op.into_rust(), result, prim, &lhs_inner.0, &rhs_inner.0);
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
        b.0.branch(&cond.0, t, f);
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
        b.0.jump(t);
        clear_error();
        LaminaStatus::Ok
    }))
}

/// A switch case passed from C: `(integer_value, target_label)`.
#[repr(C)]
pub struct LaminaSwitchCase {
    pub value: i64,
    pub label: *const c_char,
}

/// Multi-way branch. `cases` may be NULL when `case_count` is 0 (degenerates to jump to default).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_switch(
    builder: *mut LaminaBuilder,
    ty: *const LaminaType,
    value: *const LaminaValue,
    default_label: *const c_char,
    cases: *const LaminaSwitchCase,
    case_count: usize,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let t = require_ref!(ty, "ty");
        let v = require_ref!(value, "value");
        let default = require_str!(default_label, "default_label");
        let prim = match t.0 {
            OwnedType::Primitive(p) => p,
            _ => {
                set_error("switch type must be primitive");
                return LaminaStatus::ErrorInvalidArgument;
            }
        };
        if case_count > 0 && cases.is_null() {
            set_error("cases is null but case_count > 0");
            return LaminaStatus::ErrorInvalidArgument;
        }
        let mut owned_cases: Vec<(i64, String)> = Vec::with_capacity(case_count);
        for i in 0..case_count {
            let c = &*cases.add(i);
            let label = match cstr_to_str(c.label) {
                Some(s) => s.to_string(),
                None => {
                    set_error("switch case label is null or invalid UTF-8");
                    return LaminaStatus::ErrorInvalidArgument;
                }
            };
            owned_cases.push((c.value, label));
        }
        let case_refs: Vec<(i64, &str)> =
            owned_cases.iter().map(|(v, l)| (*v, l.as_str())).collect();
        b.0.switch(prim, &v.0, default, &case_refs);
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
        let res = if result.is_null() {
            None
        } else {
            cstr_to_str(result)
        };
        let owned_args = match collect_values(args, arg_count) {
            Ok(v) => v,
            Err(msg) => {
                set_error(msg);
                return LaminaStatus::ErrorInvalidArgument;
            }
        };
        b.0.call(res, fname, &owned_args);
        clear_error();
        LaminaStatus::Ok
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_copy(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
    value: *const LaminaValue,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let result = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        let v = require_ref!(value, "value");
        b.0.copy(result, &t.0, &v.0);
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
pub unsafe extern "C" fn lia_builder_stackalloc(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    ty: *const LaminaType,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let t = require_ref!(ty, "ty");
        b.0.stackalloc(r, &t.0);
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
pub unsafe extern "C" fn lia_builder_getelem(
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
            b.0.getelem(r, &ap.0, &idx.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("getelem element_type must be primitive");
            LaminaStatus::ErrorInvalidArgument
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lia_builder_getfieldptr_typed(
    builder: *mut LaminaBuilder,
    result: *const c_char,
    struct_ptr: *const LaminaValue,
    struct_type: *const LaminaType,
    field_index: usize,
) -> LaminaStatus {
    catch(AssertUnwindSafe(|| unsafe {
        let b = require_mut!(builder, "builder");
        let r = require_str!(result, "result");
        let sp = require_ref!(struct_ptr, "struct_ptr");
        let st = require_ref!(struct_type, "struct_type");
        match &st.0 {
            OwnedType::Struct(_) | OwnedType::PackedStruct(_) | OwnedType::Named(_) => {
                b.0.getfieldptr(r, &sp.0, st.0.clone(), field_index);
                clear_error();
                LaminaStatus::Ok
            }
            _ => {
                set_error("getfieldptr struct_type must be a struct or named type");
                LaminaStatus::ErrorInvalidArgument
            }
        }
    }))
}

fn is_integer_primitive(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
            | PrimitiveType::U128
    )
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
        if let OwnedType::Primitive(prim) = tt.0
            && is_integer_primitive(prim)
        {
            b.0.ptrtoint(r, &p.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("ptrtoint target_type must be an integer primitive");
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
        if let OwnedType::Primitive(prim) = tt.0
            && is_integer_primitive(prim)
        {
            b.0.inttoptr(r, &iv.0, prim);
            clear_error();
            LaminaStatus::Ok
        } else {
            set_error("inttoptr target_type must be an integer primitive");
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
                        set_error(concat!(
                            stringify!($fn_name),
                            " source and target must be primitive"
                        ));
                        LaminaStatus::ErrorInvalidArgument
                    }
                }
            }))
        }
    };
}

conv_inst!(lia_builder_zext, zext);
conv_inst!(lia_builder_sext, sext);
conv_inst!(lia_builder_trunc, trunc);
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
        let name = cstr_to_str(p.name).ok_or("parameter name is null or invalid UTF-8")?;
        let ty = unsafe { p.ty.as_ref() }
            .map(|t| t.0.clone())
            .ok_or("parameter type is null")?;
        out.push(OwnedParam {
            name: name.to_string(),
            ty,
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    fn cs(s: &str) -> CString {
        // SAFETY: test strings contain no NUL bytes.
        CString::new(s).unwrap()
    }

    #[test]
    fn builder_free_null_does_not_crash() {
        unsafe { lia_builder_free(null_mut()) };
    }

    #[test]
    fn type_named_valid_name_is_non_null() {
        unsafe {
            let ty = lia_type_named(cs("MyStruct").as_ptr());
            assert!(!ty.is_null());
            lia_type_free(ty);
        }
    }

    #[test]
    fn type_named_null_returns_null() {
        unsafe {
            let ty = lia_type_named(ptr::null());
            assert!(ty.is_null());
        }
    }

    #[test]
    fn type_array_returns_non_null() {
        unsafe {
            let elem = lia_type_i64();
            let arr = lia_type_array(elem, 8);
            assert!(!arr.is_null());
            lia_type_free(elem);
            lia_type_free(arr);
        }
    }

    #[test]
    fn type_array_null_element_returns_null() {
        unsafe {
            let arr = lia_type_array(ptr::null(), 4);
            assert!(arr.is_null());
        }
    }

    #[test]
    fn type_struct_with_one_field_returns_non_null() {
        unsafe {
            let field_ty = lia_type_i64();
            let field_name = cs("x");
            let fields = [LaminaStructField {
                name: field_name.as_ptr(),
                ty: field_ty,
            }];
            let ty = lia_type_struct(fields.as_ptr(), 1);
            assert!(!ty.is_null());
            lia_type_free(field_ty);
            lia_type_free(ty);
        }
    }

    #[test]
    fn getfieldptr_typed_form_builds() {
        unsafe {
            let builder = lia_builder_create();
            let ret = lia_type_i64();
            let fname = cs("f");
            assert_eq!(
                lia_builder_function(builder, fname.as_ptr(), std::ptr::null(), 0, ret),
                LaminaStatus::Ok
            );
            let result = cs("p");
            let ptr = lia_value_var(cs("s").as_ptr());
            let layout = lia_type_named(cs("Pair").as_ptr());
            let typed = lia_builder_getfieldptr_typed(builder, result.as_ptr(), ptr, layout, 1);
            assert_eq!(typed, LaminaStatus::Ok);
            lia_type_free(layout);
            lia_type_free(ret);
            lia_value_free(ptr);
            lia_builder_free(builder);
        }
    }

    #[test]
    fn getfieldptr_typed_rejects_a_primitive_layout() {
        unsafe {
            let builder = lia_builder_create();
            let result = cs("p");
            let ptr = lia_value_var(cs("s").as_ptr());
            let layout = lia_type_i64();
            let status = lia_builder_getfieldptr_typed(builder, result.as_ptr(), ptr, layout, 0);
            assert_eq!(status, LaminaStatus::ErrorInvalidArgument);
            lia_type_free(layout);
            lia_value_free(ptr);
            lia_builder_free(builder);
        }
    }

    #[test]
    fn type_free_null_does_not_crash() {
        unsafe { lia_type_free(null_mut()) };
    }
}
