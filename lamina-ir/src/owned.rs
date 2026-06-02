// Owned IR builder — lifetime-free API for FFI consumers.
//
// All data is heap-allocated (String, Vec). `build_ir_text()` serialises the
// accumulated builder state to Lamina IR text, which callers can then feed
// into the existing parser + codegen pipeline.

use std::fmt;

use crate::instruction::{BinaryOp, CmpOp};
use crate::types::PrimitiveType;

// ---------------------------------------------------------------------------
// Owned type and value representations
// ---------------------------------------------------------------------------

/// Owned equivalent of `Type<'a>`. Phase-1 supports primitives and void.
#[derive(Debug, Clone)]
pub enum OwnedType {
    Primitive(PrimitiveType),
    Void,
    Named(String),
    Array {
        element_type: Box<OwnedType>,
        size: u64,
    },
    Struct(Vec<OwnedStructField>),
    Tuple(Vec<OwnedType>),
}

/// A named field inside an [`OwnedType::Struct`].
#[derive(Debug, Clone)]
pub struct OwnedStructField {
    /// Field name.
    pub name: String,
    /// Field type.
    pub ty: OwnedType,
}

impl fmt::Display for OwnedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OwnedType::Primitive(p) => write!(f, "{}", p.as_str()),
            OwnedType::Void => write!(f, "void"),
            OwnedType::Named(n) => write!(f, "@{}", n),
            OwnedType::Array { element_type, size } => {
                write!(f, "[{} x {}]", size, element_type)
            }
            OwnedType::Struct(fields) => {
                write!(f, "struct {{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    write!(f, "{}: {}", field.name, field.ty)?;
                    if i + 1 < fields.len() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " }}")
            }
            OwnedType::Tuple(types) => {
                write!(f, "tuple(")?;
                for (i, ty) in types.iter().enumerate() {
                    write!(f, "{}", ty)?;
                    if i + 1 < types.len() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

/// Owned equivalent of `Value<'a>`.
#[derive(Debug, Clone)]
pub enum OwnedValue {
    Variable(String),
    Global(String),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
}

impl fmt::Display for OwnedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OwnedValue::Variable(n) => write!(f, "%{}", n),
            OwnedValue::Global(n) => write!(f, "@{}", n),
            OwnedValue::I8(v) => write!(f, "{}", v),
            OwnedValue::I16(v) => write!(f, "{}", v),
            OwnedValue::I32(v) => write!(f, "{}", v),
            OwnedValue::I64(v) => write!(f, "{}", v),
            OwnedValue::U8(v) => write!(f, "{}", v),
            OwnedValue::U16(v) => write!(f, "{}", v),
            OwnedValue::U32(v) => write!(f, "{}", v),
            OwnedValue::U64(v) => write!(f, "{}", v),
            OwnedValue::F32(v) => write!(f, "{}", v),
            OwnedValue::F64(v) => write!(f, "{}", v),
            OwnedValue::Bool(v) => write!(f, "{}", v),
            OwnedValue::Str(s) => write!(f, "\"{}\"", s),
        }
    }
}

/// A function parameter with an owned name and type.
#[derive(Debug, Clone)]
pub struct OwnedParam {
    /// Parameter name (without `%` prefix).
    pub name: String,
    /// Parameter type.
    pub ty: OwnedType,
}

// ---------------------------------------------------------------------------
// Internal builder state
// ---------------------------------------------------------------------------

struct OwnedBlock {
    instructions: Vec<String>,
}

struct OwnedFunction {
    name: String,
    params: Vec<OwnedParam>,
    return_type: OwnedType,
    is_extern: bool,
    blocks: Vec<(String, OwnedBlock)>,
    entry_block: String,
}

// ---------------------------------------------------------------------------
// OwnedIRBuilder
// ---------------------------------------------------------------------------

/// Lifetime-free IR builder for FFI use.
///
/// Accumulates functions, blocks, and instructions as owned data, then
/// serialises to Lamina IR text via [`build_ir_text`]. Callers feed that text
/// into [`lamina::parser::parse_module`] and the codegen pipeline.
pub struct OwnedIRBuilder {
    functions: Vec<OwnedFunction>,
    current_function: Option<usize>,
    current_block: Option<usize>,
}

impl Default for OwnedIRBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OwnedIRBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        OwnedIRBuilder {
            functions: Vec::new(),
            current_function: None,
            current_block: None,
        }
    }

    // -----------------------------------------------------------------------
    // Function definition
    // -----------------------------------------------------------------------

    /// Begins a new function with no parameters.
    pub fn function(&mut self, name: impl Into<String>, return_type: OwnedType) -> &mut Self {
        self.function_with_params(name, vec![], return_type)
    }

    /// Begins a new function with the given parameters.
    pub fn function_with_params(
        &mut self,
        name: impl Into<String>,
        params: Vec<OwnedParam>,
        return_type: OwnedType,
    ) -> &mut Self {
        let func = OwnedFunction {
            name: name.into(),
            params,
            return_type,
            is_extern: false,
            blocks: Vec::new(),
            entry_block: "entry".to_string(),
        };
        self.functions.push(func);
        let fi = self.functions.len() - 1;
        self.current_function = Some(fi);
        self.current_block = None;
        self.block("entry");
        self
    }

    /// Declares an external function (no body; emits `@extern` annotation).
    pub fn external_function(
        &mut self,
        name: impl Into<String>,
        params: Vec<OwnedParam>,
        return_type: OwnedType,
    ) -> &mut Self {
        let func = OwnedFunction {
            name: name.into(),
            params,
            return_type,
            is_extern: true,
            blocks: Vec::new(),
            entry_block: "entry".to_string(),
        };
        self.functions.push(func);
        self.current_function = Some(self.functions.len() - 1);
        self.current_block = None;
        self
    }

    // -----------------------------------------------------------------------
    // Block management
    // -----------------------------------------------------------------------

    /// Adds a basic block to the current function and makes it current.
    pub fn block(&mut self, name: impl Into<String>) -> &mut Self {
        if let Some(fi) = self.current_function {
            let block_name = name.into();
            self.functions[fi]
                .blocks
                .push((block_name, OwnedBlock { instructions: Vec::new() }));
            self.current_block = Some(self.functions[fi].blocks.len() - 1);
        }
        self
    }

    /// Overrides the entry block for the current function.
    pub fn set_entry_block(&mut self, name: impl Into<String>) -> &mut Self {
        if let Some(fi) = self.current_function {
            self.functions[fi].entry_block = name.into();
        }
        self
    }

    // -----------------------------------------------------------------------
    // Internal instruction push
    // -----------------------------------------------------------------------

    fn push(&mut self, text: String) {
        if let (Some(fi), Some(bi)) = (self.current_function, self.current_block) {
            self.functions[fi].blocks[bi].1.instructions.push(text);
        }
    }

    // -----------------------------------------------------------------------
    // Arithmetic & comparison
    // -----------------------------------------------------------------------

    /// Appends a binary arithmetic instruction.
    pub fn binary(
        &mut self,
        op: BinaryOp,
        result: impl AsRef<str>,
        ty: PrimitiveType,
        lhs: &OwnedValue,
        rhs: &OwnedValue,
    ) -> &mut Self {
        let t = format!("%{} = {}.{} {}, {}", result.as_ref(), op, ty.as_str(), lhs, rhs);
        self.push(t);
        self
    }

    /// Appends a comparison instruction.
    pub fn cmp(
        &mut self,
        op: CmpOp,
        result: impl AsRef<str>,
        ty: PrimitiveType,
        lhs: &OwnedValue,
        rhs: &OwnedValue,
    ) -> &mut Self {
        let t = format!("%{} = {}.{} {}, {}", result.as_ref(), op, ty.as_str(), lhs, rhs);
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    /// Appends a conditional branch.
    pub fn branch(
        &mut self,
        condition: &OwnedValue,
        true_label: impl AsRef<str>,
        false_label: impl AsRef<str>,
    ) -> &mut Self {
        let t = format!("br {}, {}, {}", condition, true_label.as_ref(), false_label.as_ref());
        self.push(t);
        self
    }

    /// Appends an unconditional jump.
    pub fn jump(&mut self, target: impl AsRef<str>) -> &mut Self {
        let t = format!("jmp {}", target.as_ref());
        self.push(t);
        self
    }

    /// Appends a function call.
    pub fn call(
        &mut self,
        result: Option<&str>,
        func_name: impl AsRef<str>,
        args: &[OwnedValue],
    ) -> &mut Self {
        let args_str = args
            .iter()
            .map(|a| format!("{}", a))
            .collect::<Vec<_>>()
            .join(", ");
        let t = if let Some(r) = result {
            format!("%{} = call @{}({})", r, func_name.as_ref(), args_str)
        } else {
            format!("call @{}({})", func_name.as_ref(), args_str)
        };
        self.push(t);
        self
    }

    /// Appends a return with a value.
    pub fn ret(&mut self, ty: &OwnedType, value: &OwnedValue) -> &mut Self {
        let t = format!("ret.{} {}", ty, value);
        self.push(t);
        self
    }

    /// Appends a void return.
    pub fn ret_void(&mut self) -> &mut Self {
        self.push("ret.void".to_string());
        self
    }

    /// Appends a phi node.
    pub fn phi(
        &mut self,
        result: impl AsRef<str>,
        ty: &OwnedType,
        incoming: &[(OwnedValue, String)],
    ) -> &mut Self {
        let pairs = incoming
            .iter()
            .map(|(v, l)| format!("[{}, {}]", v, l))
            .collect::<Vec<_>>()
            .join(", ");
        let t = format!("%{} = phi.{} {}", result.as_ref(), ty, pairs);
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------

    /// Appends a stack allocation.
    pub fn alloc_stack(&mut self, result: impl AsRef<str>, ty: &OwnedType) -> &mut Self {
        let t = format!("%{} = alloc.ptr.stack {}", result.as_ref(), ty);
        self.push(t);
        self
    }

    /// Appends a heap allocation.
    pub fn alloc_heap(&mut self, result: impl AsRef<str>, ty: &OwnedType) -> &mut Self {
        let t = format!("%{} = alloc.ptr.heap {}", result.as_ref(), ty);
        self.push(t);
        self
    }

    /// Appends a load instruction.
    pub fn load(&mut self, result: impl AsRef<str>, ty: &OwnedType, ptr: &OwnedValue) -> &mut Self {
        let t = format!("%{} = load.{} {}", result.as_ref(), ty, ptr);
        self.push(t);
        self
    }

    /// Appends a store instruction.
    pub fn store(&mut self, ty: &OwnedType, ptr: &OwnedValue, val: &OwnedValue) -> &mut Self {
        let t = format!("store.{} {}, {}", ty, ptr, val);
        self.push(t);
        self
    }

    /// Appends a heap deallocation.
    pub fn dealloc(&mut self, ptr: &OwnedValue) -> &mut Self {
        let t = format!("dealloc.heap {}", ptr);
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // Pointer operations
    // -----------------------------------------------------------------------

    /// Appends a getelementptr (array element pointer).
    pub fn getelementptr(
        &mut self,
        result: impl AsRef<str>,
        array_ptr: &OwnedValue,
        index: &OwnedValue,
        element_type: PrimitiveType,
    ) -> &mut Self {
        let t = format!(
            "%{} = getelem.ptr {}, {}, {}",
            result.as_ref(),
            array_ptr,
            index,
            element_type.as_str()
        );
        self.push(t);
        self
    }

    /// Appends a struct field pointer instruction.
    pub fn struct_gep(
        &mut self,
        result: impl AsRef<str>,
        struct_ptr: &OwnedValue,
        field_index: usize,
    ) -> &mut Self {
        let t = format!("%{} = getfield.ptr {}, {}", result.as_ref(), struct_ptr, field_index);
        self.push(t);
        self
    }

    /// Appends a pointer-to-integer cast.
    pub fn ptrtoint(
        &mut self,
        result: impl AsRef<str>,
        ptr_value: &OwnedValue,
        target_type: PrimitiveType,
    ) -> &mut Self {
        let t = format!("%{} = ptrtoint {}, {}", result.as_ref(), ptr_value, target_type.as_str());
        self.push(t);
        self
    }

    /// Appends an integer-to-pointer cast.
    pub fn inttoptr(
        &mut self,
        result: impl AsRef<str>,
        int_value: &OwnedValue,
        target_type: PrimitiveType,
    ) -> &mut Self {
        let t = format!("%{} = inttoptr {}, {}", result.as_ref(), int_value, target_type.as_str());
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // Type conversions
    // -----------------------------------------------------------------------

    /// Appends a zero-extension instruction.
    pub fn zext(
        &mut self,
        result: impl AsRef<str>,
        source: PrimitiveType,
        target: PrimitiveType,
        value: &OwnedValue,
    ) -> &mut Self {
        let t = format!(
            "%{} = zext.{}.{} {}",
            result.as_ref(),
            source.as_str(),
            target.as_str(),
            value
        );
        self.push(t);
        self
    }

    /// Appends a sign-extension instruction.
    pub fn sext(
        &mut self,
        result: impl AsRef<str>,
        source: PrimitiveType,
        target: PrimitiveType,
        value: &OwnedValue,
    ) -> &mut Self {
        let t = format!(
            "%{} = sext.{}.{} {}",
            result.as_ref(),
            source.as_str(),
            target.as_str(),
            value
        );
        self.push(t);
        self
    }

    /// Appends an integer truncation instruction.
    pub fn trunc(
        &mut self,
        result: impl AsRef<str>,
        source: PrimitiveType,
        target: PrimitiveType,
        value: &OwnedValue,
    ) -> &mut Self {
        let t = format!(
            "%{} = trunc.{}.{} {}",
            result.as_ref(),
            source.as_str(),
            target.as_str(),
            value
        );
        self.push(t);
        self
    }

    /// Appends a bitcast instruction.
    pub fn bitcast(
        &mut self,
        result: impl AsRef<str>,
        source: PrimitiveType,
        target: PrimitiveType,
        value: &OwnedValue,
    ) -> &mut Self {
        let t = format!(
            "%{} = bitcast.{}.{} {}",
            result.as_ref(),
            source.as_str(),
            target.as_str(),
            value
        );
        self.push(t);
        self
    }

    /// Appends a conditional select instruction.
    pub fn select(
        &mut self,
        result: impl AsRef<str>,
        ty: &OwnedType,
        cond: &OwnedValue,
        true_val: &OwnedValue,
        false_val: &OwnedValue,
    ) -> &mut Self {
        let t = format!(
            "%{} = select.{} {}, {}, {}",
            result.as_ref(),
            ty,
            cond,
            true_val,
            false_val
        );
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // I/O
    // -----------------------------------------------------------------------

    /// Appends a buffer write syscall.
    pub fn write(
        &mut self,
        result: impl AsRef<str>,
        buffer: &OwnedValue,
        size: &OwnedValue,
    ) -> &mut Self {
        let t = format!("%{} = write {}, {}", result.as_ref(), buffer, size);
        self.push(t);
        self
    }

    /// Appends a buffer read syscall.
    pub fn read(
        &mut self,
        result: impl AsRef<str>,
        buffer: &OwnedValue,
        size: &OwnedValue,
    ) -> &mut Self {
        let t = format!("%{} = read {}, {}", result.as_ref(), buffer, size);
        self.push(t);
        self
    }

    /// Appends a single-byte write syscall.
    pub fn write_byte(&mut self, result: impl AsRef<str>, value: &OwnedValue) -> &mut Self {
        let t = format!("%{} = writebyte {}", result.as_ref(), value);
        self.push(t);
        self
    }

    /// Appends a single-byte read syscall.
    pub fn read_byte(&mut self, result: impl AsRef<str>) -> &mut Self {
        let t = format!("%{} = readbyte", result.as_ref());
        self.push(t);
        self
    }

    /// Appends a write-pointer-value syscall.
    pub fn write_ptr(&mut self, result: impl AsRef<str>, ptr: &OwnedValue) -> &mut Self {
        let t = format!("%{} = writeptr {}", result.as_ref(), ptr);
        self.push(t);
        self
    }

    /// Appends a debug print instruction.
    pub fn print(&mut self, value: &OwnedValue) -> &mut Self {
        let t = format!("print {}", value);
        self.push(t);
        self
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Serialise the accumulated state to Lamina IR text.
    pub fn build_ir_text(&self) -> String {
        let mut out = String::new();
        for func in &self.functions {
            self.emit_function(&mut out, func);
            out.push('\n');
        }
        out
    }

    fn emit_function(&self, out: &mut String, func: &OwnedFunction) {
        if func.is_extern {
            out.push_str("@extern\n");
        }

        // fn @name(ty %param, ...) -> ret {
        out.push_str("fn @");
        out.push_str(&func.name);
        out.push('(');
        for (i, p) in func.params.iter().enumerate() {
            out.push_str(&format!("{} %{}", p.ty, p.name));
            if i + 1 < func.params.len() {
                out.push_str(", ");
            }
        }
        out.push_str(&format!(") -> {} {{\n", func.return_type));

        // entry block first, then rest
        let entry = &func.entry_block;
        for (name, block) in &func.blocks {
            if name == entry {
                out.push_str(&format!("{}:\n", name));
                for inst in &block.instructions {
                    out.push_str(&format!("  {}\n", inst));
                }
                break;
            }
        }
        for (name, block) in &func.blocks {
            if name != entry {
                out.push_str(&format!("{}:\n", name));
                for inst in &block.instructions {
                    out.push_str(&format!("  {}\n", inst));
                }
            }
        }

        out.push_str("}\n");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::BinaryOp;
    use crate::types::PrimitiveType;

    #[test]
    fn test_add_function_ir_text() {
        let mut b = OwnedIRBuilder::new();
        b.function_with_params(
            "add",
            vec![
                OwnedParam { name: "a".to_string(), ty: OwnedType::Primitive(PrimitiveType::I64) },
                OwnedParam { name: "b".to_string(), ty: OwnedType::Primitive(PrimitiveType::I64) },
            ],
            OwnedType::Primitive(PrimitiveType::I64),
        )
        .binary(
            BinaryOp::Add,
            "result",
            PrimitiveType::I64,
            &OwnedValue::Variable("a".to_string()),
            &OwnedValue::Variable("b".to_string()),
        )
        .ret(
            &OwnedType::Primitive(PrimitiveType::I64),
            &OwnedValue::Variable("result".to_string()),
        );

        let ir = b.build_ir_text();
        assert!(ir.contains("fn @add(i64 %a, i64 %b) -> i64"), "got: {}", ir);
        assert!(ir.contains("%result = add.i64 %a, %b"), "got: {}", ir);
        assert!(ir.contains("ret.i64 %result"), "got: {}", ir);
    }

    #[test]
    fn test_void_function() {
        let mut b = OwnedIRBuilder::new();
        b.function("main", OwnedType::Void).ret_void();
        let ir = b.build_ir_text();
        assert!(ir.contains("fn @main() -> void"), "got: {}", ir);
        assert!(ir.contains("ret.void"), "got: {}", ir);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(OwnedType::Void.to_string(), "void");
        assert_eq!(OwnedType::Primitive(PrimitiveType::I32).to_string(), "i32");
        assert_eq!(OwnedType::Primitive(PrimitiveType::F64).to_string(), "f64");
        assert_eq!(OwnedType::Named("Foo".to_string()).to_string(), "@Foo");
    }

    #[test]
    fn test_value_display() {
        assert_eq!(OwnedValue::Variable("x".to_string()).to_string(), "%x");
        assert_eq!(OwnedValue::Global("g".to_string()).to_string(), "@g");
        assert_eq!(OwnedValue::I32(42).to_string(), "42");
        assert_eq!(OwnedValue::Bool(true).to_string(), "true");
    }
}
