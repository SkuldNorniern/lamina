//! Encodes the text this backend writes into a WebAssembly module, in process.
//!
//! The grammar accepted is the one `generate_mir_wasm` produces, not all of WAT.
//! Anything outside it is refused by name rather than guessed at, so a module that
//! encodes here is one whose every instruction this writer knows.

use crate::error::LaminaError;
use crate::mir_codegen::CodegenError;
use std::collections::HashMap;

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

#[derive(Debug, Clone, PartialEq)]
enum Sexp {
    Atom(String),
    Str(Vec<u8>),
    List(Vec<Sexp>),
}

impl Sexp {
    fn atom(&self) -> Option<&str> {
        match self {
            Sexp::Atom(text) => Some(text),
            _ => None,
        }
    }

    fn list(&self) -> Option<&[Sexp]> {
        match self {
            Sexp::List(items) => Some(items),
            _ => None,
        }
    }

    /// The keyword a list starts with, for dispatching on its shape.
    fn head(&self) -> Option<&str> {
        self.list()?.first()?.atom()
    }
}

struct Lexer<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Lexer<'a> {
    fn skip_trivia(&mut self) -> Result<(), LaminaError> {
        loop {
            while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
                self.at += 1;
            }
            if self.bytes[self.at..].starts_with(b";;") {
                while self.at < self.bytes.len() && self.bytes[self.at] != b'\n' {
                    self.at += 1;
                }
                continue;
            }
            if self.bytes[self.at..].starts_with(b"(;") {
                let mut depth = 1usize;
                self.at += 2;
                while depth > 0 {
                    if self.at + 1 >= self.bytes.len() {
                        return Err(unsupported("WAT ends inside a block comment"));
                    }
                    if self.bytes[self.at..].starts_with(b"(;") {
                        depth += 1;
                        self.at += 2;
                    } else if self.bytes[self.at..].starts_with(b";)") {
                        depth -= 1;
                        self.at += 2;
                    } else {
                        self.at += 1;
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    fn string(&mut self) -> Result<Vec<u8>, LaminaError> {
        self.at += 1;
        let mut out = Vec::new();
        loop {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| unsupported("WAT ends inside a string"))?;
            self.at += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| unsupported("WAT ends inside a string escape"))?;
                    self.at += 1;
                    match escape {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'"' => out.push(b'"'),
                        b'\'' => out.push(b'\''),
                        b'\\' => out.push(b'\\'),
                        _ => {
                            let low = *self.bytes.get(self.at).ok_or_else(|| {
                                unsupported("WAT ends inside a hex string escape")
                            })?;
                            self.at += 1;
                            let digits = str::from_utf8(&[escape, low])
                                .ok()
                                .and_then(|text| u8::from_str_radix(text, 16).ok())
                                .ok_or_else(|| {
                                    unsupported("a string escape in the WAT is not two hex digits")
                                })?;
                            out.push(digits);
                        }
                    }
                }
                other => out.push(other),
            }
        }
    }

    fn parse(&mut self) -> Result<Sexp, LaminaError> {
        self.skip_trivia()?;
        match self.bytes.get(self.at) {
            None => Err(unsupported("the WAT is empty")),
            Some(b'(') => {
                self.at += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_trivia()?;
                    match self.bytes.get(self.at) {
                        None => return Err(unsupported("WAT ends inside a list")),
                        Some(b')') => {
                            self.at += 1;
                            return Ok(Sexp::List(items));
                        }
                        _ => items.push(self.parse()?),
                    }
                }
            }
            Some(b')') => Err(unsupported("the WAT closes a list that was never opened")),
            Some(b'"') => Ok(Sexp::Str(self.string()?)),
            _ => {
                let start = self.at;
                while let Some(byte) = self.bytes.get(self.at) {
                    if byte.is_ascii_whitespace() || matches!(byte, b'(' | b')' | b'"' | b';') {
                        break;
                    }
                    self.at += 1;
                }
                let text = str::from_utf8(&self.bytes[start..self.at])
                    .map_err(|_| unsupported("the WAT holds a non-UTF-8 token"))?;
                Ok(Sexp::Atom(text.to_string()))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ValType {
    I32,
    I64,
    F32,
    F64,
    V128,
}

impl ValType {
    fn parse(text: &str) -> Result<Self, LaminaError> {
        match text {
            "i32" => Ok(ValType::I32),
            "i64" => Ok(ValType::I64),
            "f32" => Ok(ValType::F32),
            "f64" => Ok(ValType::F64),
            "v128" => Ok(ValType::V128),
            other => Err(unsupported(format!(
                "the WAT names a value type the wasm writer does not encode: {other}"
            ))),
        }
    }

    fn byte(self) -> u8 {
        match self {
            ValType::I32 => 0x7f,
            ValType::I64 => 0x7e,
            ValType::F32 => 0x7d,
            ValType::F64 => 0x7c,
            ValType::V128 => 0x7b,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct FuncType {
    params: Vec<ValType>,
    results: Vec<ValType>,
}

fn uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn sleb(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let sign = byte & 0x40 != 0;
        if (value == 0 && !sign) || (value == -1 && sign) {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn name(out: &mut Vec<u8>, text: &str) {
    uleb(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

/// Every integer the backend writes is decimal, and an i32 slot may hold a value
/// written either signed or as the unsigned bit pattern.
fn integer(text: &str, width: u32) -> Result<i64, LaminaError> {
    let parsed = if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map(|v| v as i64)
    } else if let Some(hex) = text.strip_prefix("-0x") {
        u64::from_str_radix(hex, 16).map(|v| -(v as i64))
    } else {
        text.parse::<i64>()
            .or_else(|_| text.parse::<u64>().map(|v| v as i64))
    };
    let value =
        parsed.map_err(|_| unsupported(format!("the WAT holds a non-numeric literal: {text}")))?;
    if width == 32 {
        return Ok(i64::from(value as i32));
    }
    Ok(value)
}

struct Import {
    module: String,
    field: String,
    ty: FuncType,
}

struct Func {
    name: Option<String>,
    export: Option<String>,
    ty: FuncType,
    locals: Vec<(Option<String>, ValType)>,
    param_names: Vec<Option<String>>,
    body: Vec<Sexp>,
}

#[derive(Default)]
struct Module {
    types: Vec<FuncType>,
    type_names: HashMap<String, u32>,
    imports: Vec<Import>,
    import_names: Vec<Option<String>>,
    funcs: Vec<Func>,
    table: Option<u32>,
    memory: Option<(u64, bool)>,
    memory_export: Option<String>,
    globals: Vec<(Option<String>, ValType, bool, Sexp)>,
    exports: Vec<(String, String)>,
    elem: Vec<String>,
    data: Vec<(Sexp, Vec<u8>)>,
}

impl Module {
    fn intern(&mut self, ty: FuncType) -> u32 {
        if let Some(index) = self.types.iter().position(|known| *known == ty) {
            return index as u32;
        }
        self.types.push(ty);
        (self.types.len() - 1) as u32
    }

    fn func_index(&self, reference: &str) -> Result<u32, LaminaError> {
        if let Some(digits) = reference.strip_prefix('$') {
            let wanted = Some(digits.to_string());
            if let Some(index) = self.import_names.iter().position(|known| *known == wanted) {
                return Ok(index as u32);
            }
            if let Some(index) = self.funcs.iter().position(|func| func.name == wanted) {
                return Ok((self.imports.len() + index) as u32);
            }
            return Err(unsupported(format!(
                "the WAT names a function the wasm writer cannot find: {reference}"
            )));
        }
        integer(reference, 32).map(|value| value as u32)
    }
}

/// Reads `(param ...)`, `(result ...)` and `(local ...)` groups out of a list,
/// returning the index of the first item that is none of them.
fn signature(items: &[Sexp]) -> Result<(FuncType, Vec<Option<String>>, usize), LaminaError> {
    let mut ty = FuncType::default();
    let mut names = Vec::new();
    let mut at = 0;
    while at < items.len() {
        match items[at].head() {
            Some("param") => {
                let fields = items[at].list().unwrap_or_default();
                let mut rest = &fields[1..];
                if let Some(first) = rest.first().and_then(Sexp::atom)
                    && first.starts_with('$')
                {
                    names.push(Some(first[1..].to_string()));
                    rest = &rest[1..];
                    if rest.len() != 1 {
                        return Err(unsupported(
                            "a named (param) in the WAT declares more than one type",
                        ));
                    }
                } else {
                    for _ in rest {
                        names.push(None);
                    }
                }
                for item in rest {
                    let text = item
                        .atom()
                        .ok_or_else(|| unsupported("a (param) in the WAT holds a list"))?;
                    ty.params.push(ValType::parse(text)?);
                }
            }
            Some("result") => {
                for item in &items[at].list().unwrap_or_default()[1..] {
                    let text = item
                        .atom()
                        .ok_or_else(|| unsupported("a (result) in the WAT holds a list"))?;
                    ty.results.push(ValType::parse(text)?);
                }
            }
            _ => break,
        }
        at += 1;
    }
    Ok((ty, names, at))
}

fn collect(fields: &[Sexp]) -> Result<Module, LaminaError> {
    let mut module = Module::default();
    for field in fields {
        let items = field
            .list()
            .ok_or_else(|| unsupported("a top-level item in the WAT is not a list"))?;
        let head = items
            .first()
            .and_then(Sexp::atom)
            .ok_or_else(|| unsupported("a top-level list in the WAT has no keyword"))?;
        match head {
            "type" => {
                let label = items[1]
                    .atom()
                    .and_then(|text| text.strip_prefix('$'))
                    .ok_or_else(|| unsupported("a (type) in the WAT has no name"))?
                    .to_string();
                let body = items[2]
                    .list()
                    .filter(|body| body.first().and_then(Sexp::atom) == Some("func"))
                    .ok_or_else(|| unsupported("a (type) in the WAT does not declare a func"))?;
                let (ty, _, _) = signature(&body[1..])?;
                let index = module.intern(ty);
                module.type_names.insert(label, index);
            }
            "import" => {
                let (Sexp::Str(module_name), Sexp::Str(field_name)) = (&items[1], &items[2]) else {
                    return Err(unsupported("an (import) in the WAT has no two-part name"));
                };
                let described = items[3]
                    .list()
                    .filter(|body| body.first().and_then(Sexp::atom) == Some("func"))
                    .ok_or_else(|| unsupported("the wasm writer encodes only function imports"))?;
                let mut rest = &described[1..];
                let mut label = None;
                if let Some(first) = rest.first().and_then(Sexp::atom)
                    && first.starts_with('$')
                {
                    label = Some(first[1..].to_string());
                    rest = &rest[1..];
                }
                let (ty, _, _) = signature(rest)?;
                module.imports.push(Import {
                    module: String::from_utf8_lossy(module_name).into_owned(),
                    field: String::from_utf8_lossy(field_name).into_owned(),
                    ty,
                });
                module.import_names.push(label);
            }
            "func" => {
                let mut rest = &items[1..];
                let mut label = None;
                let mut export = None;
                if let Some(first) = rest.first().and_then(Sexp::atom)
                    && first.starts_with('$')
                {
                    label = Some(first[1..].to_string());
                    rest = &rest[1..];
                }
                if rest.first().and_then(Sexp::head) == Some("export") {
                    let Some(Sexp::Str(text)) = rest[0].list().and_then(|list| list.get(1)) else {
                        return Err(unsupported("an inline (export) in the WAT has no name"));
                    };
                    export = Some(String::from_utf8_lossy(text).into_owned());
                    rest = &rest[1..];
                }
                let (ty, param_names, mut at) = signature(rest)?;
                let mut locals = Vec::new();
                while rest.get(at).and_then(Sexp::head) == Some("local") {
                    let fields = rest[at].list().unwrap_or_default();
                    let mut group = &fields[1..];
                    let mut local_name = None;
                    if let Some(first) = group.first().and_then(Sexp::atom)
                        && first.starts_with('$')
                    {
                        local_name = Some(first[1..].to_string());
                        group = &group[1..];
                    }
                    for item in group {
                        let text = item
                            .atom()
                            .ok_or_else(|| unsupported("a (local) in the WAT holds a list"))?;
                        locals.push((local_name.take(), ValType::parse(text)?));
                    }
                    at += 1;
                }
                module.funcs.push(Func {
                    name: label,
                    export,
                    ty,
                    locals,
                    param_names,
                    body: rest[at..].to_vec(),
                });
            }
            "table" => {
                let min = items[1]
                    .atom()
                    .ok_or_else(|| unsupported("a (table) in the WAT has no size"))?;
                if items.get(2).and_then(Sexp::atom) != Some("funcref") {
                    return Err(unsupported("the wasm writer encodes only a funcref table"));
                }
                module.table = Some(integer(min, 32)? as u32);
            }
            "memory" => {
                let mut rest = &items[1..];
                if rest.first().and_then(Sexp::head) == Some("export") {
                    let Some(Sexp::Str(text)) = rest[0].list().and_then(|list| list.get(1)) else {
                        return Err(unsupported("an inline (export) in the WAT has no name"));
                    };
                    module.memory_export = Some(String::from_utf8_lossy(text).into_owned());
                    rest = &rest[1..];
                }
                let mut wide = false;
                if rest.first().and_then(Sexp::atom) == Some("i64") {
                    wide = true;
                    rest = &rest[1..];
                }
                let min = rest
                    .first()
                    .and_then(Sexp::atom)
                    .ok_or_else(|| unsupported("a (memory) in the WAT has no size"))?;
                if rest.len() > 1 {
                    return Err(unsupported(
                        "the wasm writer encodes a memory with a minimum size only",
                    ));
                }
                module.memory = Some((integer(min, 64)? as u64, wide));
            }
            "global" => {
                let mut rest = &items[1..];
                let mut label = None;
                if let Some(first) = rest.first().and_then(Sexp::atom)
                    && first.starts_with('$')
                {
                    label = Some(first[1..].to_string());
                    rest = &rest[1..];
                }
                let (ty, mutable) = match &rest[0] {
                    Sexp::Atom(text) => (ValType::parse(text)?, false),
                    Sexp::List(inner) if inner.first().and_then(Sexp::atom) == Some("mut") => (
                        ValType::parse(inner[1].atom().ok_or_else(|| {
                            unsupported("a (mut) in the WAT holds no value type")
                        })?)?,
                        true,
                    ),
                    _ => return Err(unsupported("a (global) in the WAT has no value type")),
                };
                module.globals.push((label, ty, mutable, rest[1].clone()));
            }
            "export" => {
                let Sexp::Str(text) = &items[1] else {
                    return Err(unsupported("an (export) in the WAT has no name"));
                };
                let described = items[2]
                    .list()
                    .filter(|body| body.first().and_then(Sexp::atom) == Some("func"))
                    .ok_or_else(|| unsupported("the wasm writer encodes only function exports"))?;
                let target = described[1]
                    .atom()
                    .ok_or_else(|| unsupported("an (export) in the WAT names no function"))?;
                module.exports.push((
                    String::from_utf8_lossy(text).into_owned(),
                    target.to_string(),
                ));
            }
            "elem" => {
                if items[1].head() != Some("i32.const")
                    || items[1]
                        .list()
                        .and_then(|list| list.get(1))
                        .and_then(Sexp::atom)
                        != Some("0")
                {
                    return Err(unsupported(
                        "the wasm writer encodes only an element segment based at zero",
                    ));
                }
                for item in &items[2..] {
                    let target = item.atom().ok_or_else(|| {
                        unsupported("an (elem) in the WAT names something that is not a function")
                    })?;
                    module.elem.push(target.to_string());
                }
            }
            "data" => {
                let Sexp::Str(bytes) = &items[2] else {
                    return Err(unsupported("a (data) in the WAT carries no byte string"));
                };
                module.data.push((items[1].clone(), bytes.clone()));
            }
            other => {
                return Err(unsupported(format!(
                    "the wasm writer does not encode a top-level ({other})"
                )));
            }
        }
    }
    Ok(module)
}

/// What an instruction takes after its mnemonic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    None,
    I32,
    I64,
    F32,
    F64,
    Local,
    Global,
    Func,
    Label,
    BrTable,
    Mem(u32),
    CallIndirect,
    Lane,
    V128Const,
    MemoryCopy,
    MemoryFill,
}

/// An opcode is one byte, or a prefix byte with an index after it.
#[derive(Clone, Copy)]
enum Op {
    Single(u8),
    Prefixed(u8, u32),
}

impl Op {
    fn write(self, out: &mut Vec<u8>) {
        match self {
            Op::Single(byte) => out.push(byte),
            Op::Prefixed(prefix, index) => {
                out.push(prefix);
                uleb(out, u64::from(index));
            }
        }
    }
}

/// The opcode and immediate shape of each instruction the backend writes. One
/// outside this table is refused by name rather than guessed at.
fn opcode_for(mnemonic: &str) -> Option<(Op, Imm)> {
    use Imm::*;
    const SINGLE: &[(&str, u8, Imm)] = &[
        ("unreachable", 0x00, None),
        ("nop", 0x01, None),
        ("return", 0x0f, None),
        ("br", 0x0c, Label),
        ("br_if", 0x0d, Label),
        ("br_table", 0x0e, BrTable),
        ("call", 0x10, Func),
        ("return_call", 0x12, Func),
        ("call_indirect", 0x11, CallIndirect),
        ("drop", 0x1a, None),
        ("select", 0x1b, None),
        ("local.get", 0x20, Local),
        ("local.set", 0x21, Local),
        ("local.tee", 0x22, Local),
        ("global.get", 0x23, Global),
        ("global.set", 0x24, Global),
        ("i32.load", 0x28, Mem(2)),
        ("i64.load", 0x29, Mem(3)),
        ("f32.load", 0x2a, Mem(2)),
        ("f64.load", 0x2b, Mem(3)),
        ("i32.load8_s", 0x2c, Mem(0)),
        ("i32.load8_u", 0x2d, Mem(0)),
        ("i32.load16_s", 0x2e, Mem(1)),
        ("i32.load16_u", 0x2f, Mem(1)),
        ("i64.load8_s", 0x30, Mem(0)),
        ("i64.load8_u", 0x31, Mem(0)),
        ("i64.load16_s", 0x32, Mem(1)),
        ("i64.load16_u", 0x33, Mem(1)),
        ("i64.load32_s", 0x34, Mem(2)),
        ("i64.load32_u", 0x35, Mem(2)),
        ("i32.store", 0x36, Mem(2)),
        ("i64.store", 0x37, Mem(3)),
        ("f32.store", 0x38, Mem(2)),
        ("f64.store", 0x39, Mem(3)),
        ("i32.store8", 0x3a, Mem(0)),
        ("i32.store16", 0x3b, Mem(1)),
        ("i64.store8", 0x3c, Mem(0)),
        ("i64.store16", 0x3d, Mem(1)),
        ("i64.store32", 0x3e, Mem(2)),
        ("memory.size", 0x3f, None),
        ("memory.grow", 0x40, None),
        ("i32.const", 0x41, I32),
        ("i64.const", 0x42, I64),
        ("f32.const", 0x43, F32),
        ("f64.const", 0x44, F64),
        ("i32.eqz", 0x45, None),
        ("i32.eq", 0x46, None),
        ("i32.ne", 0x47, None),
        ("i32.lt_s", 0x48, None),
        ("i32.lt_u", 0x49, None),
        ("i32.gt_s", 0x4a, None),
        ("i32.gt_u", 0x4b, None),
        ("i32.le_s", 0x4c, None),
        ("i32.le_u", 0x4d, None),
        ("i32.ge_s", 0x4e, None),
        ("i32.ge_u", 0x4f, None),
        ("i64.eqz", 0x50, None),
        ("i64.eq", 0x51, None),
        ("i64.ne", 0x52, None),
        ("i64.lt_s", 0x53, None),
        ("i64.lt_u", 0x54, None),
        ("i64.gt_s", 0x55, None),
        ("i64.gt_u", 0x56, None),
        ("i64.le_s", 0x57, None),
        ("i64.le_u", 0x58, None),
        ("i64.ge_s", 0x59, None),
        ("i64.ge_u", 0x5a, None),
        ("f32.eq", 0x5b, None),
        ("f32.ne", 0x5c, None),
        ("f32.lt", 0x5d, None),
        ("f32.gt", 0x5e, None),
        ("f32.le", 0x5f, None),
        ("f32.ge", 0x60, None),
        ("f64.eq", 0x61, None),
        ("f64.ne", 0x62, None),
        ("f64.lt", 0x63, None),
        ("f64.gt", 0x64, None),
        ("f64.le", 0x65, None),
        ("f64.ge", 0x66, None),
        ("i32.clz", 0x67, None),
        ("i32.ctz", 0x68, None),
        ("i32.popcnt", 0x69, None),
        ("i32.add", 0x6a, None),
        ("i32.sub", 0x6b, None),
        ("i32.mul", 0x6c, None),
        ("i32.div_s", 0x6d, None),
        ("i32.div_u", 0x6e, None),
        ("i32.rem_s", 0x6f, None),
        ("i32.rem_u", 0x70, None),
        ("i32.and", 0x71, None),
        ("i32.or", 0x72, None),
        ("i32.xor", 0x73, None),
        ("i32.shl", 0x74, None),
        ("i32.shr_s", 0x75, None),
        ("i32.shr_u", 0x76, None),
        ("i32.rotl", 0x77, None),
        ("i32.rotr", 0x78, None),
        ("i64.clz", 0x79, None),
        ("i64.ctz", 0x7a, None),
        ("i64.popcnt", 0x7b, None),
        ("i64.add", 0x7c, None),
        ("i64.sub", 0x7d, None),
        ("i64.mul", 0x7e, None),
        ("i64.div_s", 0x7f, None),
        ("i64.div_u", 0x80, None),
        ("i64.rem_s", 0x81, None),
        ("i64.rem_u", 0x82, None),
        ("i64.and", 0x83, None),
        ("i64.or", 0x84, None),
        ("i64.xor", 0x85, None),
        ("i64.shl", 0x86, None),
        ("i64.shr_s", 0x87, None),
        ("i64.shr_u", 0x88, None),
        ("i64.rotl", 0x89, None),
        ("i64.rotr", 0x8a, None),
        ("f32.abs", 0x8b, None),
        ("f32.neg", 0x8c, None),
        ("f32.ceil", 0x8d, None),
        ("f32.floor", 0x8e, None),
        ("f32.trunc", 0x8f, None),
        ("f32.nearest", 0x90, None),
        ("f32.sqrt", 0x91, None),
        ("f32.add", 0x92, None),
        ("f32.sub", 0x93, None),
        ("f32.mul", 0x94, None),
        ("f32.div", 0x95, None),
        ("f32.min", 0x96, None),
        ("f32.max", 0x97, None),
        ("f32.copysign", 0x98, None),
        ("f64.abs", 0x99, None),
        ("f64.neg", 0x9a, None),
        ("f64.ceil", 0x9b, None),
        ("f64.floor", 0x9c, None),
        ("f64.trunc", 0x9d, None),
        ("f64.nearest", 0x9e, None),
        ("f64.sqrt", 0x9f, None),
        ("f64.add", 0xa0, None),
        ("f64.sub", 0xa1, None),
        ("f64.mul", 0xa2, None),
        ("f64.div", 0xa3, None),
        ("f64.min", 0xa4, None),
        ("f64.max", 0xa5, None),
        ("f64.copysign", 0xa6, None),
        ("i32.wrap_i64", 0xa7, None),
        ("i32.trunc_f32_s", 0xa8, None),
        ("i32.trunc_f32_u", 0xa9, None),
        ("i32.trunc_f64_s", 0xaa, None),
        ("i32.trunc_f64_u", 0xab, None),
        ("i64.extend_i32_s", 0xac, None),
        ("i64.extend_i32_u", 0xad, None),
        ("i64.trunc_f32_s", 0xae, None),
        ("i64.trunc_f32_u", 0xaf, None),
        ("i64.trunc_f64_s", 0xb0, None),
        ("i64.trunc_f64_u", 0xb1, None),
        ("f32.convert_i32_s", 0xb2, None),
        ("f32.convert_i32_u", 0xb3, None),
        ("f32.convert_i64_s", 0xb4, None),
        ("f32.convert_i64_u", 0xb5, None),
        ("f32.demote_f64", 0xb6, None),
        ("f64.convert_i32_s", 0xb7, None),
        ("f64.convert_i32_u", 0xb8, None),
        ("f64.convert_i64_s", 0xb9, None),
        ("f64.convert_i64_u", 0xba, None),
        ("f64.promote_f32", 0xbb, None),
        ("i32.reinterpret_f32", 0xbc, None),
        ("i64.reinterpret_f64", 0xbd, None),
        ("f32.reinterpret_i32", 0xbe, None),
        ("f64.reinterpret_i64", 0xbf, None),
        ("i32.extend8_s", 0xc0, None),
        ("i32.extend16_s", 0xc1, None),
        ("i64.extend8_s", 0xc2, None),
        ("i64.extend16_s", 0xc3, None),
        ("i64.extend32_s", 0xc4, None),
    ];
    const SATURATING: &[(&str, u32, Imm)] = &[
        ("i32.trunc_sat_f32_s", 0, None),
        ("i32.trunc_sat_f32_u", 1, None),
        ("i32.trunc_sat_f64_s", 2, None),
        ("i32.trunc_sat_f64_u", 3, None),
        ("i64.trunc_sat_f32_s", 4, None),
        ("i64.trunc_sat_f32_u", 5, None),
        ("i64.trunc_sat_f64_s", 6, None),
        ("i64.trunc_sat_f64_u", 7, None),
        ("memory.copy", 10, MemoryCopy),
        ("memory.fill", 11, MemoryFill),
    ];
    const VECTOR: &[(&str, u32, Imm)] = &[
        ("v128.load", 0, Mem(4)),
        ("v128.store", 11, Mem(4)),
        ("v128.const", 12, V128Const),
        ("v128.and", 78, None),
        ("v128.or", 80, None),
        ("v128.xor", 81, None),
        ("i8x16.splat", 15, None),
        ("i8x16.extract_lane_s", 21, Lane),
        ("i8x16.extract_lane_u", 22, Lane),
        ("i8x16.replace_lane", 23, Lane),
        ("i8x16.shl", 107, None),
        ("i8x16.shr_s", 108, None),
        ("i8x16.shr_u", 109, None),
        ("i8x16.add", 110, None),
        ("i8x16.sub", 113, None),
        ("i16x8.splat", 16, None),
        ("i16x8.extract_lane_s", 24, Lane),
        ("i16x8.extract_lane_u", 25, Lane),
        ("i16x8.replace_lane", 26, Lane),
        ("i16x8.shl", 139, None),
        ("i16x8.shr_s", 140, None),
        ("i16x8.shr_u", 141, None),
        ("i16x8.add", 142, None),
        ("i16x8.sub", 145, None),
        ("i16x8.mul", 149, None),
        ("i32x4.splat", 17, None),
        ("i32x4.extract_lane", 27, Lane),
        ("i32x4.replace_lane", 28, Lane),
        ("i32x4.shl", 171, None),
        ("i32x4.shr_s", 172, None),
        ("i32x4.shr_u", 173, None),
        ("i32x4.add", 174, None),
        ("i32x4.sub", 177, None),
        ("i32x4.mul", 181, None),
        ("i64x2.splat", 18, None),
        ("i64x2.extract_lane", 29, Lane),
        ("i64x2.replace_lane", 30, Lane),
        ("i64x2.shl", 203, None),
        ("i64x2.shr_s", 204, None),
        ("i64x2.shr_u", 205, None),
        ("i64x2.add", 206, None),
        ("i64x2.sub", 209, None),
        ("i64x2.mul", 213, None),
        ("f32x4.splat", 19, None),
        ("f32x4.extract_lane", 31, Lane),
        ("f32x4.replace_lane", 32, Lane),
        ("f32x4.add", 228, None),
        ("f32x4.sub", 229, None),
        ("f32x4.mul", 230, None),
        ("f64x2.splat", 20, None),
        ("f64x2.extract_lane", 33, Lane),
        ("f64x2.replace_lane", 34, Lane),
        ("f64x2.add", 240, None),
        ("f64x2.sub", 241, None),
        ("f64x2.mul", 242, None),
    ];
    for (text, byte, imm) in SINGLE {
        if *text == mnemonic {
            return Some((Op::Single(*byte), *imm));
        }
    }
    for (text, index, imm) in SATURATING {
        if *text == mnemonic {
            return Some((Op::Prefixed(0xfc, *index), *imm));
        }
    }
    for (text, index, imm) in VECTOR {
        if *text == mnemonic {
            return Some((Op::Prefixed(0xfd, *index), *imm));
        }
    }
    Option::None
}

/// Resolves a label reference to the branch depth the binary form wants.
fn label_depth(stack: &[Option<String>], reference: &str) -> Result<u32, LaminaError> {
    if let Some(wanted) = reference.strip_prefix('$') {
        for (position, label) in stack.iter().enumerate().rev() {
            if label.as_deref() == Some(wanted) {
                return Ok((stack.len() - 1 - position) as u32);
            }
        }
        return Err(unsupported(format!(
            "the WAT branches to a label that is not open: {reference}"
        )));
    }
    Ok(integer(reference, 32)? as u32)
}

struct Body<'a> {
    module: &'a Module,
    locals: HashMap<String, u32>,
    globals: HashMap<String, u32>,
    labels: Vec<Option<String>>,
    out: Vec<u8>,
}

impl Body<'_> {
    fn local_index(&self, reference: &str) -> Result<u32, LaminaError> {
        if let Some(wanted) = reference.strip_prefix('$') {
            return self.locals.get(wanted).copied().ok_or_else(|| {
                unsupported(format!(
                    "the WAT reads a local the wasm writer cannot find: {reference}"
                ))
            });
        }
        Ok(integer(reference, 32)? as u32)
    }

    fn global_index(&self, reference: &str) -> Result<u32, LaminaError> {
        if let Some(wanted) = reference.strip_prefix('$') {
            return self.globals.get(wanted).copied().ok_or_else(|| {
                unsupported(format!(
                    "the WAT reads a global the wasm writer cannot find: {reference}"
                ))
            });
        }
        Ok(integer(reference, 32)? as u32)
    }

    /// Encodes a block type, which the backend writes as nothing or one result.
    fn block_type(&self, shape: Option<&Sexp>) -> Result<Vec<u8>, LaminaError> {
        let Some(shape) = shape else {
            return Ok(vec![0x40]);
        };
        let items = shape.list().unwrap_or_default();
        if items.first().and_then(Sexp::atom) != Some("result") || items.len() != 2 {
            return Err(unsupported(
                "the wasm writer encodes a block that yields nothing or one value",
            ));
        }
        let text = items[1]
            .atom()
            .ok_or_else(|| unsupported("a block (result) in the WAT holds a list"))?;
        Ok(vec![ValType::parse(text)?.byte()])
    }

    /// Writes one instruction's opcode and immediates, consuming the atoms that
    /// follow it. Returns how many of those atoms it took.
    fn plain(&mut self, mnemonic: &str, rest: &[Sexp]) -> Result<usize, LaminaError> {
        let (op, imm) = opcode_for(mnemonic).ok_or_else(|| {
            unsupported(format!(
                "the wasm writer does not encode the instruction {mnemonic}"
            ))
        })?;
        op.write(&mut self.out);
        let next = |at: usize| -> Result<&str, LaminaError> {
            rest.get(at).and_then(Sexp::atom).ok_or_else(|| {
                unsupported(format!("{mnemonic} in the WAT is missing an immediate"))
            })
        };
        match imm {
            Imm::None => Ok(0),
            Imm::I32 => {
                sleb(&mut self.out, integer(next(0)?, 32)?);
                Ok(1)
            }
            Imm::I64 => {
                sleb(&mut self.out, integer(next(0)?, 64)?);
                Ok(1)
            }
            Imm::F32 => {
                let value: f32 = next(0)?.parse().map_err(|_| {
                    unsupported(format!(
                        "{mnemonic} holds a float the wasm writer cannot read: {}",
                        next(0).unwrap_or("")
                    ))
                })?;
                self.out.extend_from_slice(&value.to_le_bytes());
                Ok(1)
            }
            Imm::F64 => {
                let value: f64 = next(0)?.parse().map_err(|_| {
                    unsupported(format!(
                        "{mnemonic} holds a float the wasm writer cannot read: {}",
                        next(0).unwrap_or("")
                    ))
                })?;
                self.out.extend_from_slice(&value.to_le_bytes());
                Ok(1)
            }
            Imm::Local => {
                let index = self.local_index(next(0)?)?;
                uleb(&mut self.out, u64::from(index));
                Ok(1)
            }
            Imm::Global => {
                let index = self.global_index(next(0)?)?;
                uleb(&mut self.out, u64::from(index));
                Ok(1)
            }
            Imm::Func => {
                let index = self.module.func_index(next(0)?)?;
                uleb(&mut self.out, u64::from(index));
                Ok(1)
            }
            Imm::Label => {
                let depth = label_depth(&self.labels, next(0)?)?;
                uleb(&mut self.out, u64::from(depth));
                Ok(1)
            }
            Imm::BrTable => {
                let mut targets = Vec::new();
                let mut at = 0;
                while let Some(text) = rest.get(at).and_then(Sexp::atom) {
                    if !(text.starts_with('$') || text.bytes().all(|b| b.is_ascii_digit())) {
                        break;
                    }
                    targets.push(label_depth(&self.labels, text)?);
                    at += 1;
                }
                let Some((default, table)) = targets.split_last() else {
                    return Err(unsupported("br_table in the WAT names no targets"));
                };
                uleb(&mut self.out, table.len() as u64);
                for depth in table {
                    uleb(&mut self.out, u64::from(*depth));
                }
                uleb(&mut self.out, u64::from(*default));
                Ok(at)
            }
            Imm::Mem(natural) => {
                let mut align = natural;
                let mut offset = 0u64;
                let mut at = 0;
                while let Some(text) = rest.get(at).and_then(Sexp::atom) {
                    if let Some(value) = text.strip_prefix("align=") {
                        align = (integer(value, 32)? as u32).trailing_zeros();
                    } else if let Some(value) = text.strip_prefix("offset=") {
                        offset = integer(value, 64)? as u64;
                    } else {
                        break;
                    }
                    at += 1;
                }
                uleb(&mut self.out, u64::from(align));
                uleb(&mut self.out, offset);
                Ok(at)
            }
            Imm::CallIndirect => {
                let shape = rest
                    .first()
                    .filter(|item| item.head() == Some("type"))
                    .and_then(Sexp::list)
                    .ok_or_else(|| unsupported("call_indirect in the WAT declares no type"))?;
                let reference = shape[1]
                    .atom()
                    .ok_or_else(|| unsupported("call_indirect in the WAT names no type"))?;
                let index = match reference.strip_prefix('$') {
                    Some(label) => *self.module.type_names.get(label).ok_or_else(|| {
                        unsupported(format!(
                            "call_indirect names a type the wasm writer cannot find: {reference}"
                        ))
                    })?,
                    Option::None => integer(reference, 32)? as u32,
                };
                uleb(&mut self.out, u64::from(index));
                self.out.push(0x00);
                Ok(1)
            }
            Imm::Lane => {
                let lane = integer(next(0)?, 32)?;
                self.out.push(lane as u8);
                Ok(1)
            }
            Imm::V128Const => {
                if next(0)? != "i32x4" {
                    return Err(unsupported(format!(
                        "the wasm writer encodes only an i32x4 v128.const, not {}",
                        next(0)?
                    )));
                }
                for lane in 0..4 {
                    let value = integer(next(1 + lane)?, 32)? as i32;
                    self.out.extend_from_slice(&value.to_le_bytes());
                }
                Ok(5)
            }
            Imm::MemoryCopy => {
                self.out.push(0x00);
                self.out.push(0x00);
                Ok(0)
            }
            Imm::MemoryFill => {
                self.out.push(0x00);
                Ok(0)
            }
        }
    }

    fn instructions(&mut self, items: &[Sexp]) -> Result<(), LaminaError> {
        let mut at = 0;
        while at < items.len() {
            match &items[at] {
                Sexp::Str(_) => {
                    return Err(unsupported(
                        "a string appears where the WAT expects an instruction",
                    ));
                }
                Sexp::Atom(mnemonic) => match mnemonic.as_str() {
                    "block" | "loop" | "if" => {
                        let mut cursor = at + 1;
                        let mut label = Option::None;
                        if let Some(text) = items.get(cursor).and_then(Sexp::atom)
                            && text.starts_with('$')
                        {
                            label = Some(text[1..].to_string());
                            cursor += 1;
                        }
                        let shape = items
                            .get(cursor)
                            .filter(|item| matches!(item.head(), Some("result") | Some("param")));
                        if shape.is_some() {
                            cursor += 1;
                        }
                        let header = self.block_type(shape)?;
                        self.out.push(match mnemonic.as_str() {
                            "block" => 0x02,
                            "loop" => 0x03,
                            _ => 0x04,
                        });
                        self.out.extend_from_slice(&header);
                        self.labels.push(label);
                        at = cursor;
                    }
                    "else" => {
                        self.out.push(0x05);
                        at += 1;
                    }
                    "end" => {
                        self.out.push(0x0b);
                        self.labels.pop().ok_or_else(|| {
                            unsupported("the WAT closes a block that was never opened")
                        })?;
                        at += 1;
                    }
                    _ => {
                        let taken = self.plain(mnemonic, &items[at + 1..])?;
                        at += 1 + taken;
                    }
                },
                Sexp::List(folded) => {
                    self.folded(folded)?;
                    at += 1;
                }
            }
        }
        Ok(())
    }

    /// A folded instruction puts its operands inside its own parentheses, so the
    /// operand expressions are written first and the instruction itself last.
    fn folded(&mut self, items: &[Sexp]) -> Result<(), LaminaError> {
        let mnemonic = items
            .first()
            .and_then(Sexp::atom)
            .ok_or_else(|| unsupported("a folded instruction in the WAT has no mnemonic"))?;
        match mnemonic {
            "block" | "loop" => {
                let mut cursor = 1;
                let mut label = None;
                if let Some(text) = items.get(cursor).and_then(Sexp::atom)
                    && text.starts_with('$')
                {
                    label = Some(text[1..].to_string());
                    cursor += 1;
                }
                let shape = items
                    .get(cursor)
                    .filter(|item| matches!(item.head(), Some("result") | Some("param")));
                if shape.is_some() {
                    cursor += 1;
                }
                let header = self.block_type(shape)?;
                self.out.push(if mnemonic == "block" { 0x02 } else { 0x03 });
                self.out.extend_from_slice(&header);
                self.labels.push(label);
                self.instructions(&items[cursor..])?;
                self.labels.pop();
                self.out.push(0x0b);
                Ok(())
            }
            "if" => {
                let mut cursor = 1;
                let mut label = None;
                if let Some(text) = items.get(cursor).and_then(Sexp::atom)
                    && text.starts_with('$')
                {
                    label = Some(text[1..].to_string());
                    cursor += 1;
                }
                let shape = items
                    .get(cursor)
                    .filter(|item| matches!(item.head(), Some("result") | Some("param")));
                if shape.is_some() {
                    cursor += 1;
                }
                let header = self.block_type(shape)?;
                let then_at = items[cursor..]
                    .iter()
                    .position(|item| item.head() == Some("then"))
                    .map(|offset| cursor + offset)
                    .ok_or_else(|| unsupported("a folded (if) in the WAT has no (then)"))?;
                self.instructions(&items[cursor..then_at])?;
                self.out.push(0x04);
                self.out.extend_from_slice(&header);
                self.labels.push(label);
                self.instructions(&items[then_at].list().unwrap_or_default()[1..])?;
                if let Some(otherwise) = items.get(then_at + 1) {
                    if otherwise.head() != Some("else") {
                        return Err(unsupported(
                            "a folded (if) in the WAT is followed by something other than (else)",
                        ));
                    }
                    self.out.push(0x05);
                    self.instructions(&otherwise.list().unwrap_or_default()[1..])?;
                }
                self.labels.pop();
                self.out.push(0x0b);
                Ok(())
            }
            _ => {
                let split = items[1..]
                    .iter()
                    .position(|item| item.list().is_some())
                    .map_or(items.len(), |offset| 1 + offset);
                for operand in &items[split..] {
                    let nested = operand
                        .list()
                        .ok_or_else(|| unsupported("a folded operand in the WAT is not a list"))?;
                    self.folded(nested)?;
                }
                self.plain(mnemonic, &items[1..split])?;
                Ok(())
            }
        }
    }
}

fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    if payload.is_empty() {
        return;
    }
    out.push(id);
    uleb(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// A constant initializer, which the backend writes only as a single `i32.const`
/// or `i64.const`.
fn const_expr(expr: &Sexp) -> Result<Vec<u8>, LaminaError> {
    let items = expr
        .list()
        .ok_or_else(|| unsupported("an initializer in the WAT is not an expression"))?;
    let mnemonic = items
        .first()
        .and_then(Sexp::atom)
        .ok_or_else(|| unsupported("an initializer in the WAT has no instruction"))?;
    let literal = items
        .get(1)
        .and_then(Sexp::atom)
        .ok_or_else(|| unsupported("an initializer in the WAT has no value"))?;
    let mut out = Vec::new();
    match mnemonic {
        "i32.const" => {
            out.push(0x41);
            sleb(&mut out, integer(literal, 32)?);
        }
        "i64.const" => {
            out.push(0x42);
            sleb(&mut out, integer(literal, 64)?);
        }
        other => {
            return Err(unsupported(format!(
                "the wasm writer encodes a constant initializer, not ({other})"
            )));
        }
    }
    out.push(0x0b);
    Ok(out)
}

/// Encodes the WAT this backend writes into a wasm module.
pub fn encode_module(text: &str) -> Result<Vec<u8>, LaminaError> {
    let mut lexer = Lexer {
        bytes: text.as_bytes(),
        at: 0,
    };
    let root = lexer.parse()?;
    lexer.skip_trivia()?;
    if lexer.at != lexer.bytes.len() {
        return Err(unsupported("the WAT holds more than one top-level form"));
    }
    let fields = root
        .list()
        .filter(|items| items.first().and_then(Sexp::atom) == Some("module"))
        .ok_or_else(|| unsupported("the WAT is not a (module)"))?;
    let mut module = collect(&fields[1..])?;

    let import_types: Vec<u32> = (0..module.imports.len())
        .map(|index| {
            let ty = module.imports[index].ty.clone();
            module.intern(ty)
        })
        .collect();
    let func_types: Vec<u32> = (0..module.funcs.len())
        .map(|index| {
            let ty = module.funcs[index].ty.clone();
            module.intern(ty)
        })
        .collect();

    let mut globals = HashMap::new();
    for (index, (label, ..)) in module.globals.iter().enumerate() {
        if let Some(label) = label {
            globals.insert(label.clone(), index as u32);
        }
    }

    let mut out = Vec::from(b"\0asm\x01\0\0\0");

    let mut payload = Vec::new();
    uleb(&mut payload, module.types.len() as u64);
    for ty in &module.types {
        payload.push(0x60);
        uleb(&mut payload, ty.params.len() as u64);
        payload.extend(ty.params.iter().map(|p| p.byte()));
        uleb(&mut payload, ty.results.len() as u64);
        payload.extend(ty.results.iter().map(|r| r.byte()));
    }
    section(&mut out, 1, &payload);

    let mut payload = Vec::new();
    uleb(&mut payload, module.imports.len() as u64);
    for (index, import) in module.imports.iter().enumerate() {
        name(&mut payload, &import.module);
        name(&mut payload, &import.field);
        payload.push(0x00);
        uleb(&mut payload, u64::from(import_types[index]));
    }
    if !module.imports.is_empty() {
        section(&mut out, 2, &payload);
    }

    let mut payload = Vec::new();
    uleb(&mut payload, module.funcs.len() as u64);
    for index in &func_types {
        uleb(&mut payload, u64::from(*index));
    }
    if !module.funcs.is_empty() {
        section(&mut out, 3, &payload);
    }

    if let Some(min) = module.table {
        let mut payload = Vec::new();
        uleb(&mut payload, 1);
        payload.push(0x70);
        payload.push(0x00);
        uleb(&mut payload, u64::from(min));
        section(&mut out, 4, &payload);
    }

    if let Some((min, wide)) = module.memory {
        let mut payload = Vec::new();
        uleb(&mut payload, 1);
        payload.push(if wide { 0x04 } else { 0x00 });
        uleb(&mut payload, min);
        section(&mut out, 5, &payload);
    }

    let mut payload = Vec::new();
    uleb(&mut payload, module.globals.len() as u64);
    for (_, ty, mutable, init) in &module.globals {
        payload.push(ty.byte());
        payload.push(u8::from(*mutable));
        payload.extend_from_slice(&const_expr(init)?);
    }
    if !module.globals.is_empty() {
        section(&mut out, 6, &payload);
    }

    let mut entries: Vec<(String, u8, u32)> = Vec::new();
    if let Some(label) = &module.memory_export {
        entries.push((label.clone(), 0x02, 0));
    }
    for (index, func) in module.funcs.iter().enumerate() {
        if let Some(label) = &func.export {
            entries.push((label.clone(), 0x00, (module.imports.len() + index) as u32));
        }
    }
    for (label, target) in &module.exports {
        entries.push((label.clone(), 0x00, module.func_index(target)?));
    }
    let mut payload = Vec::new();
    uleb(&mut payload, entries.len() as u64);
    for (label, kind, index) in &entries {
        name(&mut payload, label);
        payload.push(*kind);
        uleb(&mut payload, u64::from(*index));
    }
    if !entries.is_empty() {
        section(&mut out, 7, &payload);
    }

    if !module.elem.is_empty() {
        let mut payload = Vec::new();
        uleb(&mut payload, 1);
        uleb(&mut payload, 0);
        payload.push(0x41);
        sleb(&mut payload, 0);
        payload.push(0x0b);
        uleb(&mut payload, module.elem.len() as u64);
        for target in &module.elem {
            uleb(&mut payload, u64::from(module.func_index(target)?));
        }
        section(&mut out, 9, &payload);
    }

    let mut payload = Vec::new();
    uleb(&mut payload, module.funcs.len() as u64);
    for func in &module.funcs {
        let mut locals = HashMap::new();
        for (index, label) in func.param_names.iter().enumerate() {
            if let Some(label) = label {
                locals.insert(label.clone(), index as u32);
            }
        }
        for (index, (label, _)) in func.locals.iter().enumerate() {
            if let Some(label) = label {
                locals.insert(label.clone(), (func.ty.params.len() + index) as u32);
            }
        }
        let mut body = Body {
            module: &module,
            locals,
            globals: globals.clone(),
            labels: Vec::new(),
            out: Vec::new(),
        };
        body.instructions(&func.body)?;
        if !body.labels.is_empty() {
            return Err(unsupported(format!(
                "a function in the WAT leaves {} blocks open",
                body.labels.len()
            )));
        }
        let code = body.out;

        let mut groups: Vec<(u32, ValType)> = Vec::new();
        for (_, ty) in &func.locals {
            match groups.last_mut() {
                Some((count, known)) if known == ty => *count += 1,
                _ => groups.push((1, *ty)),
            }
        }
        let mut entry = Vec::new();
        uleb(&mut entry, groups.len() as u64);
        for (count, ty) in &groups {
            uleb(&mut entry, u64::from(*count));
            entry.push(ty.byte());
        }
        entry.extend_from_slice(&code);
        entry.push(0x0b);
        uleb(&mut payload, entry.len() as u64);
        payload.extend_from_slice(&entry);
    }
    if !module.funcs.is_empty() {
        section(&mut out, 10, &payload);
    }

    let mut payload = Vec::new();
    uleb(&mut payload, module.data.len() as u64);
    for (offset, bytes) in &module.data {
        uleb(&mut payload, 0);
        payload.extend_from_slice(&const_expr(offset)?);
        uleb(&mut payload, bytes.len() as u64);
        payload.extend_from_slice(bytes);
    }
    if !module.data.is_empty() {
        section(&mut out, 11, &payload);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_module_starts_with_the_wasm_magic_and_version() {
        let encoded = encode_module("(module (func (export \"main\") (result i64) i64.const 7))")
            .expect("the writer should encode");
        assert_eq!(&encoded[..8], b"\0asm\x01\0\0\0");
    }

    #[test]
    fn an_instruction_outside_the_table_is_refused_by_name() {
        let error = encode_module("(module (func $f i32.const 0 i32.atomic.load drop))")
            .expect_err("an unknown instruction must be refused");
        assert!(error.to_string().contains("i32.atomic.load"), "{error}");
    }

    #[test]
    fn a_shared_memory_is_refused_rather_than_written_unshared() {
        let error = encode_module("(module (memory 1 1 shared))")
            .expect_err("a shared memory must be refused");
        assert!(error.to_string().contains("minimum size only"), "{error}");
    }

    #[test]
    fn a_branch_to_a_label_that_is_not_open_is_refused() {
        let error = encode_module("(module (func $f (block $a) br $a))")
            .expect_err("a dangling branch must be refused");
        assert!(error.to_string().contains("$a"), "{error}");
    }
}
