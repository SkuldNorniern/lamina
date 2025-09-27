use std::{
    borrow::Cow, fmt::{Debug, Display}, ops::{Deref, DerefMut}
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericType {
    I32,
    I64,
    F32,
    F64,
}

impl Display for NumericType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntegerType {
    I32,
    I64,
}

impl TryFrom<NumericType> for IntegerType {
    type Error = &'static str;
    fn try_from(value: NumericType) -> Result<Self, Self::Error> {
        match value {
            NumericType::I32 => Ok(Self::I32),
            NumericType::I64 => Ok(Self::I64),
            _ => Err("invalid numeric type for integer type")
        }
    }
}

impl Display for IntegerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatType {
    F32,
    F64,
}

impl Display for FloatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericConstant {
    I32(u32),
    I64(u64),
    F32(f32),
    F64(f64),
}

impl NumericConstant {
    pub fn is_zero(self) -> bool {
        match self {
            Self::I32(v) => v == 0,
            Self::I64(v) => v == 0,
            Self::F32(v) => v == 0.0,
            Self::F64(v) => v == 0.0,
        }
    }
}

impl Display for NumericConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::I32(v) => format_args!("i32.const {}", *v),
            Self::I64(v) => format_args!("i64.const {}", *v),
            Self::F32(v) => format_args!("f32.const {}", *v),
            Self::F64(v) => format_args!("f64.const {}", *v),
        };
        f.write_fmt(v)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Identifier<'a> {
    Index(u64),
    Name(Cow<'a, str>),
}

impl Display for Identifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Index(v) => format_args!("{}", *v),
            Self::Name(v) => format_args!("${}", *v),
        };
        f.write_fmt(v)
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq)]
pub enum WasmInstruction<'a> {
    /// Comments
    Comment(String),
    /// Returns a static constant.
    Const(NumericConstant),

    Clz(NumericType),
    Ctz(NumericType),
    PopCnt(NumericType),

    Add(NumericType),
    Sub(NumericType),
    Mul(NumericType),

    DivF(FloatType),
    DivS(IntegerType),
    DivU(IntegerType),

    RemS(IntegerType),
    RemU(IntegerType),

    And(IntegerType),
    Or(IntegerType),
    Xor(IntegerType),
    Shl(IntegerType),

    ShrS(IntegerType),
    ShrU(IntegerType),

    RotL(IntegerType),
    RotR(IntegerType),

    Abs(FloatType),
    Neg(FloatType),
    Sqrt(FloatType),
    Ceil(FloatType),
    Floor(FloatType),
    Trunc(FloatType),
    Nearest(FloatType),

    Min(FloatType),
    Max(FloatType),
    CopySign(FloatType),

    Eqz(IntegerType),
    Eq(NumericType),
    Ne(NumericType),

    LtF(FloatType),
    LtU(IntegerType),
    LtS(IntegerType),

    GtF(FloatType),
    GtU(IntegerType),
    GtS(IntegerType),

    LeF(FloatType),
    LeU(IntegerType),
    LeS(IntegerType),

    GeF(FloatType),
    GeU(IntegerType),
    GeS(IntegerType),

    Extend8S(IntegerType),
    Extend16S(IntegerType),

    I32_WrapI64,

    I64_ExtendI32U,
    I64_ExtendI32S,

    TruncS(IntegerType, FloatType),
    TruncU(IntegerType, FloatType),

    /// Convert a f32 into a f64.
    F32_Demote_F64,
    /// Convert a f64 into a f32.
    F64_Promote_F32,

    ConvertU(FloatType, IntegerType),
    ConvertS(FloatType, IntegerType),

    /// Provided type is what to go FROM. Converts to a float or an integer depending on the type.
    Reinterpret(NumericType),

    /// The drop instruction simply throws away a single operand.
    Drop,
    /// The select instruction selects one of its first two operands based on whether its third operand is zero or not.
    Select,

    LocalGet(Identifier<'a>),
    LocalSet(Identifier<'a>),
    /// Peek at the top argument, set the local to it, and keep it on the stack.
    LocalTee(Identifier<'a>),

    GlobalGet(Identifier<'a>),
    GlobalSet(Identifier<'a>),

    Load(NumericType, Option<Identifier<'a>>),
    Store(NumericType, Option<Identifier<'a>>),

    Load8S(IntegerType, Option<Identifier<'a>>),
    Load8U(IntegerType, Option<Identifier<'a>>),
    Load16S(IntegerType, Option<Identifier<'a>>),
    Load16U(IntegerType, Option<Identifier<'a>>),
    I64_Load32S(Option<Identifier<'a>>),
    I64_Load32U(Option<Identifier<'a>>),

    Store8(IntegerType, Option<Identifier<'a>>),
    Store16(IntegerType, Option<Identifier<'a>>),
    I64_Store32(Option<Identifier<'a>>),

    /// In pages.
    MemorySize,

    /// Grow the memory by the specified number of pages.
    MemoryGrow,

    /// Copy some bytes of memory.
    MemoryCopy,

    Nop,
    Unreachable,

    Block(Option<String>, Instructions<'a>),
    If {
        identifier: Option<&'a str>,
        result: Option<Vec<NumericType>>,
        then: Instructions<'a>,
        r#else: Option<Instructions<'a>>,
    },
    Loop {
        identifier: Option<&'a str>,
        contents: Instructions<'a>,
    },
    /// Inside a block or if statement: branches out of it
    ///
    /// Inside a loop: Continues to the next iteration
    Br(Identifier<'a>),
    /// See [`WasmInstruction::Br`]. Checks a single argument and only branches if true.
    BrIf(Identifier<'a>),
    /// See [`WasmInstruction::Br`]. Each identifier triggers if the top argument is it's index.
    BrTable(Vec<Identifier<'a>>),

    Return,

    Call(Identifier<'a>),
    /// Each identifier triggers if the top argument is it's index.
    CallIndirect {
        parameters: Vec<NumericType>,
        result: Option<Vec<NumericType>>,
        table: Option<u32>,
    },

    /// Equivalent to a [`WasmInstruction::Call`] followed by a [`WasmInstruction::Return`].
    ReturnCall(Identifier<'a>),
    /// Equivalent to a [`WasmInstruction::CallIndirect`] followed by a [`WasmInstruction::Return`].
    /// Each identifier triggers if the top argument is it's index.
    ReturnCallIndirect {
        parameters: Vec<NumericType>,
        result: Option<Vec<NumericType>>,
        table: Option<Identifier<'a>>,
    },
}

impl Display for WasmInstruction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            Self::Comment(data) => format!("(; {data} ;)"),
            Self::Abs(ty) => format!("{ty}.abs"),
            Self::Add(ty) => format!("{ty}.add"),
            Self::And(ty) => format!("{ty}.and"),
            Self::Ceil(ty) => format!("{ty}.ceil"),
            Self::Block(name, instructions) => format!(
                "(block {}\n{instructions})",
                if let Some(name) = name {
                    format!("${name} ")
                } else {
                    String::new()
                }
            ),
            Self::Br(id) => format!("br {id}"),
            Self::BrIf(id) => format!("br_if {id}"),
            Self::BrTable(ids) => format!(
                "(br_table {})",
                ids.iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Self::Call(id) => format!("call {id}"),
            Self::CallIndirect {
                parameters,
                result,
                table,
            } => format!(
                "(call_indirect (func {}{}){})",
                parameters
                    .iter()
                    .map(|v| format!("(param {v})"))
                    .collect::<Vec<_>>()
                    .join(" "),
                if let Some(result) = result {
                    format!(
                        " (result {})",
                        result
                            .iter()
                            .map(|v| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                } else {
                    String::new()
                },
                if let Some(table) = table {
                    format!(" (table {table})")
                } else {
                    String::new()
                }
            ),
            Self::Clz(ty) => format!("{ty}.clz"),
            Self::Const(con) => format!("{con}"),
            Self::ConvertS(ty1, ty2) => format!("{ty1}.convert_{ty2}_s"),
            Self::ConvertU(ty1, ty2) => format!("{ty1}.convert_{ty2}_u"),
            Self::CopySign(ty) => format!("{ty}.copysign"),
            Self::Ctz(ty) => format!("{ty}.ctz"),
            Self::DivF(ty) => format!("{ty}.div"),
            Self::DivS(ty) => format!("{ty}.div_s"),
            Self::DivU(ty) => format!("{ty}.div_u"),
            Self::Drop => format!("drop"),
            Self::Eq(ty) => format!("{ty}.eq"),
            Self::Eqz(ty) => format!("{ty}.eqz"),
            Self::Extend8S(ty) => format!("{ty}.extend8_s"),
            Self::Extend16S(ty) => format!("{ty}.extend16_s"),
            Self::F32_Demote_F64 => format!("f32.demote_f64"),
            Self::F64_Promote_F32 => format!("f64.promote_f32"),
            Self::Reinterpret(ty) => format!(
                "{ty}.reinterpret_{}",
                match ty {
                    NumericType::I32 => "f32",
                    NumericType::F32 => "i32",
                    NumericType::I64 => "f64",
                    NumericType::F64 => "i64",
                }
            ),
            Self::Floor(ty) => format!("{ty}.floor"),

            Self::GeF(ty) => format!("{ty}.ge"),
            Self::GeS(ty) => format!("{ty}.ge_s"),
            Self::GeU(ty) => format!("{ty}.ge_u"),
            Self::GtF(ty) => format!("{ty}.gt"),
            Self::GtS(ty) => format!("{ty}.gt_s"),
            Self::GtU(ty) => format!("{ty}.gt_u"),
            Self::LeF(ty) => format!("{ty}.le"),
            Self::LeS(ty) => format!("{ty}.le_s"),
            Self::LeU(ty) => format!("{ty}.le_u"),
            Self::LtF(ty) => format!("{ty}.lt"),
            Self::LtS(ty) => format!("{ty}.lt_s"),
            Self::LtU(ty) => format!("{ty}.lt_u"),

            Self::GlobalGet(id) => format!("global.get {id}"),
            Self::GlobalSet(id) => format!("global.set {id}"),
            Self::I32_WrapI64 => format!("i32.wrap_i64"),
            Self::I64_ExtendI32S => format!("i64.extend_i32_s"),
            Self::I64_ExtendI32U => format!("i64.extend_i32_u"),
            Self::I64_Load32S(id) => format!(
                "i64.load_i32_s{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::I64_Load32U(id) => format!(
                "i64.load_i32_u{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::I64_Store32(id) => format!(
                "i64.store32{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::If {
                identifier,
                result,
                then,
                r#else,
            } => format!(
                "(if {}{}(then {}){})",
                if let Some(identifier) = identifier {
                    format!("${identifier} ")
                } else {
                    String::new()
                },
                if let Some(result) = result {
                    format!(
                        "(result {}) ",
                        result
                            .iter()
                            .map(|v| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                } else {
                    String::new()
                },
                then,
                if let Some(r#else) = r#else {
                    format!(" (else {else})")
                } else {
                    String::new()
                }
            ),
            Self::Loop { identifier, contents } => {
                format!("(loop {}\n{contents})", if let Some(identifier) = identifier {
                    format!("${identifier} ")
                } else {
                    String::new()
                },)
            },
            Self::Load(ty, id) => format!(
                "{ty}.load{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Load16S(ty, id) => format!(
                "{ty}.load16_s{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Load16U(ty, id) => format!(
                "{ty}.load16_u{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Load8S(ty, id) => format!(
                "{ty}.load8_s{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Load8U(ty, id) => format!(
                "{ty}.load8_u{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::LocalGet(id) => format!("local.get {id}"),
            Self::LocalSet(id) => format!("local.set {id}"),
            Self::LocalTee(id) => format!("local.tee {id}"),
            Self::Max(ty) => format!("{ty}.max"),
            Self::MemoryGrow => format!("memory.grow"),
            Self::MemorySize => format!("memory.size"),
            Self::MemoryCopy => format!("memory.copy"),
            Self::Min(ty) => format!("{ty}.min"),
            Self::Mul(ty) => format!("{ty}.mul"),
            Self::Ne(ty) => format!("{ty}.ne"),
            Self::Nearest(ty) => format!("{ty}.nearest"),
            Self::Neg(ty) => format!("{ty}.neg"),
            Self::Nop => format!("nop"),
            Self::Or(ty) => format!("{ty}.or"),
            Self::PopCnt(ty) => format!("{ty}.popcnt"),
            Self::RemS(ty) => format!("{ty}.rem_s"),
            Self::RemU(ty) => format!("{ty}.rem_u"),
            Self::Return => format!("return"),
            Self::ReturnCall(id) => format!("return_call {id}"),
            Self::ReturnCallIndirect {
                parameters,
                result,
                table,
            } => format!(
                "(return_call_indirect (func {}{}){})",
                parameters
                    .iter()
                    .map(|v| format!("(param {v})"))
                    .collect::<Vec<_>>()
                    .join(" "),
                if let Some(result) = result {
                    format!(
                        " (result {})",
                        result
                            .iter()
                            .map(|v| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                } else {
                    String::new()
                },
                if let Some(table) = table {
                    format!(" (table {table})")
                } else {
                    String::new()
                }
            ),
            Self::RotL(ty) => format!("{ty}.rotl"),
            Self::RotR(ty) => format!("{ty}.rotr"),
            Self::Select => format!("select"),
            Self::Shl(ty) => format!("{ty}.shl"),
            Self::ShrS(ty) => format!("{ty}.shr_s"),
            Self::ShrU(ty) => format!("{ty}.shr_u"),
            Self::Sqrt(ty) => format!("{ty}.sqrt"),
            Self::Store(ty, id) => format!(
                "{ty}.store{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Store16(ty, id) => format!(
                "{ty}.store16{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Store8(ty, id) => format!(
                "{ty}.store8{}",
                if let Some(id) = id {
                    format!(" (memory {id})")
                } else {
                    String::new()
                }
            ),
            Self::Sub(ty) => format!("{ty}.sub"),
            Self::Trunc(ty) => format!("{ty}.trunc"),
            Self::TruncS(ty1, ty2) => format!("{ty1}.trunc_{ty2}_s"),
            Self::TruncU(ty1, ty2) => format!("{ty1}.trunc_{ty2}_u"),
            Self::Unreachable => format!("unreachable"),
            Self::Xor(ty) => format!("{ty}.xor"),
        })
    }
}

#[derive(Clone, PartialEq)]
pub struct Instructions<'a> {
    data: Box<[WasmInstruction<'a>]>,
}

impl<'a> From<Vec<WasmInstruction<'a>>> for Instructions<'a> {
    fn from(value: Vec<WasmInstruction<'a>>) -> Self {
        Self {
            data: value.into_boxed_slice()
        }
    }
}

impl Debug for Instructions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl Display for Instructions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data
            .iter()
            .map(|v| f.write_fmt(format_args!("{v}\n")))
            .collect::<Result<Vec<()>, _>>()?;
        Ok(())
    }
}

impl<'a> Deref for Instructions<'a> {
    type Target = [WasmInstruction<'a>];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> DerefMut for Instructions<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FuncType {
    ExternRef,
    AnyFunc,
}

impl Display for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ExternRef => "externref",
            Self::AnyFunc => "anyfunc",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElemType {
    FuncRef,
    ExternRef,
}

impl Display for ElemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::FuncRef => "funcref",
            Self::ExternRef => "externref",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModuleExpression<'a> {
    Function {
        name: Option<&'a str>,
        export: Option<&'a str>,
        parameters: Vec<(Option<&'a str>, NumericType)>,
        result: Option<Vec<NumericType>>,
        locals: Vec<(Option<&'a str>, NumericType)>,
        instructions: Instructions<'a>,
    },

    ExportFunc(Identifier<'a>, &'a str),
    ExportTable(Identifier<'a>, &'a str),
    ExportMem(Identifier<'a>, &'a str),
    ExportGlobal(Identifier<'a>, &'a str),

    ImportFunc {
        namespace: &'a str,
        name: &'a str,
        import_name: Option<&'a str>,
        parameters: Vec<NumericType>,
        result: Option<Vec<NumericType>>,
    },
    ImportTable {
        namespace: &'a str,
        name: &'a str,
        import_name: Option<&'a str>,
        length: usize,
        func: FuncType,
    },
    ImportMem {
        namespace: &'a str,
        name: &'a str,
        import_name: Option<&'a str>,
        min_size: usize,
        max_size: Option<usize>,
    },
    ImportGlobal {
        namespace: &'a str,
        name: &'a str,
        import_name: Option<&'a str>,
        // mutable: bool,
        ty: NumericType,
    },

    /// In pages.
    Memory(usize),

    Global {
        name: Option<&'a str>,
        ty: NumericType,
    },

    /// Immediately run the function.
    Start(Identifier<'a>),

    NumericType {
        name: Option<&'a str>,
        params: Vec<NumericType>,
        results: Option<Vec<NumericType>>,
    },

    Table {
        name: Option<&'a str>,
        min: usize,
        max: Option<usize>,
        elem_type: ElemType, // usually funcref
    },

    Elem {
        table: Identifier<'a>,
        offset: Instructions<'a>, // typically i32.const
        funcs: &'a [Identifier<'a>],
    },

    Data {
        memory: Identifier<'a>,
        offset: Instructions<'a>, // typically i32.const
        bytes: &'a [u8],
    },
}

impl Display for ModuleExpression<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Function {
                name,
                export,
                parameters,
                result,
                locals,
                instructions,
            } => {
                let mut out = String::from("(func ");
                if let Some(name) = name {
                    out.push_str(&format!("${name} "));
                }
                if let Some(export) = export {
                    out.push_str(&format!("(export \"{export}\") "));
                }

                for (name, ty) in parameters {
                    out.push_str("(param ");
                    if let Some(name) = name {
                        out.push_str(&format!("${name} "));
                    }
                    out.push_str(&format!("{ty}) "));
                }

                if let Some(result) = result {
                    out.push_str("(result ");
                    for ty in result {
                        out.push_str(&format!("{ty} "));
                    }
                    out.pop();
                    out.push_str(")");
                }

                for (name, ty) in locals {
                    out.push_str("(local ");
                    if let Some(name) = name {
                        out.push_str(&format!("${name} "));
                    }
                    out.push_str(&format!("{ty}) "));
                }

                out.push_str(&format!("{instructions})"));

                format!("{out}")
            }

            Self::ExportFunc(id, name) => format!("(export \"{}\" (func {id}))", name),
            Self::ExportTable(id, name) => format!("(export \"{}\" (table {id}))", name),
            Self::ExportMem(id, name) => format!("(export \"{}\" (memory {id}))", name),
            Self::ExportGlobal(id, name) => format!("(export \"{}\" (global {id}))", name),

            Self::ImportFunc {
                namespace,
                name,
                import_name,
                parameters,
                result,
            } => {
                let mut out = format!("(import \"{namespace}\" \"{name}\" (func ");

                if let Some(import_name) = import_name {
                    out.push_str(&format!("${import_name} "));
                }

                for ty in parameters {
                    out.push_str(&format!("(param {ty}) "));
                }

                if let Some(result) = result {
                    out.push_str("(result ");
                    for ty in result {
                        out.push_str(&format!("{ty} "));
                    }
                    out.pop();
                    out.push_str(")");
                }

                format!("{out}))")
            }

            Self::ImportGlobal {
                namespace,
                name,
                import_name,
                ty,
            } => {
                format!(
                    "(import \"{}\" \"{}\" (global {}{}))",
                    namespace,
                    name,
                    if let Some(import_name) = import_name {
                        format!("${import_name} ")
                    } else {
                        String::new()
                    },
                    ty.clone()
                )
            }
            Self::ImportMem {
                namespace,
                name,
                import_name,
                min_size,
                max_size,
            } => format!(
                "(import \"{}\" \"{}\" (memory {}{}{}))",
                namespace,
                name,
                if let Some(import_name) = import_name {
                    format!("{import_name} ")
                } else {
                    String::new()
                },
                min_size.clone(),
                if let Some(max_size) = max_size {
                    format!(" {max_size}")
                } else {
                    String::new()
                }
            ),
            Self::ImportTable {
                namespace,
                name,
                import_name,
                length,
                func,
            } => format!(
                "(import \"{}\" \"{}\" (table {}{} {}))",
                namespace,
                name,
                if let Some(import_name) = import_name {
                    format!("{import_name} ")
                } else {
                    String::new()
                },
                length.clone(),
                func.clone()
            ),

            Self::Memory(size) => format!("(memory {})", size.clone()),

            Self::Global { name, ty } => format!(
                "(global {}{})",
                if let Some(name) = name {
                    format!("{name} ")
                } else {
                    String::new()
                },
                ty.clone()
            ),

            Self::Start(id) => format!("(start {})", id.clone()),

            Self::Data {
                memory,
                offset,
                bytes,
            } => format!(
                "(data {memory} ({offset}) \"{}\")",
                bytes
                    .iter()
                    .flat_map(|v| std::ascii::escape_default(*v))
                    .map(|v| char::from_u32(v as u32).unwrap())
                    .collect::<String>()
            ),
            Self::Elem {
                table,
                offset,
                funcs,
            } => format!(
                "(elem {} ({offset}) {})",
                match table {
                    Identifier::Name(name) => format!("${name}"),
                    Identifier::Index(i) => format!("(i32.const {i})"),
                },
                funcs
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),

            Self::NumericType { name, params, results } => {
                let mut out = String::new();

                for ty in params {
                    out.push_str(&format!("(param {ty}) "));
                }

                if let Some(results) = results {
                    out.push_str("(result ");
                    for ty in results {
                        out.push_str(&format!("{ty} "));
                    }
                    out.pop();
                    out.push_str(")");
                }
                format!("(type {}{out})", if let Some(name) = name {
                    format!("${name} ")
                } else {
                    String::new()
                })
            },
            Self::Table { name, min, max, elem_type } => format!("(table {}{min}{} {elem_type})", if let Some(name) = name {
                    format!("${name} ")
                } else {
                    String::new()
                }, if let Some(max) = max {
                    format!(" {max}")
                } else {
                    String::new()
                })
        };
        f.write_str(&v)
    }
}
