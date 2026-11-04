//! What a debugger is told about a module, built from MIR and handed to the assembler.
//!
//! The DWARF bytes are written by `ras`; this names the subprograms and the types in
//! terms the writer can encode. Nothing here states where a value lives at run time.
//! That needs the register allocator's result, which does not outlive the backends, and
//! a location guessed from anything less makes a debugger print a confident wrong
//! answer. A parameter therefore arrives with a name and a type and no location, which
//! a debugger reports as optimized out.
//!
//! When the backends do publish their allocation, a `location: Option<Vec<u8>>` on
//! [`ras::FormalParameter`] and a `locals` list beside `params` are where it attaches;
//! every subprogram already carries `DW_AT_frame_base`, so a frame-relative expression
//! has something to be relative to.

use lamina_mir::{
    Module as MirModule, Parameter, ScalarType, Signedness, SymbolVisibility, types::MirType,
};
use lamina_platform::TargetOperatingSystem;
use ras::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions,
    Subprogram,
};

/// Whether the object this target writes carries DWARF sections.
pub fn emits_dwarf(target_os: TargetOperatingSystem) -> bool {
    matches!(
        target_os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::MacOS
    )
}

/// Whether the object this target writes carries CodeView debug sections, which the
/// linker gathers into a PDB.
pub fn emits_codeview(target_os: TargetOperatingSystem) -> bool {
    target_os == TargetOperatingSystem::Windows
}

/// Mach-O labels a function's code with its name under a leading underscore, which the
/// debugger must not show and the assembler must be asked for by.
fn object_symbol(target_os: TargetOperatingSystem, name: &str) -> String {
    match target_os {
        TargetOperatingSystem::MacOS => format!("_{name}"),
        _ => name.to_string(),
    }
}

/// Fill in what the assembler needs to describe this module's functions. On a DWARF
/// target the symbol types and sizes go in whether or not debug info was asked for,
/// and the DWARF only under `-g`. COFF takes a symbol's type from its section and has
/// no symbol size, so there the whole request is `-g` only and a build without it is
/// byte-identical.
pub fn apply(
    options: &mut ObjectWriteOptions,
    target_os: TargetOperatingSystem,
    debug: bool,
    module: &MirModule,
) {
    if !emits_dwarf(target_os) && !(debug && emits_codeview(target_os)) {
        return;
    }
    options.function_symbols = function_symbols(module)
        .iter()
        .map(|name| object_symbol(target_os, name))
        .collect();
    if debug {
        options.debug_info = Some(describe(target_os, module));
    }
}

/// Names of the functions this module defines, for the assembler to mark `STT_FUNC` and
/// size. A declaration has no body here and is left out.
pub fn function_symbols(module: &MirModule) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter(|(name, func)| !module.is_external(name) && !func.blocks.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    names.sort_unstable();
    names
}

pub fn describe(target_os: TargetOperatingSystem, module: &MirModule) -> DebugInfoRequest {
    let mut subprograms: Vec<Subprogram> = module
        .functions
        .iter()
        .filter(|(name, func)| !module.is_external(name) && !func.blocks.is_empty())
        .map(|(name, func)| Subprogram {
            name: name.clone(),
            symbol: Some(object_symbol(target_os, name)).filter(|symbol| symbol != name),
            external: is_linkable(module, name, func.sig.visibility),
            decl: func.sig.decl.map(|loc| DeclSite {
                file: loc.file.get(),
                line: loc.line.get(),
            }),
            ret: func
                .sig
                .ret_aggregate
                .is_none()
                .then(|| {
                    func.sig
                        .ret_ty
                        .and_then(|ty| debug_type(ty, func.sig.ret_signedness))
                })
                .flatten(),
            params: func.sig.params.iter().map(parameter).collect(),
            variadic: func.sig.variadic,
        })
        .collect();
    subprograms.sort_by(|left, right| left.name.cmp(&right.name));
    DebugInfoRequest {
        producer: Some(format!("lamina {}", env!("CARGO_PKG_VERSION"))),
        subprograms,
    }
}

fn is_linkable(module: &MirModule, name: &str, visibility: Option<SymbolVisibility>) -> bool {
    name == "main"
        || module.exported_functions.contains(name)
        || matches!(
            visibility,
            Some(SymbolVisibility::Hidden | SymbolVisibility::Protected)
        )
}

/// An aggregate parameter keeps its name and loses its type: `reg` holds the address of
/// the callee's copy, and MIR records the aggregate's size and alignment but not its
/// members, so neither a struct nor the pointer would describe what the source declared.
fn parameter(param: &Parameter) -> FormalParameter {
    FormalParameter {
        name: param.name.clone(),
        ty: match param.aggregate {
            Some(_) => None,
            None => debug_type(param.ty, param.signedness),
        },
    }
}

fn debug_type(ty: MirType, signedness: Signedness) -> Option<DebugType> {
    let unsigned = signedness == Signedness::Unsigned;
    let scalar = match ty {
        MirType::Scalar(scalar) => scalar,
        MirType::Vector(_) => return None,
    };
    let base = match scalar {
        ScalarType::I1 => BaseType::Bool,
        ScalarType::I8 if unsigned => BaseType::U8,
        ScalarType::I8 => BaseType::I8,
        ScalarType::I16 if unsigned => BaseType::U16,
        ScalarType::I16 => BaseType::I16,
        ScalarType::I32 if unsigned => BaseType::U32,
        ScalarType::I32 => BaseType::I32,
        ScalarType::I64 if unsigned => BaseType::U64,
        ScalarType::I64 => BaseType::I64,
        ScalarType::F32 => BaseType::F32,
        ScalarType::F64 => BaseType::F64,
        ScalarType::Ptr | ScalarType::PtrIn(_) => return Some(DebugType::Pointer),
    };
    Some(DebugType::Base(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::codegen::from_ir_for_target;
    use crate::parser::parse_module;
    use lamina_platform::Target;
    use std::str::FromStr;

    const SOURCE: &str = r#"@extern fn @outside(i64 %n) -> i64

type !pair = struct { a: i64, b: i64 }

fn @scale(f64 %x, u32 %times) -> f64 {
  entry:
    ret.f64 %x
}

fn @widen(!pair %p) -> i64 {
  entry:
    ret.i64 0
}

fn @main() -> i64 {
  entry:
    ret.i64 0
}
"#;

    fn described() -> DebugInfoRequest {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        describe(TargetOperatingSystem::Linux, &mir)
    }

    fn subprogram(request: &DebugInfoRequest, name: &str) -> Subprogram {
        request
            .subprograms
            .iter()
            .find(|sub| sub.name == name)
            .unwrap_or_else(|| panic!("no subprogram for {name}"))
            .clone()
    }

    #[test]
    fn a_declaration_without_a_body_is_not_described() {
        let request = described();
        assert!(request.subprograms.iter().all(|sub| sub.name != "outside"));
        assert_eq!(request.subprograms.len(), 3);
    }

    #[test]
    fn a_function_carries_the_line_its_fn_keyword_is_on() {
        let request = described();
        let decl = subprogram(&request, "scale").decl.expect("decl");
        assert_eq!(decl.line, 5);
        assert_eq!(subprogram(&request, "main").decl.expect("decl").line, 15);
    }

    #[test]
    fn only_a_linkable_function_is_external() {
        let request = described();
        assert!(subprogram(&request, "main").external);
        assert!(!subprogram(&request, "scale").external);
    }

    #[test]
    fn a_parameter_keeps_its_source_name_and_signedness() {
        let params = subprogram(&described(), "scale").params;
        let names: Vec<Option<&str>> = params.iter().map(|p| p.name.as_deref()).collect();
        assert_eq!(names, vec![Some("x"), Some("times")]);
        assert_eq!(params[0].ty, Some(DebugType::Base(BaseType::F64)));
        assert_eq!(params[1].ty, Some(DebugType::Base(BaseType::U32)));
    }

    /// MIR records an aggregate's size and alignment but not its members, so neither
    /// a struct nor the pointer that carries it describes what the source declared.
    #[test]
    fn an_aggregate_parameter_is_named_and_left_untyped() {
        let params = subprogram(&described(), "widen").params;
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name.as_deref(), Some("p"));
        assert_eq!(params[0].ty, None);
    }

    #[test]
    fn the_function_symbols_are_the_functions_with_a_body() {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        assert_eq!(function_symbols(&mir), vec!["main", "scale", "widen"]);
    }

    #[test]
    fn a_target_whose_object_carries_no_debug_info_is_described_to_no_one() {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        let mut options = ObjectWriteOptions::default();
        apply(&mut options, TargetOperatingSystem::Orbis, true, &mir);
        assert!(options.function_symbols.is_empty());
        assert!(options.debug_info.is_none());
    }

    /// COFF takes a symbol's type from its section and has no symbol size, so there is
    /// nothing for the assembler to fill in unless CodeView was asked for.
    #[test]
    fn a_windows_build_is_described_only_under_g() {
        let target = Target::from_str("x86_64_windows").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");

        let mut plain = ObjectWriteOptions::default();
        apply(&mut plain, TargetOperatingSystem::Windows, false, &mir);
        assert!(plain.function_symbols.is_empty());
        assert!(plain.debug_info.is_none());

        let mut debug = ObjectWriteOptions::default();
        apply(&mut debug, TargetOperatingSystem::Windows, true, &mir);
        assert_eq!(debug.function_symbols, vec!["main", "scale", "widen"]);
        let request = debug.debug_info.expect("a request");
        // COFF spells a function's symbol exactly as the source named it.
        assert!(request.subprograms.iter().all(|sub| sub.symbol.is_none()));
    }

    /// Mach-O labels a function's code with a leading underscore and a debugger must
    /// not show it, so the assembler is asked for one name and tells the debugger the
    /// other.
    #[test]
    fn a_mach_o_subprogram_keeps_its_source_name_and_names_its_underscored_symbol() {
        let target = Target::from_str("x86_64_macos").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        let mut options = ObjectWriteOptions::default();
        apply(&mut options, TargetOperatingSystem::MacOS, true, &mir);

        assert_eq!(options.function_symbols, vec!["_main", "_scale", "_widen"]);
        let request = options.debug_info.expect("a described module");
        let main = subprogram(&request, "main");
        assert_eq!(main.symbol.as_deref(), Some("_main"));
        assert!(
            request
                .subprograms
                .iter()
                .all(|sub| !sub.name.starts_with('_'))
        );
    }

    #[test]
    fn an_elf_subprogram_names_no_separate_symbol() {
        let request = described();
        assert!(request.subprograms.iter().all(|sub| sub.symbol.is_none()));
    }
}
