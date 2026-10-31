//! Layer 1 of the runtime: the compiler runtime routines lamina ships itself.
//!
//! `builtins/` holds twenty-four `@export`ed routines written in Lamina IR for the
//! 128-bit helpers that i128 lowering emits and the 64-bit helpers a target with
//! 32-bit integer registers needs for every `i64`. They are embedded in the compiler
//! here, compiled for the requested target when a program calls one, and handed
//! to the linker ahead of any host archive. A program that never reaches one of
//! these symbols builds and links exactly as before.
//!
//! A routine the target's backend cannot yet compile is left out and resolved
//! from the host compiler runtime instead, so a backend gap costs the routine
//! rather than the build.
//!
//! See `Plan/runtime-layering.md` for the split between this layer and libc.

use crate::{
    error::LaminaError,
    mir::{
        Instruction, Module, TransformPipeline, codegen::from_ir_for_target,
        transform::vector_unit_for,
    },
    mir_codegen::{
        MirCodegenSettings,
        assemble::{
            AssemblerBackend, assemble_with_ras_object_options, get_assembly_output_extension,
            get_intermediate_extension,
        },
        generate_mir_to_target_with_settings,
    },
    parser::parse_module,
};
use lamina_codegen::abi::{BuiltinLibrary, RuntimeSymbol};
use lamina_platform::Target;
use ras::ObjectWriteOptions;
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const ROUTINES: &[(RuntimeSymbol, &str, &str)] = &[
    (
        RuntimeSymbol::Udivti3,
        "udivti3",
        include_str!("../../builtins/udivti3.lamina"),
    ),
    (
        RuntimeSymbol::Umodti3,
        "umodti3",
        include_str!("../../builtins/umodti3.lamina"),
    ),
    (
        RuntimeSymbol::Divti3,
        "divti3",
        include_str!("../../builtins/divti3.lamina"),
    ),
    (
        RuntimeSymbol::Modti3,
        "modti3",
        include_str!("../../builtins/modti3.lamina"),
    ),
    (
        RuntimeSymbol::Floatuntidf,
        "floatuntidf",
        include_str!("../../builtins/floatuntidf.lamina"),
    ),
    (
        RuntimeSymbol::Floattidf,
        "floattidf",
        include_str!("../../builtins/floattidf.lamina"),
    ),
    (
        RuntimeSymbol::Floatuntisf,
        "floatuntisf",
        include_str!("../../builtins/floatuntisf.lamina"),
    ),
    (
        RuntimeSymbol::Floattisf,
        "floattisf",
        include_str!("../../builtins/floattisf.lamina"),
    ),
    (
        RuntimeSymbol::Fixunsdfti,
        "fixunsdfti",
        include_str!("../../builtins/fixunsdfti.lamina"),
    ),
    (
        RuntimeSymbol::Fixdfti,
        "fixdfti",
        include_str!("../../builtins/fixdfti.lamina"),
    ),
    (
        RuntimeSymbol::Fixunssfti,
        "fixunssfti",
        include_str!("../../builtins/fixunssfti.lamina"),
    ),
    (
        RuntimeSymbol::Fixsfti,
        "fixsfti",
        include_str!("../../builtins/fixsfti.lamina"),
    ),
    (
        RuntimeSymbol::Udivdi3,
        "udivdi3",
        include_str!("../../builtins/udivdi3.lamina"),
    ),
    (
        RuntimeSymbol::Umoddi3,
        "umoddi3",
        include_str!("../../builtins/umoddi3.lamina"),
    ),
    (
        RuntimeSymbol::Divdi3,
        "divdi3",
        include_str!("../../builtins/divdi3.lamina"),
    ),
    (
        RuntimeSymbol::Moddi3,
        "moddi3",
        include_str!("../../builtins/moddi3.lamina"),
    ),
    (
        RuntimeSymbol::Floatundidf,
        "floatundidf",
        include_str!("../../builtins/floatundidf.lamina"),
    ),
    (
        RuntimeSymbol::Floatdidf,
        "floatdidf",
        include_str!("../../builtins/floatdidf.lamina"),
    ),
    (
        RuntimeSymbol::Floatundisf,
        "floatundisf",
        include_str!("../../builtins/floatundisf.lamina"),
    ),
    (
        RuntimeSymbol::Floatdisf,
        "floatdisf",
        include_str!("../../builtins/floatdisf.lamina"),
    ),
    (
        RuntimeSymbol::Fixunsdfdi,
        "fixunsdfdi",
        include_str!("../../builtins/fixunsdfdi.lamina"),
    ),
    (
        RuntimeSymbol::Fixdfdi,
        "fixdfdi",
        include_str!("../../builtins/fixdfdi.lamina"),
    ),
    (
        RuntimeSymbol::Fixunssfdi,
        "fixunssfdi",
        include_str!("../../builtins/fixunssfdi.lamina"),
    ),
    (
        RuntimeSymbol::Fixsfdi,
        "fixsfdi",
        include_str!("../../builtins/fixsfdi.lamina"),
    ),
];

fn routine(symbol: RuntimeSymbol) -> Option<(&'static str, &'static str)> {
    ROUTINES
        .iter()
        .find(|(candidate, _, _)| *candidate == symbol)
        .map(|(_, stem, source)| (*stem, *source))
}

/// Runtime symbols the module calls, leaves undefined, and has not redirected
/// through `LAMINA_BUILTINS`. Ordered by [`ROUTINES`], so a link line does not
/// depend on how a hash happened to fall.
fn wanted(module: &Module, builtins: &BuiltinLibrary) -> Vec<RuntimeSymbol> {
    let called = |symbol: RuntimeSymbol| {
        let name = symbol.c_name();
        if builtins.c_name(symbol) != name
            || module
                .functions
                .get(name)
                .is_some_and(|defined| !defined.blocks.is_empty())
        {
            return false;
        }
        module.functions.values().any(|function| {
            function.blocks.iter().any(|block| {
                block.instructions.iter().any(|instruction| {
                    matches!(
                        &instruction.instr,
                        Instruction::Call { name: called, .. }
                            | Instruction::TailCall { name: called, .. }
                        if called == name
                    )
                })
            })
        })
    };
    ROUTINES
        .iter()
        .map(|(symbol, _, _)| *symbol)
        .filter(|symbol| called(*symbol))
        .collect()
}

/// Objects built for one compilation, in link order. They live in a directory of
/// their own so a failed link leaves them to look at and a successful one can
/// drop them.
pub struct RuntimeObjects {
    directory: PathBuf,
    written: Vec<PathBuf>,
    objects: Vec<PathBuf>,
}

impl RuntimeObjects {
    pub fn objects(&self) -> &[PathBuf] {
        &self.objects
    }

    pub fn discard(self) {
        for path in &self.written {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_dir(&self.directory);
    }
}

/// A directory beside the program's own object file, so two compilations running
/// at once never write the same intermediate.
fn scratch_directory(object_path: &Path) -> Result<PathBuf, LaminaError> {
    let stem = object_path
        .file_stem()
        .map_or_else(|| "a".to_string(), |s| s.to_string_lossy().to_string());
    let directory = object_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.lamina-runtime"));
    fs::create_dir_all(&directory).map_err(|e| {
        LaminaError::ValidationError(format!("cannot create {}: {e}", directory.display()))
    })?;
    Ok(directory)
}

/// Compiles the layer-1 routines the module reaches, and the routines those
/// reach in turn, for `target`. Returns `None` when the module needs none.
#[allow(clippy::too_many_arguments)]
pub fn build(
    module: &Module,
    object_path: &Path,
    target: Target,
    settings: &MirCodegenSettings,
    opt_level: u8,
    codegen_units: usize,
    assembler: Option<AssemblerBackend>,
    assembler_flags: &[String],
    verbose: bool,
    ras_object_write_options: &ObjectWriteOptions,
) -> Result<Option<RuntimeObjects>, LaminaError> {
    let builtins = BuiltinLibrary::from_env(target.operating_system);
    let mut pending = wanted(module, &builtins);
    if pending.is_empty() {
        return Ok(None);
    }

    let mut built = RuntimeObjects {
        directory: scratch_directory(object_path)?,
        written: Vec::new(),
        objects: Vec::new(),
    };
    let mut seen = pending.clone();
    let mut unbuilt: Vec<(RuntimeSymbol, LaminaError)> = Vec::new();
    while let Some(symbol) = pending.pop() {
        let Some((stem, source)) = routine(symbol) else {
            continue;
        };
        let routine_module = match lower(source, stem, target, settings, opt_level) {
            Ok(module) => module,
            Err(reason) => {
                unbuilt.push((symbol, reason));
                continue;
            }
        };
        for dependency in wanted(&routine_module, &builtins) {
            if !seen.contains(&dependency) {
                seen.push(dependency);
                pending.push(dependency);
            }
        }
        match emit(
            &routine_module,
            stem,
            &built.directory,
            target,
            settings,
            codegen_units,
            assembler,
            assembler_flags,
            verbose,
            ras_object_write_options,
            &mut built.written,
        ) {
            Ok(object) => built.objects.push(object),
            Err(reason) => unbuilt.push((symbol, reason)),
        }
    }
    if verbose {
        println!(
            "[VERBOSE] Compiler runtime for {target}: {}",
            if built.objects.is_empty() {
                "nothing built".to_string()
            } else {
                built
                    .objects
                    .iter()
                    .filter_map(|o| o.file_stem().map(|s| s.to_string_lossy().to_string()))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        for (symbol, reason) in &unbuilt {
            println!(
                "[VERBOSE] Compiler runtime: {} left to the host runtime: {reason}",
                symbol.c_name()
            );
        }
    }
    if built.objects.is_empty() {
        built.discard();
        return Ok(None);
    }
    Ok(Some(built))
}

fn lower(
    source: &'static str,
    stem: &str,
    target: Target,
    settings: &MirCodegenSettings,
    opt_level: u8,
) -> Result<Module, LaminaError> {
    let ir = parse_module(source).map_err(|e| {
        LaminaError::ValidationError(format!("compiler runtime {stem} failed to parse: {e}"))
    })?;
    let mut module = from_ir_for_target(&ir, stem, target.data_layout()).map_err(|e| {
        LaminaError::ValidationError(format!("compiler runtime {stem} failed to lower: {e}"))
    })?;
    if opt_level > 0 {
        TransformPipeline::default_for_opt_level_on(
            opt_level,
            vector_unit_for(target.architecture, settings.x86_cpu_level()),
        )
        .apply_to_module(&mut module)
        .map_err(|e| {
            LaminaError::ValidationError(format!("compiler runtime {stem} failed to optimize: {e}"))
        })?;
    }
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
fn emit(
    module: &Module,
    stem: &str,
    directory: &Path,
    target: Target,
    settings: &MirCodegenSettings,
    codegen_units: usize,
    assembler: Option<AssemblerBackend>,
    assembler_flags: &[String],
    verbose: bool,
    ras_object_write_options: &ObjectWriteOptions,
    written: &mut Vec<PathBuf>,
) -> Result<PathBuf, LaminaError> {
    let mut assembly = Vec::<u8>::new();
    generate_mir_to_target_with_settings(module, &mut assembly, &target, codegen_units, settings)?;

    let assembly_path = directory
        .join(stem)
        .with_extension(get_intermediate_extension(
            target.architecture,
            target.operating_system,
        ));
    written.push(assembly_path.clone());
    File::create(&assembly_path)
        .and_then(|mut file| file.write_all(&assembly))
        .map_err(|e| {
            LaminaError::ValidationError(format!("cannot write {}: {e}", assembly_path.display()))
        })?;

    let object_path = directory
        .join(stem)
        .with_extension(get_assembly_output_extension(target.architecture));
    written.push(object_path.clone());
    assemble_with_ras_object_options(
        &assembly_path,
        &object_path,
        target,
        assembler,
        assembler_flags,
        verbose,
        ras_object_write_options.clone(),
    )
    .map(|result| result.output_path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{ROUTINES, routine, wanted};
    use crate::mir::codegen::from_ir_for_target;
    use crate::parser::parse_module;
    use lamina_codegen::abi::{BuiltinLibrary, RuntimeSymbol};
    use lamina_platform::{Target, TargetOperatingSystem};
    use std::collections::HashMap;
    use std::str::FromStr;

    fn target() -> Target {
        Target::from_str("x86_64_linux").unwrap()
    }

    #[test]
    fn every_routine_parses_and_exports_its_symbol() {
        for (symbol, stem, source) in ROUTINES {
            let ir = parse_module(source).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let module = from_ir_for_target(&ir, stem, target().data_layout())
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(
                module.exported_functions.contains(symbol.c_name()),
                "{stem} does not export {}",
                symbol.c_name()
            );
        }
    }

    #[test]
    fn signed_division_pulls_in_the_unsigned_routine() {
        let (stem, source) = routine(RuntimeSymbol::Divti3).unwrap();
        let ir = parse_module(source).unwrap();
        let module = from_ir_for_target(&ir, stem, target().data_layout()).unwrap();
        let builtins = BuiltinLibrary::from_env(TargetOperatingSystem::Linux);
        assert!(wanted(&module, &builtins).contains(&RuntimeSymbol::Udivti3));
    }

    #[test]
    fn signed_64bit_division_pulls_in_the_unsigned_routine() {
        let (stem, source) = routine(RuntimeSymbol::Divdi3).unwrap();
        let ir = parse_module(source).unwrap();
        let module = from_ir_for_target(&ir, stem, target().data_layout()).unwrap();
        let builtins = BuiltinLibrary::from_env(TargetOperatingSystem::Linux);
        assert!(wanted(&module, &builtins).contains(&RuntimeSymbol::Udivdi3));
    }

    #[test]
    fn every_runtime_symbol_the_compiler_emits_has_a_routine() {
        for symbol in RuntimeSymbol::ALL {
            if !symbol.c_name().starts_with("__") {
                continue;
            }
            assert!(
                routine(symbol).is_some(),
                "{} has no routine",
                symbol.c_name()
            );
        }
    }

    #[test]
    fn a_program_without_wide_arithmetic_wants_nothing() {
        let source = "fn @main() -> i32 {\nentry:\n  ret.i32 0\n}\n";
        let ir = parse_module(source).unwrap();
        let module = from_ir_for_target(&ir, "main", target().data_layout()).unwrap();
        let builtins = BuiltinLibrary::from_env(TargetOperatingSystem::Linux);
        assert!(wanted(&module, &builtins).is_empty());
    }

    #[test]
    fn an_overridden_symbol_is_left_to_the_embedder() {
        let (stem, source) = routine(RuntimeSymbol::Divti3).unwrap();
        let ir = parse_module(source).unwrap();
        let module = from_ir_for_target(&ir, stem, target().data_layout()).unwrap();
        let overrides = HashMap::from([(RuntimeSymbol::Udivti3, "my_udiv".to_string())]);
        let builtins = BuiltinLibrary::new(TargetOperatingSystem::Linux, overrides);
        assert!(wanted(&module, &builtins).is_empty());
    }
}
