//! Stable entry point: IR module in, relocatable object bytes out.
//!
//! This module is the covered surface of the crate for object emission. Its items
//! are [`compile_module_to_object`], [`ObjectError`], and the [`Module`] and [`Target`]
//! types that appear in its signature.
//!
//! Versioning: the crate follows semantic versioning as practised within the 0.x
//! series. Within a 0.x minor version, items here are not removed or given a changed
//! signature. [`ObjectError`] is `#[non_exhaustive]`, so new failure kinds can be added
//! without a breaking change; match it with a wildcard arm.
//!
//! Everything outside this module, including the other items at the crate root, the
//! `mir`, `mir_codegen` and `runtime` modules, and the assembly-text functions, may
//! change without notice.

use crate::ObjectOptions;
pub use crate::ir::module::Module;
use crate::mir::TransformPipeline;
use crate::mir::codegen::from_ir_for_target;
use crate::mir::transform::vector_unit_for;
use crate::mir_codegen::generate_mir_to_target_with_settings;
pub use crate::target::Target;
#[cfg(feature = "ssa-round-trip")]
use lamina_mir::ssa::round_trip_module;
use ras::{ObjectWriteOptions, Ras, is_object_file_supported};
use std::{error::Error as StdError, fmt};

const OPT_LEVEL: u8 = 1;

/// Failure modes of [`compile_module_to_object`].
///
/// Non-exhaustive: match with a wildcard arm so that new kinds of failure can be
/// added in a later release.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObjectError {
    /// The target has no object-file writer. Carries the rejected target.
    UnsupportedTarget(Target),
    /// The IR module could not be lowered to MIR, or the MIR optimizer failed.
    Lowering(String),
    /// The MIR could not be generated as assembly text for the target.
    Codegen(String),
    /// The assembly text could not be assembled into an object file.
    Assembly(String),
}

impl fmt::Display for ObjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectError::UnsupportedTarget(target) => {
                write!(f, "target {target} does not produce object files")
            }
            ObjectError::Lowering(msg) => write!(f, "lowering failed: {msg}"),
            ObjectError::Codegen(msg) => write!(f, "code generation failed: {msg}"),
            ObjectError::Assembly(msg) => write!(f, "assembly failed: {msg}"),
        }
    }
}

impl StdError for ObjectError {}

/// Compiles an IR module to relocatable object-file bytes for `target`.
///
/// The object format follows the target: ELF for Linux, BSD and Redox targets,
/// Mach-O for macOS, and COFF for Windows. The bytes are returned in memory and
/// nothing is written to disk. No external assembler or linker is run; the assembly
/// stage is the in-process `ras` encoder.
///
/// The module is compiled at optimization level 1, the `lamina` command-line default.
///
/// Targets that have no object writer are refused with
/// [`ObjectError::UnsupportedTarget`] before any work is done. A target can be built
/// from a string such as `"x86_64_linux"` with `str::parse`.
///
/// # Example
///
/// ```rust
/// use lamina::api::{compile_module_to_object, Target};
/// use std::str::FromStr;
///
/// let module = lamina::parser::parse_module(r#"
/// fn @main() -> i64 {
/// entry:
///     %result = add.i64 42, 8
///     ret.i64 %result
/// }
/// "#)?;
/// let target = Target::from_str("x86_64_linux")?;
/// let object = compile_module_to_object(&module, target)?;
/// assert_eq!(&object[..4], b"\x7fELF");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_module_to_object(module: &Module, target: Target) -> Result<Vec<u8>, ObjectError> {
    if !is_object_file_supported(target.architecture, target.operating_system) {
        return Err(ObjectError::UnsupportedTarget(target));
    }

    let options = ObjectOptions::for_target(&target, OPT_LEVEL);

    let mut mir_module = from_ir_for_target(module, "module", target.data_layout())
        .map_err(|e| ObjectError::Lowering(e.to_string()))?;

    TransformPipeline::default_for_opt_level_on(
        OPT_LEVEL,
        vector_unit_for(target.architecture, options.settings.x86_cpu_level()),
    )
    .apply_to_module(&mut mir_module)
    .map_err(|e| ObjectError::Lowering(format!("MIR optimization failed: {e}")))?;

    #[cfg(feature = "ssa-round-trip")]
    round_trip_module(&mut mir_module).map_err(|e| ObjectError::Lowering(e.to_string()))?;

    let mut text = Vec::new();
    generate_mir_to_target_with_settings(
        &mir_module,
        &mut text,
        &target,
        options.codegen_units,
        &options.settings,
    )
    .map_err(|e| ObjectError::Codegen(e.to_string()))?;
    let text = String::from_utf8(text)
        .map_err(|e| ObjectError::Codegen(format!("generated assembly is not UTF-8: {e}")))?;

    let mut assembler = Ras::with_object_write_options(
        target.architecture,
        target.operating_system,
        ObjectWriteOptions::default(),
    )
    .map_err(|e| ObjectError::Assembly(e.to_string()))?;
    assembler
        .assemble_to_bytes(&text)
        .map_err(|e| ObjectError::Assembly(e.to_string()))
}
