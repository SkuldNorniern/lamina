//! PowerPC code generation backends.

pub mod powerpc64;

pub use powerpc64::{
    Ppc64Codegen, generate_mir_ppc64, generate_mir_ppc64_with_units,
    generate_mir_ppc64_with_units_and_settings,
};
