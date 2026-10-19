use crate::cli::{compile::compile as run_compiler, options::parse_args_from};
use std::{error::Error, path::Path};

pub(crate) fn compile(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    flags: &[&str],
) -> Result<(), Box<dyn Error>> {
    let args = [
        "lamina".to_owned(),
        source.as_ref().to_string_lossy().into_owned(),
        "-o".to_owned(),
        output.as_ref().to_string_lossy().into_owned(),
    ]
    .into_iter()
    .chain(flags.iter().map(|flag| (*flag).to_owned()))
    .collect();
    run_compiler(parse_args_from(args)?)
}
