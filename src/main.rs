use lamina::cli::{
    self,
    options::{CliError, parse_args, print_usage},
};
use std::{error::Error, process};

fn main() -> Result<(), Box<dyn Error>> {
    let options = match parse_args() {
        Ok(opts) => opts,
        Err(CliError::Help) => {
            print_usage();
            return Ok(());
        }
        Err(CliError::Usage(message)) => {
            eprintln!("Error: {message}");
            print_usage();
            process::exit(1);
        }
        Err(CliError::Diagnosis(message)) => {
            eprintln!("Error: {message}");
            process::exit(1);
        }
    };

    cli::compile::compile(options)
}
