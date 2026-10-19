use lamina::cli::{
    self,
    options::{parse_args, print_usage},
};
use std::{error::Error, process};

fn main() -> Result<(), Box<dyn Error>> {
    let options = match parse_args() {
        Ok(opts) => opts,
        Err(e) => {
            if e == "help" {
                print_usage();
                return Ok(());
            } else {
                eprintln!("Error: {e}");
                print_usage();
                process::exit(1);
            }
        }
    };

    cli::compile::compile(options)
}
