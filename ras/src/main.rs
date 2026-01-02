//! ras - Raw Assembler
//!
//! Standalone assembler tool, compatible with `as` and `gas` command-line interface.

use std::env;
use std::path::PathBuf;

fn print_usage() {
    eprintln!("Usage: ras [options] <input.s> -o <output.o>");
    eprintln!("Options:");
    eprintln!("  -o, --output <file>     Specify output object file");
    eprintln!("  --target <arch_os>      Specify target (e.g., x86_64_linux)");
    eprintln!("  -v, --verbose           Enable verbose output");
    eprintln!("  -h, --help              Display this help message");
    eprintln!();
    eprintln!("ras - Raw Assembler for Lamina");
    eprintln!("Cross-platform assembler that generates object files from assembly text.");
}

struct Options {
    input_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    target: Option<String>,
    verbose: bool,
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = env::args().collect();
    let mut options = Options {
        input_file: None,
        output_file: None,
        target: None,
        verbose: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for output file".to_string());
                }
                options.output_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for target".to_string());
                }
                options.target = Some(args[i + 1].clone());
                i += 2;
            }
            "-v" | "--verbose" => {
                options.verbose = true;
                i += 1;
            }
            "-h" | "--help" => {
                return Err("help".to_string());
            }
            _ => {
                if args[i].starts_with('-') {
                    return Err(format!("Unknown option: {}", args[i]));
                }
                if options.input_file.is_none() {
                    options.input_file = Some(PathBuf::from(&args[i]));
                } else {
                    return Err(format!("Unexpected argument: {}", args[i]));
                }
                i += 1;
            }
        }
    }

    if options.input_file.is_none() {
        return Err("No input file specified".to_string());
    }

    if options.output_file.is_none() {
        return Err("No output file specified (use -o)".to_string());
    }

    Ok(options)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = match parse_args() {
        Ok(opts) => opts,
        Err(e) => {
            if e == "help" {
                print_usage();
                return Ok(());
            } else {
                eprintln!("Error: {}", e);
                print_usage();
                std::process::exit(1);
            }
        }
    };

    let input_path = options.input_file.as_ref().unwrap();
    if !input_path.exists() {
        eprintln!("Error: Input file '{}' does not exist.", input_path.display());
        std::process::exit(1);
    }

    // Determine target
    use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};
    use std::str::FromStr;
    let target = if let Some(target_str) = &options.target {
        Target::from_str(target_str)
            .map_err(|e| format!("Invalid target '{}': {}", target_str, e))?
    } else {
        Target::detect_host()
    };

    if options.verbose {
        println!("[ras] Assembling {} -> {}", input_path.display(), options.output_file.as_ref().unwrap().display());
        println!("[ras] Target: {}", target);
    }

    // Create assembler
    let mut ras = ras::Ras::new(target.architecture, target.operating_system)
        .map_err(|e| format!("Failed to create assembler: {}", e))?;

    // Assemble
    ras.assemble_file(input_path, options.output_file.as_ref().unwrap())
        .map_err(|e| format!("Assembly failed: {}", e))?;

    if options.verbose {
        println!("[ras] Assembly completed successfully");
    }

    Ok(())
}
