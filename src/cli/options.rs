//! CLI option parsing for the lamina compiler driver.

use std::path::PathBuf;

pub struct CompileOptions {
    pub input_file: PathBuf,
    pub output_file: Option<PathBuf>,
    pub verbose: bool,
    pub forced_compiler: Option<String>,
    pub assembler: Option<String>,
    pub assembler_flags: Vec<String>,
    pub linker_flags: Vec<String>,
    pub emit_asm_only: bool,
    pub emit_mir: bool,
    pub emit_mir_asm: Option<String>,
    pub target_arch: Option<String>,
    pub opt_level: u8,
    pub jit: bool,
    pub sandbox: bool,
    pub codegen_units: Option<usize>,
}

pub fn toolchain_backends(
    forced_compiler: &Option<String>,
    assembler_override: &Option<String>,
) -> (
    Option<lamina::mir_codegen::assemble::AssemblerBackend>,
    Option<lamina::mir_codegen::link::LinkerBackend>,
) {
    use lamina::mir_codegen::assemble::AssemblerBackend;
    use lamina::mir_codegen::link::LinkerBackend;

    let assem = assembler_override
        .as_deref()
        .and_then(|s| match s {
            "ras" => Some(AssemblerBackend::Ras),
            "gas" | "as" => Some(AssemblerBackend::Gas),
            "clang" => Some(AssemblerBackend::Lld),
            _ => None,
        })
        .or_else(|| {
            forced_compiler.as_deref().and_then(|s| match s {
                "gnu" => Some(AssemblerBackend::Gas),
                "clang" => Some(AssemblerBackend::Lld),
                "lamina" => Some(AssemblerBackend::Ras),
                _ => None,
            })
        });

    let link = forced_compiler.as_deref().and_then(|s| match s {
        "gnu" => Some(LinkerBackend::Ld),
        "clang" => Some(LinkerBackend::Lld),
        "msvc" => Some(LinkerBackend::Msvc),
        "lamina" => Some(LinkerBackend::Weld),
        "ld" => Some(LinkerBackend::Ld),
        "lld" => Some(LinkerBackend::Lld),
        "mold" => Some(LinkerBackend::Mold),
        "weld" => Some(LinkerBackend::Weld),
        "link" | "cl" => Some(LinkerBackend::Msvc),
        _ => None,
    });

    (assem, link)
}

pub fn print_usage() {
    eprintln!("Usage: lamina <input.lamina> [options]");
    eprintln!("Options:");
    eprintln!("  -o, --output <file>     Specify output executable name");
    eprintln!("  -v, --verbose           Enable verbose output");
    eprintln!(
        "  -c, --compiler <n>      Toolchain: gnu | clang | msvc | lamina. Or linker only: ld, lld, mold, weld, link, cl"
    );
    eprintln!("  --assembler <n>         Override assembler: ras, gas, clang");
    eprintln!("  -Wl,<flag>              Pass flag(s) to linker (GCC/Clang compatible)");
    eprintln!("  -Wa,<flag>              Pass flag(s) to assembler (GCC/Clang compatible)");
    eprintln!("  --emit-asm              Only emit assembly file without compiling");
    eprintln!("  --target <arch_os>      Specify target (e.g., x86_64_linux)");
    eprintln!("  --emit-mir              Only emit MIR (.lumir) and exit");
    eprintln!(
        "  --emit-mir-asm          EXPERIMENTAL: emit assembly from MIR (uses --target for OS, architecture)"
    );
    eprintln!(
        "  --opt-level <n>         EXPERIMENTAL(mir only): Set optimization level (0-3, default: 1)"
    );
    eprintln!("  --jit                   Compile and execute using runtime compilation (JIT)");
    eprintln!("  --sandbox               Enable sandbox for secure execution (requires --jit)");
    eprintln!(
        "  -j, --jobs <n>          Number of parallel compilation threads (default: max - 2)"
    );
    eprintln!("  -h, --help              Display this help message");
}

pub fn parse_args() -> Result<CompileOptions, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Err("Not enough arguments".to_string());
    }

    if args[1] == "-h" || args[1] == "--help" {
        return Err("help".to_string());
    }

    let mut options = CompileOptions {
        input_file: PathBuf::new(),
        output_file: None,
        verbose: false,
        forced_compiler: None,
        assembler: None,
        assembler_flags: Vec::new(),
        linker_flags: Vec::new(),
        emit_asm_only: false,
        emit_mir: false,
        emit_mir_asm: None,
        target_arch: None,
        opt_level: 1,
        jit: false,
        sandbox: false,
        codegen_units: None,
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
            "-v" | "--verbose" => {
                options.verbose = true;
                i += 1;
            }
            "-c" | "--compiler" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for compiler".to_string());
                }
                options.forced_compiler = Some(args[i + 1].clone());
                i += 2;
            }
            "--assembler" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for --assembler".to_string());
                }
                options.assembler = Some(args[i + 1].clone());
                i += 2;
            }
            "--emit-asm" => {
                options.emit_asm_only = true;
                i += 1;
            }
            "--emit-mir" => {
                options.emit_mir = true;
                i += 1;
            }
            "--emit-mir-asm" => {
                options.emit_mir_asm = Some("default".to_string());
                i += 1;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for target architecture".to_string());
                }
                let requested_target = args[i + 1].to_lowercase();
                if !requested_target.contains('_')
                    || !lamina_platform::HOST_ARCH_LIST
                        .iter()
                        .any(|&supported| supported == requested_target)
                {
                    return Err(format!(
                        "Unsupported target '{}'. Supported values:\n{}",
                        requested_target,
                        lamina_platform::HOST_ARCH_LIST.join(", ")
                    ));
                }
                options.target_arch = Some(requested_target);
                i += 2;
            }
            "--opt-level" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for optimization level".to_string());
                }
                let level = args[i + 1]
                    .parse::<u8>()
                    .map_err(|_| "Invalid --opt-level value (must be 0-3)".to_string())?;
                if level > 3 {
                    return Err("--opt-level must be between 0 and 3".to_string());
                }
                options.opt_level = level;
                i += 2;
            }
            "--jit" => {
                options.jit = true;
                i += 1;
            }
            "--sandbox" => {
                options.sandbox = true;
                i += 1;
            }
            "-j" | "--jobs" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for -j/--jobs".to_string());
                }
                let jobs = args[i + 1].parse::<usize>().map_err(|_| {
                    "Invalid -j/--jobs value (must be a positive integer)".to_string()
                })?;
                if jobs == 0 {
                    return Err("-j/--jobs must be at least 1".to_string());
                }
                options.codegen_units = Some(jobs);
                i += 2;
            }
            "--version" => {
                println!("lamina {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            arg => {
                if arg.starts_with("-Wl,") {
                    options
                        .linker_flags
                        .extend(arg[4..].split(',').map(|s| s.to_string()));
                    i += 1;
                } else if arg.starts_with("-Wa,") {
                    options
                        .assembler_flags
                        .extend(arg[4..].split(',').map(|s| s.to_string()));
                    i += 1;
                } else if arg == "-Wl" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wl".to_string());
                    }
                    options
                        .linker_flags
                        .extend(args[i + 1].split(',').map(|s| s.to_string()));
                    i += 2;
                } else if arg == "-Wa" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wa".to_string());
                    }
                    options
                        .assembler_flags
                        .extend(args[i + 1].split(',').map(|s| s.to_string()));
                    i += 2;
                } else if options.input_file.as_os_str().is_empty() {
                    // First non-flag argument is the input file, regardless of position.
                    options.input_file = PathBuf::from(arg);
                    i += 1;
                } else {
                    return Err(format!("Unknown argument: {}", arg));
                }
            }
        }
    }

    if options.input_file.as_os_str().is_empty() {
        return Err("No input file specified".to_string());
    }

    Ok(options)
}
