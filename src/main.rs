use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

// Import the library crate

fn print_usage() {
    eprintln!("Usage: lamina <input.lamina> [options]");
    eprintln!("Options:");
    eprintln!("  -o, --output <file>     Specify output executable name");
    eprintln!("  -v, --verbose           Enable verbose output");
    eprintln!("  -c, --compiler <n>      Force specific compiler (gcc, clang, cl)");
    eprintln!("  -f, --flag <flag>       Pass additional flag to compiler");
    eprintln!("  --emit-asm              Only emit assembly file without compiling");
    eprintln!("  --target <arch>         Specify target architecture (x86_64, aarch64)");
    eprintln!("  --emit-mir              Only emit MIR (.mlamina) and exit");
    eprintln!(
        "  --emit-mir-asm <os>     EXPERIMENTAL: emit AArch64 asm from MIR (os: macos|linux|windows)"
    );
    eprintln!("  --opt-level <n>         Set optimization level (0-3, default: 1)");
    eprintln!("  --timeout <secs>        Abort after N seconds (best-effort)");
    eprintln!("  -h, --help              Display this help message");
}

struct CompileOptions {
    input_file: PathBuf,
    output_file: Option<PathBuf>,
    verbose: bool,
    forced_compiler: Option<String>,
    compiler_flags: Vec<String>,
    emit_asm_only: bool,
    emit_mir: bool,
    emit_mir_asm: Option<String>,
    target_arch: Option<String>,
    timeout_secs: Option<u64>,
    opt_level: u8,
}

fn parse_args() -> Result<CompileOptions, String> {
    let args: Vec<String> = env::args().collect();

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
        compiler_flags: Vec::new(),
        emit_asm_only: false,
        emit_mir: false,
        emit_mir_asm: None,
        target_arch: None,
        timeout_secs: None,
        opt_level: 1, // Default optimization level
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
            "-f" | "--flag" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for compiler flag".to_string());
                }
                options.compiler_flags.push(args[i + 1].clone());
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
                if i + 1 >= args.len() {
                    return Err("Missing argument for --emit-mir-asm".to_string());
                }
                options.emit_mir_asm = Some(args[i + 1].to_lowercase());
                i += 2;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for target architecture".to_string());
                }
                let target = args[i + 1].to_lowercase();

                if !lamina::HOST_ARCH_LIST.contains(&target.as_str()) {
                    return Err(format!(
                        "Unsupported target architecture: {}\nSupported values: \n{}",
                        target,
                        lamina::HOST_ARCH_LIST.join(",\n")
                    ));
                }
                options.target_arch = Some(target);
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
            "--timeout" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for --timeout".to_string());
                }
                let secs = args[i + 1]
                    .parse::<u64>()
                    .map_err(|_| "Invalid --timeout value (must be integer seconds)".to_string())?;
                options.timeout_secs = Some(secs);
                i += 2;
            }
            _ => {
                if options.input_file.as_os_str().is_empty() {
                    options.input_file = PathBuf::from(&args[i]);
                    i += 1;
                } else {
                    return Err(format!("Unknown argument: {}", args[i]));
                }
            }
        }
    }

    // Check if input file is specified
    if options.input_file.as_os_str().is_empty() {
        return Err("No input file specified".to_string());
    }

    Ok(options)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
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

    let input_path = &options.input_file;
    if !input_path.exists() {
        eprintln!(
            "Error: Input file '{}' does not exist.",
            input_path.display()
        );
        std::process::exit(1);
    }

    if input_path.extension().is_none_or(|ext| ext != "lamina") {
        eprintln!(
            "Warning: Input file '{}' does not have .lamina extension.",
            input_path.display()
        );
    }

    // Determine default executable extension based on platform
    // It's not used, but we keep it for reference and maybe in the future
    let _exe_extension = if cfg!(windows) { ".exe" } else { "" };

    // Get output name
    let output_stem = if let Some(out_path) = &options.output_file {
        out_path.clone()
    } else {
        // Default output name based on input file
        let stem = input_path.file_stem().map_or_else(
            || "a".into(), // "a.out" on Unix, "a.exe" on Windows
            PathBuf::from,
        );

        // If output name doesn't have an extension and we're on Windows, add .exe
        if cfg!(windows) && stem.extension().is_none() {
            let mut stem_with_ext = stem.clone();
            stem_with_ext.set_extension("exe");
            stem_with_ext
        } else {
            stem
        }
    };

    // Create assembly file path using same directory as output
    let asm_extension = if cfg!(windows) { "asm" } else { "s" };
    let mut asm_path = output_stem.clone();
    asm_path.set_extension(asm_extension);

    // Ensure output executable has correct extension
    let mut exec_path = output_stem.clone();
    if cfg!(windows) && exec_path.extension().is_none() {
        exec_path.set_extension("exe");
    }

    println!(
        "[INFO] Compiling {} -> {} -> {}",
        input_path.display(),
        asm_path.display(),
        exec_path.display()
    );

    if options.verbose {
        println!("[VERBOSE] Compiler options:");
        println!("  Verbose mode: {}", options.verbose);
        println!("  Optimization level: {}", options.opt_level);
        if let Some(compiler) = &options.forced_compiler {
            println!("  Forced compiler: {}", compiler);
        }
        if !options.compiler_flags.is_empty() {
            println!("  Additional compiler flags: {:?}", options.compiler_flags);
        }
    }

    // 1. Read Input Lamina IR
    let mut input_file = match File::open(input_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("[ERROR] Failed to open input file: {}", e);
            std::process::exit(1);
        }
    };

    let mut ir_source = String::new();
    if let Err(e) = input_file.read_to_string(&mut ir_source) {
        eprintln!("[ERROR] Failed to read input file: {}", e);
        std::process::exit(1);
    }

    // 2. Compile IR to Assembly
    let mut asm_buffer = Vec::<u8>::new();

    // 2-1 Optionally lower the IR to MIR and emit (.lumir) or experimental MIR->AArch64 asm
    if options.emit_mir || options.emit_mir_asm.is_some() {
        // Parse IR and lower to MIR, then exit early
        let ir_mod = lamina::parser::parse_module(&ir_source)
            .map_err(|e| format!("IR parse failed: {}", e))?;
        let mut mir_mod =
            lamina::mir::codegen::from_ir(&ir_mod, input_path.to_string_lossy().as_ref())
                .map_err(|e| format!("MIR lowering failed: {}", e))?;

        // Apply MIR optimizations
        if options.opt_level > 0 {
            let pipeline = lamina::mir::TransformPipeline::default_for_opt_level(options.opt_level);
            let transform_stats = pipeline
                .apply_to_module(&mut mir_mod)
                .map_err(|e| format!("MIR optimization failed: {}", e))?;

            // Apply function inlining at higher optimization levels
            let mut inlined_count = 0;
            if options.opt_level >= 3 {
                let inliner = lamina::mir::ModuleInlining::new();
                inlined_count = inliner
                    .inline_functions(&mut mir_mod)
                    .map_err(|e| format!("Function inlining failed: {}", e))?;
            }

            if options.verbose {
                println!(
                    "[VERBOSE] MIR optimizations: {} transforms run, {} made changes, {} functions inlined",
                    transform_stats.transforms_run,
                    transform_stats.transforms_changed,
                    inlined_count
                );
            }
        }
        if options.emit_mir {
            let mut mir_path = output_stem.clone();
            mir_path.set_extension("lumir");
            let mut mir_file =
                File::create(&mir_path).map_err(|e| format!("Failed to create MIR file: {}", e))?;
            let mir_text = format!("{}", mir_mod);
            mir_file
                .write_all(mir_text.as_bytes())
                .map_err(|e| format!("Failed to write MIR file: {}", e))?;
            println!("[INFO] MIR written to {}", mir_path.display());
        }

        if let Some(os) = &options.emit_mir_asm {
            let mut out_path = input_path.clone();
            out_path.set_extension("s");
            let mut out = Vec::<u8>::new();
            lamina::generate_mir_to_aarch64(&mir_mod, &mut out, os)
                .map_err(|e| format!("MIR→AArch64 emission failed: {}", e))?;
            File::create(&asm_path)
                .and_then(|mut f| f.write_all(&out))
                .map_err(|e| format!("Failed to write MIR asm: {}", e))?;
            println!(
                "[INFO] MIR AArch64 asm (experimental) written to {}",
                asm_path.display()
            );
        }

        // return Ok(());
    }
    // if mir asm is not emmit
    if !options.emit_mir_asm.is_some() && !options.emit_mir {
        // Choose compilation method based on target
        let compilation_result = if let Some(target) = &options.target_arch {
            if options.verbose {
                println!("[VERBOSE] Using explicit target architecture: {}", target);
            }
            lamina::compile_lamina_ir_to_target_assembly(&ir_source, &mut asm_buffer, target)
        } else {
            // Get the architecture that will be used by default (from the detect_host_architecture function)
            let default_arch = lamina::detect_host_architecture();
            if options.verbose {
                println!("[VERBOSE] Using host architecture: {}", default_arch);
            }
            lamina::compile_lamina_ir_to_assembly(&ir_source, &mut asm_buffer)
        };

        match compilation_result {
            Ok(_) => {
                println!(
                    "[INFO] Assembly generated successfully ({} bytes).",
                    asm_buffer.len()
                );
            }
            Err(e) => {
                eprintln!("\n[ERROR] Compilation failed: {}", e);
                eprintln!(
                    "[HINT] If you see dependency errors, make sure all required crates are in Cargo.toml"
                );
                std::process::exit(1);
            }
        }
        // 3. Write Assembly to file
        match File::create(&asm_path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(&asm_buffer) {
                    eprintln!("[ERROR] Failed to write assembly file: {}", e);
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("[ERROR] Failed to create assembly file: {}", e);
                std::process::exit(1);
            }
        };

        println!("[INFO] Assembly written to {}", asm_path.display());
    }
    // Skip compilation step if --emit-asm flag is present
    if options.emit_asm_only {
        println!("[INFO] Skipping compilation as requested (--emit-asm flag)");
        return Ok(());
    }

    // 4. Determine which compiler to use
    let (compiler_name, compiler_args) = match &options.forced_compiler {
        Some(forced) => {
            // Check if forced compiler exists
            if Command::new(forced).arg("--version").output().is_err() {
                eprintln!("[ERROR] Forced compiler '{}' not found", forced);
                std::process::exit(1);
            }
            // Get appropriate flags for forced compiler
            match forced.as_str() {
                "cl" => (forced.as_str(), vec!["/nologo".to_string()]),
                _ => (forced.as_str(), Vec::<String>::new()),
            }
        }
        None => match detect_compiler() {
            Ok((name, args)) => {
                // Convert &'static str to String instead of trying to collect to Vec<&str>
                let string_args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
                (name, string_args)
            }
            Err(e) => {
                eprintln!("[ERROR] {}", e);
                std::process::exit(1);
            }
        },
    };

    // 5. Assemble and Link
    println!("[INFO] Assembling and linking with {}...", compiler_name);

    // Build command with appropriate arguments
    let mut cmd = Command::new(compiler_name);

    // Add any compiler-specific flags first
    for arg in compiler_args {
        cmd.arg(arg);
    }

    // Add user-specified flags
    for flag in &options.compiler_flags {
        cmd.arg(flag);
    }

    // Add input and output files
    cmd.arg(&asm_path).arg("-o").arg(&exec_path);

    if options.verbose {
        println!("[VERBOSE] Executing: {:?}", cmd);
    }

    // Execute the compiler command
    let compiler_output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            eprintln!("[ERROR] Failed to execute compiler: {}", e);
            std::process::exit(1);
        }
    };

    if compiler_output.status.success() {
        println!(
            "[INFO] Executable '{}' created successfully.",
            exec_path.display()
        );
    } else {
        eprintln!("[ERROR] Compiler failed:");
        eprintln!("--- stdout ---");
        eprintln!("{}", String::from_utf8_lossy(&compiler_output.stdout));
        eprintln!("--- stderr ---");
        eprintln!("{}", String::from_utf8_lossy(&compiler_output.stderr));
        std::process::exit(1);
    }

    Ok(())
}

/// Detect available compiler and return its name and any needed flags
fn detect_compiler() -> Result<(&'static str, Vec<&'static str>), Box<dyn std::error::Error>> {
    // Check for available compilers based on platform
    if cfg!(windows) {
        // First try MSVC compiler
        if Command::new("cl").arg("/?").output().is_ok() {
            return Ok(("cl", vec!["/nologo"]));
        }

        // Then try GCC in MinGW
        if Command::new("gcc").arg("--version").output().is_ok() {
            return Ok(("gcc", vec![]));
        }

        // Then try Clang
        if Command::new("clang").arg("--version").output().is_ok() {
            return Ok(("clang", vec![]));
        }

        eprintln!("No suitable compiler found on Windows. Please install GCC, Clang, or MSVC.");
    } else if cfg!(target_os = "macos") {
        // On macOS, prefer Clang
        if Command::new("clang").arg("--version").output().is_ok() {
            return Ok(("clang", vec![]));
        }

        // Then try GCC
        if Command::new("gcc").arg("--version").output().is_ok() {
            return Ok(("gcc", vec![]));
        }

        eprintln!("No suitable compiler found on macOS. Please install Clang.");
    } else {
        // On Unix-like systems, prefer GCC, fallback to Clang
        if Command::new("gcc").arg("--version").output().is_ok() {
            return Ok(("gcc", vec![]));
        }

        if Command::new("clang").arg("--version").output().is_ok() {
            return Ok(("clang", vec![]));
        }

        eprintln!("No suitable compiler found. Please install GCC or Clang.");
    }

    Err("No suitable compiler found. Please install GCC, Clang, or MSVC (on Windows).".into())
}
