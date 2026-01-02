use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

fn print_usage() {
    eprintln!("Usage: lamina <input.lamina> [options]");
    eprintln!("Options:");
    eprintln!("  -o, --output <file>     Specify output executable name");
    eprintln!("  -v, --verbose           Enable verbose output");
    eprintln!("  -c, --compiler <n>      Force specific compiler (gcc, clang, cl)");
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
    eprintln!("  -h, --help              Display this help message");
}

struct CompileOptions {
    input_file: PathBuf,
    output_file: Option<PathBuf>,
    verbose: bool,
    forced_compiler: Option<String>,
    assembler_flags: Vec<String>,
    linker_flags: Vec<String>,
    emit_asm_only: bool,
    emit_mir: bool,
    emit_mir_asm: Option<String>,
    target_arch: Option<String>,
    opt_level: u8,
    jit: bool,
    sandbox: bool,
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
        assembler_flags: Vec::new(),
        linker_flags: Vec::new(),
        emit_asm_only: false,
        emit_mir: false,
        emit_mir_asm: None,
        target_arch: None,
        opt_level: 1,
        jit: false,
        sandbox: false,
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
            "--version" => {
                println!("lamina {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ => {
                if args[i].starts_with("-Wl,") {
                    let flags = args[i][4..].split(',');
                    options.linker_flags.extend(flags.map(|s| s.to_string()));
                    i += 1;
                } else if args[i].starts_with("-Wa,") {
                    let flags = args[i][4..].split(',');
                    options.assembler_flags.extend(flags.map(|s| s.to_string()));
                    i += 1;
                } else if args[i] == "-Wl" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wl".to_string());
                    }
                    let flags = args[i + 1].split(',');
                    options.linker_flags.extend(flags.map(|s| s.to_string()));
                    i += 2;
                } else if args[i] == "-Wa" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wa".to_string());
                    }
                    let flags = args[i + 1].split(',');
                    options.assembler_flags.extend(flags.map(|s| s.to_string()));
                    i += 2;
                } else if options.input_file.as_os_str().is_empty() {
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

    // Determine target for extension logic
    let target_for_extensions = if let Some(target_str) = &options.target_arch {
        lamina_platform::Target::from_str(target_str)
            .unwrap_or_else(|_| lamina_platform::Target::detect_host())
    } else {
        lamina_platform::Target::detect_host()
    };

    // Get output name
    let output_stem = if let Some(out_path) = &options.output_file {
        out_path.clone()
    } else {
        // Default output name based on input file
        let stem = input_path.file_stem().map_or_else(
            || "a".into(), // "a.out" on Unix, "a.exe" on Windows
            PathBuf::from,
        );

        // If output name doesn't have an extension and target is Windows, add .exe
        if target_for_extensions.operating_system == lamina_platform::TargetOperatingSystem::Windows
            && stem.extension().is_none()
        {
            let mut stem_with_ext = stem.clone();
            stem_with_ext.set_extension("exe");
            stem_with_ext
        } else {
            stem
        }
    };

    // Output message will be printed after we determine the actual target
    // (handled in the compilation section below)

    if options.verbose {
        println!("[VERBOSE] Compiler options:");
        println!("  Verbose mode: {}", options.verbose);
        println!("  Optimization level: {}", options.opt_level);
        if let Some(compiler) = &options.forced_compiler {
            println!("  Forced compiler: {}", compiler);
        }
        if !options.assembler_flags.is_empty() {
            println!(
                "  Additional assembler flags: {:?}",
                options.assembler_flags
            );
        }
        if !options.linker_flags.is_empty() {
            println!("  Additional linker flags: {:?}", options.linker_flags);
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

    // 2. Compile IR to Assembly/Binary (handled below)

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
            // ModuleInlining disabled - complex recursive functions cause control flow issues
            let inlined_count = 0;
            // if options.opt_level >= 3 {
            //     let inliner = lamina::mir::ModuleInlining::new();
            //     inlined_count = inliner
            //         .inline_functions(&mut mir_mod)
            //         .map_err(|e| format!("Function inlining failed: {}", e))?;
            // }

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

        if options.emit_mir_asm.is_some() {
            let default_target = lamina_platform::Target::detect_host();
            let default_target_str = default_target.to_str();
            let target_str = options
                .target_arch
                .as_deref()
                .unwrap_or(&default_target_str);
            let target = lamina_platform::Target::from_str(target_str)
                .unwrap_or_else(|_| lamina_platform::Target::detect_host());

            // Determine output extension based on target
            let asm_extension = match target.architecture {
                lamina_platform::TargetArchitecture::Wasm32
                | lamina_platform::TargetArchitecture::Wasm64 => "wat", // WebAssembly text format
                _ => {
                    if target.operating_system == lamina_platform::TargetOperatingSystem::Windows {
                        "asm"
                    } else {
                        "s"
                    }
                }
            };
            let mut mir_asm_path = output_stem.clone();
            mir_asm_path.set_extension(asm_extension);

            let mut out = Vec::<u8>::new();

            // Use OS from the target
            let target_os = target.operating_system;

            lamina::mir_codegen::generate_mir_to_target(
                &mir_mod,
                &mut out,
                target.architecture,
                target_os,
            )
            .map_err(|e| format!("MIR→{} emission failed: {}", target.architecture, e))?;

            let arch_name = match target.architecture {
                lamina_platform::TargetArchitecture::X86_64 => "x86_64",
                lamina_platform::TargetArchitecture::Wasm32
                | lamina_platform::TargetArchitecture::Wasm64 => "WASM",
                lamina_platform::TargetArchitecture::Aarch64 => "AArch64",
                lamina_platform::TargetArchitecture::Riscv32 => "RISC-V32",
                lamina_platform::TargetArchitecture::Riscv64 => "RISC-V64",
                #[cfg(feature = "nightly")]
                lamina_platform::TargetArchitecture::Riscv128 => "RISC-V128",
                _ => "unknown",
            };

            println!(
                "[INFO] MIR {} asm (experimental) written to {}",
                arch_name,
                mir_asm_path.display()
            );

            File::create(&mir_asm_path)
                .and_then(|mut f| f.write_all(&out))
                .map_err(|e| format!("Failed to write MIR output: {}", e))?;
        }
    }
    // Handle JIT compilation mode
    if options.jit {
        return handle_jit_compilation(&ir_source, input_path, &options);
    }

    // Always compile to binary unless only MIR is requested
    if !options.emit_mir {
        // Determine target
        let target = if let Some(target_str) = &options.target_arch {
            if options.verbose {
                println!("[VERBOSE] Using explicit target: {}", target_str);
            }
            lamina_platform::Target::from_str(target_str).unwrap_or_else(|e| {
                eprintln!(
                    "Warning: Invalid target '{}': {}. Using host target.",
                    target_str, e
                );
                lamina_platform::Target::detect_host()
            })
        } else {
            let default_target = lamina_platform::Target::detect_host();
            if options.verbose {
                println!("[VERBOSE] Using host target: {}", default_target);
            }
            default_target
        };

        // Parse IR and lower to MIR
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

            if options.verbose {
                println!(
                    "[VERBOSE] MIR optimizations: {} transforms run, {} made changes",
                    transform_stats.transforms_run, transform_stats.transforms_changed
                );
            }
        }

        // Generate intermediate format (assembly or WAT)
        let mut intermediate_buffer = Vec::<u8>::new();
        lamina::mir_codegen::generate_mir_to_target(
            &mir_mod,
            &mut intermediate_buffer,
            target.architecture,
            target.operating_system,
        )
        .map_err(|e| format!("Code generation failed: {}", e))?;

        // Determine intermediate file path and extension
        let intermediate_ext =
            lamina::mir_codegen::assemble::get_intermediate_extension(target.architecture);
        let mut intermediate_path = output_stem.clone();
        intermediate_path.set_extension(intermediate_ext);

        // Determine final output path for display
        let final_ext = lamina::mir_codegen::link::get_output_extension(
            target.architecture,
            target.operating_system,
        );
        let mut final_output_display = output_stem.clone();
        if !final_ext.is_empty() {
            final_output_display.set_extension(final_ext);
        }

        // Print compilation message
        let intermediate_name = if matches!(
            target.architecture,
            lamina_platform::TargetArchitecture::Wasm32 | lamina_platform::TargetArchitecture::Wasm64
        ) {
            println!(
                "[INFO] Compiling {} -> {}",
                input_path.display(),
                final_output_display.display()
            );
            "WAT"
        } else {
            println!(
                "[INFO] Compiling {} -> {} -> {}",
                input_path.display(),
                intermediate_path.display(),
                final_output_display.display()
            );
            "Assembly"
        };

        // Write intermediate format to file
        File::create(&intermediate_path)
            .and_then(|mut f| f.write_all(&intermediate_buffer))
            .map_err(|e| format!("Failed to write intermediate file: {}", e))?;

        println!(
            "[INFO] {} generated successfully ({} bytes).",
            intermediate_name,
            intermediate_buffer.len()
        );
        println!(
            "[INFO] {} written to {}",
            intermediate_name,
            intermediate_path.display()
        );

        // Skip assembly/linking if --emit-asm flag is present
        if options.emit_asm_only {
            println!("[INFO] Skipping assembly and linking as requested (--emit-asm flag)");
            return Ok(());
        }

        // Assemble intermediate format to binary/object file
        let assembly_output_ext =
            lamina::mir_codegen::assemble::get_assembly_output_extension(target.architecture);
        let mut assembly_output_path = output_stem.clone();
        assembly_output_path.set_extension(assembly_output_ext);

        let assemble_result = lamina::mir_codegen::assemble::assemble(
            &intermediate_path,
            &assembly_output_path,
            target.architecture,
            target.operating_system,
            None,
            &options.assembler_flags,
            options.verbose,
        )
        .map_err(|e| format!("Assembly failed: {}", e))?;

        println!(
            "[INFO] {} assembled successfully.",
            if matches!(
                target.architecture,
                lamina_platform::TargetArchitecture::Wasm32
                    | lamina_platform::TargetArchitecture::Wasm64
            ) {
                "WASM binary"
            } else {
                "Object file"
            }
        );

        // Link if needed (native targets only)
        if assemble_result.needs_linking {
            let final_output_ext = lamina::mir_codegen::link::get_output_extension(
                target.architecture,
                target.operating_system,
            );
            let mut final_output_path = output_stem.clone();
            if !final_output_ext.is_empty() {
                final_output_path.set_extension(final_output_ext);
            }

            // Determine linker backend from forced compiler if specified
            let linker_backend = if let Some(ref compiler) = options.forced_compiler {
                match compiler.as_str() {
                    "ld" => Some(lamina::mir_codegen::link::LinkerBackend::Ld),
                    "lld" => Some(lamina::mir_codegen::link::LinkerBackend::Lld),
                    "mold" => Some(lamina::mir_codegen::link::LinkerBackend::Mold),
                    "link" | "cl" => Some(lamina::mir_codegen::link::LinkerBackend::Msvc),
                    _ => None,
                }
            } else {
                None
            };

            lamina::mir_codegen::link::link(
                &assemble_result.output_path,
                &final_output_path,
                target.architecture,
                target.operating_system,
                linker_backend,
                &options.linker_flags,
                options.verbose,
            )
            .map_err(|e| format!("Linking failed: {}", e))?;

            println!(
                "[INFO] Executable '{}' created successfully.",
                final_output_path.display()
            );
        } else {
            // WASM doesn't need linking
            println!(
                "[INFO] {} '{}' created successfully.",
                if matches!(
                    target.architecture,
                    lamina_platform::TargetArchitecture::Wasm32
                        | lamina_platform::TargetArchitecture::Wasm64
                ) {
                    "WASM binary"
                } else {
                    "Binary"
                },
                assembly_output_path.display()
            );
        }
    }

    Ok(())
}

/// Handle JIT compilation and execution
fn handle_jit_compilation(
    ir_source: &str,
    input_path: &std::path::Path,
    options: &CompileOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    use lamina_platform::Target;
    use std::str::FromStr;

    // Determine target
    let target = if let Some(target_str) = &options.target_arch {
        Target::from_str(target_str)
            .map_err(|e| format!("Invalid target '{}': {}", target_str, e))?
    } else {
        Target::detect_host()
    };

    if options.verbose {
        println!("[JIT] Compiling {} for runtime execution", input_path.display());
        println!("[JIT] Target: {}", target);
        if options.sandbox {
            println!("[JIT] Sandbox: enabled");
        }
    }

    // Parse IR and lower to MIR
    let ir_mod = lamina::parser::parse_module(ir_source)
        .map_err(|e| format!("IR parse failed: {}", e))?;
    let mut mir_mod = lamina::mir::codegen::from_ir(&ir_mod, input_path.to_string_lossy().as_ref())
        .map_err(|e| format!("MIR lowering failed: {}", e))?;

    // Apply MIR optimizations
    if options.opt_level > 0 {
        let pipeline = lamina::mir::TransformPipeline::default_for_opt_level(options.opt_level);
        let transform_stats = pipeline
            .apply_to_module(&mut mir_mod)
            .map_err(|e| format!("MIR optimization failed: {}", e))?;

        if options.verbose {
            println!(
                "[JIT] MIR optimizations: {} transforms run, {} made changes",
                transform_stats.transforms_run, transform_stats.transforms_changed
            );
        }
    }

    // Check if target architecture is supported for JIT
    match target.architecture {
        lamina_platform::TargetArchitecture::X86_64 => {
            // Supported - continue
        }
        _ => {
            return Err(format!(
                "JIT compilation is not yet supported for architecture {:?}.\n\
                 Currently only x86_64 is supported for JIT compilation.\n\
                 You can:\n\
                 - Use AOT compilation instead (remove --jit flag)\n\
                 - Cross-compile to x86_64 using --target x86_64_{}",
                target.architecture,
                target.operating_system
            ).into());
        }
    }

    // Compile using runtime compilation
    if options.sandbox {
        // Use sandbox for secure execution
        use lamina::runtime::{Sandbox, SandboxConfig};
        
        let config = if options.verbose {
            SandboxConfig::default()
        } else {
            SandboxConfig::restrictive()
        };
        
        let mut sandbox = Sandbox::new(target.architecture, target.operating_system, config);
        
        // Find main function or use first function
        let function_name = mir_mod.functions.keys().next()
            .ok_or("No functions found in module")?;
        
        if options.verbose {
            println!("[JIT] Executing function '{}' in sandbox", function_name);
        }
        
        // Execute in sandbox (placeholder - returns default value for now)
        let _result: i64 = sandbox.execute(&mir_mod, function_name)
            .map_err(|e| format!("Sandbox execution failed: {}", e))?;
        
        if options.verbose {
            println!("[JIT] Execution completed successfully");
        }
    } else {
        // Direct runtime compilation (no sandbox)
        use lamina::runtime::compile_to_runtime;
        
        let runtime_result = compile_to_runtime(&mir_mod, target.architecture, target.operating_system)
            .map_err(|e| format!("Runtime compilation failed: {}", e))?;
        
        if options.verbose {
            println!("[JIT] Code compiled to executable memory");
            println!("[JIT] Function pointer: {:p}", runtime_result.function_ptr);
        }
        
        // Note: To actually call the function, we need to know its signature
        // This is a placeholder - real implementation would:
        // 1. Parse function signature from MIR
        // 2. Create appropriate function pointer type
        // 3. Call the function
        // 4. Print or return the result
        
        println!("[JIT] Runtime compilation successful (function not executed - signature unknown)");
    }

    Ok(())
}
