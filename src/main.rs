mod cli;

use cli::jit::handle_jit_compilation;
use cli::options::{parse_args, print_usage, toolchain_backends};
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

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

    let input_path = &options.input_file;
    if !input_path.exists() {
        eprintln!(
            "Error: Input file '{}' does not exist.",
            input_path.display()
        );
        process::exit(1);
    }

    if input_path.extension().is_none_or(|ext| ext != "lamina") {
        eprintln!(
            "Warning: Input file '{}' does not have .lamina extension.",
            input_path.display()
        );
    }

    let _exe_extension = if cfg!(windows) { ".exe" } else { "" };

    let target_for_extensions = if let Some(target_str) = &options.target_arch {
        lamina_platform::Target::from_str(target_str)
            .unwrap_or_else(|_| lamina_platform::Target::detect_host())
    } else {
        lamina_platform::Target::detect_host()
    };

    let output_stem = if let Some(out_path) = &options.output_file {
        out_path.clone()
    } else {
        let stem = input_path
            .file_stem()
            .map_or_else(|| "a".into(), PathBuf::from);
        if target_for_extensions.operating_system == lamina_platform::TargetOperatingSystem::Windows
            && stem.extension().is_none()
        {
            let mut stem_with_ext = stem;
            stem_with_ext.set_extension("exe");
            stem_with_ext
        } else {
            stem
        }
    };

    let codegen_units = options.codegen_units.unwrap_or_else(|| {
        let max_threads = lamina_platform::cpu_count();
        if max_threads > 2 { max_threads - 2 } else { 1 }
    });

    if options.verbose {
        println!("[VERBOSE] Compiler options:");
        println!("  Verbose mode: {}", options.verbose);
        println!("  Optimization level: {}", options.opt_level);
        println!("  Codegen units (parallel threads): {codegen_units}");
        if let Some(compiler) = &options.forced_compiler {
            println!("  Forced compiler: {compiler}");
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

    let mut input_file = match File::open(input_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("[ERROR] Failed to open input file: {e}");
            process::exit(1);
        }
    };

    let mut ir_source = String::new();
    if let Err(e) = input_file.read_to_string(&mut ir_source) {
        eprintln!("[ERROR] Failed to read input file: {e}");
        process::exit(1);
    }

    if options.emit_mir || options.emit_mir_asm.is_some() {
        let ir_mod = lamina::parser::parse_module(&ir_source)
            .map_err(|e| format!("IR parse failed: {e}"))?;
        let mut mir_mod =
            lamina::mir::codegen::from_ir(&ir_mod, input_path.to_string_lossy().as_ref())
                .map_err(|e| format!("MIR lowering failed: {e}"))?;

        if options.opt_level > 0 {
            let pipeline = lamina::mir::TransformPipeline::default_for_opt_level(options.opt_level);
            let transform_stats = pipeline
                .apply_to_module(&mut mir_mod)
                .map_err(|e| format!("MIR optimization failed: {e}"))?;

            let mut inlined_count = 0;
            if options.opt_level >= 2 {
                let inliner = lamina::mir::ModuleInlining::new();
                if let Ok(count) = inliner.inline_functions(&mut mir_mod) {
                    inlined_count = count;
                }
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
                File::create(&mir_path).map_err(|e| format!("Failed to create MIR file: {e}"))?;
            let mir_text = format!("{mir_mod}");
            mir_file
                .write_all(mir_text.as_bytes())
                .map_err(|e| format!("Failed to write MIR file: {e}"))?;
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

            let asm_extension = match target.architecture {
                lamina_platform::TargetArchitecture::Wasm32
                | lamina_platform::TargetArchitecture::Wasm64 => "wat",
                _ => {
                    if target.operating_system == lamina_platform::TargetOperatingSystem::Windows {
                        "asm"
                    } else {
                        "s"
                    }
                }
            };
            let mut mir_asm_path = output_stem;
            mir_asm_path.set_extension(asm_extension);

            let mut out = Vec::<u8>::new();
            let target_os = target.operating_system;

            lamina::mir_codegen::generate_mir_to_target_with_settings(
                &mir_mod,
                &mut out,
                target.architecture,
                target_os,
                codegen_units,
                &options.mir_codegen_settings,
            )
            .map_err(|e| format!("MIR→{} emission failed: {}", target.architecture, e))?;

            let arch_name = match target.architecture {
                lamina_platform::TargetArchitecture::X86_64 => "x86_64",
                lamina_platform::TargetArchitecture::Wasm32
                | lamina_platform::TargetArchitecture::Wasm64 => "WASM",
                lamina_platform::TargetArchitecture::Aarch64 => "AArch64",
                lamina_platform::TargetArchitecture::Arx64 => "ARX64",
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
                .map_err(|e| format!("Failed to write MIR output: {e}"))?;
        }

        return Ok(());
    }

    if options.jit {
        return Ok(handle_jit_compilation(&ir_source, input_path, &options)?);
    }

    if !options.emit_mir {
        let target = if let Some(target_str) = &options.target_arch {
            if options.verbose {
                println!("[VERBOSE] Using explicit target: {target_str}");
            }
            lamina_platform::Target::from_str(target_str).unwrap_or_else(|e| {
                eprintln!("Warning: Invalid target '{target_str}': {e}. Using host target.");
                lamina_platform::Target::detect_host()
            })
        } else {
            let default_target = lamina_platform::Target::detect_host();
            if options.verbose {
                println!("[VERBOSE] Using host target: {default_target}");
            }
            default_target
        };

        let ir_mod = lamina::parser::parse_module(&ir_source)
            .map_err(|e| format!("IR parse failed: {e}"))?;
        let mut mir_mod =
            lamina::mir::codegen::from_ir(&ir_mod, input_path.to_string_lossy().as_ref())
                .map_err(|e| format!("MIR lowering failed: {e}"))?;

        if options.opt_level > 0 {
            let pipeline = lamina::mir::TransformPipeline::default_for_opt_level(options.opt_level);
            let transform_stats = pipeline
                .apply_to_module(&mut mir_mod)
                .map_err(|e| format!("MIR optimization failed: {e}"))?;

            if options.verbose {
                println!(
                    "[VERBOSE] MIR optimizations: {} transforms run, {} made changes",
                    transform_stats.transforms_run, transform_stats.transforms_changed
                );
            }
        }

        let mut intermediate_buffer = Vec::<u8>::new();
        lamina::mir_codegen::generate_mir_to_target_with_settings(
            &mir_mod,
            &mut intermediate_buffer,
            target.architecture,
            target.operating_system,
            codegen_units,
            &options.mir_codegen_settings,
        )
        .map_err(|e| format!("Code generation failed: {e}"))?;

        let intermediate_ext = lamina::mir_codegen::assemble::get_intermediate_extension(
            target.architecture,
            target.operating_system,
        );
        let mut intermediate_path = output_stem.clone();
        intermediate_path.set_extension(intermediate_ext);

        let final_ext = lamina::mir_codegen::link::get_output_extension(
            target.architecture,
            target.operating_system,
        );
        let mut final_output_display = output_stem.clone();
        if !final_ext.is_empty() {
            final_output_display.set_extension(final_ext);
        }

        let intermediate_name = if matches!(
            target.architecture,
            lamina_platform::TargetArchitecture::Wasm32
                | lamina_platform::TargetArchitecture::Wasm64
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

        File::create(&intermediate_path)
            .and_then(|mut f| f.write_all(&intermediate_buffer))
            .map_err(|e| format!("Failed to write intermediate file: {e}"))?;

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

        if options.emit_asm_only {
            println!("[INFO] Skipping assembly and linking as requested (--emit-asm flag)");
            return Ok(());
        }

        let assembly_output_ext =
            lamina::mir_codegen::assemble::get_assembly_output_extension(target.architecture);
        let mut assembly_output_path = output_stem.clone();
        assembly_output_path.set_extension(assembly_output_ext);

        let (assembler_backend, linker_backend) =
            toolchain_backends(&options.forced_compiler, &options.assembler);
        let assemble_result = lamina::mir_codegen::assemble::assemble_with_ras_object_options(
            &intermediate_path,
            &assembly_output_path,
            target.architecture,
            target.operating_system,
            assembler_backend,
            &options.assembler_flags,
            options.verbose,
            options.ras_object_write_options(target.operating_system),
        )
        .map_err(|e| format!("Assembly failed: {e}"))?;

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

        if assemble_result.needs_linking {
            let final_output_ext = lamina::mir_codegen::link::get_output_extension(
                target.architecture,
                target.operating_system,
            );
            let mut final_output_path = output_stem;
            if !final_output_ext.is_empty() {
                final_output_path.set_extension(final_output_ext);
            }

            lamina::mir_codegen::link::link(
                &assemble_result.output_path,
                &final_output_path,
                target.architecture,
                target.operating_system,
                linker_backend,
                &options.linker_flags,
                options.verbose,
            )
            .map_err(|e| format!("Linking failed: {e}"))?;

            println!(
                "[INFO] Executable '{}' created successfully.",
                final_output_path.display()
            );
        } else {
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
