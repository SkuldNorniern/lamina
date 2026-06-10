//! JIT compilation and execution pipeline.

use crate::cli::options::{CompileOptions, toolchain_backends};

use lamina::runtime::{Sandbox, SandboxConfig, compile_to_runtime};
use lamina_platform::Target;

use std::process::Command;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

/// Handle JIT compilation and execution
pub fn handle_jit_compilation(
    ir_source: &str,
    input_path: &std::path::Path,
    options: &CompileOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let target = if let Some(target_str) = &options.target_arch {
        Target::from_str(target_str).map_err(|e| format!("Invalid target '{target_str}': {e}"))?
    } else {
        Target::detect_host()
    };

    if options.verbose {
        println!(
            "[JIT] Compiling {} for runtime execution",
            input_path.display()
        );
        println!("[JIT] Target: {target}");
        if options.sandbox {
            println!("[JIT] Sandbox: enabled");
        }
    }

    let ir_mod =
        lamina::parser::parse_module(ir_source).map_err(|e| format!("IR parse failed: {e}"))?;
    let mut mir_mod = lamina::mir::codegen::from_ir(&ir_mod, input_path.to_string_lossy().as_ref())
        .map_err(|e| format!("MIR lowering failed: {e}"))?;

    if options.opt_level > 0 {
        let pipeline = lamina::mir::TransformPipeline::default_for_opt_level(options.opt_level);
        let transform_stats = pipeline
            .apply_to_module(&mut mir_mod)
            .map_err(|e| format!("MIR optimization failed: {e}"))?;

        if options.verbose {
            println!(
                "[JIT] MIR optimizations: {} transforms run, {} made changes",
                transform_stats.transforms_run, transform_stats.transforms_changed
            );
        }
    }

    if options.sandbox {
        let config = if options.verbose {
            SandboxConfig::default()
        } else {
            SandboxConfig::restrictive()
        };

        let mut sandbox = Sandbox::new(target.architecture, target.operating_system, config);

        let function_name = if mir_mod.functions.contains_key("main") {
            "main"
        } else {
            mir_mod
                .functions
                .keys()
                .next()
                .ok_or("No functions found in module")?
                .as_str()
        };

        if options.verbose {
            println!("[JIT] Executing function '{function_name}' in sandbox");
        }

        let result: i64 = sandbox
            .execute_i64(&mir_mod, function_name)
            .map_err(|e| format!("Sandbox execution failed: {e}"))?;

        if options.verbose {
            println!("[JIT] Execution completed successfully; return={result}");
        }

        if function_name == "main" {
            std::process::exit(result as i32);
        }
    } else {
        let codegen_units = options.codegen_units.unwrap_or_else(|| {
            let max_threads = lamina_platform::cpu_count();
            if max_threads > 2 { max_threads - 2 } else { 1 }
        });

        let jit_function_name = if mir_mod.functions.contains_key("main") {
            "main"
        } else {
            mir_mod
                .functions
                .keys()
                .find(|name| name.contains("main") || name.contains("matmul"))
                .map(String::as_str)
                .or_else(|| mir_mod.functions.keys().next().map(String::as_str))
                .ok_or("No functions found in module")?
        };

        let func = mir_mod
            .functions
            .get(jit_function_name)
            .ok_or("Function not found in module")?;

        let runtime_result = compile_to_runtime(
            &mir_mod,
            target.architecture,
            target.operating_system,
            Some(jit_function_name),
        )
        .map_err(|e| format!("Runtime compilation failed: {e}"));

        let runtime_result = match runtime_result {
            Ok(result) => Some(result),
            Err(err) => {
                if options.verbose {
                    eprintln!(
                        "[JIT] In-memory compilation failed; falling back to AOT execution.\n  Reason: {err}"
                    );
                }
                None
            }
        };

        if let Some(runtime_result) = runtime_result {
            if options.verbose {
                println!("[JIT] Code compiled to executable memory");
                println!("[JIT] Function pointer: {:p}", runtime_result.function_ptr);
                println!("[JIT] Executing function '{jit_function_name}'");
                println!(
                    "[JIT] Signature: {} params, return type: {:?}",
                    func.sig.params.len(),
                    func.sig.ret_ty
                );
                println!(
                    "[JIT] All functions in module: {:?}",
                    mir_mod.function_names()
                );
            }

            unsafe {
                let result = lamina::runtime::execute_jit_function(
                    &func.sig,
                    runtime_result.function_ptr,
                    None,
                    options.verbose,
                    Some(func),
                )?;
                if let Some(ret) = result {
                    if options.verbose {
                        println!("[JIT] Function returned {ret}");
                    }
                    if jit_function_name == "main" {
                        std::process::exit(ret as i32);
                    }
                } else if jit_function_name == "main" {
                    std::process::exit(0);
                }
            }
        } else {
            // AOT fallback: compile to a temp executable and run it.
            if matches!(
                target.architecture,
                lamina_platform::TargetArchitecture::Wasm32
                    | lamina_platform::TargetArchitecture::Wasm64
            ) {
                return Err("JIT fallback: cannot execute WASM targets directly".into());
            }

            let pid = std::process::id();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let tmp_dir = std::env::temp_dir().join(format!("lamina_jit_{pid}_{nanos}"));
            std::fs::create_dir_all(&tmp_dir)?;

            let intermediate_ext = lamina::mir_codegen::assemble::get_intermediate_extension(
                target.architecture,
                target.operating_system,
            );
            let assembly_output_ext =
                lamina::mir_codegen::assemble::get_assembly_output_extension(target.architecture);
            let final_ext = lamina::mir_codegen::link::get_output_extension(
                target.architecture,
                target.operating_system,
            );

            let mut intermediate_path = tmp_dir.join("module");
            intermediate_path.set_extension(intermediate_ext);
            let mut object_path = tmp_dir.join("module");
            object_path.set_extension(assembly_output_ext);
            let mut exe_path = tmp_dir.join("module");
            if !final_ext.is_empty() {
                exe_path.set_extension(final_ext);
            }

            let mut intermediate = Vec::<u8>::new();
            lamina::mir_codegen::generate_mir_to_target_with_settings(
                &mir_mod,
                &mut intermediate,
                target.architecture,
                target.operating_system,
                codegen_units,
                &options.mir_codegen_settings,
            )
            .map_err(|e| format!("JIT fallback: codegen failed: {e}"))?;

            std::fs::write(&intermediate_path, &intermediate)?;

            let (assembler_backend, linker_backend) =
                toolchain_backends(&options.forced_compiler, &options.assembler);
            let assemble_result = lamina::mir_codegen::assemble::assemble_with_ras_object_options(
                &intermediate_path,
                &object_path,
                target.architecture,
                target.operating_system,
                assembler_backend,
                &options.assembler_flags,
                options.verbose,
                options.ras_object_write_options(target.operating_system),
            )
            .map_err(|e| format!("JIT fallback: assembly failed: {e}"))?;

            if assemble_result.needs_linking {
                lamina::mir_codegen::link::link(
                    &assemble_result.output_path,
                    &exe_path,
                    target.architecture,
                    target.operating_system,
                    linker_backend,
                    &options.linker_flags,
                    options.verbose,
                )
                .map_err(|e| format!("JIT fallback: linking failed: {e}"))?;
            } else {
                exe_path = assemble_result.output_path;
            }

            if options.verbose {
                println!("[JIT] Running {}", exe_path.display());
            }
            let status = Command::new(&exe_path).status()?;
            if !status.success() {
                if let Some(code) = status.code() {
                    if options.verbose {
                        eprintln!("[JIT] Program exited with status {code}");
                    }
                    std::process::exit(code);
                }
                return Err(format!("JIT fallback: program terminated: {status}").into());
            }

            let _ = std::fs::remove_dir_all(&tmp_dir);
        }
    }

    Ok(())
}
