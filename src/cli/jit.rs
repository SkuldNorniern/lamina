//! JIT compilation and execution pipeline.

use crate::cli::host_cpu::check_host_runs_level;
use crate::cli::options::{CompileOptions, toolchain_backends};
use crate::mir::transform::vector_unit_for;

use crate::{
    mir::{Module, TransformPipeline, codegen::from_ir_for_target},
    mir_codegen::{
        assemble::{
            assemble_with_ras_object_options, assembler_input_extension, assembler_output_extension,
        },
        generate_mir_to_target_with_settings,
        link::{get_output_extension, link_with_runtime},
        runtime_builtins,
    },
    parser::parse_module,
    runtime::{Sandbox, SandboxConfig, compile_to_runtime, execute_jit_function},
};
use lamina_platform::{Target, TargetArchitecture, cpu_count};

use crate::LaminaError;
use crate::cache::lamina_cache_dir;
use std::{
    fs::{create_dir_all, remove_dir_all, write},
    path::Path,
    process::{self, Command},
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

/// The function `--jit` runs: `@main`, or the module's only function.
///
/// A module with neither is refused by name rather than run by map order.
fn jit_entry_point(module: &Module) -> Result<&str, LaminaError> {
    if module.functions.contains_key("main") {
        return Ok("main");
    }
    let mut names: Vec<&str> = module.functions.keys().map(String::as_str).collect();
    names.sort_unstable();
    match names.as_slice() {
        [] => Err(LaminaError::ValidationError(
            "No functions found in module".to_owned(),
        )),
        [only] => Ok(only),
        _ => Err(LaminaError::ValidationError(format!(
            "--jit has no entry point to run: the module declares no @main, and its {} functions ({}) give no way to choose one",
            names.len(),
            names.join(", ")
        ))),
    }
}

/// Handle JIT compilation and execution
pub fn handle_jit_compilation(
    ir_source: &str,
    input_path: &Path,
    options: &CompileOptions,
) -> Result<(), LaminaError> {
    let target = if let Some(target_str) = &options.target_arch {
        Target::from_str(target_str).map_err(|e| {
            LaminaError::ValidationError(format!("Invalid target '{target_str}': {e}"))
        })?
    } else {
        Target::detect_host().map_err(LaminaError::ValidationError)?
    };

    let host = Target::detect_host().map_err(LaminaError::ValidationError)?;
    if target.architecture != host.architecture || target.operating_system != host.operating_system
    {
        return Err(LaminaError::ValidationError(format!(
            "--jit cannot run code for {target}: JIT executes on the host, which is {host}. Drop --jit to cross-compile for {target}"
        )));
    }

    if target.architecture == TargetArchitecture::X86_64 {
        check_host_runs_level(
            options.mir_codegen_settings.x86_cpu_level(),
            "--jit",
            "pick a lower --cpu-level",
        )?;
    }

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

    let ir_mod = parse_module(ir_source)
        .map_err(|e| LaminaError::ParsingError(format!("IR parse failed: {e}")))?;
    let mut mir_mod = from_ir_for_target(
        &ir_mod,
        input_path.to_string_lossy().as_ref(),
        target.data_layout(),
    )
    .map_err(|e| LaminaError::MirError(format!("MIR lowering failed: {e}")))?;

    if let Some(name) = mir_mod
        .functions
        .iter()
        .filter(|(_, function)| function.sig.kernel.is_some())
        .map(|(name, _)| name)
        .min()
    {
        return Err(LaminaError::ValidationError(format!(
            "@kernel function '{name}' requires a GPU target, which --jit cannot be: JIT executes on the host"
        )));
    }

    if options.opt_level > 0 {
        let pipeline = TransformPipeline::default_for_opt_level_on(
            options.opt_level,
            vector_unit_for(
                target.architecture,
                options.mir_codegen_settings.x86_cpu_level(),
            ),
        );
        let transform_stats = pipeline
            .apply_to_module(&mut mir_mod)
            .map_err(|e| LaminaError::MirError(format!("MIR optimization failed: {e}")))?;

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

        let function_name = jit_entry_point(&mir_mod)?;

        if options.verbose {
            println!("[JIT] Executing function '{function_name}' in sandbox");
        }

        let result: i64 = sandbox.execute_i64(&mir_mod, function_name)?;

        if options.verbose {
            println!("[JIT] Execution completed successfully; return={result}");
        }

        if function_name == "main" {
            process::exit(result as i32);
        }
    } else {
        let codegen_units = options.codegen_units.unwrap_or_else(|| {
            let max_threads = cpu_count();
            if max_threads > 2 { max_threads - 2 } else { 1 }
        });

        let jit_function_name = jit_entry_point(&mir_mod)?;

        let func = mir_mod.functions.get(jit_function_name).ok_or_else(|| {
            LaminaError::ValidationError("Function not found in module".to_owned())
        })?;

        let (runtime_result, in_memory_failure) = match compile_to_runtime(
            &mir_mod,
            target.architecture,
            target.operating_system,
            Some(jit_function_name),
        ) {
            Ok(result) => (Some(result), None),
            Err(err) => {
                if options.verbose {
                    eprintln!(
                        "[JIT] In-memory compilation failed; falling back to AOT execution.\n  Reason: {err}"
                    );
                }
                (None, Some(err.to_string()))
            }
        };
        let aot_fallback = |error: LaminaError| match &in_memory_failure {
            Some(reason) => LaminaError::ValidationError(format!(
                "--jit fell back to AOT execution because in-memory compilation failed ({reason}), and the fallback failed too: {error}"
            )),
            None => error,
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
                let result = execute_jit_function(
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
                        process::exit(ret as i32);
                    }
                } else if jit_function_name == "main" {
                    process::exit(0);
                }
            }
        } else {
            let fallback = (|| -> Result<(), LaminaError> {
                if matches!(
                    target.architecture,
                    TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64
                ) {
                    return Err(LaminaError::ValidationError(
                        "JIT fallback: cannot execute WASM targets directly".to_owned(),
                    ));
                }

                let pid = process::id();
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let tmp_dir = lamina_cache_dir()?.join(format!("jit_{pid}_{nanos}"));
                create_dir_all(&tmp_dir)?;

                let intermediate_ext =
                    assembler_input_extension(target.architecture, target.operating_system);
                let assembly_output_ext = assembler_output_extension(target.architecture);
                let final_ext = get_output_extension(target.architecture, target.operating_system);

                let mut intermediate_path = tmp_dir.join("module");
                intermediate_path.set_extension(intermediate_ext);
                let mut object_path = tmp_dir.join("module");
                object_path.set_extension(assembly_output_ext);
                let mut exe_path = tmp_dir.join("module");
                if !final_ext.is_empty() {
                    exe_path.set_extension(final_ext);
                }

                let mut intermediate = Vec::<u8>::new();
                let codegen_settings = options
                    .mir_codegen_settings
                    .clone()
                    .with_target_variant(target.variant)
                    .with_environment(target.environment);
                generate_mir_to_target_with_settings(
                    &mir_mod,
                    &mut intermediate,
                    &target,
                    codegen_units,
                    &codegen_settings,
                )?;

                write(&intermediate_path, &intermediate)?;

                let (assembler_backend, linker_backend) =
                    toolchain_backends(&options.forced_compiler, &options.assembler);
                let assemble_result = assemble_with_ras_object_options(
                    &intermediate_path,
                    &object_path,
                    target,
                    assembler_backend,
                    &options.assembler_flags,
                    options.verbose,
                    options.ras_object_write_options(target.operating_system, &mir_mod),
                )?;

                if assemble_result.needs_linking {
                    let runtime = runtime_builtins::build(
                        &mir_mod,
                        &assemble_result.output_path,
                        target,
                        &codegen_settings,
                        options.opt_level,
                        codegen_units,
                        assembler_backend,
                        &options.assembler_flags,
                        options.verbose,
                        &options.ras_object_write_options(target.operating_system, &mir_mod),
                    )?;
                    link_with_runtime(
                        &assemble_result.output_path,
                        &exe_path,
                        &target,
                        linker_backend,
                        runtime
                            .as_ref()
                            .map_or(&[][..], runtime_builtins::RuntimeObjects::objects),
                        &options.linker_flags,
                        options.verbose,
                    )?;
                    if let Some(runtime) = runtime {
                        runtime.discard();
                    }
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
                        process::exit(code);
                    }
                    #[cfg(unix)]
                    {
                        use std::os::unix::process::ExitStatusExt;
                        if let Some(signal) = status.signal() {
                            eprintln!("JIT fallback: program terminated: {status}");
                            process::exit(128 + signal);
                        }
                    }
                    return Err(LaminaError::RuntimeError(format!(
                        "JIT fallback: program terminated: {status}"
                    )));
                }

                let _ = remove_dir_all(&tmp_dir);
                Ok(())
            })();
            fallback.map_err(aot_fallback)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::options::CompileOptions;
    use crate::mir_codegen::MirCodegenSettings;
    use std::path::Path;

    fn minimal_options() -> CompileOptions {
        CompileOptions {
            input_file: "test.lir".into(),
            output_file: None,
            verbose: false,
            forced_compiler: None,
            assembler: None,
            assembler_flags: Vec::new(),
            linker_flags: Vec::new(),
            emit_asm_only: false,
            emit_mir: false,
            target_arch: None,
            opt_level: 0,
            jit: true,
            sandbox: false,
            codegen_units: None,
            cpu_level_explicit: false,
            mir_codegen_settings: MirCodegenSettings::default(),
        }
    }

    const SIMPLE_ADD_IR: &str = r#"
fn @add(i64 %a, i64 %b) -> i64 {
entry:
    %r = add.i64 %a, %b
    ret.i64 %r
}
"#;

    #[test]
    fn jit_refuses_a_target_that_is_not_the_host() {
        let host = Target::detect_host().expect("host");
        let foreign = if host.architecture == TargetArchitecture::Aarch64 {
            "x86_64_linux"
        } else {
            "aarch64_linux"
        };
        let mut opts = minimal_options();
        opts.target_arch = Some(foreign.to_owned());
        let error = handle_jit_compilation(SIMPLE_ADD_IR, Path::new("test.lir"), &opts)
            .expect_err("a foreign target must be refused");
        let message = error.to_string();
        assert!(message.contains(foreign), "{message}");
        assert!(message.contains("host"), "{message}");
    }

    #[test]
    fn jit_refuses_a_module_with_no_entry_point_to_choose() {
        let ir =
            "fn @a() -> i64 {\nentry:\n ret.i64 1\n}\nfn @b() -> i64 {\nentry:\n ret.i64 2\n}\n";
        let error = handle_jit_compilation(ir, Path::new("test.lir"), &minimal_options())
            .expect_err("two functions and no @main give no entry point");
        let message = error.to_string();
        assert!(message.contains("no entry point"), "{message}");
        assert!(message.contains('a') && message.contains('b'), "{message}");
    }

    #[test]
    fn jit_refuses_a_kernel_function() {
        let error = handle_jit_compilation(
            "@kernel fn @main() -> void {\nentry:\nret.void\n}\n",
            Path::new("test.lir"),
            &minimal_options(),
        )
        .expect_err("a GPU kernel cannot run on the host");
        let message = error.to_string();
        assert!(message.contains("@kernel function 'main'"), "{message}");
    }

    #[test]
    fn invalid_target_returns_error() {
        let mut opts = minimal_options();
        opts.target_arch = Some("invalid_target_xyz".to_owned());
        let result = handle_jit_compilation(SIMPLE_ADD_IR, Path::new("test.lir"), &opts);
        assert!(matches!(result, Err(LaminaError::ValidationError(_))));
    }

    #[test]
    fn malformed_ir_returns_parse_error() {
        let result = handle_jit_compilation(
            "this is not valid IR at all!!!",
            Path::new("test.lir"),
            &minimal_options(),
        );
        assert!(result.is_err());
    }
}
