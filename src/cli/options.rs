//! CLI option parsing for the lamina compiler driver.

use std::{env::args, path::PathBuf, process::exit};

use crate::mir_codegen::{
    MirCodegenSettings, RegallocStrategy, amdgpu::AmdGpuTarget, assemble::AssemblerBackend,
    link::LinkerBackend,
};
use lamina_platform::{HOST_ARCH_LIST, Target, TargetOperatingSystem, X86CpuLevel};
use ras::ObjectWriteOptions;

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
    pub target_arch: Option<String>,
    pub opt_level: u8,
    pub jit: bool,
    pub sandbox: bool,
    pub codegen_units: Option<usize>,
    pub mir_codegen_settings: MirCodegenSettings,
}

impl CompileOptions {
    pub fn ras_object_write_options(&self, target_os: TargetOperatingSystem) -> ObjectWriteOptions {
        let mut o = ObjectWriteOptions::default();
        let elfish = matches!(
            target_os,
            TargetOperatingSystem::Linux
                | TargetOperatingSystem::FreeBSD
                | TargetOperatingSystem::OpenBSD
                | TargetOperatingSystem::NetBSD
                | TargetOperatingSystem::DragonFly
                | TargetOperatingSystem::Redox
        );
        if self.mir_codegen_settings.emit_asm_debug_lines && elfish {
            o.emit_minimal_dwarf_sections = true;
            o.dwarf_decl_file_name = self.mir_codegen_settings.debug_file_tag.clone();
        }
        o
    }
}

pub fn toolchain_backends(
    forced_compiler: &Option<String>,
    assembler_override: &Option<String>,
) -> (Option<AssemblerBackend>, Option<LinkerBackend>) {
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
        "  -c, --compiler <n>      Toolchain: lamina (default: ras + weld, falling back to GNU as/ld with a [WARN]) | gnu | clang | msvc. Or linker only: ld, lld, mold, weld, link, cl"
    );
    eprintln!("  --assembler <n>         Override assembler: ras (default), gas, clang");
    eprintln!("  -Wl,<flag>              Pass flag(s) to linker (GCC/Clang compatible)");
    eprintln!("  -Wa,<flag>              Pass flag(s) to assembler (GCC/Clang compatible)");
    eprintln!("  --emit-asm              Only emit assembly file without compiling");
    eprintln!("  --mcpu <gpu>            AMDGPU processor: gfx900 | gfx1200 (default: gfx1200)");
    eprintln!("  --target <arch_os>      Specify target (e.g., x86_64_linux)");
    eprintln!(
        "                          A sub-architecture may replace the arch: rv32imac, rv64gc,\n\
         \x20                         armv7r, armv7a-neon, aarch64v8.2, ppc64le-power9"
    );
    eprintln!(
        "  --cpu-level <v>         x86-64 level the output may use: v1 | v2 | v3 | v4 (default: v2, x86_64 targets only)"
    );
    eprintln!("  --emit-mir              Only emit MIR (.lumir) and exit");
    eprintln!(
        "  --opt-level <n>         EXPERIMENTAL(mir only): Set optimization level (0-3, default: 1)"
    );
    eprintln!("  --jit                   Compile and execute using runtime compilation (JIT)");
    eprintln!("  --sandbox               Enable sandbox for secure execution (requires --jit)");
    eprintln!(
        "  -j, --jobs <n>          Number of parallel compilation threads (default: max - 2)"
    );
    eprintln!(
        "  -g                      Emit assembler .file/.loc and minimal ELF DWARF when using ras"
    );
    eprintln!(
        "  --regalloc <m>          MIR regalloc: incremental | linear | graph (default: linear at O1+ on x86_64, incremental otherwise)"
    );
    eprintln!("  -h, --help              Display this help message");
}

pub fn parse_args() -> Result<CompileOptions, String> {
    parse_args_from(args().collect())
}

pub(crate) fn parse_args_from(args: Vec<String>) -> Result<CompileOptions, String> {
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
        target_arch: None,
        opt_level: 1,
        jit: false,
        sandbox: false,
        codegen_units: None,
        mir_codegen_settings: MirCodegenSettings::default(),
    };

    let mut regalloc_explicit = false;
    let mut cpu_level_explicit = false;
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
                eprintln!(
                    "Warning: --emit-mir-asm is deprecated and will be removed; use --emit-asm"
                );
                options.emit_asm_only = true;
                i += 1;
            }
            "--mcpu" => {
                options.mir_codegen_settings.amdgpu_target =
                    match args.get(i + 1).map(String::as_str) {
                        Some("gfx900") => AmdGpuTarget::Gfx900,
                        Some("gfx1200") => AmdGpuTarget::Gfx1200,
                        _ => return Err("--mcpu requires gfx900 or gfx1200".into()),
                    };
                i += 2;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for target architecture".to_string());
                }
                let requested_target = args[i + 1].to_lowercase();
                // A sub-architecture spelling such as rv64gc_linux or aarch64v8.2_linux
                // is the same target with a variant chosen, so the list is checked
                // against the bare form it resolves to.
                let parsed = requested_target.parse::<Target>();
                let listed = parsed.as_ref().is_ok_and(|target| {
                    let bare = Target::new(target.architecture, target.operating_system);
                    HOST_ARCH_LIST.contains(&bare.to_string().as_str())
                });
                if !requested_target.contains('_') || !listed {
                    let reason = parsed.as_ref().map_or_else(
                        |e| format!(": {e}"),
                        |_| {
                            ": the architecture and the operating system are each known, but \
                             lamina has no backend for them together"
                                .to_string()
                        },
                    );
                    return Err(format!(
                        "Unsupported target '{}'{reason}. Supported values:\n{}\n\
                         A sub-architecture may be named instead: rv32imac, rv64gc, armv7r, \
                         armv7a-neon, aarch64v8.2, ppc64le-power9",
                        requested_target,
                        HOST_ARCH_LIST.join(", ")
                    ));
                }
                options.target_arch = Some(requested_target);
                i += 2;
            }
            "--cpu-level" => {
                if i + 1 >= args.len() {
                    return Err("Missing argument for --cpu-level".to_string());
                }
                options.mir_codegen_settings.x86_cpu_level = args[i + 1]
                    .parse::<X86CpuLevel>()
                    .map_err(|e| format!("--cpu-level: {e}"))?;
                cpu_level_explicit = true;
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
            "-g" => {
                options.mir_codegen_settings.emit_asm_debug_lines = true;
                i += 1;
            }
            "--regalloc" => {
                regalloc_explicit = true;
                if i + 1 >= args.len() {
                    return Err("Missing argument for --regalloc".to_string());
                }
                let m = args[i + 1].to_lowercase();
                options.mir_codegen_settings.regalloc = match m.as_str() {
                    "incremental" => RegallocStrategy::Incremental,
                    "linear" => RegallocStrategy::LinearScanGlobal,
                    "graph" => RegallocStrategy::GraphColorGlobal,
                    _ => {
                        return Err("--regalloc must be incremental, linear, or graph".to_string());
                    }
                };
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
                exit(0);
            }
            arg => {
                if let Some(rest) = arg.strip_prefix("-Wl,") {
                    options
                        .linker_flags
                        .extend(rest.split(',').map(ToString::to_string));
                    i += 1;
                } else if let Some(rest) = arg.strip_prefix("-Wa,") {
                    options
                        .assembler_flags
                        .extend(rest.split(',').map(ToString::to_string));
                    i += 1;
                } else if arg == "-Wl" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wl".to_string());
                    }
                    options
                        .linker_flags
                        .extend(args[i + 1].split(',').map(ToString::to_string));
                    i += 2;
                } else if arg == "-Wa" {
                    if i + 1 >= args.len() {
                        return Err("Missing argument for -Wa".to_string());
                    }
                    options
                        .assembler_flags
                        .extend(args[i + 1].split(',').map(ToString::to_string));
                    i += 2;
                } else if options.input_file.as_os_str().is_empty() {
                    // First non-flag argument is the input file, regardless of position.
                    options.input_file = PathBuf::from(arg);
                    i += 1;
                } else {
                    return Err(format!("Unknown argument: {arg}"));
                }
            }
        }
    }

    if options.input_file.as_os_str().is_empty() {
        return Err("No input file specified".to_string());
    }

    let x86_64 = options
        .target_arch
        .as_deref()
        .map_or(cfg!(target_arch = "x86_64"), |target| {
            target.starts_with("x86_64_")
        });
    if options.mir_codegen_settings.emit_asm_debug_lines {
        options.mir_codegen_settings.debug_file_tag =
            options.input_file.to_string_lossy().into_owned();
    }
    if cpu_level_explicit && !x86_64 {
        return Err("--cpu-level applies only to x86_64 targets".to_string());
    }
    if !regalloc_explicit && x86_64 {
        options.mir_codegen_settings.regalloc = RegallocStrategy::for_opt_level(options.opt_level);
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<CompileOptions, String> {
        let mut all = vec!["lamina".to_string(), "prog.lamina".to_string()];
        all.extend(args.iter().map(ToString::to_string));
        parse_args_from(all)
    }

    #[test]
    fn freebsd_is_accepted_by_the_driver() {
        let options = parse(&["--target", "x86_64_freebsd"]).expect("x86_64_freebsd is a target");
        assert_eq!(options.target_arch.as_deref(), Some("x86_64_freebsd"));
        parse(&["--target", "x86_64_bsd"]).expect("the bsd alias resolves to freebsd");
    }

    #[test]
    fn a_refused_target_names_the_reason() {
        let error = parse(&["--target", "x86_64_openbsd"])
            .err()
            .expect("openbsd is refused");
        assert!(error.contains("is not implemented"), "{error}");
        assert!(error.contains("Hint:"), "{error}");
        assert!(error.contains("freebsd"), "{error}");

        let error = parse(&["--target", "aarch64_freebsd"])
            .err()
            .expect("aarch64 freebsd has no backend");
        assert!(error.contains("no backend for them together"), "{error}");
    }

    #[test]
    fn cpu_level_defaults_to_v2() {
        let options = parse(&[]).expect("default arguments parse");
        assert_eq!(options.mir_codegen_settings.x86_cpu_level, X86CpuLevel::V2);
    }

    #[test]
    fn cpu_level_flag_selects_the_level() {
        let options =
            parse(&["--cpu-level", "v1", "--target", "x86_64_linux"]).expect("v1 parses on x86_64");
        assert_eq!(options.mir_codegen_settings.x86_cpu_level, X86CpuLevel::V1);
    }

    #[test]
    fn cpu_level_rejects_unknown_values() {
        let error = parse(&["--cpu-level", "native"])
            .err()
            .expect("native is not a level");
        assert!(error.contains("native"), "{error}");
    }

    #[test]
    fn cpu_level_is_rejected_off_x86_64() {
        let error = parse(&["--cpu-level", "v1", "--target", "aarch64_linux"])
            .err()
            .expect("cpu level is x86_64 only");
        assert!(error.contains("x86_64"), "{error}");
    }
}
