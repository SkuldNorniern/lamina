//! CLI option parsing for the lamina compiler driver.

use std::{env::args, error::Error, fmt, path::PathBuf, process::exit, str::FromStr};

use crate::debug_info;
use crate::mir_codegen::{
    MirCodegenSettings, RegallocStrategy, amdgpu::AmdGpuTarget, assemble::AssemblerBackend,
    link::LinkerBackend,
};
use lamina_mir::Module as MirModule;
use lamina_platform::{
    ArchVariant, Target, TargetArchitecture, TargetOperatingSystem, X86CpuLevel, host_arch_list,
    target_alias_list,
};
use ras::ObjectWriteOptions;

/// A refusal from the command line.
///
/// The variant says what the driver should print after it. A command line of the wrong
/// shape is answered with the usage text; a message that already states the diagnosis
/// is answered with itself, because burying four lines of diagnosis under eighteen
/// lines of usage hides it.
#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    /// `-h` or `--help`: print the usage text and exit cleanly.
    Help,
    /// The command line has the wrong shape: a missing argument, an unknown flag.
    Usage(String),
    /// The message names what is wrong and what to write instead.
    Diagnosis(String),
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    fn diagnosis(message: impl Into<String>) -> Self {
        Self::Diagnosis(message.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Help => write!(f, "help"),
            CliError::Usage(message) | CliError::Diagnosis(message) => write!(f, "{message}"),
        }
    }
}

impl Error for CliError {}

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
    /// True when `--cpu-level` named the x86-64 level, false when it was defaulted.
    /// A defaulted level is checked against the host before an AOT build for the host.
    pub cpu_level_explicit: bool,
    pub mir_codegen_settings: MirCodegenSettings,
}

impl CompileOptions {
    pub fn ras_object_write_options(
        &self,
        target_os: TargetOperatingSystem,
        module: &MirModule,
    ) -> ObjectWriteOptions {
        let mut o = ObjectWriteOptions::default();
        let debug = self.mir_codegen_settings.emit_asm_debug_lines;
        if debug && debug_info::emits_dwarf(target_os) {
            o.emit_minimal_dwarf_sections = true;
            o.dwarf_decl_file_name = self.mir_codegen_settings.debug_file_tag.clone();
        }
        if debug && debug_info::emits_codeview(target_os) {
            o.debug_object_name = self.input_file.with_extension("obj").display().to_string();
        }
        debug_info::apply(&mut o, target_os, debug, module);
        o
    }
}

/// The architecture the driver will build for, as far as the arguments say.
///
/// `None` when neither the named target nor the host resolves; the driver reports that
/// for itself, and the register allocator then stays on its portable default.
fn selected_architecture(options: &CompileOptions) -> Option<TargetArchitecture> {
    let named = options
        .target_arch
        .as_deref()
        .and_then(|target| Target::from_str(target).ok());
    match named {
        Some(target) => Some(target.architecture),
        None => Target::detect_host().ok().map(|host| host.architecture),
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

/// The usage text, one line per option.
pub fn usage_text() -> String {
    [
        "Usage: lamina <input.lamina> [options]",
        "Options:",
        "  -o, --output <file>     Specify output executable name",
        "  -v, --verbose           Enable verbose output",
        "  -c, --compiler <n>      Toolchain: lamina (default: ras + weld, falling back to GNU as/ld with a [WARN]) | gnu | clang | msvc. Or linker only: ld, lld, mold, weld, link, cl",
        "  --assembler <n>         Override assembler: ras (default), gas, clang",
        "  -Wl,<flag>              Pass flag(s) to linker (GCC/Clang compatible)",
        "  -Wa,<flag>              Pass flag(s) to assembler (GCC/Clang compatible)",
        "  --emit-asm              Only emit assembly file without compiling",
        "  --mcpu <gpu>            AMDGPU processor: gfx900 | gfx1200 (default: gfx1200)",
        "  --target <arch_os[_env]> Specify target (e.g., x86_64_linux, x86_64_linux_musl)",
        "                          A sub-architecture may replace the arch: rv32imac, rv64gc,\n\
         \x20                         armv7r, armv7a-neon, aarch64v8.2, ppc64le-power9\n\
         \x20                         An environment may follow the OS: gnu | musl | lamina |\n\
         \x20                         eabi | eabihf | none. Left out, each target declares its own.",
        "  --cpu-level <v>         x86-64 level the output may use: v1 | v2 | v3 | v4 (default: v2, x86_64 targets only)",
        "  --emit-mir              Only emit MIR (.lumir) and exit",
        "  --opt-level <n>         EXPERIMENTAL(mir only): Set optimization level (0-3, default: 1)",
        "  --jit                   Compile and execute using runtime compilation (JIT)",
        "  --sandbox               Enable sandbox for secure execution (requires --jit)",
        "  -j, --jobs <n>          Number of parallel compilation threads (default: max - 2)",
        "  -g                      Emit assembler .file/.loc directives, with ras or with gas; with ras, also minimal DWARF in ELF and Mach-O objects",
        "  --regalloc <m>          MIR regalloc: incremental | linear | graph (default: linear at O1+ where the backend implements it, incremental otherwise)",
        "  -h, --help              Display this help message",
        "  --version               Print the compiler version and exit",
    ]
    .join("\n")
}

pub fn print_usage() {
    eprintln!("{}", usage_text());
}

pub fn parse_args() -> Result<CompileOptions, CliError> {
    parse_args_from(args().collect())
}

pub(crate) fn parse_args_from(args: Vec<String>) -> Result<CompileOptions, CliError> {
    if args.len() < 2 {
        return Err(CliError::usage("Not enough arguments"));
    }

    if args[1] == "-h" || args[1] == "--help" {
        return Err(CliError::Help);
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
        cpu_level_explicit: false,
        mir_codegen_settings: MirCodegenSettings::default(),
    };

    let mut regalloc_explicit = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 >= args.len() {
                    return Err(CliError::usage("Missing argument for output file"));
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
                    return Err(CliError::usage("Missing argument for compiler"));
                }
                options.forced_compiler = Some(args[i + 1].clone());
                i += 2;
            }
            "--assembler" => {
                if i + 1 >= args.len() {
                    return Err(CliError::usage("Missing argument for --assembler"));
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
                        _ => return Err(CliError::diagnosis("--mcpu requires gfx900 or gfx1200")),
                    };
                i += 2;
            }
            "--target" => {
                if i + 1 >= args.len() {
                    return Err(CliError::usage("Missing argument for target architecture"));
                }
                let requested_target = args[i + 1].to_lowercase();
                // A sub-architecture spelling such as rv64gc_linux or aarch64v8.2_linux
                // is the same target with a variant chosen, so the list is checked
                // against the bare form it resolves to.
                let parsed = requested_target.parse::<Target>();
                let listed = parsed.as_ref().is_ok_and(|target| {
                    let bare = Target::new(target.architecture, target.operating_system);
                    host_arch_list().contains(&bare.to_string().as_str())
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
                    return Err(CliError::Diagnosis(format!(
                        "Unsupported target '{}'{reason}. Supported values:\n{}\n\
                         Also accepted, as second spellings of the same targets:\n{}\n\
                         A RISC-V ISA string may be named instead: rv32imac, rv64gc, and an \
                         environment may follow the operating system: gnu, musl, lamina, \
                         eabi, eabihf, none",
                        requested_target,
                        host_arch_list().join(", "),
                        target_alias_list().join(", ")
                    )));
                }
                options.target_arch = Some(requested_target);
                i += 2;
            }
            "--cpu-level" => {
                if i + 1 >= args.len() {
                    return Err(CliError::usage("Missing argument for --cpu-level"));
                }
                options.mir_codegen_settings.arch_variant = ArchVariant::X86(
                    args[i + 1]
                        .parse::<X86CpuLevel>()
                        .map_err(|e| CliError::Diagnosis(format!("--cpu-level: {e}")))?,
                );
                options.cpu_level_explicit = true;
                i += 2;
            }
            "--opt-level" => {
                if i + 1 >= args.len() {
                    return Err(CliError::usage("Missing argument for optimization level"));
                }
                let level = args[i + 1]
                    .parse::<u8>()
                    .map_err(|_| CliError::diagnosis("Invalid --opt-level value (must be 0-3)"))?;
                if level > 3 {
                    return Err(CliError::diagnosis("--opt-level must be between 0 and 3"));
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
                    return Err(CliError::usage("Missing argument for --regalloc"));
                }
                let m = args[i + 1].to_lowercase();
                options.mir_codegen_settings.regalloc = match m.as_str() {
                    "incremental" => RegallocStrategy::Incremental,
                    "linear" => RegallocStrategy::LinearScanGlobal,
                    "graph" => RegallocStrategy::GraphColorGlobal,
                    _ => {
                        return Err(CliError::diagnosis(
                            "--regalloc must be incremental, linear, or graph",
                        ));
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
                    return Err(CliError::usage("Missing argument for -j/--jobs"));
                }
                let jobs = args[i + 1].parse::<usize>().map_err(|_| {
                    CliError::diagnosis("Invalid -j/--jobs value (must be a positive integer)")
                })?;
                if jobs == 0 {
                    return Err(CliError::diagnosis("-j/--jobs must be at least 1"));
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
                        return Err(CliError::usage("Missing argument for -Wl"));
                    }
                    options
                        .linker_flags
                        .extend(args[i + 1].split(',').map(ToString::to_string));
                    i += 2;
                } else if arg == "-Wa" {
                    if i + 1 >= args.len() {
                        return Err(CliError::usage("Missing argument for -Wa"));
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
                    return Err(CliError::Usage(format!("Unknown argument: {arg}")));
                }
            }
        }
    }

    if options.input_file.as_os_str().is_empty() {
        return Err(CliError::usage("No input file specified"));
    }

    if options.jit && options.emit_asm_only {
        return Err(CliError::diagnosis(
            "--emit-asm and --jit ask for different things: --jit runs the code and writes no assembly file. Pick one",
        ));
    }
    if options.jit && options.emit_mir {
        return Err(CliError::diagnosis(
            "--emit-mir and --jit ask for different things: --jit runs the code and writes no MIR file. Pick one",
        ));
    }
    if options.sandbox && !options.jit {
        return Err(CliError::diagnosis(
            "--sandbox applies only to --jit, which is what it sandboxes",
        ));
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
    if options.cpu_level_explicit && !x86_64 {
        return Err(CliError::diagnosis(
            "--cpu-level applies only to x86_64 targets",
        ));
    }
    if !regalloc_explicit && let Some(architecture) = selected_architecture(&options) {
        options.mir_codegen_settings.regalloc =
            RegallocStrategy::default_for(architecture, options.opt_level);
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<CompileOptions, String> {
        let mut all = vec!["lamina".to_string(), "prog.lamina".to_string()];
        all.extend(args.iter().map(ToString::to_string));
        parse_args_from(all).map_err(|error| error.to_string())
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
    fn jit_refuses_the_flags_that_would_emit_nothing() {
        for flag in ["--emit-asm", "--emit-mir"] {
            let error = parse(&["--jit", flag]).err().expect("refused");
            assert!(error.contains("--jit"), "{error}");
            assert!(error.contains(flag), "{error}");
        }
    }

    #[test]
    fn sandbox_without_jit_is_refused() {
        let error = parse(&["--sandbox"]).err().expect("refused");
        assert!(error.contains("--jit"), "{error}");
        parse(&["--sandbox", "--jit"]).expect("--sandbox is for --jit");
    }

    #[test]
    fn regalloc_default_follows_the_backend_not_the_architecture_name() {
        let aarch64 = parse(&["--target", "aarch64_linux", "--opt-level", "2"])
            .expect("aarch64 at O2 parses");
        assert_eq!(
            aarch64.mir_codegen_settings.regalloc,
            RegallocStrategy::LinearScanGlobal
        );
        let s390x =
            parse(&["--target", "s390x_linux", "--opt-level", "2"]).expect("s390x at O2 parses");
        assert_eq!(
            s390x.mir_codegen_settings.regalloc,
            RegallocStrategy::Incremental
        );
        let aarch64_o0 = parse(&["--target", "aarch64_linux", "--opt-level", "0"])
            .expect("aarch64 at O0 parses");
        assert_eq!(
            aarch64_o0.mir_codegen_settings.regalloc,
            RegallocStrategy::Incremental
        );
    }

    #[test]
    fn cpu_level_defaults_to_v2() {
        let options = parse(&[]).expect("default arguments parse");
        assert!(!options.cpu_level_explicit);
        assert_eq!(
            options.mir_codegen_settings.x86_cpu_level(),
            X86CpuLevel::V2
        );
    }

    #[test]
    fn cpu_level_flag_selects_the_level() {
        let options =
            parse(&["--cpu-level", "v1", "--target", "x86_64_linux"]).expect("v1 parses on x86_64");
        assert!(options.cpu_level_explicit);
        assert_eq!(
            options.mir_codegen_settings.x86_cpu_level(),
            X86CpuLevel::V1
        );
    }

    #[test]
    fn cpu_level_rejects_unknown_values() {
        let error = parse(&["--cpu-level", "native"])
            .err()
            .expect("native is not a level");
        assert!(error.contains("native"), "{error}");
    }

    #[test]
    fn the_unknown_target_message_lists_every_accepted_alias() {
        let error = parse(&["--target", "x86_64_plan9"])
            .err()
            .expect("plan9 is not an operating system");
        for alias in target_alias_list() {
            assert!(error.contains(&alias), "{alias} is missing from:\n{error}");
            parse(&["--target", &alias]).unwrap_or_else(|e| panic!("{alias} is refused: {e}"));
        }
        for witness in [
            "x86_64_bsd",
            "i386_linux",
            "x86_linux",
            "systemz_linux",
            "loong64_linux",
            "mips64le_linux",
            "powerpc64_linux",
            "armv7m_unknown",
            "cortex-m3_unknown",
            "ppc64le-power10_linux",
        ] {
            assert!(
                error.contains(witness),
                "{witness} is missing from:\n{error}"
            );
        }
    }

    #[test]
    fn a_diagnosis_is_not_answered_with_the_usage_text() {
        let diagnoses = [
            vec!["--target", "x86_64_plan9"],
            vec!["--opt-level", "9"],
            vec!["--regalloc", "none"],
            vec!["--cpu-level", "native"],
            vec!["--sandbox"],
            vec!["--jit", "--emit-asm"],
            vec!["--mcpu", "gfx42"],
            vec!["-j", "0"],
        ];
        for args in diagnoses {
            let mut all = vec!["lamina".to_string(), "prog.lamina".to_string()];
            all.extend(args.iter().map(ToString::to_string));
            let error = parse_args_from(all).err().expect("refused");
            assert!(
                matches!(error, CliError::Diagnosis(_)),
                "{args:?} should diagnose, not print usage: {error}"
            );
        }

        let shapes = [
            vec!["--bogus"],
            vec!["-o"],
            vec!["--target"],
            vec!["--regalloc"],
        ];
        for args in shapes {
            let mut all = vec!["lamina".to_string(), "prog.lamina".to_string()];
            all.extend(args.iter().map(ToString::to_string));
            let error = parse_args_from(all).err().expect("refused");
            assert!(
                matches!(error, CliError::Usage(_)),
                "{args:?} is a usage error: {error}"
            );
        }

        assert_eq!(
            parse_args_from(vec!["lamina".to_string(), "--help".to_string()]).err(),
            Some(CliError::Help)
        );
    }

    #[test]
    fn the_usage_text_names_version_and_the_assemblers_that_honour_g() {
        let text = usage_text();
        assert!(text.contains("  --version "), "{text}");
        assert!(
            text.lines()
                .any(|line| line.starts_with("  -g ") && line.contains("gas")),
            "{text}"
        );
    }

    #[test]
    fn cpu_level_is_rejected_off_x86_64() {
        let error = parse(&["--cpu-level", "v1", "--target", "aarch64_linux"])
            .err()
            .expect("cpu level is x86_64 only");
        assert!(error.contains("x86_64"), "{error}");
    }
}
