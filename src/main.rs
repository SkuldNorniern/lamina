use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

// Import the library crate
use lamina;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: lamina <input.lamina> [output_executable]");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    if !input_path.exists() {
        eprintln!("Error: Input file '{}' does not exist.", args[1]);
        std::process::exit(1);
    }
    if input_path.extension().map_or(true, |ext| ext != "lamina") {
        eprintln!(
            "Warning: Input file '{}' does not have .lamina extension.",
            args[1]
        );
    }

    // Determine default executable extension based on platform
    let exe_extension = if cfg!(windows) { ".exe" } else { "" };
    
    // Get output name
    let output_stem = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        // Default output name based on input file
        let stem = input_path.file_stem().map_or_else(
            || "a".into(), // "a.out" on Unix, "a.exe" on Windows
            |s| PathBuf::from(s),
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
    let mut exec_path = output_stem;
    if cfg!(windows) && exec_path.extension().is_none() {
        exec_path.set_extension("exe");
    }

    println!(
        "[INFO] Compiling {} -> {} -> {}",
        input_path.display(), asm_path.display(), exec_path.display()
    );

    // 1. Read Input Lamina IR
    let mut input_file = File::open(input_path)?;
    let mut ir_source = String::new();
    input_file.read_to_string(&mut ir_source)?;

    // 2. Compile IR to Assembly
    let mut asm_buffer = Vec::<u8>::new();
    match lamina::compile_lamina_ir_to_assembly(&ir_source, &mut asm_buffer) {
        Ok(_) => {
            println!(
                "[INFO] Assembly generated successfully ({} bytes).",
                asm_buffer.len()
            );
        }
        Err(e) => {
            eprintln!("\n[ERROR] Compilation failed: {}", e);
            std::process::exit(1);
        }
    }

    // 3. Write Assembly to file
    let mut asm_file = File::create(&asm_path)?;
    asm_file.write_all(&asm_buffer)?;
    println!("[INFO] Assembly written to {}", asm_path.display());

    // 4. Determine which compiler to use
    let (compiler, args) = detect_compiler()?;
    
    // 5. Assemble and Link
    println!("[INFO] Assembling and linking with {}...", compiler);
    
    // Build command with appropriate arguments
    let mut cmd = Command::new(compiler);
    
    // Add any compiler-specific flags first
    for arg in args {
        cmd.arg(arg);
    }
    
    // Add input and output files
    let compiler_output = cmd
        .arg(&asm_path)
        .arg("-o")
        .arg(&exec_path)
        .output()?;

    if compiler_output.status.success() {
        println!("[INFO] Executable '{}' created successfully.", exec_path.display());
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
