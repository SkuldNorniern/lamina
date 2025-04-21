#!/usr/bin/env python3

import subprocess
import os
import sys
import time
import stat
from pathlib import Path

# --- Configuration ---
BENCHMARK_DIR = Path("benchmarks/2Dmatmul") # Use Path for directory handling
LAMINA_EXECUTABLE = "./target/release/lamina"
LAMINA_SOURCE = "benchmarks/2Dmatmul/2Dmatmul.lamina" # Keep lamina source path relative to root
LAMINA_OUTPUT_BINARY = "2Dmatmul_lamina" # Consistent naming

# --- Colors (ANSI escape codes) ---
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m' # No Color

def print_color(color, text):
    """Prints text in the specified color."""
    print(f"{color}{text}{NC}")

def run_command(command, check=True, cwd=None):
    """Runs a command using subprocess, optionally checking for errors."""
    # Convert all command parts to strings for display and execution
    command_str_list = [str(item) for item in command]
    cmd_str = ' '.join(command_str_list)
    cwd_str = f" in {cwd}" if cwd else ""
    print_color(YELLOW, f"Running command: {cmd_str}{cwd_str}")
    try:
        result = subprocess.run(command_str_list, check=check, capture_output=True, text=True, cwd=cwd)
        # Limit noisy output from benchmarks themselves
        # if result.stdout:
        #     print(result.stdout)
        if result.stderr:
            # Filter common verbose compiler messages unless it's an error
             if "warning:" not in result.stderr.lower() and "note:" not in result.stderr.lower() or result.returncode != 0 :
                 # Print entire stderr if it's likely an error or failure
                 print_color(RED if result.returncode != 0 else YELLOW, f"Stderr: {result.stderr.strip()}")
        if check and result.returncode != 0:
             print_color(RED, f"Command failed with exit code {result.returncode}")
             # Optionally print full stderr on failure if not already printed
             # if result.stderr and ("warning:" in result.stderr.lower() or "note:" in result.stderr.lower()):
             #    print(result.stderr, file=sys.stderr)
             sys.exit(result.returncode)
        return result
    except FileNotFoundError:
        print_color(RED, f"Error: Command not found: {command[0]}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print_color(RED, f"Command failed: {' '.join(command_str_list)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print_color(RED, f"An unexpected error occurred: {e}")
        sys.exit(1)


def ensure_executable(file_path):
    """Ensures a file has execute permissions."""
    if not os.path.exists(file_path):
        return False
    try:
        st = os.stat(file_path)
        # Ensure the file path is treated as Path object for consistency if needed
        file_path_obj = Path(file_path)
        # Set execute permissions for user, group, and others
        os.chmod(file_path_obj, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print_color(GREEN, f"Set execute permission for {file_path_obj.name}")
        return True
    except Exception as e:
        print_color(RED, f"Error setting executable permission for {file_path}: {e}")
        return False

def run_single_benchmark(label, command_list, cwd=None):
    """Runs and times a benchmark command list."""
    print_color(GREEN, f"\n--- Running {label} Implementation ---")

    start_time = time.perf_counter()
    # Run without check=True to capture potentially non-zero exit codes from the benchmark itself
    result = run_command(command_list, check=False, cwd=cwd)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print_color(YELLOW, f"{label} Execution Time: {elapsed_time:.4f} seconds")
    if result.returncode != 0:
         print_color(RED, f"{label} Benchmark failed with Exit Code: {result.returncode}")
         # Optionally print output/error from the benchmark on failure
         # if result.stdout: print(result.stdout)
         # if result.stderr: print(result.stderr, file=sys.stderr)
         return None, result.returncode # Indicate failure with None time
    else:
         print_color(YELLOW, f"{label} Benchmark Exit Code: {result.returncode}")

    return elapsed_time, result.returncode


def create_csharp_project_file(benchmark_dir):
    """Creates a minimal C# project file for the benchmark."""
    csproj_path = benchmark_dir / "2Dmatmul.csproj"
    csproj_content = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AssemblyName>2Dmatmul</AssemblyName>
    <PlatformTarget>AnyCPU</PlatformTarget>
    <Optimize>true</Optimize>
    <DebugType>None</DebugType>
  </PropertyGroup>
</Project>
"""
    try:
        with open(csproj_path, 'w') as f:
            f.write(csproj_content)
        print_color(GREEN, f"Created C# project file: {csproj_path}")
        return True
    except Exception as e:
        print_color(RED, f"Error creating C# project file: {e}")
        return False


def compile_and_run(target, results, cwd):
    """Compiles (if needed) and runs a benchmark target."""
    label = target['lang']
    source_file_rel = target['source'] # Source file relative to BENCHMARK_DIR
    source_file_abs = cwd / source_file_rel # Absolute path
    output_path_rel = target.get('output') # Output path relative to BENCHMARK_DIR (can be dir or file)
    compile_cmd_template = target.get('compile')
    run_cmd_template = target['run']

    print_color(YELLOW, f"\nProcessing {label} ({source_file_rel})...")

    # Check if source file exists
    if not source_file_abs.exists():
        print_color(RED, f"Source file not found: {source_file_abs}")
        results[label] = {'time': None, 'exit_code': -1, 'error': 'Source not found'}
        return

    # --- Special handling for C# ---
    if label == "C#":
        # Create a proper project file for dotnet run
        if not create_csharp_project_file(cwd):
            results[label] = {'time': None, 'exit_code': -1, 'error': 'Failed to create project file'}
            return

    # --- Determine Paths Relative to CWD for commands ---
    # Use relative paths for commands executed within the benchmark directory
    src_for_cmd = source_file_rel
    out_for_cmd = output_path_rel

    # --- Compilation Step ---
    if compile_cmd_template:
        print_color(YELLOW, f"Compiling {label}...")
        # Replace placeholders with paths relative to the cwd (BENCHMARK_DIR)
        compile_cmd_full = [
            str(c).replace('{src}', src_for_cmd).replace('{out}', str(out_for_cmd) if out_for_cmd else "")
            for c in compile_cmd_template
        ]

        compile_result = run_command(compile_cmd_full, check=False, cwd=cwd) # Run compile command in BENCHMARK_DIR

        if compile_result.returncode != 0:
            print_color(RED, f"{label} compilation failed.")
            results[label] = {'time': None, 'exit_code': compile_result.returncode, 'error': 'Compilation failed'}
            return

        # Verify compilation output if expected
        if output_path_rel and label != "C#":  # Skip output check for C# as it's handled differently
            output_binary_abs = cwd / output_path_rel
            if not output_binary_abs.exists():
                print_color(RED, f"Error: {label} compilation did not produce expected output: {output_binary_abs}")
                results[label] = {'time': None, 'exit_code': -1, 'error': 'Output binary missing'}
                return
            ensure_executable(output_binary_abs)

        print_color(GREEN, f"{label} compilation successful.")


    # --- Execution Step ---
    # Determine the actual command and working directory for execution
    exec_cwd = None # Default: run from project root (for interpreters)
    run_cmd_actual = []

    # Special case: Java needs to run 'java ClassName' from the directory containing the .class file
    if label == "Java":
        # run_cmd_template is already correctly defined as ['java', '2Dmatmul']
        run_cmd_actual = run_cmd_template
        exec_cwd = cwd # Run 'java' command from BENCHMARK_DIR
    # Special case: C# with dotnet run
    elif label == "C#":
        # We need to run dotnet from the benchmark directory where the .csproj is located
        run_cmd_actual = run_cmd_template
        exec_cwd = cwd # Run from benchmark dir where .csproj is located
    # Handle other compiled languages
    elif output_path_rel:
        # Use relative path for execution command when running within BENCHMARK_DIR
        exec_path_rel = f"./{output_path_rel}" # e.g., ./2Dmatmul_c or ./bin_cs/2Dmatmul
        # The first element of run_cmd_template is usually the placeholder for the executable itself
        run_cmd_actual = [exec_path_rel] + run_cmd_template[1:]
        exec_cwd = cwd # Run the executable from within BENCHMARK_DIR
    # Handle interpreted languages
    else:
        # Replace {src} with the source *relative to the execution directory* (project root)
        # Here, source_file_abs is the correct path relative to project root
        run_cmd_actual = [
             str(c).replace('{src}', str(source_file_abs))
             for c in run_cmd_template
        ]
        exec_cwd = None # Run interpreter from project root

    # Ensure executable permission again just before running (if applicable)
    # This check might be redundant if ensure_executable was called after compilation, but safe to keep
    if output_path_rel and label != "C#": # Skip for C# - it's handled by dotnet
        exec_file_abs = cwd / output_path_rel
        if not ensure_executable(exec_file_abs):
             print_color(RED, f"Could not ensure {exec_file_abs} is executable.")
             results[label] = {'time': None, 'exit_code': -1, 'error': 'Execution permission error'}
             return

    print_color(YELLOW, f"Executing {label} with command: {' '.join(str(c) for c in run_cmd_actual)} {'in '+str(exec_cwd) if exec_cwd else ''}")
    exec_time, exit_code = run_single_benchmark(label, run_cmd_actual, cwd=exec_cwd)
    results[label] = {'time': exec_time, 'exit_code': exit_code}


def main():
    """Main function to run the benchmark steps."""
    print_color(BLUE, "===== Multi-Language Matrix Multiplication Benchmark =====")

    # Ensure benchmark directory exists
    if not BENCHMARK_DIR.is_dir():
        print_color(RED, f"Benchmark directory not found: {BENCHMARK_DIR}")
        sys.exit(1)

    # --- Clean previous build artifacts (optional but recommended) ---
    print_color(YELLOW, "Cleaning previous build artifacts...")
    artifacts_to_clean = [
        "2Dmatmul_lamina", # Lamina output at root
        BENCHMARK_DIR / "2Dmatmul_c",
        BENCHMARK_DIR / "2Dmatmul_cpp",
        BENCHMARK_DIR / "2Dmatmul_rs",
        BENCHMARK_DIR / "2Dmatmul_go",
        BENCHMARK_DIR / "2Dmatmul_zig",
        BENCHMARK_DIR / "2Dmatmul.class",
        BENCHMARK_DIR / "bin_cs", # C# output directory
        BENCHMARK_DIR / "2Dmatmul.csproj", # Temporary C# project file
        BENCHMARK_DIR / "obj", # C# build objects directory
        # Add other potential artifacts if needed
    ]
    for item in artifacts_to_clean:
        try:
            item_path = Path(item)
            if item_path.is_file():
                item_path.unlink()
                print(f"Removed file: {item_path.name}")
            elif item_path.is_dir():
                # Simple directory removal, adjust if complex structure
                import shutil
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path.name}")
        except FileNotFoundError:
            pass # Ignore if artifact doesn't exist
        except Exception as e:
            print_color(RED, f"Error cleaning artifact {item}: {e}")
    print_color(GREEN, "Cleanup complete.")
    # --- End Cleaning ---


    results = {} # Store results: {lang: {'time': float|None, 'exit_code': int, 'error': str|None}}

    # --- Compile and Run Lamina (Baseline) ---
    print_color(YELLOW, "\nCompiling Lamina file...")
    # Ensure LAMINA_SOURCE is the correct path relative to root
    lamina_source_abs = Path(LAMINA_SOURCE)
    if not lamina_source_abs.exists():
         print_color(RED, f"Lamina source file not found: {lamina_source_abs}")
         sys.exit(1)

    # Update Lamina command to use the new CLI with -o/--output flag
    lamina_compile_cmd = [LAMINA_EXECUTABLE, str(lamina_source_abs), "--output", LAMINA_OUTPUT_BINARY]
    compile_result = run_command(lamina_compile_cmd, check=False, cwd=None) # Run lamina compiler from project root

    if compile_result.returncode != 0:
         print_color(RED, f"Lamina compilation failed.")
         results['Lamina'] = {'time': None, 'exit_code': compile_result.returncode, 'error': 'Compilation failed'}
    else:
        lamina_output_path = Path(LAMINA_OUTPUT_BINARY) # Output is at project root
        if not lamina_output_path.exists():
            print_color(RED, f"Error: Lamina compilation failed or output binary '{lamina_output_path}' not created.")
            results['Lamina'] = {'time': None, 'exit_code': -1, 'error': 'Output binary missing'}
        else:
            print_color(GREEN, "Lamina compilation successful.")
            if not ensure_executable(lamina_output_path):
                 results['Lamina'] = {'time': None, 'exit_code': -1, 'error': 'Execution permission error'}
            else:
                # Run Lamina executable from project root
                lamina_time, lamina_exit_code = run_single_benchmark("Lamina", [f"./{LAMINA_OUTPUT_BINARY}"], cwd=None)
                results['Lamina'] = {'time': lamina_time, 'exit_code': lamina_exit_code}


    # --- Define Other Benchmark Targets ---
    # {out} placeholder for output path relative to BENCHMARK_DIR
    # {src} placeholder for source file relative to BENCHMARK_DIR
    # All compile/run commands here assume execution *within* BENCHMARK_DIR unless cwd=None specified elsewhere
    targets = [
        {'lang': 'C',     'source': '2Dmatmul.c', 'output': '2Dmatmul_c',
         'compile': ['gcc', '-O3', '-o', '{out}', '{src}'], 'run': ['./{out}']},
        {'lang': 'C++',   'source': '2Dmatmul.cpp', 'output': '2Dmatmul_cpp',
         'compile': ['g++', '-O3', '-o', '{out}', '{src}'], 'run': ['./{out}']},
         {'lang': 'Rust',  'source': '2Dmatmul.rs', 'output': '2Dmatmul_rs',
          'compile': ['rustc', '-C', 'opt-level=3', '--out-dir', '.', '-o', '{out}', '{src}'], 'run': ['./{out}']}, # Added --out-dir .
        {'lang': 'Go',    'source': '2Dmatmul.go', 'output': '2Dmatmul_go',
         'compile': ['go', 'build', '-o', '{out}', '{src}'], 'run': ['./{out}']},
        {'lang': 'Zig',   'source': '2Dmatmul.zig', 'output': '2Dmatmul_zig',
         'compile': ['zig', 'build-exe', '{src}', '-O', 'ReleaseFast', '--name', '{out}'], 'run': ['./{out}']},
        {'lang': 'Java',  'source': 'MatMul2D.java', 'output': 'MatMul2D.class', # Compile produces .class
         'compile': ['javac', '{src}'],
         # Run needs class name (no extension), executed from BENCHMARK_DIR
         'run': ['java', 'MatMul2D']},
        # C# uses dotnet publish, outputs to a directory specified by --output
        # 'output' here specifies the relative path to the *executable file* within BENCHMARK_DIR
        #{'lang': 'C#',    'source': '2Dmatmul.cs', 'output': 'bin_cs/2Dmatmul',
        # 'compile': ['dotnet', 'publish', '{src}', '-c', 'Release', '--framework', 'net8.0', # Specify framework if needed
        #             '--output', 'bin_cs', # Output directory relative to BENCHMARK_DIR
        #             '/p:AssemblyName=2Dmatmul'], # Ensure executable name is predictable
        # 'run': ['./{out}']}, # Run command uses the 'output' path which points to the exe
        # --- Try C# with a temporary project file ---
        {'lang': 'C#', 'source': '2Dmatmul.cs',
         'compile': ['dotnet', 'new', 'console', '-n', '2Dmatmul', '--force', '--no-restore'], # Create project file first
         # Run using the specific project file name
         'run': ['dotnet', 'run', '-c', 'Release', '--project', '2Dmatmul.csproj']},
        # Interpreted languages run from project root (cwd=None in run_single_benchmark)
        # {src} will be replaced with full path relative to root for these
        {'lang': 'Python', 'source': '2Dmatmul.py', 'run': ['python3', '{src}']},
        {'lang': 'JavaScript', 'source': '2Dmatmul.js', 'run': ['node', '{src}']},
        {'lang': 'Ruby', 'source': '2Dmatmul.rb', 'run': ['ruby', '{src}']},
        {'lang': 'PHP', 'source': '2Dmatmul.php', 'run': ['php', '{src}']},
    ]

    # --- Compile and Run Other Benchmarks ---
    for target in targets:
        # Pass BENCHMARK_DIR as the cwd for compile_and_run context
        compile_and_run(target, results, BENCHMARK_DIR)


    # --- Summary ---\
    print_color(BLUE, "\\n===== Benchmark Summary =====")
    lamina_result = results.get('Lamina', {})
    lamina_time = lamina_result.get('time')

    # Table setup
    hdr_lang = "Language"
    hdr_time = "Time (s)"
    hdr_ratio = "Ratio vs Lamina"
    col_lang_width = 15
    col_time_width = 10
    col_ratio_width = 16
    total_width = col_lang_width + col_time_width + col_ratio_width + 4 # | Lang | Time | Ratio |

    # Box drawing characters
    T_DOWN = '┬'
    T_UP = '┴'
    T_CROSS = '┼'
    L_VERT = '│'
    L_HORZ = '─'
    C_TL = '┌'
    C_TR = '┐'
    C_BL = '└'
    C_BR = '┘'
    T_LEFT = '├'
    T_RIGHT = '┤'

    # Header
    print(f"{C_TL}{L_HORZ * total_width}{C_TR}")
    title = " Benchmark Results (Higher Ratio is Better) "
    print(f"{L_VERT}{title.center(total_width)}{L_VERT}")
    print(f"{T_LEFT}{L_HORZ*col_lang_width}{T_DOWN}{L_HORZ*(col_time_width+2)}{T_DOWN}{L_HORZ*col_ratio_width}{T_RIGHT}")

    if lamina_time is not None:
        lamina_line = f"{L_VERT} {GREEN}{'Lamina (Base)':<{col_lang_width-2}}{NC} {L_VERT} {lamina_time:>{col_time_width}.4f} {L_VERT} {'1.00x':>{col_ratio_width-2}} {L_VERT}"
        print(lamina_line)
    else:
        err_msg = lamina_result.get('error', 'Unknown')
        lamina_fail_line = f"{L_VERT} {RED}{'Lamina (Base)':<{col_lang_width-2}}{NC} {L_VERT} {'FAILED':^{col_time_width}} {L_VERT} {'N/A':^{col_ratio_width}} {L_VERT}"
        print(lamina_fail_line)
        print_color(RED, f"    Error: {err_msg}")
        print_color(RED, "    Cannot calculate ratios without successful Lamina baseline.")

    print(f"{T_LEFT}{L_HORZ*col_lang_width}{T_CROSS}{L_HORZ*(col_time_width+2)}{T_CROSS}{L_HORZ*col_ratio_width}{T_RIGHT}") # Separator after baseline

    successful_results = {}
    failed_results = {}

    for lang, result in results.items():
        if lang == 'Lamina': continue
        # Check for non-None time AND exit code 0 for success
        if result.get('time') is not None and result.get('exit_code', -1) == 0:
            successful_results[lang] = result
        else:
            # Ensure an error message exists if none was explicitly set
            if not result.get('error'):
                 result['error'] = f"Non-zero exit code ({result.get('exit_code', 'N/A')})"
            failed_results[lang] = result

    # Calculate ratios for successful runs if baseline exists
    ratios = {}
    if lamina_time is not None and lamina_time > 0:
        for lang, result in successful_results.items():
            lang_time = result['time']
            if lang_time is not None and lang_time >= 0: # Allow 0 time, treat as Inf ratio
                ratios[lang] = lang_time / lamina_time if lang_time > 0 else float('inf')
            else:
                 ratios[lang] = float('inf') # Handle None time

    # Sort successful results by ratio (lower is better)
    sorted_success = sorted(successful_results.keys(), key=lambda l: ratios.get(l, float('inf')))

    if sorted_success:
        for lang in sorted_success:
            result = successful_results[lang]
            lang_time = result['time']
            ratio = ratios.get(lang)

            ratio_str = "N/A"
            ratio_color = NC
            if ratio is not None:
                if ratio == float('inf'):
                    ratio_str = "Inf"
                    ratio_color = RED
                else:
                    ratio_str = f"{ratio:>7.2f}x"
                    # Updated color logic
                    if ratio < 0.95:
                         ratio_color = RED # Faster than baseline is suspicious/red
                    elif ratio < 1.05:
                         ratio_color = GREEN # Similar speed
                    elif ratio <= 10.0:
                         ratio_color = YELLOW # Moderately slower
                    else: # > 10.0x
                         ratio_color = GREEN # Significantly slower is GOOD
            else: # Should not happen if baseline exists, but safety check
                 ratio_color = NC
                 ratio_str = "Error"


            print(f"{L_VERT} {lang:<{col_lang_width-2}} {L_VERT} {lang_time:>{col_time_width}.4f} {L_VERT} {ratio_color}{ratio_str:>{col_ratio_width-2}}{NC} {L_VERT}")

    # Footer
    print(f"{C_BL}{L_HORZ*col_lang_width}{T_UP}{L_HORZ*(col_time_width+2)}{T_UP}{L_HORZ*col_ratio_width}{C_BR}")

    # Print failed benchmarks separately
    if failed_results:
        print_color(RED, "\n--- Failed Benchmarks ---")
        for lang, result in failed_results.items():
             exit_code = result.get('exit_code', 'N/A')
             error = result.get('error', 'Unknown Error')
             # Indent failed lines slightly
             print(f"  {RED}* {lang:<{col_lang_width}}: FAILED (Exit: {exit_code}, Error: {error}){NC}")
        # print("-" * 60)

    print_color(BLUE, "\\n===== Benchmark Complete =====")

if __name__ == "__main__":
    main()