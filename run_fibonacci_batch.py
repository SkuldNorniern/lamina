#!/usr/bin/env python3

import subprocess
import os
import sys
import time
import stat
import platform
import statistics
import numpy as np
from pathlib import Path

# --- Configuration ---
BENCHMARK_DIR = Path("benchmarks/fibonacci") # Use Path for directory handling
LAMINA_EXECUTABLE = "./target/release/lamina"
LAMINA_SOURCE = "benchmarks/fibonacci/fibonacci.lamina" # Keep lamina source path relative to root
LAMINA_OUTPUT_BINARY = "fibonacci_lamina" # Consistent naming
BATCH_RUNS = 100  # Number of times to run each benchmark (fewer for fibonacci)
TIME_THRESHOLD = 1.0  # Seconds - fibonacci is faster than matrix multiplication

# --- Colors (ANSI escape codes) ---
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
MAGENTA = '\033[0;35m'  # For single-run markers
NC = '\033[0m' # No Color

# Box drawing characters
T_DOWN = '┬'
T_UP = '┴'
T_CROSS = '┼'
V = '│'  # Vertical line
L_VERT = '│'
L_HORZ = '─'
C_TL = '┌'  # Top left corner
C_TR = '┐'  # Top right corner
C_BL = '└'  # Bottom left corner
C_BR = '┘'  # Bottom right corner
L_LEFT = '├'  # T-shape pointing right
L_RIGHT = '┤'  # T-shape pointing left
C_CR = '┼'  # Cross

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
    """Runs and times a benchmark command list with memory usage tracking.
    Performs multiple runs for more accurate measurements unless the benchmark is too slow."""
    print_color(GREEN, f"\n--- Running {label} Implementation ---")

    # Determine the proper time command based on the OS
    is_mac = platform.system() == 'Darwin'
    time_flag = '-l' if is_mac else '-v'
    
    # Prefix with /usr/bin/time to measure memory usage
    time_command = ['/usr/bin/time', time_flag]
    
    # Full command with time prefix
    full_command = time_command + command_list

    # First run to check if it's too slow
    print_color(YELLOW, f"Initial run (1/{BATCH_RUNS}) for {label}...")
    start_time = time.perf_counter()
    result = run_command(full_command, check=False, cwd=cwd)
    end_time = time.perf_counter()
    first_run_time = end_time - start_time
    
    # Check if the benchmark failed
    if result.returncode != 0:
        print_color(RED, f"{label} Benchmark failed with Exit Code: {result.returncode}")
        return None, result.returncode, None, True, [], 0
    
    # Parse memory usage from stderr
    max_memory = parse_memory_usage(result.stderr, is_mac)
    
    # Store results of first run
    times = [first_run_time]
    memories = [max_memory] if max_memory is not None else []
    
    # Check if benchmark is too slow to run multiple times
    single_run_only = first_run_time > TIME_THRESHOLD
    if single_run_only:
        print_color(MAGENTA, f"{label} took {first_run_time:.4f}s > {TIME_THRESHOLD}s threshold, skipping additional runs.")
        print_color(YELLOW, f"{label} (Single Run) Execution Time: {first_run_time:.4f} seconds")
        if max_memory is not None:
            print_color(YELLOW, f"{label} Peak Memory Usage: {max_memory:.2f} MB")
        return first_run_time, result.returncode, max_memory, True, times, 0
    
    # Perform additional runs for more accurate measurements
    for run in range(2, BATCH_RUNS + 1):
        print_color(YELLOW, f"Run {run}/{BATCH_RUNS} for {label}...")
        start_time = time.perf_counter()
        result = run_command(full_command, check=False, cwd=cwd)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        # Check if the benchmark failed in subsequent runs
        if result.returncode != 0:
            print_color(RED, f"{label} Benchmark failed on run {run} with Exit Code: {result.returncode}")
            # We still have data from previous successful runs, so don't return None
            break
        
        times.append(run_time)
        mem = parse_memory_usage(result.stderr, is_mac)
        if mem is not None:
            memories.append(mem)
    
    # Calculate statistics
    if len(times) > 0:
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        median_time = statistics.median(times)
        
        # Detect outliers using IQR method if we have enough samples
        outliers = []
        if len(times) >= 10:
            q1 = np.percentile(times, 25)
            q3 = np.percentile(times, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [t for t in times if t < lower_bound or t > upper_bound]
            if outliers:
                print_color(YELLOW, f"Detected {len(outliers)} outliers in {label} runs: {[f'{o:.4f}' for o in outliers]}")
                
                # If there are outliers, also calculate stats without outliers for comparison
                clean_times = [t for t in times if t not in outliers]
                clean_min = min(clean_times)
                clean_max = max(clean_times)
                clean_avg = sum(clean_times) / len(clean_times)
                clean_median = statistics.median(clean_times)
                print_color(YELLOW, f"{label} Clean Times (no outliers): Min: {clean_min:.4f}, Max: {clean_max:.4f}, Avg: {clean_avg:.4f}, Med: {clean_median:.4f}")
        
        # Add percentile calculations
        if len(times) > 5:  # Only calculate percentiles with enough samples
            p90_time = np.percentile(times, 90)
            p95_time = np.percentile(times, 95)
            print_color(YELLOW, f"{label} Times (seconds): Min: {min_time:.4f}, Max: {max_time:.4f}, Avg: {avg_time:.4f}, Med: {median_time:.4f}, P90: {p90_time:.4f}, P95: {p95_time:.4f}")
        else:
            print_color(YELLOW, f"{label} Times (seconds): Min: {min_time:.4f}, Max: {max_time:.4f}, Avg: {avg_time:.4f}, Median: {median_time:.4f}")
        
        if memories:
            avg_memory = sum(memories) / len(memories)
            print_color(YELLOW, f"{label} Avg Peak Memory Usage: {avg_memory:.2f} MB")
        else:
            avg_memory = None
            print_color(YELLOW, f"{label} Peak Memory Usage: Unknown")
        
        # If there were outliers, also report the version without them
        if outliers and len(times) >= 10:
            return clean_avg, result.returncode, avg_memory, single_run_only, clean_times if clean_times else times, len(outliers)
        else:
            return avg_time, result.returncode, avg_memory, single_run_only, times, 0
    else:
        print_color(RED, f"{label} No successful runs")
        return None, -1, None, True, [], 0

def parse_memory_usage(stderr, is_mac):
    """Parse memory usage from stderr output of time command."""
    if not stderr:
        return None
        
    max_memory = None
    if is_mac:
        # On macOS, parse memory from the -l output format
        for line in stderr.splitlines():
            if 'maximum resident set size' in line.lower():
                try:
                    # macOS reports in bytes
                    max_memory = int(line.split()[0]) / 1024 / 1024  # Convert to MB
                    break
                except (ValueError, IndexError):
                    pass
    else:
        # On Linux, parse memory from the -v output format
        for line in stderr.splitlines():
            if 'maximum resident set size' in line.lower():
                try:
                    # Linux typically reports in KB
                    max_memory = float(line.split(':')[1].strip()) / 1024  # Convert to MB
                    break
                except (ValueError, IndexError):
                    pass
    
    return max_memory

def create_csharp_project_file(benchmark_dir):
    """Creates a minimal C# project file for the benchmark."""
    csproj_path = benchmark_dir / "fibonacci.csproj"
    csproj_content = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AssemblyName>Fibonacci</AssemblyName>
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
        results[label] = {'time': None, 'exit_code': -1, 'error': 'Source not found', 'memory': None, 
                         'single_run': True, 'times': []}
        return

    # --- Special handling for C# ---
    if label == "C#":
        # Create a proper project file for dotnet run
        if not create_csharp_project_file(cwd):
            results[label] = {'time': None, 'exit_code': -1, 'error': 'Failed to create project file', 
                             'memory': None, 'single_run': True, 'times': []}
            return

    # --- Determine Paths Relative to CWD for commands ---
    # Use relative paths for commands executed within the benchmark directory
    src_for_cmd = source_file_rel
    out_for_cmd = output_path_rel

    # --- Compilation Step ---
    if compile_cmd_template:
        print_color(YELLOW, f"Compiling {label}...")
        
        # Handle special case for C# with multiple commands
        if label == "C#" and isinstance(compile_cmd_template, list) and len(compile_cmd_template) > 1:
            # For C#, we run multiple commands in sequence
            
            # First, create the project (if not already done in special handling)
            create_cmd = ['dotnet', 'new', 'console', '-n', 'Fibonacci', '--force', '--no-restore']
            print_color(YELLOW, f"Creating C# project with command: {' '.join(create_cmd)}")
            create_result = run_command(create_cmd, check=False, cwd=cwd)
            
            if create_result.returncode != 0:
                print_color(RED, f"{label} project creation failed.")
                results[label] = {'time': None, 'exit_code': create_result.returncode, 
                                 'error': 'Project creation failed', 'memory': None,
                                 'single_run': True, 'times': []}
                return
                
            # Then build the project in Release mode
            build_cmd = ['dotnet', 'build', '-c', 'Release', 'fibonacci.csproj']
            print_color(YELLOW, f"Building C# project with command: {' '.join(build_cmd)}")
            compile_result = run_command(build_cmd, check=False, cwd=cwd)
            
            if compile_result.returncode != 0:
                print_color(RED, f"{label} compilation failed.")
                results[label] = {'time': None, 'exit_code': compile_result.returncode, 
                                 'error': 'Compilation failed', 'memory': None,
                                 'single_run': True, 'times': []}
                return
        else:
            # Regular single compilation command for other languages
            # Replace placeholders with paths relative to the cwd (BENCHMARK_DIR)
            compile_cmd_full = [
                str(c).replace('{src}', src_for_cmd).replace('{out}', str(out_for_cmd) if out_for_cmd else "")
                for c in compile_cmd_template
            ]

            compile_result = run_command(compile_cmd_full, check=False, cwd=cwd) # Run compile command in BENCHMARK_DIR

            if compile_result.returncode != 0:
                print_color(RED, f"{label} compilation failed.")
                results[label] = {'time': None, 'exit_code': compile_result.returncode, 
                                 'error': 'Compilation failed', 'memory': None,
                                 'single_run': True, 'times': []}
                return

        # Verify compilation output if expected
        if output_path_rel and label != "C#":  # Skip output check for C# as it's handled differently
            output_binary_abs = cwd / output_path_rel
            if not output_binary_abs.exists():
                print_color(RED, f"Error: {label} compilation did not produce expected output: {output_binary_abs}")
                results[label] = {'time': None, 'exit_code': -1, 'error': 'Output binary missing', 
                                 'memory': None, 'single_run': True, 'times': []}
                return
            ensure_executable(output_binary_abs)
        
        # For C#, check the output explicitly
        if label == "C#" and output_path_rel:
            output_binary_abs = cwd / output_path_rel
            if not output_binary_abs.exists():
                print_color(RED, f"Error: {label} compilation did not produce expected output: {output_binary_abs}")
                results[label] = {'time': None, 'exit_code': -1, 'error': 'Output binary missing', 
                                 'memory': None, 'single_run': True, 'times': []}
                return
            ensure_executable(output_binary_abs)

        print_color(GREEN, f"{label} compilation successful.")


    # --- Execution Step ---
    # Determine the actual command and working directory for execution
    exec_cwd = None # Default: run from project root (for interpreters)
    run_cmd_actual = []

    # Special case: Java needs to run 'java ClassName' from the directory containing the .class file
    if label == "Java":
        # run_cmd_template is already correctly defined as ['java', 'Fibonacci']
        run_cmd_actual = run_cmd_template
        exec_cwd = cwd # Run 'java' command from BENCHMARK_DIR
    # Handle C# special case
    elif label == "C#":
        # For C# we need to use the directly built executable
        if output_path_rel:
            exec_path_rel = f"./{output_path_rel}" # e.g., ./bin/Release/net9.0/Fibonacci
            run_cmd_actual = [exec_path_rel] + run_cmd_template[1:]
            exec_cwd = cwd # Run the executable from within BENCHMARK_DIR
            
            # Make sure the executable exists and has execution permissions
            exec_file_abs = cwd / output_path_rel
            print_color(YELLOW, f"Looking for C# executable at: {exec_file_abs}")
            if not exec_file_abs.exists():
                print_color(RED, f"Could not find C# executable: {exec_file_abs}")
                results[label] = {'time': None, 'exit_code': -1, 'error': 'Executable not found', 
                                 'memory': None, 'single_run': True, 'times': []}
                return
            ensure_executable(exec_file_abs)
        else:
            print_color(RED, f"No output path specified for C#")
            results[label] = {'time': None, 'exit_code': -1, 'error': 'No output path', 
                             'memory': None, 'single_run': True, 'times': []}
            return
    # Handle other compiled languages
    elif output_path_rel:
        # Use relative path for execution command when running within BENCHMARK_DIR
        exec_path_rel = f"./{output_path_rel}" # e.g., ./fibonacci_c or ./fibonacci
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
             results[label] = {'time': None, 'exit_code': -1, 'error': 'Execution permission error', 
                              'memory': None, 'single_run': True, 'times': []}
             return

    print_color(YELLOW, f"Executing {label} with command: {' '.join(str(c) for c in run_cmd_actual)} {'in '+str(exec_cwd) if exec_cwd else ''}")
    exec_time, exit_code, memory_mb, single_run, times, outliers = run_single_benchmark(label, run_cmd_actual, cwd=exec_cwd)
    results[label] = {
        'time': exec_time, 
        'exit_code': exit_code, 
        'memory': memory_mb, 
        'single_run': single_run,
        'times': times,
        'outliers': outliers
    }


def main():
    """Main function to run the benchmark steps."""
    print("===== Multi-Language Fibonacci Benchmark =====")
    
    # Check for command-line arguments
    test_single_lang = None
    if len(sys.argv) > 1:
        test_single_lang = sys.argv[1]
        print(f"Testing only: {test_single_lang}")
        
    print("Directory check...")

    # Ensure benchmark directory exists
    if not BENCHMARK_DIR.is_dir():
        print(f"Benchmark directory not found: {BENCHMARK_DIR}")
        sys.exit(1)

    # --- Clean previous build artifacts (optional but recommended) ---
    print_color(YELLOW, "Cleaning previous build artifacts...")
    artifacts_to_clean = [
        "fibonacci_lamina", # Lamina output at root
        BENCHMARK_DIR / "fibonacci_c",
        BENCHMARK_DIR / "fibonacci_cpp",
        BENCHMARK_DIR / "fibonacci_rs",
        BENCHMARK_DIR / "fibonacci_go",
        BENCHMARK_DIR / "fibonacci_zig",
        BENCHMARK_DIR / "Fibonacci.class",
        BENCHMARK_DIR / "bin", # C# output directory
        BENCHMARK_DIR / "fibonacci.csproj", # Temporary C# project file
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


    results = {} # Store results: {lang: {'time': float|None, 'exit_code': int, 'error': str|None, 'memory': float|None}}

    # --- Compile and Run Lamina (Baseline) ---
    if not test_single_lang or test_single_lang.lower() == "lamina":
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
             results['Lamina'] = {'time': None, 'exit_code': compile_result.returncode, 
                                 'error': 'Compilation failed', 'memory': None,
                                 'single_run': True, 'times': []}
        else:
            lamina_output_path = Path(LAMINA_OUTPUT_BINARY) # Output is at project root
            if not lamina_output_path.exists():
                print_color(RED, f"Error: Lamina compilation failed or output binary '{lamina_output_path}' not created.")
                results['Lamina'] = {'time': None, 'exit_code': -1, 'error': 'Output binary missing', 
                                    'memory': None, 'single_run': True, 'times': []}
            else:
                print_color(GREEN, "Lamina compilation successful.")
                if not ensure_executable(lamina_output_path):
                     results['Lamina'] = {'time': None, 'exit_code': -1, 'error': 'Execution permission error', 
                                         'memory': None, 'single_run': True, 'times': []}
                else:
                    # Run Lamina executable from project root
                    lamina_time, lamina_exit_code, lamina_memory, single_run, times, outliers = run_single_benchmark(
                        "Lamina", [f"./{LAMINA_OUTPUT_BINARY}"], cwd=None)
                    results['Lamina'] = {
                        'time': lamina_time, 
                        'exit_code': lamina_exit_code, 
                        'memory': lamina_memory,
                        'single_run': single_run,
                        'times': times,
                        'outliers': outliers
                    }
    else:
        print_color(YELLOW, "Skipping Lamina compilation (not testing)")
        # Add a placeholder result for Lamina
        results['Lamina'] = {'time': 1.0, 'exit_code': 0, 'memory': 100.0, 'single_run': True, 'times': [1.0], 'outliers': 0}


    # --- Define Other Benchmark Targets ---
    # {out} placeholder for output path relative to BENCHMARK_DIR
    # {src} placeholder for source file relative to BENCHMARK_DIR
    # All compile/run commands here assume execution *within* BENCHMARK_DIR unless cwd=None specified elsewhere
    targets = [
        {'lang': 'C',     'source': 'fibonacci.c', 'output': 'fibonacci_c',
         'compile': ['gcc', '-O3', '-o', '{out}', '{src}'], 'run': ['./{out}']},
        {'lang': 'C++',   'source': 'fibonacci.cpp', 'output': 'fibonacci_cpp',
         'compile': ['g++', '-O3', '-o', '{out}', '{src}'], 'run': ['./{out}']},
         {'lang': 'Rust',  'source': 'fibonacci.rs', 'output': 'fibonacci_rs',
          'compile': ['rustc', '-C', 'opt-level=3', '--out-dir', '.', '-o', '{out}', '{src}'], 'run': ['./{out}']}, # Added --out-dir .
        {'lang': 'Go',    'source': 'fibonacci.go', 'output': 'fibonacci_go',
         'compile': ['go', 'build', '-o', '{out}', '{src}'], 'run': ['./{out}']},
        {'lang': 'Zig',   'source': 'fibonacci.zig', 'output': 'fibonacci_zig',
         'compile': ['zig', 'build-exe', '{src}', '-O', 'ReleaseFast', '--name', '{out}'], 'run': ['./{out}']},
        {'lang': 'Nim',   'source': 'fibonacci.nim', 'output': 'fibonacci',
         'compile': ['nim', 'c', '-d:release', '--out:{out}', '{src}'], 'run': ['./{out}']},
        {'lang': 'Java',  'source': 'Fibonacci.java', 'output': 'Fibonacci.class', # Compile produces .class
         'compile': ['javac', '{src}'],
         # Run needs class name (no extension), executed from BENCHMARK_DIR
         'run': ['java', 'Fibonacci']},
        # Updated C# configuration to build and then run the executable directly
        {'lang': 'C#', 'source': 'fibonacci.cs', 'output': 'bin/Release/net9.0/Fibonacci',
         'compile': ['dotnet', 'build'],  # This is a placeholder, actual commands handled in compile_and_run
         'run': ['./{out}']},
        # Interpreted languages run from project root (cwd=None in run_single_benchmark)
        # {src} will be replaced with full path relative to root for these
        {'lang': 'Python', 'source': 'fibonacci.py', 'run': ['python3', '{src}']},
        {'lang': 'JavaScript', 'source': 'fibonacci.js', 'run': ['node', '{src}']},
        {'lang': 'Ruby', 'source': 'fibonacci.rb', 'run': ['ruby', '{src}']},
        {'lang': 'PHP', 'source': 'fibonacci.php', 'run': ['php', '{src}']},
    ]

    # --- Compile and Run Other Benchmarks ---
    for target in targets:
        lang = target['lang']
        # Skip languages that don't match the test_single_lang if it's set
        if test_single_lang and lang.lower() != test_single_lang.lower():
            print_color(YELLOW, f"Skipping {lang} (not testing)")
            continue
            
        # Pass BENCHMARK_DIR as the cwd for compile_and_run context
        compile_and_run(target, results, BENCHMARK_DIR)


    # --- Summary ---\
    print_color(BLUE, "===== Benchmark Summary =====")
    lamina_result = results.get('Lamina', {})
    lamina_time = lamina_result.get('time')
    lamina_memory = lamina_result.get('memory')
    lamina_single_run = lamina_result.get('single_run', True)
    lamina_times = lamina_result.get('times', [])
    lamina_outliers = lamina_result.get('outliers', 0)

    # Table setup
    col_lang_width = 15
    col_time_width = 9
    col_time_stats_width = 28  # Increased to accommodate outlier info
    col_ratio_width = 9
    col_ratio_stats_width = 26
    col_memory_width = 10
    col_mem_ratio_width = 10
    total_width = col_lang_width + col_time_width + col_time_stats_width + col_ratio_width + col_ratio_stats_width + col_memory_width + col_mem_ratio_width + 14  # Added border characters

    # Table headers
    print(f"{C_TL}{L_HORZ*total_width}{C_TR}")
    print(f"{V}{f'Fibonacci Sequence Computation Benchmark Results (Higher ratio is Better for Lamina) [{BATCH_RUNS} runs]':^{total_width}}{V}")
    print(f"{L_LEFT}{L_HORZ*col_lang_width}{T_DOWN}{L_HORZ*(col_time_width+2)}{T_DOWN}{L_HORZ*(col_time_stats_width+2)}{T_DOWN}{L_HORZ*(col_ratio_width+2)}{T_DOWN}{L_HORZ*(col_ratio_stats_width+2)}{T_DOWN}{L_HORZ*col_memory_width}{T_DOWN}{L_HORZ*col_mem_ratio_width}{L_RIGHT}")
    print(f"{V} {'Language':<{col_lang_width-2}} {V} {'Time (s)':<{col_time_width}} {V} {'Time Stats':<{col_time_stats_width}} {V} {'Ratio':<{col_ratio_width}} {V} {'Ratio Stats':<{col_ratio_stats_width}} {V} {'Memory':<{col_memory_width-2}} {V} {'MemRatio':<{col_mem_ratio_width-2}} {V}")
    print(f"{L_LEFT}{L_HORZ*col_lang_width}{C_CR}{L_HORZ*(col_time_width+2)}{C_CR}{L_HORZ*(col_time_stats_width+2)}{C_CR}{L_HORZ*(col_ratio_width+2)}{C_CR}{L_HORZ*(col_ratio_stats_width+2)}{C_CR}{L_HORZ*col_memory_width}{C_CR}{L_HORZ*col_mem_ratio_width}{L_RIGHT}")

    if lamina_time is not None:
        memory_str = f"{lamina_memory:.2f}" if lamina_memory is not None else "N/A"
        
        # Format time value
        time_value = f"{lamina_time:.4f}"
        if lamina_single_run:
            time_value += "*"
        elif lamina_outliers > 0:
            time_value += "†"  # Mark if outliers were removed
            
        # Format time stats
        if lamina_times and len(lamina_times) > 1:
            min_time = min(lamina_times)
            max_time = max(lamina_times)
            median_time = statistics.median(lamina_times)
            p95_time = np.percentile(lamina_times, 95) if len(lamina_times) > 5 else max_time
            time_stats = f"{min_time:.4f}/{median_time:.4f}/{lamina_time:.4f}/{p95_time:.4f}"
        else:
            time_stats = "N/A (single run)"
            
        lamina_line = f"{L_VERT} {GREEN}{'Lamina (Base)':<{col_lang_width-2}}{NC} {L_VERT} {time_value:^{col_time_width}} {L_VERT} {time_stats:^{col_time_stats_width}} {L_VERT} {'1.00x':^{col_ratio_width}} {L_VERT} {'N/A':^{col_ratio_stats_width}} {L_VERT} {memory_str:>{col_memory_width-2}} {L_VERT} {'1.00x':>{col_mem_ratio_width-2}} {L_VERT}"
        print(lamina_line)
    else:
        err_msg = lamina_result.get('error', 'Unknown')
        lamina_fail_line = f"{L_VERT} {RED}{'Lamina (Base)':<{col_lang_width-2}}{NC} {L_VERT} {'FAILED':^{col_time_width}} {L_VERT} {'N/A':^{col_time_stats_width}} {L_VERT} {'N/A':^{col_ratio_width}} {L_VERT} {'N/A':^{col_ratio_stats_width}} {L_VERT} {'N/A':^{col_memory_width-2}} {L_VERT} {'N/A':^{col_mem_ratio_width-2}} {L_VERT}"
        print(lamina_fail_line)
        print_color(RED, f"    Error: {err_msg}")
        print_color(RED, "    Cannot calculate ratios without successful Lamina baseline.")

    print(f"{L_LEFT}{L_HORZ*col_lang_width}{T_CROSS}{L_HORZ*(col_time_width+2)}{T_CROSS}{L_HORZ*(col_time_stats_width+2)}{T_CROSS}{L_HORZ*(col_ratio_width+2)}{T_CROSS}{L_HORZ*(col_ratio_stats_width+2)}{T_CROSS}{L_HORZ*col_memory_width}{T_CROSS}{L_HORZ*col_mem_ratio_width}{L_RIGHT}") # Separator after baseline

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
    time_ratios = {}
    ratio_stats = {}  # To store min/avg/med/max ratio stats
    memory_ratios = {}
    if lamina_time is not None and lamina_time > 0 and lamina_times:
        lamina_avg_time = lamina_time  # This is already the average time
        
        for lang, result in successful_results.items():
            lang_time = result['time']
            lang_times = result.get('times', [])
            
            if lang_time is not None and lang_time > 0 and lang_times: 
                # Calculate individual ratios for each run
                individual_ratios = []
                for run_time in lang_times:
                    if run_time > 0:  # Avoid division by zero
                        ratio = run_time / lamina_avg_time
                        individual_ratios.append(ratio)
                
                if individual_ratios:
                    # Calculate ratio statistics
                    min_ratio = min(individual_ratios)
                    max_ratio = max(individual_ratios)
                    avg_ratio = sum(individual_ratios) / len(individual_ratios)
                    med_ratio = statistics.median(individual_ratios)
                    
                    # Add percentile calculations if enough samples
                    p90_ratio = p95_ratio = None
                    if len(individual_ratios) > 5:
                        p90_ratio = np.percentile(individual_ratios, 90)
                        p95_ratio = np.percentile(individual_ratios, 95)
                    
                    # Store the average ratio for sorting
                    time_ratios[lang] = avg_ratio
                    # Store all ratio stats
                    ratio_stats[lang] = {
                        'min': min_ratio,
                        'max': max_ratio,
                        'avg': avg_ratio,
                        'med': med_ratio,
                        'p90': p90_ratio,
                        'p95': p95_ratio
                    }
                else:
                    time_ratios[lang] = None
                    ratio_stats[lang] = None
            else:
                # If time is 0 or None, can't determine ratio
                time_ratios[lang] = None
                ratio_stats[lang] = None

    if lamina_memory is not None and lamina_memory > 0:
        for lang, result in successful_results.items():
            lang_memory = result['memory']
            if lang_memory is not None and lang_memory > 0:
                # Calculate how much more memory the other language uses (higher is better for Lamina)
                memory_ratios[lang] = lang_memory / lamina_memory
            else:
                memory_ratios[lang] = None

    # Sort successful results by time ratio (higher is better for Lamina now)
    sorted_success = sorted(successful_results.keys(), key=lambda l: time_ratios.get(l, 0.0), reverse=True)

    if sorted_success:
        for lang in sorted_success:
            result = successful_results[lang]
            lang_time = result['time']
            lang_memory = result['memory']
            lang_single_run = result.get('single_run', True)
            lang_times = result.get('times', [])
            time_ratio = time_ratios.get(lang)
            ratio_stat = ratio_stats.get(lang)
            memory_ratio = memory_ratios.get(lang)

            # Format time value
            time_value = f"{lang_time:.4f}"
            if lang_single_run:
                time_value += "*"
            elif result.get('outliers', 0) > 0:
                time_value += "†"  # Mark if outliers were removed
                
            # Format time stats
            if lang_times and len(lang_times) > 1 and not lang_single_run:
                min_time = min(lang_times)
                median_time = statistics.median(lang_times)
                p95_time = np.percentile(lang_times, 95) if len(lang_times) > 5 else max(lang_times)
                time_stats = f"{min_time:.4f}/{median_time:.4f}/{lang_time:.4f}/{p95_time:.4f}"
            else:
                time_stats = "N/A (single run)"

            # Format ratio value
            ratio_value = "N/A"
            ratio_stats_str = "N/A"
            ratio_color = NC
            
            if ratio_stat is not None:
                # Main ratio value (avg)
                ratio_value = f"{time_ratio:.2f}x"
                
                # Color based on average ratio
                if time_ratio > 5.0:
                    ratio_color = GREEN     # Other language is much slower (great for Lamina)
                elif time_ratio > 1.05:
                    ratio_color = YELLOW    # Other language is somewhat slower (good for Lamina)
                else:                           
                    ratio_color = RED       # Other language is similar or faster (bad for Lamina)
                
                # Format ratio stats
                if len(lang_times) > 1 and not lang_single_run:
                    p95_ratio = ratio_stat['p95'] if ratio_stat['p95'] is not None else ratio_stat['max']
                    ratio_stats_str = f"{ratio_stat['min']:.2f}/{ratio_stat['med']:.2f}/{ratio_stat['avg']:.2f}/{p95_ratio:.2f}x"
            
            # Format memory stats
            memory_str = "N/A"
            memory_ratio_str = "N/A"
            memory_ratio_color = NC
            
            if lang_memory is not None:
                memory_str = f"{lang_memory:.2f}"
                
                if memory_ratio is not None:
                    memory_ratio_str = f"{memory_ratio:.2f}x"
                    # Memory color logic (higher is better for Lamina)
                    if memory_ratio > 2.0:
                        memory_ratio_color = GREEN    # Other language uses much more memory
                    elif memory_ratio > 1.05:
                        memory_ratio_color = YELLOW   # Other language uses somewhat more memory
                    else:
                        memory_ratio_color = RED      # Other language uses similar or less memory
                
            print(f"{L_VERT} {lang:<{col_lang_width-2}} {L_VERT} {time_value:^{col_time_width}} {L_VERT} {time_stats:^{col_time_stats_width}} {L_VERT} {ratio_color}{ratio_value:^{col_ratio_width}}{NC} {L_VERT} {ratio_color}{ratio_stats_str:^{col_ratio_stats_width}}{NC} {L_VERT} {memory_str:>{col_memory_width-2}} {L_VERT} {memory_ratio_color}{memory_ratio_str:>{col_mem_ratio_width-2}}{NC} {L_VERT}")

    # Footer
    print(f"{C_BL}{L_HORZ*col_lang_width}{T_UP}{L_HORZ*(col_time_width+2)}{T_UP}{L_HORZ*(col_time_stats_width+2)}{T_UP}{L_HORZ*(col_ratio_width+2)}{T_UP}{L_HORZ*(col_ratio_stats_width+2)}{T_UP}{L_HORZ*col_memory_width}{T_UP}{L_HORZ*col_mem_ratio_width}{C_BR}")
    print(f"Stats format: min/median/avg/p95  (* benchmark only ran once due to exceeding the {TIME_THRESHOLD}s threshold)")
    print(f"† Statistical outliers were detected and removed using IQR method")
    
    # Print outlier summary if any exist
    outlier_langs = [(lang, result.get('outliers', 0)) for lang, result in results.items() if result.get('outliers', 0) > 0]
    if outlier_langs:
        outlier_summary = ", ".join([f"{lang}: {count}" for lang, count in outlier_langs])
        print(f"Outliers removed: {outlier_summary}")
        
    # Print failed benchmarks separately
    if failed_results:
        print_color(RED, "--- Failed Benchmarks ---")
        for lang, result in failed_results.items():
            exit_code = result.get('exit_code', 'N/A')
            error = result.get('error', 'Unknown Error')
            # Indent failed lines slightly
            print(f"  {RED}* {lang:<{col_lang_width}}: FAILED (Exit: {exit_code}, Error: {error}){NC}")

    print_color(BLUE, "===== Benchmark Complete =====")

if __name__ == "__main__":
    main()