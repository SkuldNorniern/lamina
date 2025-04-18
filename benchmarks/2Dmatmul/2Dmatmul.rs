// tensor_benchmark.rs
use std::env; // Not needed for logic, but often included
use std::process;
use std::time::Instant; // For potential internal timing (optional)

// --- Configuration ---
const N_ROWS: usize = 256;
const K_DIM: usize = 256;
const N_COLS: usize = 256;

// --- Markers ---
// Use strings for consistency with other scripts' output parsing if needed,
// though direct printing of numbers is also fine.
const HEADER_MARKER: i64 = 123456789;
const START_MARKER: i64 = 987654321;
const END_MARKER: i64 = 987654322;
const STATUS_MARKER: i64 = 987654323;

// Generates element A[i,k] deterministically.
// Use i64 to match Lamina/C/Go for potential large intermediate values.
#[inline] // Suggest inlining for performance critical function
fn get_matrix_a_element(i: i64, k: i64) -> i64 {
	(i * k) + 1
}

// Generates element B[k,j] deterministically.
#[inline]
fn get_matrix_b_element(k: i64, j: i64) -> i64 {
	(k * j) + 1
}

// Performs matrix multiplication using standard Rust loops.
fn matmul_rust(n_rows: usize, k_dim: usize, n_cols: usize) -> i64 {
	println!("{}", n_rows);
	println!("{}", k_dim);
	println!("{}", n_cols);

	// Use i64 for calculations to prevent potential overflow and match others
	let result_size: i64 = (n_rows * n_cols) as i64;
	let mut total_sum: i64 = 0;

	println!("{}", START_MARKER); // Start timing after setup

	// --- Standard Rust Implementation ---
	// Generate elements on the fly
	for i_idx in 0..n_rows {
		for j_idx in 0..n_cols {
			let mut cell_sum: i64 = 0;
			let i = i_idx as i64;
			let j = j_idx as i64;
			for k_idx in 0..k_dim {
				let k = k_idx as i64;
				let a_elem = get_matrix_a_element(i, k);
				let b_elem = get_matrix_b_element(k, j);
				// Use wrapping_add/mul if overflow is a possibility and desired behavior,
				// otherwise default Rust checks for overflow in debug, panics.
				// Release builds (-O) might remove checks depending on settings.
				// Let's assume standard ops are okay for this benchmark logic.
				cell_sum = cell_sum.saturating_add(a_elem.saturating_mul(b_elem));
			}
			total_sum = total_sum.saturating_add(cell_sum);
			// Note: No progress reporting added here for simplicity
		}
	}
	// --- End of benchmark operation ---

	// Calculate operations
	let ops_per_cell: i64 = (k_dim * 2) as i64;
	let total_ops: i64 = result_size.saturating_mul(ops_per_cell);

	println!("{}", END_MARKER); // End timing before final prints
	println!("{}", total_sum);
	println!("{}", total_ops);

	return total_sum;
}


fn main() {
	println!("{}", HEADER_MARKER);
	let _result = matmul_rust(N_ROWS, K_DIM, N_COLS); // Run the benchmark
	println!("{}", STATUS_MARKER);
	// Rust automatically exits 0 on successful main completion
}
