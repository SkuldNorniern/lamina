// tensor_benchmark.go
package main

import (
	"fmt"
	// "os" // Removed unused import
	// "strconv" // Removed unused import
)

// --- Configuration ---
const N_ROWS = 256
const K_DIM = 256
const N_COLS = 256

// --- Markers ---
// Use strings for large constant markers to avoid potential int conversion issues if very large
const HEADER_MARKER = "123456789"
const START_MARKER = "987654321"
const END_MARKER = "987654322"
const STATUS_MARKER = "987654323"

// Generates element A[i,k] deterministically.
// Using int64 to match Lamina/C potentially large intermediate values.
func getMatrixAElement(i, k int64) int64 {
	return (i * k) + 1
}

// Generates element B[k,j] deterministically.
func getMatrixBElement(k, j int64) int64 {
	return (k * j) + 1
}

// Performs matrix multiplication using standard Go loops.
func matmulGo(n_rows, k_dim, n_cols int) int64 {
	fmt.Println(n_rows)
	fmt.Println(k_dim)
	fmt.Println(n_cols)

	// Use int64 for calculations to prevent potential overflow
	resultSize := int64(n_rows) * int64(n_cols)
	var totalSum int64 = 0

	fmt.Println(START_MARKER) // Start timing after setup

	// --- Standard Go Implementation ---
	// Generate elements on the fly
	for i_idx := 0; i_idx < n_rows; i_idx++ {
		for j_idx := 0; j_idx < n_cols; j_idx++ {
			var cellSum int64 = 0
			i := int64(i_idx)
			j := int64(j_idx)
			for k_idx := 0; k_idx < k_dim; k_idx++ {
				k := int64(k_idx)
				aElem := getMatrixAElement(i, k)
				bElem := getMatrixBElement(k, j)
				cellSum += aElem * bElem
			}
			totalSum += cellSum
			// Note: No progress reporting added here for simplicity
		}
	}
	// --- End of benchmark operation ---

	// Calculate operations
	opsPerCell := int64(k_dim) * 2
	totalOps := resultSize * opsPerCell

	fmt.Println(END_MARKER) // End timing before final prints
	fmt.Println(totalSum)
	fmt.Println(totalOps)

	return totalSum
}

func main() {
	fmt.Println(HEADER_MARKER)
	_ = matmulGo(N_ROWS, K_DIM, N_COLS) // Run the benchmark
	fmt.Println(STATUS_MARKER)
	// Go programs automatically exit 0 on successful completion without explicit return
}
