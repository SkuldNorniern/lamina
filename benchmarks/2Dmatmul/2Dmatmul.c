#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Get element from matrix A at position [i,k]
int64_t get_matrix_a_element(int64_t i, int64_t k) {
    // Same deterministic pattern as in Lamina version
    return i * k + 1;
}

// Get element from matrix B at position [k,j]
int64_t get_matrix_b_element(int64_t k, int64_t j) {
    // Same deterministic pattern as in Lamina version
    return k * j + 1;
}

// Compute a single cell of the result matrix C[i,j]
int64_t compute_matrix_cell(int64_t i, int64_t j, int64_t k_dim) {
    int64_t sum = 0;
    
    // Standard matrix multiplication formula:
    // C[i,j] = sum(A[i,k] * B[k,j]) for all k
    for (int64_t k = 0; k < k_dim; k++) {
        int64_t a_elem = get_matrix_a_element(i, k);
        int64_t b_elem = get_matrix_b_element(k, j);
        sum += a_elem * b_elem;
    }
    
    return sum;
}

// Benchmark matrix multiplication C = A @ B
// A is (n_rows × k_dim), B is (k_dim × n_cols), C is (n_rows × n_cols)
int64_t matmul_2d(int64_t n_rows, int64_t k_dim, int64_t n_cols) {
    // Print matrix dimensions for verification
    printf("Matrix A: %ld × %ld\n", n_rows, k_dim);
    printf("Matrix B: %ld × %ld\n", k_dim, n_cols);
    printf("Result C: %ld × %ld\n", n_rows, n_cols);
    
    // Start timing
    printf("Starting computation...\n");
    clock_t start_time = clock();
    
    // Calculate expected result size and operations
    int64_t result_size = n_rows * n_cols;
    int64_t total_ops = result_size * k_dim;
    int64_t ops_done = 0;
    
    // Progress tracking
    int64_t cells_done = 0;
    int64_t next_report = result_size / 20;  // Report every 5%
    
    // Total sum for verification
    int64_t total_sum = 0;
    
    // Perform matrix multiplication
    for (int64_t i = 0; i < n_rows; i++) {
        for (int64_t j = 0; j < n_cols; j++) {
            // Compute this cell of the result matrix
            int64_t cell_result = compute_matrix_cell(i, j, k_dim);
            
            // Add to total sum
            total_sum += cell_result;
            
            // Update operations counter
            ops_done += k_dim;
            
            // Update completed cells counter
            cells_done++;
            
            // Report progress every 5%
            if (cells_done >= next_report) {
                double percent = (double)cells_done * 100.0 / result_size;
                double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                printf("Progress: %.1f%% (%.2f seconds elapsed)\n", percent, elapsed);
                next_report += result_size / 20;
            }
        }
    }
    
    // Calculate total time
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    
    // Print results
    printf("\nComputation complete!\n");
    printf("Total sum: %ld\n", total_sum);
    printf("Total operations: %ld\n", ops_done);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Operations per second: %.2f million\n", ops_done / (total_time * 1000000));
    
    return total_sum;
}

int main() {
    // Set matrix dimensions
    int64_t n_rows = 256;  // Rows in A
    int64_t k_dim = 256;   // Cols in A = Rows in B
    int64_t n_cols = 256;  // Cols in B
    
    printf("2D Matrix Multiplication Benchmark (NumPy matmul equivalent)\n");
    printf("===============================================\n");
    
    // Run the benchmark
    int64_t result = matmul_2d(n_rows, k_dim, n_cols);
    
    return 0;
}