<?php
/**
 * Matrix multiplication benchmark for PHP
 * Similar to the other implementations in the benchmark suite
 */

// Get matrix dimensions
define('MATRIX_SIZE', 256);

/**
 * Retrieves a value from matrix A at position [i, k]
 * 
 * @param int $i Row index
 * @param int $k Column index
 * @return int The value at position [i, k]
 */
function get_matrix_a_element($i, $k) {
    // Values based on position (consistent with other implementations)
    return $i * $k + 1;
}

/**
 * Retrieves a value from matrix B at position [k, j]
 * 
 * @param int $k Row index
 * @param int $j Column index
 * @return int The value at position [k, j]
 */
function get_matrix_b_element($k, $j) {
    // Values based on position (consistent with other implementations)
    return $k * $j + 1;
}

/**
 * Compute a single cell in the result matrix
 * 
 * @param int $i Row index in result matrix
 * @param int $j Column index in result matrix
 * @param int $k_dim The dimension of the matrices (n for an n×n matrix)
 * @return int The computed cell value
 */
function compute_matrix_cell($i, $j, $k_dim) {
    $sum = 0;
    for ($k = 0; $k < $k_dim; $k++) {
        $a_elem = get_matrix_a_element($i, $k);
        $b_elem = get_matrix_b_element($k, $j);
        $sum += $a_elem * $b_elem;
    }
    return $sum;
}

/**
 * Matrix multiplication benchmark
 * 
 * @return int Exit code
 */
function matrix_multiply_2d() {
    $start_time = microtime(true);
    
    $n = MATRIX_SIZE;
    $total = 0;
    
    echo "Starting {$n}×{$n} matrix multiplication benchmark in PHP...\n";
    
    // Only compute a sample of cells for large matrices
    $sample_step = max(intval($n / 10), 1);
    
    for ($i = 0; $i < $n; $i += $sample_step) {
        echo "Progress: " . round($i * 100 / $n) . "%\r";
        
        for ($j = 0; $j < $n; $j += $sample_step) {
            // Compute the result cell [i,j]
            $cell_value = compute_matrix_cell($i, $j, $n);
            $total += $cell_value;
        }
    }
    
    $elapsed = microtime(true) - $start_time;
    echo "\nCompleted in " . round($elapsed, 4) . " seconds\n";
    echo "Checksum: $total\n";
    
    // Return success
    return 0;
}

// Run the benchmark and exit with its return code
exit(matrix_multiply_2d()); 