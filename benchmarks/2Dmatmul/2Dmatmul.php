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
    
    // Create a result matrix to store all computed values
    $result = array();
    $total = 0;
    
    echo "Starting {$n}×{$n} matrix multiplication benchmark in PHP...\n";
    
    // Progress tracking
    $progress_step = max(intval($n / 10), 1);
    $last_progress = 0;
    
    for ($i = 0; $i < $n; $i++) {
        // Show progress every 10% with a newline to ensure visibility
        if ($i % $progress_step == 0) {
            $progress = intval($i * 100 / $n);
            echo "Progress: {$progress}%\n";
            $last_progress = $progress;
        }
        
        // Initialize the row in the result matrix
        $result[$i] = array();
        
        for ($j = 0; $j < $n; $j++) {
            // Compute the result cell [i,j]
            $result[$i][$j] = compute_matrix_cell($i, $j, $n);
            $total += $result[$i][$j];
        }
    }
    
    // Make sure we show 100% progress if we didn't already
    if ($last_progress < 100) {
        echo "Progress: 100%\n";
    }
    
    $elapsed = microtime(true) - $start_time;
    echo "\nCompleted in " . round($elapsed, 4) . " seconds\n";
    echo "Total sum: $total\n";
    
    // Print some sample values from the result matrix
    echo "\nSample result values:\n";
    echo "result[0][0] = " . $result[0][0] . "\n";
    echo "result[1][1] = " . $result[1][1] . "\n";
    echo "result[10][10] = " . $result[10][10] . "\n";
    echo "result[100][100] = " . $result[100][100] . "\n";
    echo "result[" . ($n-1) . "][" . ($n-1) . "] = " . $result[$n-1][$n-1] . "\n";
    
    // Return success
    return 0;
}

// Run the benchmark and exit with its return code
exit(matrix_multiply_2d()); 