<?php
/**
 * Factorial benchmark for PHP
 */

// Configuration
define('N1', 10);
define('N2', 12);
define('N3', 15);
define('N4', 18);

// Markers matching Lamina version
define('HEADER_MARKER', 123456789);
define('FOOTER_MARKER', 987654321);

/**
 * Compute factorial iteratively: n! = n * (n-1) * ... * 2 * 1
 *
 * @param int $n
 * @return int
 */
function factorial_iterative($n) {
    if ($n == 0 || $n == 1) {
        return 1;
    }

    $result = 1;
    for ($i = 2; $i <= $n; $i++) {
        $result *= $i;
    }
    return $result;
}

// Print header marker
echo HEADER_MARKER . "\n";

// Compute and print factorial values
echo factorial_iterative(N1) . "\n";
echo factorial_iterative(N2) . "\n";
echo factorial_iterative(N3) . "\n";
echo factorial_iterative(N4) . "\n";

// Print footer marker
echo FOOTER_MARKER . "\n";
?>



