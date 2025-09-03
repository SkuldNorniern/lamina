<?php

// Configuration
define('N1', 10);
define('N2', 20);
define('N3', 30);
define('N4', 35);

// Markers matching Lamina version
define('HEADER_MARKER', 123456789);
define('FOOTER_MARKER', 987654321);

function fibonacci_iterative($n) {
    if ($n == 0) return 0;
    if ($n == 1) return 1;

    $a = 0;
    $b = 1;
    for ($i = 2; $i <= $n; $i++) {
        $temp = $a + $b;
        $a = $b;
        $b = $temp;
    }
    return $b;
}

// Print header marker
echo HEADER_MARKER . "\n";

// Compute and print fibonacci numbers
echo fibonacci_iterative(N1) . "\n";
echo fibonacci_iterative(N2) . "\n";
echo fibonacci_iterative(N3) . "\n";
echo fibonacci_iterative(N4) . "\n";

// Print footer marker
echo FOOTER_MARKER . "\n";

?>

