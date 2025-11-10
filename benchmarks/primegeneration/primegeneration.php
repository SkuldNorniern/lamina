<?php
/**
 * Prime Generation benchmark for PHP (simplified counting)
 */

// Configuration - limits for counting
define('LIMIT1', 100);
define('LIMIT2', 1000);
define('LIMIT3', 10000);
define('LIMIT4', 50000);

// Markers matching Lamina version
define('HEADER_MARKER', 123456789);
define('FOOTER_MARKER', 987654321);

/**
 * Check if a number is prime
 *
 * @param int $n
 * @return bool
 */
function isPrime($n) {
    if ($n <= 1) return false;
    if ($n <= 3) return true;
    if ($n % 2 == 0) return false;

    for ($i = 3; $i * $i <= $n; $i += 2) {
        if ($n % $i == 0) return false;
    }
    return true;
}

/**
 * Count primes from 2 to limit inclusive
 *
 * @param int $limit
 * @return int
 */
function countPrimes($limit) {
    if ($limit < 2) return 0;
    $count = 0;
    for ($i = 2; $i <= $limit; $i++) {
        if (isPrime($i)) $count++;
    }
    return $count;
}

// Print header marker
echo HEADER_MARKER . "\n";

// Count and print prime counts for different limits
echo countPrimes(LIMIT1) . "\n";
echo countPrimes(LIMIT2) . "\n";
echo countPrimes(LIMIT3) . "\n";
echo countPrimes(LIMIT4) . "\n";

// Print footer marker
echo FOOTER_MARKER . "\n";
?>
