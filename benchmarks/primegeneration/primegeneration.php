<?php

const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

function is_prime(int $n): bool {
    if ($n <= 1) return false;
    if ($n == 2 || $n == 3) return true;
    if ($n % 2 == 0) return false;
    for ($d = 3; $d * $d <= $n; $d += 2) {
        if ($n % $d == 0) return false;
    }
    return true;
}

function count_primes(int $limit): int {
    $count = 0;
    for ($current = 2; $current <= $limit; $current++) {
        if (is_prime($current)) $count++;
    }
    return $count;
}

echo HEADER_MARKER, "\n";
echo count_primes(100), "\n";
echo count_primes(1000), "\n";
echo count_primes(10000), "\n";
echo count_primes(1000000), "\n";
echo FOOTER_MARKER, "\n";
