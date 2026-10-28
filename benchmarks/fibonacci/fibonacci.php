<?php

const N1 = 10;
const N2 = 20;
const N3 = 30;
const N4 = 35;
const SWEEP_COUNT = 200000;
const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

function add_wrap(int $a, int $b): int {
    $lo = ($a & 0xFFFFFFFF) + ($b & 0xFFFFFFFF);
    $hi = (($a >> 32) & 0xFFFFFFFF) + (($b >> 32) & 0xFFFFFFFF) + ($lo >> 32);
    return (($hi & 0xFFFFFFFF) << 32) | ($lo & 0xFFFFFFFF);
}

function fibonacci_iterative(int $n): int {
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

echo HEADER_MARKER, "\n";
echo fibonacci_iterative(N1), "\n";
echo fibonacci_iterative(N2), "\n";
echo fibonacci_iterative(N3), "\n";
echo fibonacci_iterative(N4), "\n";

$acc = 0;
for ($s = 0; $s < SWEEP_COUNT; $s++) {
    $acc = add_wrap($acc, fibonacci_iterative($s % 90));
}
echo sprintf("%u", $acc), "\n";
echo FOOTER_MARKER, "\n";
