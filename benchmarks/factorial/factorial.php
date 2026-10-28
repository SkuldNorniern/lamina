<?php

const N1 = 10;
const N2 = 12;
const N3 = 15;
const N4 = 18;
const SWEEP_COUNT = 4000000;
const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

function add_wrap(int $a, int $b): int {
    $lo = ($a & 0xFFFFFFFF) + ($b & 0xFFFFFFFF);
    $hi = (($a >> 32) & 0xFFFFFFFF) + (($b >> 32) & 0xFFFFFFFF) + ($lo >> 32);
    return (($hi & 0xFFFFFFFF) << 32) | ($lo & 0xFFFFFFFF);
}

function factorial_iterative(int $n): int {
    if ($n == 0 || $n == 1) return 1;

    $result = 1;
    for ($i = 2; $i <= $n; $i++) {
        $result *= $i;
    }
    return $result;
}

echo HEADER_MARKER, "\n";
echo factorial_iterative(N1), "\n";
echo factorial_iterative(N2), "\n";
echo factorial_iterative(N3), "\n";
echo factorial_iterative(N4), "\n";

$acc = 0;
for ($s = 0; $s < SWEEP_COUNT; $s++) {
    $acc = add_wrap($acc, factorial_iterative($s % 20));
}
echo sprintf("%u", $acc), "\n";
echo FOOTER_MARKER, "\n";
