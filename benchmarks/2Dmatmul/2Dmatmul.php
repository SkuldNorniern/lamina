<?php

const N_ROWS = 256;
const K_DIM = 256;
const N_COLS = 256;

function mix(int $x): int {
    $a = ($x * 1103515245 + 12345) & 2147483647;
    $b = $a ^ ($a >> 7);
    $c = ($b * 1103515245 + 12345) & 2147483647;
    $d = $c ^ ($c >> 11);
    return $d & 65535;
}

function get_matrix_a_element(int $i, int $k): int {
    return mix($i * K_DIM + $k);
}

function get_matrix_b_element(int $k, int $j): int {
    return mix($k * N_COLS + $j);
}

function compute_matrix_cell(int $i, int $j, int $k_dim): int {
    $sum = 0;
    for ($k = 0; $k < $k_dim; $k++) {
        $sum += get_matrix_a_element($i, $k) * get_matrix_b_element($k, $j);
    }
    return $sum;
}

function matmul_2d(int $n_rows, int $k_dim, int $n_cols): int {
    echo $n_rows, "\n";
    echo $k_dim, "\n";
    echo $n_cols, "\n";

    $total_sum = 0;
    for ($i = 0; $i < $n_rows; $i++) {
        for ($j = 0; $j < $n_cols; $j++) {
            $total_sum += compute_matrix_cell($i, $j, $k_dim);
        }
    }

    $total_ops = $n_rows * $n_cols * ($k_dim * 2);
    echo $total_sum, "\n";
    echo $total_ops, "\n";
    return $total_sum;
}

echo "123456789\n";
$result = matmul_2d(N_ROWS, K_DIM, N_COLS);
echo $result, "\n";
echo "987654323\n";
exit(0);
