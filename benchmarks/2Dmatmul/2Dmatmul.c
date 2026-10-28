#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

static int64_t get_matrix_a_element(int64_t i, int64_t k) { return i * k + 1; }
static int64_t get_matrix_b_element(int64_t k, int64_t j) { return k * j + 1; }

static int64_t compute_matrix_cell(int64_t i, int64_t j, int64_t k_dim) {
    uint64_t sum = 0;
    for (int64_t k = 0; k < k_dim; k++)
        sum += (uint64_t)get_matrix_a_element(i, k) * (uint64_t)get_matrix_b_element(k, j);
    return (int64_t)sum;
}

static int64_t matmul_2d(int64_t n_rows, int64_t k_dim, int64_t n_cols) {
    printf("%lld\n", (long long)n_rows);
    printf("%lld\n", (long long)k_dim);
    printf("%lld\n", (long long)n_cols);

    uint64_t total_sum = 0;
    for (int64_t i = 0; i < n_rows; i++)
        for (int64_t j = 0; j < n_cols; j++)
            total_sum += (uint64_t)compute_matrix_cell(i, j, k_dim);

    int64_t result_size = n_rows * n_cols;
    int64_t total_ops = result_size * (k_dim * 2);
    printf("%lld\n", (long long)total_sum);
    printf("%lld\n", (long long)total_ops);
    return (int64_t)total_sum;
}

int main(int argc, char **argv) {
    int64_t n_rows = arg(argc, argv, 1);
    int64_t k_dim = arg(argc, argv, 2);
    int64_t n_cols = arg(argc, argv, 3);

    printf("123456789\n");
    int64_t result = matmul_2d(n_rows, k_dim, n_cols);
    printf("%lld\n", (long long)result);
    printf("987654323\n");
    return 0;
}
