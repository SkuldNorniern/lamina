#include <cstdint>
#include <cstdio>

static const int64_t N_ROWS = 256;
static const int64_t K_DIM = 256;
static const int64_t N_COLS = 256;

static int64_t mix(int64_t x) {
    int64_t a = (x * 1103515245 + 12345) & 2147483647;
    int64_t b = a ^ (a >> 7);
    int64_t c = (b * 1103515245 + 12345) & 2147483647;
    int64_t d = c ^ (c >> 11);
    return d & 65535;
}

static int64_t get_matrix_a_element(int64_t i, int64_t k) { return mix(i * K_DIM + k); }
static int64_t get_matrix_b_element(int64_t k, int64_t j) { return mix(k * N_COLS + j); }

static int64_t compute_matrix_cell(int64_t i, int64_t j) {
    uint64_t sum = 0;
    for (int64_t k = 0; k < K_DIM; k++)
        sum += (uint64_t)get_matrix_a_element(i, k) * (uint64_t)get_matrix_b_element(k, j);
    return (int64_t)sum;
}

static int64_t matmul_2d(int64_t n_rows, int64_t k_dim, int64_t n_cols) {
    std::printf("%lld\n", (long long)n_rows);
    std::printf("%lld\n", (long long)k_dim);
    std::printf("%lld\n", (long long)n_cols);

    uint64_t total_sum = 0;
    for (int64_t i = 0; i < n_rows; i++)
        for (int64_t j = 0; j < n_cols; j++)
            total_sum += (uint64_t)compute_matrix_cell(i, j);

    int64_t result_size = n_rows * n_cols;
    int64_t total_ops = result_size * (k_dim * 2);
    std::printf("%lld\n", (long long)total_sum);
    std::printf("%lld\n", (long long)total_ops);
    return (int64_t)total_sum;
}

int main() {
    std::printf("123456789\n");
    int64_t result = matmul_2d(N_ROWS, K_DIM, N_COLS);
    std::printf("%lld\n", (long long)result);
    std::printf("987654323\n");
    return 0;
}
