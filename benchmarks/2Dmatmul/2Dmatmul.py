import sys

N_ROWS = 256
K_DIM = 256
N_COLS = 256

HEADER_MARKER = 123456789
STATUS_MARKER = 987654323


def mix(x):
    a = (x * 1103515245 + 12345) & 2147483647
    b = a ^ (a >> 7)
    c = (b * 1103515245 + 12345) & 2147483647
    d = c ^ (c >> 11)
    return d & 65535


def get_matrix_a_element(i, k):
    return mix(i * K_DIM + k)


def get_matrix_b_element(k, j):
    return mix(k * N_COLS + j)


def matmul_pure_python(n_rows, k_dim, n_cols):
    print(n_rows)
    print(k_dim)
    print(n_cols)

    total_sum = 0
    for i in range(n_rows):
        for j in range(n_cols):
            cell_sum = 0
            for k in range(k_dim):
                cell_sum += get_matrix_a_element(i, k) * get_matrix_b_element(k, j)
            total_sum += cell_sum

    total_ops = n_rows * n_cols * (k_dim * 2)
    print(total_sum)
    print(total_ops)
    return total_sum


def main():
    print(HEADER_MARKER)
    result = matmul_pure_python(N_ROWS, K_DIM, N_COLS)
    print(result)
    print(STATUS_MARKER)
    sys.exit(0)


if __name__ == "__main__":
    main()
