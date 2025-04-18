# tensor_benchmark.py
# Pure Python implementation (no NumPy)
import time
import sys

# --- Configuration --- 
N_ROWS = 256  # Rows in A 
K_DIM = 256   # Cols in A = Rows in B 
N_COLS = 256  # Cols in B

# Remove the 64x64 override and warnings
# N_ROWS = 64 
# K_DIM = 64
# N_COLS = 64
# print(f"WARNING: Running pure Python benchmark with reduced dimensions: {N_ROWS}x{K_DIM}x{N_COLS}", file=sys.stderr)
# print(f"WARNING: Original dimensions ({1024}x{1024}x{1024}) would take an extremely long time.", file=sys.stderr)


# Unique markers matching Lamina/C if possible
HEADER_MARKER = 123456789
START_MARKER = 987654321
END_MARKER = 987654322
STATUS_MARKER = 987654323

def get_matrix_a_element(i, k):
  """Generates element A[i,k] deterministically."""
  # Python's default integers handle arbitrary size
  return (i * k) + 1

def get_matrix_b_element(k, j):
  """Generates element B[k,j] deterministically."""
  return (k * j) + 1

def matmul_pure_python(n_rows, k_dim, n_cols):
    """Performs matrix multiplication using pure Python lists and loops."""
    print(n_rows)
    print(k_dim)
    print(n_cols)
    print(START_MARKER) # Start timing after setup

    # --- Generate matrices (less efficient than NumPy, but part of the setup) ---
    # For extremely large N, this generation itself could be slow / memory intensive
    # matrix_a = [[get_matrix_a_element(i, k) for k in range(k_dim)] for i in range(n_rows)]
    # matrix_b = [[get_matrix_b_element(k, j) for j in range(n_cols)] for k in range(k_dim)]
    # Let's generate on the fly within the main loop to be closer to Lamina/C

    # --- Pure Python MatMul Implementation --- 
    result_matrix = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    total_sum = 0

    for i in range(n_rows):
        for j in range(n_cols):
            cell_sum = 0
            for k in range(k_dim):
                a_elem = get_matrix_a_element(i, k)
                b_elem = get_matrix_b_element(k, j)
                cell_sum += a_elem * b_elem
            # result_matrix[i][j] = cell_sum # Storing result is optional for benchmark
            total_sum += cell_sum
    # --- End of benchmark operation --- 

    # Calculate operations (approximately)
    ops_per_cell = k_dim * 2
    result_size = n_rows * n_cols
    total_ops = result_size * ops_per_cell

    print(END_MARKER) # End timing before final prints
    print(total_sum) 
    print(total_ops)

    return total_sum

def main():
    print(HEADER_MARKER)
    result = matmul_pure_python(N_ROWS, K_DIM, N_COLS)
    print(STATUS_MARKER)
    sys.exit(0) # Explicitly exit 0

if __name__ == "__main__":
    main()
