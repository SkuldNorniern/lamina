const N_ROWS: i64 = 512;
const K_DIM: i64 = 512;
const N_COLS: i64 = 512;

fn mix(x: i64) -> i64 {
    let a = x.wrapping_mul(1103515245).wrapping_add(12345) & 2147483647;
    let b = a ^ (a >> 7);
    let c = b.wrapping_mul(1103515245).wrapping_add(12345) & 2147483647;
    let d = c ^ (c >> 11);
    d & 65535
}

fn get_matrix_a_element(i: i64, k: i64) -> i64 {
    mix(i * K_DIM + k)
}

fn get_matrix_b_element(k: i64, j: i64) -> i64 {
    mix(k * N_COLS + j)
}

fn compute_matrix_cell(i: i64, j: i64) -> i64 {
    let mut sum: i64 = 0;
    for k in 0..K_DIM {
        sum = sum.wrapping_add(get_matrix_a_element(i, k).wrapping_mul(get_matrix_b_element(k, j)));
    }
    sum
}

fn matmul_2d(n_rows: i64, k_dim: i64, n_cols: i64) -> i64 {
    println!("{}", n_rows);
    println!("{}", k_dim);
    println!("{}", n_cols);

    let mut total_sum: i64 = 0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            total_sum = total_sum.wrapping_add(compute_matrix_cell(i, j));
        }
    }

    let result_size = n_rows.wrapping_mul(n_cols);
    let total_ops = result_size.wrapping_mul(k_dim.wrapping_mul(2));
    println!("{}", total_sum);
    println!("{}", total_ops);
    total_sum
}

fn main() {
    println!("123456789");
    let result = matmul_2d(N_ROWS, K_DIM, N_COLS);
    println!("{}", result);
    println!("987654323");
}
