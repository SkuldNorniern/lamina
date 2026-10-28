use std::env;

fn arg(i: usize) -> i64 {
    env::args().nth(i).expect("missing argument").parse().expect("bad integer")
}

fn get_matrix_a_element(i: i64, k: i64) -> i64 {
    i.wrapping_mul(k).wrapping_add(1)
}

fn get_matrix_b_element(k: i64, j: i64) -> i64 {
    k.wrapping_mul(j).wrapping_add(1)
}

fn compute_matrix_cell(i: i64, j: i64, k_dim: i64) -> i64 {
    let mut sum: i64 = 0;
    for k in 0..k_dim {
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
            total_sum = total_sum.wrapping_add(compute_matrix_cell(i, j, k_dim));
        }
    }

    let result_size = n_rows.wrapping_mul(n_cols);
    let total_ops = result_size.wrapping_mul(k_dim.wrapping_mul(2));
    println!("{}", total_sum);
    println!("{}", total_ops);
    total_sum
}

fn main() {
    let n_rows = arg(1);
    let k_dim = arg(2);
    let n_cols = arg(3);

    println!("123456789");
    let result = matmul_2d(n_rows, k_dim, n_cols);
    println!("{}", result);
    println!("987654323");
}
