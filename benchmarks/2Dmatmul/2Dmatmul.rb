N_ROWS = 256
K_DIM = 256
N_COLS = 256

def mix(x)
  a = (x * 1103515245 + 12345) & 2147483647
  b = a ^ (a >> 7)
  c = (b * 1103515245 + 12345) & 2147483647
  d = c ^ (c >> 11)
  d & 65535
end

def get_matrix_a_element(i, k)
  mix(i * K_DIM + k)
end

def get_matrix_b_element(k, j)
  mix(k * N_COLS + j)
end

def compute_matrix_cell(i, j, k_dim)
  sum = 0
  (0...k_dim).each do |k|
    sum += get_matrix_a_element(i, k) * get_matrix_b_element(k, j)
  end
  sum
end

def matmul_2d(n_rows, k_dim, n_cols)
  puts n_rows
  puts k_dim
  puts n_cols

  total_sum = 0
  (0...n_rows).each do |i|
    (0...n_cols).each do |j|
      total_sum += compute_matrix_cell(i, j, k_dim)
    end
  end

  total_ops = n_rows * n_cols * (k_dim * 2)
  puts total_sum
  puts total_ops
  total_sum
end

puts 123456789
result = matmul_2d(N_ROWS, K_DIM, N_COLS)
puts result
puts 987654323
