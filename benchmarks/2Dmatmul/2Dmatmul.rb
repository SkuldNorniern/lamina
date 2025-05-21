#!/usr/bin/env ruby

# Matrix multiplication benchmark for Ruby
# Similar to the other implementations in the benchmark suite

# Get matrix dimensions
MATRIX_SIZE = 256

# Retrieves a value from matrix A at position [i, k]
def get_matrix_a_element(i, k)
  # Values based on position (consistent with other implementations)
  i * k + 1
end

# Retrieves a value from matrix B at position [k, j]
def get_matrix_b_element(k, j)
  # Values based on position (consistent with other implementations)
  k * j + 1
end

# Compute a single cell in the result matrix
def compute_matrix_cell(i, j, k_dim)
  sum = 0
  (0...k_dim).each do |k|
    a_elem = get_matrix_a_element(i, k)
    b_elem = get_matrix_b_element(k, j)
    sum += a_elem * b_elem
  end
  sum
end

# Matrix multiplication benchmark
def matrix_multiply_2d
  start_time = Time.now

  n = MATRIX_SIZE
  total = 0
  
  puts "Starting #{n}x#{n} matrix multiplication benchmark in Ruby..."
  
  # Compute the full matrix multiplication
  (0...n).each do |i|
    # Print progress every 10% to avoid excessive output
    if i % (n / 10) == 0
      print "Progress: #{(i * 100.0 / n).round}%\r"
      $stdout.flush
    end
    
    (0...n).each do |j|
      # Compute the result cell [i,j]
      cell_value = compute_matrix_cell(i, j, n)
      total += cell_value
    end
  end
  
  elapsed = Time.now - start_time
  puts "\nCompleted in #{elapsed.round(4)} seconds"
  puts "Checksum: #{total}"
  
  # Return success
  return 0
end

# Run the benchmark
exit(matrix_multiply_2d) 