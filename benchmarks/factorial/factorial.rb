#!/usr/bin/env ruby

# Configuration
N1 = 10
N2 = 12
N3 = 15
N4 = 18

# Markers matching Lamina version
HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def factorial_iterative(n)
  if n == 0 || n == 1
    return 1
  end

  result = 1
  for i in 2..n
    result *= i
  end
  return result
end

# Print header marker
puts HEADER_MARKER

# Compute and print factorial values
puts factorial_iterative(N1)
puts factorial_iterative(N2)
puts factorial_iterative(N3)
puts factorial_iterative(N4)

# Print footer marker
puts FOOTER_MARKER



