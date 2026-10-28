#!/usr/bin/env ruby

N1 = 10
N2 = 12
N3 = 15
N4 = 18
SWEEP_COUNT = 4000000
MASK = (1 << 64) - 1

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

puts HEADER_MARKER
puts factorial_iterative(N1)
puts factorial_iterative(N2)
puts factorial_iterative(N3)
puts factorial_iterative(N4)

acc = 0
SWEEP_COUNT.times do |s|
  acc = (acc + factorial_iterative(s % 20)) & MASK
end
puts acc
puts FOOTER_MARKER
