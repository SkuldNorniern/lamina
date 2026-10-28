#!/usr/bin/env ruby

N1 = 10
N2 = 20
N3 = 30
N4 = 35
SWEEP_COUNT = 200000
MASK = (1 << 64) - 1

HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def fibonacci_iterative(n)
  if n == 0
    return 0
  end
  if n == 1
    return 1
  end

  a = 0
  b = 1
  (2..n).each do |i|
    temp = a + b
    a = b
    b = temp
  end
  return b
end

puts HEADER_MARKER
puts fibonacci_iterative(N1)
puts fibonacci_iterative(N2)
puts fibonacci_iterative(N3)
puts fibonacci_iterative(N4)

acc = 0
SWEEP_COUNT.times do |s|
  acc = (acc + fibonacci_iterative(s % 90)) & MASK
end
puts acc
puts FOOTER_MARKER
