#!/usr/bin/env ruby

HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def is_prime(n)
  if n <= 1
    return false
  end
  if n <= 3
    return true
  end
  if n % 2 == 0
    return false
  end

  i = 3
  while i * i <= n
    if n % i == 0
      return false
    end
    i += 2
  end
  return true
end

def count_primes(limit)
  count = 0
  for i in 2..limit
    if is_prime(i)
      count += 1
    end
  end
  return count
end

puts HEADER_MARKER
puts count_primes(100)
puts count_primes(1000)
puts count_primes(10000)
puts count_primes(1000000)
puts FOOTER_MARKER
