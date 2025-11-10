#!/usr/bin/env ruby

# Configuration - limits for counting
LIMIT1 = 100
LIMIT2 = 1000
LIMIT3 = 10000
LIMIT4 = 50000

# Markers matching Lamina version
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
  if limit < 2
    return 0
  end
  count = 0
  for i in 2..limit
    if is_prime(i)
      count += 1
    end
  end
  return count
end

# Print header marker
puts HEADER_MARKER

# Count and print prime counts for different limits
puts count_primes(LIMIT1)
puts count_primes(LIMIT2)
puts count_primes(LIMIT3)
puts count_primes(LIMIT4)

# Print footer marker
puts FOOTER_MARKER
