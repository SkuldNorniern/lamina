# primegeneration.nim

# Configuration - limits for counting
const LIMIT1 = 100
const LIMIT2 = 1000
const LIMIT3 = 10000
const LIMIT4 = 50000

# Markers matching Lamina version
const HEADER_MARKER: int64 = 123456789
const FOOTER_MARKER: int64 = 987654321

proc isPrime(n: int): bool =
  if n <= 1:
    return false
  if n <= 3:
    return true
  if n mod 2 == 0:
    return false

  var i = 3
  while i * i <= n:
    if n mod i == 0:
      return false
    i += 2
  return true

proc countPrimes(limit: int): int64 =
  if limit < 2:
    return 0
  var count: int64 = 0
  for i in 2..limit:
    if isPrime(i):
      count += 1
  return count

when isMainModule:
  # Print header marker
  echo HEADER_MARKER

  # Count and print prime counts for different limits
  echo countPrimes(LIMIT1)
  echo countPrimes(LIMIT2)
  echo countPrimes(LIMIT3)
  echo countPrimes(LIMIT4)

  # Print footer marker
  echo FOOTER_MARKER
