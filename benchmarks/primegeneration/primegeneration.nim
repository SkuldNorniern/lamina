const
  HEADER_MARKER: int64 = 123456789
  FOOTER_MARKER: int64 = 987654321

proc isPrime(n: int64): bool =
  if n <= 1:
    return false
  if n == 2 or n == 3:
    return true
  if n mod 2 == 0:
    return false
  var d: int64 = 3
  while d * d <= n:
    if n mod d == 0:
      return false
    d += 2
  return true

proc countPrimes(limit: int64): int64 =
  var count: int64 = 0
  for current in 2'i64 .. limit:
    if isPrime(current):
      count += 1
  return count

when isMainModule:
  echo HEADER_MARKER
  echo countPrimes(100)
  echo countPrimes(1000)
  echo countPrimes(10000)
  echo countPrimes(1000000)
  echo FOOTER_MARKER
