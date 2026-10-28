const
  N1 = 10
  N2 = 12
  N3 = 15
  N4 = 18
  SweepCount = 4000000

const
  HEADER_MARKER: int64 = 123456789
  FOOTER_MARKER: int64 = 987654321

proc factorial_iterative(n: int): int64 =
  if n == 0 or n == 1:
    return 1

  var result: int64 = 1
  for i in 2..n:
    result *= i.int64
  return result

when isMainModule:
  echo HEADER_MARKER
  echo factorial_iterative(N1)
  echo factorial_iterative(N2)
  echo factorial_iterative(N3)
  echo factorial_iterative(N4)

  var acc: uint64 = 0
  var s = 0
  while s < SweepCount:
    acc += uint64(factorial_iterative(s mod 20))
    s += 1
  echo acc
  echo FOOTER_MARKER
