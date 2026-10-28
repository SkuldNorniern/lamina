const
  N1 = 10'u64
  N2 = 20'u64
  N3 = 30'u64
  N4 = 35'u64
  SweepCount = 200000'u64

const
  HEADER_MARKER = 123456789'i64
  FOOTER_MARKER = 987654321'i64

proc fibonacciIterative(n: uint64): uint64 =
  if n == 0:
    return 0
  if n == 1:
    return 1

  var a: uint64 = 0
  var b: uint64 = 1
  for i in 2'u64 .. n:
    let temp = a + b
    a = b
    b = temp
  return b

when isMainModule:
  echo HEADER_MARKER
  echo fibonacciIterative(N1)
  echo fibonacciIterative(N2)
  echo fibonacciIterative(N3)
  echo fibonacciIterative(N4)

  var acc: uint64 = 0
  var s: uint64 = 0
  while s < SweepCount:
    acc += fibonacciIterative(s mod 90'u64)
    s += 1
  echo acc
  echo FOOTER_MARKER
