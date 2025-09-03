# Fibonacci sequence benchmark for Nim
# Similar to the other implementations in the benchmark suite

# Configuration
const
  N1 = 10
  N2 = 20
  N3 = 30
  N4 = 35

# Markers matching Lamina version
const
  HEADER_MARKER = 123456789'i64
  FOOTER_MARKER = 987654321'i64

proc fibonacciIterative(n: int): int64 =
  if n == 0:
    return 0
  if n == 1:
    return 1

  var a = 0'i64
  var b = 1'i64
  for i in 2..n:
    let temp = a + b
    a = b
    b = temp
  return b

when isMainModule:
  # Print header marker
  echo HEADER_MARKER

  # Compute and print fibonacci numbers
  echo fibonacciIterative(N1)
  echo fibonacciIterative(N2)
  echo fibonacciIterative(N3)
  echo fibonacciIterative(N4)

  # Print footer marker
  echo FOOTER_MARKER

