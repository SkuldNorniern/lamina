# factorial.nim
import strformat

# Configuration
const N1 = 10
const N2 = 12
const N3 = 15
const N4 = 18

# Markers matching Lamina version
const HEADER_MARKER: int64 = 123456789
const FOOTER_MARKER: int64 = 987654321

proc factorial_iterative(n: int): int64 =
  if n == 0 or n == 1:
    return 1

  var result: int64 = 1
  for i in 2..n:
    result *= i.int64
  return result

when isMainModule:
  # Print header marker
  echo HEADER_MARKER

  # Compute and print factorial values
  echo factorial_iterative(N1)
  echo factorial_iterative(N2)
  echo factorial_iterative(N3)
  echo factorial_iterative(N4)

  # Print footer marker
  echo FOOTER_MARKER



