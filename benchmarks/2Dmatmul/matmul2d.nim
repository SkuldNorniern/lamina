const
  N_ROWS = 256'i64
  K_DIM = 256'i64
  N_COLS = 256'i64

proc mix(x: int64): int64 =
  let a = ((x * 1103515245'i64) + 12345'i64) and 2147483647'i64
  let b = a xor (a shr 7)
  let c = ((b * 1103515245'i64) + 12345'i64) and 2147483647'i64
  let d = c xor (c shr 11)
  d and 65535'i64

proc getMatrixAElement(i, k: int64): int64 {.inline.} =
  mix((i * K_DIM) + k)

proc getMatrixBElement(k, j: int64): int64 {.inline.} =
  mix((k * N_COLS) + j)

proc computeMatrixCell(i, j: int64): int64 =
  var sum: int64 = 0
  for k in 0'i64 ..< K_DIM:
    sum += getMatrixAElement(i, k) * getMatrixBElement(k, j)
  sum

proc matmulNim(nRows, kDim, nCols: int64): int64 =
  echo nRows
  echo kDim
  echo nCols

  var totalSum: int64 = 0
  for i in 0'i64 ..< nRows:
    for j in 0'i64 ..< nCols:
      totalSum += computeMatrixCell(i, j)

  let totalOps = (nRows * nCols) * (kDim * 2)
  echo totalSum
  echo totalOps
  totalSum

when isMainModule:
  echo 123456789'i64
  let result = matmulNim(N_ROWS, K_DIM, N_COLS)
  echo result
  echo 987654323'i64
