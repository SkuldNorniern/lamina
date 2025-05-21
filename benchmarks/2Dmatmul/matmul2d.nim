# matmul2d.nim
import times

# --- Configuration ---
const
  N_ROWS = 256
  K_DIM = 256
  N_COLS = 256

# --- Markers ---
const
  HEADER_MARKER = 123456789'i64
  START_MARKER = 987654321'i64
  END_MARKER = 987654322'i64
  STATUS_MARKER = 987654323'i64

# Generates element A[i,k] deterministically
proc getMatrixAElement(i, k: int64): int64 {.inline.} =
  (i * k) + 1

# Generates element B[k,j] deterministically
proc getMatrixBElement(k, j: int64): int64 {.inline.} =
  (k * j) + 1

# Performs matrix multiplication using standard Nim loops
proc matmulNim(nRows, kDim, nCols: int): int64 =
  echo nRows
  echo kDim
  echo nCols

  # Use int64 for calculations to prevent potential overflow
  let resultSize = (nRows * nCols).int64
  var totalSum: int64 = 0

  echo START_MARKER # Start timing after setup

  # --- Standard Nim Implementation ---
  # Generate elements on the fly
  for iIdx in 0..<nRows:
    for jIdx in 0..<nCols:
      var cellSum: int64 = 0
      let i = iIdx.int64
      let j = jIdx.int64
      for kIdx in 0..<kDim:
        let k = kIdx.int64
        let aElem = getMatrixAElement(i, k)
        let bElem = getMatrixBElement(k, j)
        # Use +% and *% for wrapping arithmetic to prevent overflow
        cellSum += aElem *% bElem
      totalSum += cellSum

  # --- End of benchmark operation ---

  # Calculate operations
  let opsPerCell = (kDim * 2).int64
  let totalOps = resultSize *% opsPerCell

  echo END_MARKER # End timing before final prints
  echo totalSum
  echo totalOps

  return totalSum

when isMainModule:
  echo HEADER_MARKER
  discard matmulNim(N_ROWS, K_DIM, N_COLS) # Run the benchmark
  echo STATUS_MARKER 