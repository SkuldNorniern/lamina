#include <iostream>
#include <vector> // Not needed for on-the-fly, but often included

// --- Configuration ---
const int N_ROWS = 256;
const int K_DIM = 256;
const int N_COLS = 256;

// --- Markers ---
// Using long long to match others, ensure sufficient size
const long long HEADER_MARKER = 123456789LL;
const long long START_MARKER = 987654321LL;
const long long END_MARKER = 987654322LL;
const long long STATUS_MARKER = 987654323LL;

// Generates element A[i,k] deterministically.
// Use long long (typically 64-bit) to match others.
inline long long getMatrixAElement(long long i, long long k) {
    return (i * k) + 1LL;
}

// Generates element B[k,j] deterministically.
inline long long getMatrixBElement(long long k, long long j) {
    return (k * j) + 1LL;
}

// Performs matrix multiplication using standard C++ loops.
long long matmulCpp(int nRows, int kDim, int nCols) {
    std::cout << nRows << std::endl;
    std::cout << kDim << std::endl;
    std::cout << nCols << std::endl;

    // Use long long for calculations
    long long resultSize = static_cast<long long>(nRows) * nCols;
    long long totalSum = 0LL;

    std::cout << START_MARKER << std::endl; // Start timing after setup

    // --- Standard C++ Implementation ---
    // Generate elements on the fly
    for (int iIdx = 0; iIdx < nRows; ++iIdx) {
        for (int jIdx = 0; jIdx < nCols; ++jIdx) {
            long long cellSum = 0LL;
            long long i = iIdx;
            long long j = jIdx;
            for (int kIdx = 0; kIdx < kDim; ++kIdx) {
                long long k = kIdx;
                long long aElem = getMatrixAElement(i, k);
                long long bElem = getMatrixBElement(k, j);
                // Standard C++ integer overflow wraps (like C)
                cellSum += aElem * bElem;
            }
            totalSum += cellSum;
            // Note: No progress reporting added here for simplicity
        }
    }
    // --- End of benchmark operation ---

    // Calculate operations
    long long opsPerCell = static_cast<long long>(kDim) * 2;
    long long totalOps = resultSize * opsPerCell;

    std::cout << END_MARKER << std::endl; // End timing before final prints
    std::cout << totalSum << std::endl;
    std::cout << totalOps << std::endl;

    return totalSum;
}


int main() {
    std::cout << HEADER_MARKER << std::endl;
    matmulCpp(N_ROWS, K_DIM, N_COLS); // Run the benchmark
    std::cout << STATUS_MARKER << std::endl;
    return 0; // Explicit return 0 in C++ main
}
