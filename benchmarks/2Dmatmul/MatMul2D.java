import java.time.Instant; // Not used for benchmark timing, runner handles it

public class MatMul2D {

    // --- Configuration ---
    static final int N_ROWS = 256;
    static final int K_DIM = 256;
    static final int N_COLS = 256;

    // --- Markers ---
    static final long HEADER_MARKER = 123456789L;
    static final long START_MARKER = 987654321L;
    static final long END_MARKER = 987654322L;
    static final long STATUS_MARKER = 987654323L;

    // Generates element A[i,k] deterministically.
    // Use long to match others for potential large intermediate values.
    static long getMatrixAElement(long i, long k) {
        return (i * k) + 1L;
    }

    // Generates element B[k,j] deterministically.
    static long getMatrixBElement(long k, long j) {
        return (k * j) + 1L;
    }

    // Performs matrix multiplication using standard Java loops.
    static long matmulJava(int nRows, int kDim, int nCols) {
        System.out.println(nRows);
        System.out.println(kDim);
        System.out.println(nCols);

        // Use long for calculations
        long resultSize = (long)nRows * nCols;
        long totalSum = 0L;

        System.out.println(START_MARKER); // Start timing after setup

        // --- Standard Java Implementation ---
        // Generate elements on the fly
        for (int iIdx = 0; iIdx < nRows; iIdx++) {
            for (int jIdx = 0; jIdx < nCols; jIdx++) {
                long cellSum = 0L;
                long i = iIdx; // Implicit cast to long
                long j = jIdx; // Implicit cast to long
                for (int kIdx = 0; kIdx < kDim; kIdx++) {
                    long k = kIdx; // Implicit cast to long
                    long aElem = getMatrixAElement(i, k);
                    long bElem = getMatrixBElement(k, j);
                    // Standard Java long arithmetic handles overflow by wrapping (like C/C++)
                    // Could use Math.addExact/multiplyExact if overflow detection was needed
                    cellSum += aElem * bElem;
                }
                totalSum += cellSum;
                // Note: No progress reporting added here for simplicity
            }
        }
        // --- End of benchmark operation ---

        // Calculate operations
        long opsPerCell = (long)kDim * 2;
        long totalOps = resultSize * opsPerCell;

        System.out.println(END_MARKER); // End timing before final prints
        System.out.println(totalSum);
        System.out.println(totalOps);

        return totalSum;
    }


    public static void main(String[] args) {
        System.out.println(HEADER_MARKER);
        long result = matmulJava(N_ROWS, K_DIM, N_COLS); // Run the benchmark
        System.out.println(STATUS_MARKER);
        // Java automatically exits 0 on successful main completion
    }
}
