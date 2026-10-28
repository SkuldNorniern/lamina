public class MatMul2D {

    static final int N_ROWS = 256;
    static final int K_DIM = 256;
    static final int N_COLS = 256;

    static long mix(long x) {
        long a = (x * 1103515245L + 12345L) & 2147483647L;
        long b = a ^ (a >> 7);
        long c = (b * 1103515245L + 12345L) & 2147483647L;
        long d = c ^ (c >> 11);
        return d & 65535L;
    }

    static long getMatrixAElement(long i, long k) {
        return mix(i * K_DIM + k);
    }

    static long getMatrixBElement(long k, long j) {
        return mix(k * N_COLS + j);
    }

    static long computeMatrixCell(long i, long j) {
        long sum = 0L;
        for (long k = 0; k < K_DIM; k++) {
            sum += getMatrixAElement(i, k) * getMatrixBElement(k, j);
        }
        return sum;
    }

    static long matmulJava(int nRows, int kDim, int nCols) {
        System.out.println(nRows);
        System.out.println(kDim);
        System.out.println(nCols);

        long totalSum = 0L;
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                totalSum += computeMatrixCell(i, j);
            }
        }

        long resultSize = (long) nRows * nCols;
        long totalOps = resultSize * ((long) kDim * 2);
        System.out.println(totalSum);
        System.out.println(totalOps);
        return totalSum;
    }

    public static void main(String[] args) {
        System.out.println(123456789L);
        long result = matmulJava(N_ROWS, K_DIM, N_COLS);
        System.out.println(result);
        System.out.println(987654323L);
    }
}
