using System;

namespace MatMul2D
{
    class Program
    {
        private const long N_ROWS = 256;
        private const long K_DIM = 256;
        private const long N_COLS = 256;

        static long Mix(long x)
        {
            long a = (x * 1103515245L + 12345L) & 2147483647L;
            long b = a ^ (a >> 7);
            long c = (b * 1103515245L + 12345L) & 2147483647L;
            long d = c ^ (c >> 11);
            return d & 65535L;
        }

        static long GetMatrixAElement(long i, long k) => Mix(i * K_DIM + k);

        static long GetMatrixBElement(long k, long j) => Mix(k * N_COLS + j);

        static long ComputeMatrixCell(long i, long j, long kDim)
        {
            long sum = 0;
            for (long k = 0; k < kDim; k++)
            {
                sum += GetMatrixAElement(i, k) * GetMatrixBElement(k, j);
            }
            return sum;
        }

        static long MatmulCs(long nRows, long kDim, long nCols)
        {
            Console.WriteLine(nRows);
            Console.WriteLine(kDim);
            Console.WriteLine(nCols);

            long totalSum = 0;
            for (long i = 0; i < nRows; i++)
            {
                for (long j = 0; j < nCols; j++)
                {
                    totalSum += ComputeMatrixCell(i, j, kDim);
                }
            }

            long totalOps = nRows * nCols * (kDim * 2);
            Console.WriteLine(totalSum);
            Console.WriteLine(totalOps);
            return totalSum;
        }

        static int Main()
        {
            Console.WriteLine(123456789L);
            long result = MatmulCs(N_ROWS, K_DIM, N_COLS);
            Console.WriteLine(result);
            Console.WriteLine(987654323L);
            return 0;
        }
    }
}
