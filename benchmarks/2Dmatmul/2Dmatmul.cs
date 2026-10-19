using System;
using System.Diagnostics;

namespace MatMul2D
{
    /// <summary>
    /// Matrix multiplication benchmark for C#
    /// Similar to the other implementations in the benchmark suite
    /// </summary>
    class Program
    {
        // Matrix dimensions
        private const int MATRIX_SIZE = 256;

        /// <summary>
        /// Retrieves a value from matrix A at position [i, k]
        /// </summary>
        /// <param name="i">Row index</param>
        /// <param name="k">Column index</param>
        /// <returns>The value at position [i, k]</returns>
        static long GetMatrixAElement(long i, long k)
        {
            // Values based on position (consistent with other implementations)
            return i * k + 1;
        }

        /// <summary>
        /// Retrieves a value from matrix B at position [k, j]
        /// </summary>
        /// <param name="k">Row index</param>
        /// <param name="j">Column index</param>
        /// <returns>The value at position [k, j]</returns>
        static long GetMatrixBElement(long k, long j)
        {
            // Values based on position (consistent with other implementations)
            return k * j + 1;
        }

        /// <summary>
        /// Compute a single cell in the result matrix
        /// </summary>
        /// <param name="i">Row index in result matrix</param>
        /// <param name="j">Column index in result matrix</param>
        /// <param name="kDim">The dimension of the matrices (n for an n×n matrix)</param>
        /// <returns>The computed cell value</returns>
        static long ComputeMatrixCell(long i, long j, long kDim)
        {
            long sum = 0;
            for (long k = 0; k < kDim; k++)
            {
                long aElem = GetMatrixAElement(i, k);
                long bElem = GetMatrixBElement(k, j);
                sum += aElem * bElem;
            }
            return sum;
        }

        /// <summary>
        /// Matrix multiplication benchmark
        /// </summary>
        static int Main()
        {
            long n = MATRIX_SIZE;
            long total = 0;

            Console.WriteLine($"Starting {n}×{n} matrix multiplication benchmark in C#...");

            // Start timing
            var stopwatch = Stopwatch.StartNew();

            // Report progress every 5%, matching the C version
            long resultSize = n * n;
            long cellsDone = 0;
            long nextReport = resultSize / 20;

            for (long i = 0; i < n; i++)
            {
                for (long j = 0; j < n; j++)
                {
                    // Compute the result cell [i,j]
                    total += ComputeMatrixCell(i, j, n);

                    cellsDone++;
                    if (cellsDone >= nextReport)
                    {
                        Console.WriteLine($"Progress: {cellsDone * 100 / resultSize}%");
                        nextReport += resultSize / 20;
                    }
                }
            }

            // End timing
            stopwatch.Stop();
            double elapsed = stopwatch.ElapsedMilliseconds / 1000.0;

            Console.WriteLine($"\nCompleted in {elapsed:F4} seconds");
            Console.WriteLine($"Checksum: {total}");

            // Return success
            return 0;
        }
    }
}
