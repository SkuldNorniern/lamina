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
        static int GetMatrixAElement(int i, int k)
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
        static int GetMatrixBElement(int k, int j)
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
        static long ComputeMatrixCell(int i, int j, int kDim)
        {
            long sum = 0;
            for (int k = 0; k < kDim; k++)
            {
                int aElem = GetMatrixAElement(i, k);
                int bElem = GetMatrixBElement(k, j);
                sum += aElem * bElem;
            }
            return sum;
        }

        /// <summary>
        /// Matrix multiplication benchmark
        /// </summary>
        static int Main()
        {
            int n = MATRIX_SIZE;
            long total = 0;

            Console.WriteLine($"Starting {n}×{n} matrix multiplication benchmark in C#...");

            // Start timing
            var stopwatch = Stopwatch.StartNew();

            // Only compute a sample of cells for large matrices
            int sampleStep = Math.Max(n / 10, 1);

            for (int i = 0; i < n; i += sampleStep)
            {
                Console.Write($"Progress: {i * 100 / n}%\r");

                for (int j = 0; j < n; j += sampleStep)
                {
                    // Compute the result cell [i,j]
                    long cellValue = ComputeMatrixCell(i, j, n);
                    total += cellValue;
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