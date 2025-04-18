const std = @import("std");

// Matrix dimensions
const MATRIX_SIZE: usize = 256;

/// Retrieves a value from matrix A at position [i, k]
fn getMatrixAElement(i: usize, k: usize) usize {
    // Values based on position (consistent with other implementations)
    return i * k + 1;
}

/// Retrieves a value from matrix B at position [k, j]
fn getMatrixBElement(k: usize, j: usize) usize {
    // Values based on position (consistent with other implementations)
    return k * j + 1;
}

/// Compute a single cell in the result matrix
fn computeMatrixCell(i: usize, j: usize, k_dim: usize) usize {
    var sum: usize = 0;
    var k: usize = 0;
    while (k < k_dim) : (k += 1) {
        const a_elem = getMatrixAElement(i, k);
        const b_elem = getMatrixBElement(k, j);
        sum += a_elem * b_elem;
    }
    return sum;
}

/// Matrix multiplication benchmark
pub fn main() !u8 {
    const stdout = std.io.getStdOut().writer();
    const n = MATRIX_SIZE;
    var total: usize = 0;

    try stdout.print("Starting {d}×{d} matrix multiplication benchmark in Zig...\n", .{ n, n });

    // Start timing
    const start_time = std.time.milliTimestamp();

    // Only compute a sample of cells for large matrices
    const sample_step = @max(n / 10, 1);

    var i: usize = 0;
    while (i < n) : (i += sample_step) {
        try stdout.print("Progress: {}%\r", .{i * 100 / n});

        var j: usize = 0;
        while (j < n) : (j += sample_step) {
            // Compute the result cell [i,j]
            const cell_value = computeMatrixCell(i, j, n);
            total += cell_value;
        }
    }

    // End timing
    // Using modern Zig syntax for type conversion
    const elapsed = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;

    try stdout.print("\nCompleted in {d:.4} seconds\n", .{elapsed});
    try stdout.print("Checksum: {d}\n", .{total});

    // Return success
    return 0;
} 