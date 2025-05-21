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
    
    // Create a result matrix to store values and prevent optimization from eliminating the work
    var result: [MATRIX_SIZE][MATRIX_SIZE]usize = undefined;
    var total: usize = 0;

    try stdout.print("Starting {d}×{d} matrix multiplication benchmark in Zig...\n", .{ n, n });

    // Start timing
    const start_time = std.time.milliTimestamp();

    // Progress tracking
    const progress_step = @max(n / 10, 1);
    var last_progress: usize = 0;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        // Show progress every 10% with a newline to ensure visibility
        if (i % progress_step == 0) {
            const progress = i * 100 / n;
            try stdout.print("Progress: {d}%\n", .{progress});
            last_progress = progress;
        }

        var j: usize = 0;
        while (j < n) : (j += 1) {
            // Compute the result cell [i,j]
            result[i][j] = computeMatrixCell(i, j, n);
            total += result[i][j];
        }
    }

    // Make sure we show 100% progress if we didn't already
    if (last_progress < 100) {
        try stdout.print("Progress: 100%\n", .{});
    }

    // End timing
    // Using modern Zig syntax for type conversion
    const elapsed = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time)) / 1000.0;

    try stdout.print("\nCompleted in {d:.4} seconds\n", .{elapsed});
    try stdout.print("Total sum: {d}\n", .{total});
    
    // Print some sample values from the result matrix
    try stdout.print("\nSample result values:\n", .{});
    try stdout.print("result[0][0] = {d}\n", .{result[0][0]});
    try stdout.print("result[1][1] = {d}\n", .{result[1][1]});
    try stdout.print("result[10][10] = {d}\n", .{result[10][10]});
    try stdout.print("result[100][100] = {d}\n", .{result[100][100]});
    try stdout.print("result[{d}][{d}] = {d}\n", .{n-1, n-1, result[n-1][n-1]});

    // Return success
    return 0;
} 