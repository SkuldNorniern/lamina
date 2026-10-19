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
pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const n = MATRIX_SIZE;
    var total: usize = 0;

    try stdout.print("Starting {d}x{d} matrix multiplication benchmark in Zig...\n", .{ n, n });

    // Start timing
    const start_ns = std.Io.Clock.awake.now(init.io).nanoseconds;

    // Report progress every 5%, matching the C version
    const result_size = n * n;
    var cells_done: usize = 0;
    var next_report = result_size / 20;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            total += computeMatrixCell(i, j, n);

            cells_done += 1;
            if (cells_done >= next_report) {
                try stdout.print("Progress: {d}%\n", .{cells_done * 100 / result_size});
                next_report += result_size / 20;
            }
        }
    }

    // End timing
    const end_ns = std.Io.Clock.awake.now(init.io).nanoseconds;
    const elapsed = @as(f64, @floatFromInt(end_ns - start_ns)) / 1_000_000_000.0;

    try stdout.print("\nCompleted in {d:.4} seconds\n", .{elapsed});
    try stdout.print("Total sum: {d}\n", .{total});
    try stdout.flush();
}
