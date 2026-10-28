const std = @import("std");

const N_ROWS: u64 = 256;
const K_DIM: u64 = 256;
const N_COLS: u64 = 256;

fn mix(x: u64) u64 {
    const a = ((x *% 1103515245) +% 12345) & 2147483647;
    const b = a ^ (a >> 7);
    const c = ((b *% 1103515245) +% 12345) & 2147483647;
    const d = c ^ (c >> 11);
    return d & 65535;
}

fn getMatrixAElement(i: u64, k: u64) u64 {
    return mix(i * K_DIM + k);
}

fn getMatrixBElement(k: u64, j: u64) u64 {
    return mix(k * N_COLS + j);
}

fn computeMatrixCell(i: u64, j: u64) u64 {
    var sum: u64 = 0;
    var k: u64 = 0;
    while (k < K_DIM) : (k += 1) {
        sum +%= getMatrixAElement(i, k) *% getMatrixBElement(k, j);
    }
    return sum;
}

fn matmulZig(stdout: *std.Io.Writer, n_rows: u64, k_dim: u64, n_cols: u64) !u64 {
    try stdout.print("{d}\n{d}\n{d}\n", .{ n_rows, k_dim, n_cols });

    var total_sum: u64 = 0;
    var i: u64 = 0;
    while (i < n_rows) : (i += 1) {
        var j: u64 = 0;
        while (j < n_cols) : (j += 1) {
            total_sum +%= computeMatrixCell(i, j);
        }
    }

    const total_ops = n_rows * n_cols * (k_dim * 2);
    try stdout.print("{d}\n{d}\n", .{ total_sum, total_ops });
    return total_sum;
}

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("123456789\n", .{});
    const result = try matmulZig(stdout, N_ROWS, K_DIM, N_COLS);
    try stdout.print("{d}\n", .{result});
    try stdout.print("987654323\n", .{});
    try stdout.flush();
}
