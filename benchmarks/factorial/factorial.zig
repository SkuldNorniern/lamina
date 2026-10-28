const std = @import("std");

fn factorial_iterative(n: u64) u64 {
    if (n == 0 or n == 1) return 1;
    var result: u64 = 1;
    var i: u64 = 2;
    while (i <= n) : (i += 1) {
        result *= i;
    }
    return result;
}

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("123456789\n", .{});
    try stdout.print("{d}\n", .{factorial_iterative(10)});
    try stdout.print("{d}\n", .{factorial_iterative(12)});
    try stdout.print("{d}\n", .{factorial_iterative(15)});
    try stdout.print("{d}\n", .{factorial_iterative(18)});

    var acc: u64 = 0;
    var s: u64 = 0;
    while (s < 4000000) : (s += 1) {
        acc +%= factorial_iterative(s % 20);
    }
    try stdout.print("{d}\n", .{acc});
    try stdout.print("987654321\n", .{});
    try stdout.flush();
}
