const std = @import("std");

noinline fn arg(comptime T: type, args: []const [:0]const u8, i: usize, fallback: T) T {
    if (i >= args.len) return fallback;
    return std.fmt.parseInt(T, args[i], 10) catch @panic("bad integer");
}

fn isPrime(n: i64) bool {
    if (n <= 1) return false;
    if (n == 2 or n == 3) return true;
    if (@rem(n, 2) == 0) return false;
    var d: i64 = 3;
    while (d * d <= n) : (d += 2) {
        if (@rem(n, d) == 0) return false;
    }
    return true;
}

fn countPrimes(limit: i64) i64 {
    var count: i64 = 0;
    var c: i64 = 2;
    while (c <= limit) : (c += 1) {
        if (isPrime(c)) count += 1;
    }
    return count;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const defaults = [4]i64{ 100, 1000, 10000, 2000000 };

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("123456789\n", .{});
    var i: usize = 1;
    while (i <= 4) : (i += 1) {
        try stdout.print("{d}\n", .{countPrimes(arg(i64, args, i, defaults[i - 1]))});
    }
    try stdout.print("987654321\n", .{});
    try stdout.flush();
}
