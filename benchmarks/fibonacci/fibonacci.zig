const std = @import("std");

noinline fn arg(comptime T: type, args: []const [:0]const u8, i: usize, fallback: T) T {
    if (i >= args.len) return fallback;
    return std.fmt.parseInt(T, args[i], 10) catch @panic("bad integer");
}

fn fibonacciIterative(n: u64) u64 {
    if (n == 0) return 0;
    if (n == 1) return 1;
    var a: u64 = 0;
    var b: u64 = 1;
    var i: u64 = 2;
    while (i <= n) : (i += 1) {
        const temp = a +% b;
        a = b;
        b = temp;
    }
    return b;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const n1 = arg(u64, args, 1, 10);
    const n2 = arg(u64, args, 2, 20);
    const n3 = arg(u64, args, 3, 30);
    const n4 = arg(u64, args, 4, 35);
    const sweep_count = arg(u64, args, 5, 12000000);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("123456789\n", .{});
    try stdout.print("{d}\n", .{fibonacciIterative(n1)});
    try stdout.print("{d}\n", .{fibonacciIterative(n2)});
    try stdout.print("{d}\n", .{fibonacciIterative(n3)});
    try stdout.print("{d}\n", .{fibonacciIterative(n4)});

    var acc: u64 = 0;
    var s: u64 = 0;
    while (s < sweep_count) : (s += 1) {
        acc +%= fibonacciIterative(s % 90);
    }
    try stdout.print("{d}\n", .{acc});
    try stdout.print("987654321\n", .{});
    try stdout.flush();
}
