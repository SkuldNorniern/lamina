const std = @import("std");

noinline fn arg(comptime T: type, args: []const [:0]const u8, i: usize, fallback: T) T {
    if (i >= args.len) return fallback;
    return std.fmt.parseInt(T, args[i], 10) catch @panic("bad integer");
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const iters = arg(u64, args, 1, 100000000);
    var x = arg(u64, args, 2, 12345);
    var y: u64 = 0;
    var z: u64 = 0;

    var i: u64 = 0;
    while (i < iters) : (i += 1) {
        const m = x *% 1103515245 +% 12345;
        x = m & 2147483647;
        const s = x >> 7;
        const s2 = y << 3;
        y = (s ^ s2) & 2147483647;
        const w = y ^ x;
        z +%= w;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("{d}\n{d}\n{d}\n", .{ x, y, z });
    try stdout.flush();
}
