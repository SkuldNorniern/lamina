const std = @import("std");

fn arg(comptime T: type, args: []const [:0]const u8, i: usize) T {
    return std.fmt.parseInt(T, args[i], 10) catch @panic("bad integer");
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const n = arg(usize, args, 1);
    const passes = arg(u64, args, 2);

    const arr = try init.gpa.alloc(u64, n);
    defer init.gpa.free(arr);
    for (arr, 0..) |*slot, i| {
        slot.* = @as(u64, i) *% 7 +% 3;
    }

    var total: u64 = 0;
    var pass: u64 = 0;
    while (pass < passes) : (pass += 1) {
        const scale = pass *% 3;
        for (arr) |x| {
            total +%= x *% scale;
        }
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("{d}\n", .{total});
    try stdout.flush();
}
