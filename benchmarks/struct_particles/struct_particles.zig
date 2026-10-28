const std = @import("std");

fn arg(comptime T: type, args: []const [:0]const u8, i: usize) T {
    return std.fmt.parseInt(T, args[i], 10) catch @panic("bad integer");
}

const Particle = struct {
    x: i64,
    y: i64,
    vx: i64,
    vy: i64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const steps = arg(i64, args, 1);
    var p = Particle{ .x = 0, .y = 0, .vx = arg(i64, args, 2), .vy = arg(i64, args, 3) };
    var flips: i64 = 0;

    var i: i64 = 0;
    while (i < steps) : (i += 1) {
        p.x +%= p.vx;
        p.y +%= p.vy;
        if (p.x *% p.x > 1000000) {
            p.vx = -p.vx;
            flips += 1;
        }
        if (p.y *% p.y > 1000000) {
            p.vy = -p.vy;
            flips += 1;
        }
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("{d}\n{d}\n{d}\n", .{ p.x, p.y, flips });
    try stdout.flush();
}
