const std = @import("std");

// Configuration
const N1 = 10;
const N2 = 12;
const N3 = 15;
const N4 = 18;

// Markers matching Lamina version
const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

/// Computes factorial iteratively: n! = n * (n-1) * ... * 2 * 1
fn factorial_iterative(n: u32) u64 {
    if (n == 0 or n == 1) {
        return 1;
    }

    var result: u64 = 1;
    var i: u32 = 2;
    while (i <= n) : (i += 1) {
        result *= i;
    }
    return result;
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Print header marker
    try stdout.print("{}\n", .{HEADER_MARKER});

    // Compute and print factorial values
    try stdout.print("{}\n", .{factorial_iterative(N1)});
    try stdout.print("{}\n", .{factorial_iterative(N2)});
    try stdout.print("{}\n", .{factorial_iterative(N3)});
    try stdout.print("{}\n", .{factorial_iterative(N4)});

    // Print footer marker
    try stdout.print("{}\n", .{FOOTER_MARKER});
}



