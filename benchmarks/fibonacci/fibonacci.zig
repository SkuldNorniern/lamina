const std = @import("std");

// Configuration
const N1 = 10;
const N2 = 20;
const N3 = 30;
const N4 = 35;

// Markers matching Lamina version
const HEADER_MARKER: i64 = 123456789;
const FOOTER_MARKER: i64 = 987654321;

fn fibonacciIterative(n: u32) i64 {
    if (n == 0) return 0;
    if (n == 1) return 1;

    var a: i64 = 0;
    var b: i64 = 1;
    var i: u32 = 2;
    while (i <= n) : (i += 1) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

pub fn main() void {
    // Print header marker
    std.debug.print("{d}\n", .{HEADER_MARKER});

    // Compute and print fibonacci numbers
    std.debug.print("{d}\n", .{fibonacciIterative(N1)});
    std.debug.print("{d}\n", .{fibonacciIterative(N2)});
    std.debug.print("{d}\n", .{fibonacciIterative(N3)});
    std.debug.print("{d}\n", .{fibonacciIterative(N4)});

    // Print footer marker
    std.debug.print("{d}\n", .{FOOTER_MARKER});
}
