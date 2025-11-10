const std = @import("std");

// Configuration - limits for counting
const LIMIT1 = 100;
const LIMIT2 = 1000;
const LIMIT3 = 10000;
const LIMIT4 = 50000;

// Markers matching Lamina version
const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

/// Checks if a number is prime
fn isPrime(n: usize) bool {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    var i: usize = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// Counts primes from 2 to limit inclusive
fn countPrimes(limit: usize) u64 {
    if (limit < 2) return 0;
    var count: u64 = 0;
    var i: usize = 2;
    while (i <= limit) : (i += 1) {
        if (isPrime(i)) count += 1;
    }
    return count;
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Print header marker
    try stdout.print("{}\n", .{HEADER_MARKER});

    // Count and print prime counts for different limits
    try stdout.print("{}\n", .{countPrimes(LIMIT1)});
    try stdout.print("{}\n", .{countPrimes(LIMIT2)});
    try stdout.print("{}\n", .{countPrimes(LIMIT3)});
    try stdout.print("{}\n", .{countPrimes(LIMIT4)});

    // Print footer marker
    try stdout.print("{}\n", .{FOOTER_MARKER});
}
