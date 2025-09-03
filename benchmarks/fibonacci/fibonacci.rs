// fibonacci_benchmark.rs
// Rust implementation of Fibonacci sequence benchmark

// Configuration
const N1: i32 = 10;
const N2: i32 = 20;
const N3: i32 = 30;
const N4: i32 = 35;

// Markers matching Lamina version
const HEADER_MARKER: i64 = 123456789;
const FOOTER_MARKER: i64 = 987654321;

fn fibonacci_iterative(n: i32) -> i64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut a: i64 = 0;
    let mut b: i64 = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

fn main() {
    // Print header marker
    println!("{}", HEADER_MARKER);

    // Compute and print fibonacci numbers
    println!("{}", fibonacci_iterative(N1));
    println!("{}", fibonacci_iterative(N2));
    println!("{}", fibonacci_iterative(N3));
    println!("{}", fibonacci_iterative(N4));

    // Print footer marker
    println!("{}", FOOTER_MARKER);
}
