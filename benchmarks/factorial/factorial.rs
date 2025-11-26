// factorial.rs
// Rust implementation of factorial benchmark

// Configuration
const N1: i32 = 10;
const N2: i32 = 12;
const N3: i32 = 15;
const N4: i32 = 18;

// Markers matching Lamina version
const HEADER_MARKER: i64 = 123456789;
const FOOTER_MARKER: i64 = 987654321;

fn factorial_iterative(n: i32) -> i64 {
    if n == 0 || n == 1 {
        return 1;
    }

    let mut result: i64 = 1;
    for i in 2..=n {
        result = result.saturating_mul(i as i64);
    }
    result
}

fn main() {
    // Print header marker
    println!("{}", HEADER_MARKER);

    // Compute and print factorial values
    println!("{}", factorial_iterative(N1));
    println!("{}", factorial_iterative(N2));
    println!("{}", factorial_iterative(N3));
    println!("{}", factorial_iterative(N4));

    // Print footer marker
    println!("{}", FOOTER_MARKER);
}



