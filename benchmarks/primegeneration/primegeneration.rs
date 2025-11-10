// primegeneration.rs
// Rust implementation of prime generation benchmark

// Configuration - limits for counting
const LIMIT1: usize = 100;
const LIMIT2: usize = 1000;
const LIMIT3: usize = 10000;
const LIMIT4: usize = 50000;

// Markers matching Lamina version
const HEADER_MARKER: i64 = 123456789;
const FOOTER_MARKER: i64 = 987654321;

fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

fn count_primes(limit: usize) -> i64 {
    if limit < 2 {
        return 0;
    }
    let mut count = 0;
    for i in 2..=limit {
        if is_prime(i) {
            count += 1;
        }
    }
    count
}

fn main() {
    // Print header marker
    println!("{}", HEADER_MARKER);

    // Count and print prime counts for different limits
    println!("{}", count_primes(LIMIT1));
    println!("{}", count_primes(LIMIT2));
    println!("{}", count_primes(LIMIT3));
    println!("{}", count_primes(LIMIT4));

    // Print footer marker
    println!("{}", FOOTER_MARKER);
}
