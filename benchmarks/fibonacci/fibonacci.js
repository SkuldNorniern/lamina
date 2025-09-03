// fibonacci_benchmark.js
// JavaScript implementation of Fibonacci sequence benchmark

// Configuration
const N1 = 10;
const N2 = 20;
const N3 = 30;
const N4 = 35;

// Markers matching Lamina version
const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

function fibonacci_iterative(n) {
    if (n === 0) return 0;
    if (n === 1) return 1;

    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

function main() {
    // Print header marker
    console.log(HEADER_MARKER);

    // Compute and print fibonacci numbers
    console.log(fibonacci_iterative(N1));
    console.log(fibonacci_iterative(N2));
    console.log(fibonacci_iterative(N3));
    console.log(fibonacci_iterative(N4));

    // Print footer marker
    console.log(FOOTER_MARKER);
}

main();
