// factorial.js
// JavaScript implementation of factorial benchmark

// Configuration
const N1 = 10;
const N2 = 12;
const N3 = 15;
const N4 = 18;

// Markers matching Lamina version
const HEADER_MARKER = 123456789n;
const FOOTER_MARKER = 987654321n;

function factorial_iterative(n) {
    if (n === 0 || n === 1) {
        return 1n;
    }

    let result = 1n;
    for (let i = 2; i <= n; i++) {
        result *= BigInt(i);
    }
    return result;
}

function main() {
    // Print header marker
    console.log(HEADER_MARKER.toString());

    // Compute and print factorial values
    console.log(factorial_iterative(N1).toString());
    console.log(factorial_iterative(N2).toString());
    console.log(factorial_iterative(N3).toString());
    console.log(factorial_iterative(N4).toString());

    // Print footer marker
    console.log(FOOTER_MARKER.toString());
}

main();



