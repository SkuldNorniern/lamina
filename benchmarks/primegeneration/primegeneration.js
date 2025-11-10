// primegeneration.js
// JavaScript implementation of prime generation benchmark

// Configuration - limits for counting
const LIMIT1 = 100;
const LIMIT2 = 1000;
const LIMIT3 = 10000;
const LIMIT4 = 50000;

// Markers matching Lamina version
const HEADER_MARKER = 123456789n;
const FOOTER_MARKER = 987654321n;

function isPrime(n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 === 0) return false;

    for (let i = 3; i * i <= n; i += 2) {
        if (n % i === 0) return false;
    }
    return true;
}

function countPrimes(limit) {
    if (limit < 2) return 0n;
    let count = 0n;
    for (let i = 2; i <= limit; i++) {
        if (isPrime(i)) count++;
    }
    return count;
}

function main() {
    // Print header marker
    console.log(HEADER_MARKER.toString());

    // Count and print prime counts for different limits
    console.log(countPrimes(LIMIT1).toString());
    console.log(countPrimes(LIMIT2).toString());
    console.log(countPrimes(LIMIT3).toString());
    console.log(countPrimes(LIMIT4).toString());

    // Print footer marker
    console.log(FOOTER_MARKER.toString());
}

main();
