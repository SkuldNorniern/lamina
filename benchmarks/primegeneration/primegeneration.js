const HEADER_MARKER = 123456789;
const FOOTER_MARKER = 987654321;

function isPrime(n) {
    if (n <= 1) return false;
    if (n === 2 || n === 3) return true;
    if (n % 2 === 0) return false;
    for (let d = 3; d * d <= n; d += 2) {
        if (n % d === 0) return false;
    }
    return true;
}

function countPrimes(limit) {
    let count = 0;
    for (let current = 2; current <= limit; current++) {
        if (isPrime(current)) count++;
    }
    return count;
}

function main() {
    console.log(HEADER_MARKER.toString());
    console.log(countPrimes(100).toString());
    console.log(countPrimes(1000).toString());
    console.log(countPrimes(10000).toString());
    console.log(countPrimes(1000000).toString());
    console.log(FOOTER_MARKER.toString());
}

main();
