const N1 = 10n;
const N2 = 20n;
const N3 = 30n;
const N4 = 35n;
const SWEEP_COUNT = 200000n;

const HEADER_MARKER = 123456789n;
const FOOTER_MARKER = 987654321n;

function fibonacci_iterative(n) {
    if (n === 0n) return 0n;
    if (n === 1n) return 1n;

    let a = 0n, b = 1n;
    for (let i = 2n; i <= n; i++) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

function main() {
    console.log(HEADER_MARKER.toString());
    console.log(fibonacci_iterative(N1).toString());
    console.log(fibonacci_iterative(N2).toString());
    console.log(fibonacci_iterative(N3).toString());
    console.log(fibonacci_iterative(N4).toString());

    let acc = 0n;
    for (let s = 0n; s < SWEEP_COUNT; s++) {
        acc = BigInt.asUintN(64, acc + fibonacci_iterative(s % 90n));
    }
    console.log(acc.toString());
    console.log(FOOTER_MARKER.toString());
}

main();
