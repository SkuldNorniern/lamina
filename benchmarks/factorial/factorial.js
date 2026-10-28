const N1 = 10n;
const N2 = 12n;
const N3 = 15n;
const N4 = 18n;
const SWEEP_COUNT = 4000000n;

const HEADER_MARKER = 123456789n;
const FOOTER_MARKER = 987654321n;

function factorial_iterative(n) {
    if (n === 0n || n === 1n) return 1n;

    let result = 1n;
    for (let i = 2n; i <= n; i++) {
        result *= i;
    }
    return result;
}

function main() {
    console.log(HEADER_MARKER.toString());
    console.log(factorial_iterative(N1).toString());
    console.log(factorial_iterative(N2).toString());
    console.log(factorial_iterative(N3).toString());
    console.log(factorial_iterative(N4).toString());

    let acc = 0n;
    for (let s = 0n; s < SWEEP_COUNT; s++) {
        acc = BigInt.asUintN(64, acc + factorial_iterative(s % 20n));
    }
    console.log(acc.toString());
    console.log(FOOTER_MARKER.toString());
}

main();
