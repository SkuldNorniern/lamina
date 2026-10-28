const N_ROWS = 256n;
const K_DIM = 256n;
const N_COLS = 256n;

function mix(x) {
    const a = (x * 1103515245n + 12345n) & 2147483647n;
    const b = a ^ (a >> 7n);
    const c = (b * 1103515245n + 12345n) & 2147483647n;
    const d = c ^ (c >> 11n);
    return d & 65535n;
}

function getMatrixAElement(i, k) {
    return mix(i * K_DIM + k);
}

function getMatrixBElement(k, j) {
    return mix(k * N_COLS + j);
}

function computeMatrixCell(i, j) {
    let sum = 0n;
    for (let k = 0n; k < K_DIM; k++) {
        sum += getMatrixAElement(i, k) * getMatrixBElement(k, j);
    }
    return sum;
}

function matmulJS(nRows, kDim, nCols) {
    console.log(nRows.toString());
    console.log(kDim.toString());
    console.log(nCols.toString());

    let totalSum = 0n;
    for (let i = 0n; i < nRows; i++) {
        for (let j = 0n; j < nCols; j++) {
            totalSum += computeMatrixCell(i, j);
        }
    }

    const totalOps = nRows * nCols * (kDim * 2n);
    console.log(totalSum.toString());
    console.log(totalOps.toString());
    return totalSum;
}

function main() {
    console.log("123456789");
    const result = matmulJS(N_ROWS, K_DIM, N_COLS);
    console.log(result.toString());
    console.log("987654323");
    process.exit(0);
}

main();
