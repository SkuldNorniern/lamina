// tensor_benchmark.js

// --- Configuration ---
const N_ROWS = 256; // Rows in A
const K_DIM = 256;  // Cols in A = Rows in B
const N_COLS = 256; // Cols in B

// Markers (using BigInt for large numbers)
const HEADER_MARKER = 123456789n;
const START_MARKER = 987654321n;
const END_MARKER = 987654322n;
const STATUS_MARKER = 987654323n;

// Use BigInt for calculations as intermediate/final sums can exceed Number.MAX_SAFE_INTEGER
function getMatrixAElement(i, k) {
    return (i * k) + 1n; // Use BigInt literals
}

function getMatrixBElement(k, j) {
    return (k * j) + 1n; // Use BigInt literals
}

function computeMatrixCell(i_idx, j_idx, k_dim) {
    let sum = 0n; // Use BigInt
    const i = BigInt(i_idx);
    const j = BigInt(j_idx);
    for (let k_idx = 0; k_idx < k_dim; k_idx++) {
        const k = BigInt(k_idx);
        const aElem = getMatrixAElement(i, k);
        const bElem = getMatrixBElement(k, j);
        sum += aElem * bElem;
    }
    return sum;
}

function matmulJS(n_rows, k_dim, n_cols) {
    console.log(n_rows);
    console.log(k_dim);
    console.log(n_cols);

    const resultSize = BigInt(n_rows) * BigInt(n_cols);
    let totalSum = 0n;

    console.log(START_MARKER.toString()); // Start timing after setup

    // --- Standard JS Implementation ---
    for (let i = 0; i < n_rows; i++) {
        for (let j = 0; j < n_cols; j++) {
            const cellResult = computeMatrixCell(i, j, k_dim);
            totalSum += cellResult;
             // Note: No progress reporting added here for simplicity, unlike Lamina version
        }
    }
     // --- End of benchmark operation ---

    // Calculate operations
    const opsPerCell = BigInt(k_dim) * 2n;
    const totalOps = resultSize * opsPerCell;

    console.log(END_MARKER.toString()); // End timing before final prints
    console.log(totalSum.toString());
    console.log(totalOps.toString());

    return totalSum;
}

function main() {
    console.log(HEADER_MARKER.toString());
    const result = matmulJS(N_ROWS, K_DIM, N_COLS);
    console.log(STATUS_MARKER.toString());
    process.exit(0); // Explicitly exit 0
}

main();
