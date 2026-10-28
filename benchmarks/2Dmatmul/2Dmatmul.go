package main

import "fmt"

const N_ROWS = 256
const K_DIM = 256
const N_COLS = 256

func mix(x int64) int64 {
	a := (x*1103515245 + 12345) & 2147483647
	b := a ^ (a >> 7)
	c := (b*1103515245 + 12345) & 2147483647
	d := c ^ (c >> 11)
	return d & 65535
}

func getMatrixAElement(i, k int64) int64 {
	return mix(i*K_DIM + k)
}

func getMatrixBElement(k, j int64) int64 {
	return mix(k*N_COLS + j)
}

func computeMatrixCell(i, j int64) int64 {
	var sum int64 = 0
	for k := int64(0); k < K_DIM; k++ {
		sum += getMatrixAElement(i, k) * getMatrixBElement(k, j)
	}
	return sum
}

func matmulGo(nRows, kDim, nCols int64) int64 {
	fmt.Println(nRows)
	fmt.Println(kDim)
	fmt.Println(nCols)

	var totalSum int64 = 0
	for i := int64(0); i < nRows; i++ {
		for j := int64(0); j < nCols; j++ {
			totalSum += computeMatrixCell(i, j)
		}
	}

	resultSize := nRows * nCols
	totalOps := resultSize * (kDim * 2)
	fmt.Println(totalSum)
	fmt.Println(totalOps)
	return totalSum
}

func main() {
	fmt.Println(123456789)
	result := matmulGo(N_ROWS, K_DIM, N_COLS)
	fmt.Println(result)
	fmt.Println(987654323)
}
