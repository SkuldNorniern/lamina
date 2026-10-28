package main

import (
	"fmt"
	"os"
	"strconv"
)

//go:noinline
func arg(i int, fallback int64) int64 {
	if i >= len(os.Args) {
		return fallback
	}
	v, err := strconv.ParseInt(os.Args[i], 10, 64)
	if err != nil {
		panic(err)
	}
	return v
}

func fibonacciIterative(n uint64) uint64 {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	a, b := uint64(0), uint64(1)
	for i := uint64(2); i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

func main() {
	n1, n2, n3, n4 := uint64(arg(1, 10)), uint64(arg(2, 20)), uint64(arg(3, 30)), uint64(arg(4, 35))
	sweepCount := arg(5, 200000)

	fmt.Println(123456789)
	fmt.Println(fibonacciIterative(n1))
	fmt.Println(fibonacciIterative(n2))
	fmt.Println(fibonacciIterative(n3))
	fmt.Println(fibonacciIterative(n4))

	var acc uint64
	for s := int64(0); s < sweepCount; s++ {
		acc += fibonacciIterative(uint64(s) % 90)
	}
	fmt.Println(acc)
	fmt.Println(987654321)
}
