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

func factorial_iterative(n int) int64 {
	if n == 0 || n == 1 {
		return 1
	}
	result := int64(1)
	for i := 2; i <= n; i++ {
		result *= int64(i)
	}
	return result
}

func main() {
	n1, n2, n3, n4 := int(arg(1, 10)), int(arg(2, 12)), int(arg(3, 15)), int(arg(4, 18))
	sweepCount := arg(5, 50000000)

	fmt.Println(123456789)
	fmt.Println(factorial_iterative(n1))
	fmt.Println(factorial_iterative(n2))
	fmt.Println(factorial_iterative(n3))
	fmt.Println(factorial_iterative(n4))

	var acc uint64
	for s := int64(0); s < sweepCount; s++ {
		acc += uint64(factorial_iterative(int(s % 20)))
	}
	fmt.Println(acc)
	fmt.Println(987654321)
}
