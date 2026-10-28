package main

import "fmt"

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
	fmt.Println(123456789)
	fmt.Println(factorial_iterative(10))
	fmt.Println(factorial_iterative(12))
	fmt.Println(factorial_iterative(15))
	fmt.Println(factorial_iterative(18))

	var acc uint64
	for s := 0; s < 4000000; s++ {
		acc += uint64(factorial_iterative(s % 20))
	}
	fmt.Println(acc)
	fmt.Println(987654321)
}
