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

func isPrime(n int64) bool {
	if n <= 1 {
		return false
	}
	if n == 2 || n == 3 {
		return true
	}
	if n%2 == 0 {
		return false
	}
	for d := int64(3); d*d <= n; d += 2 {
		if n%d == 0 {
			return false
		}
	}
	return true
}

func countPrimes(limit int64) int64 {
	var count int64
	for c := int64(2); c <= limit; c++ {
		if isPrime(c) {
			count++
		}
	}
	return count
}

func main() {
	defaults := [4]int64{100, 1000, 10000, 1000000}
	fmt.Println(123456789)
	for i := 1; i <= 4; i++ {
		fmt.Println(countPrimes(arg(i, defaults[i-1])))
	}
	fmt.Println(987654321)
}
