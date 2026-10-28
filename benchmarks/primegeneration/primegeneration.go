package main

import (
	"fmt"
	"os"
	"strconv"
)

func arg(i int) int64 {
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
	fmt.Println(123456789)
	for i := 1; i <= 4; i++ {
		fmt.Println(countPrimes(arg(i)))
	}
	fmt.Println(987654321)
}
