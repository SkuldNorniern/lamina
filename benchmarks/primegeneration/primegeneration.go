// primegeneration.go
package main

import "fmt"

// Configuration - limits for counting
const LIMIT1 = 100
const LIMIT2 = 1000
const LIMIT3 = 10000
const LIMIT4 = 50000

// Markers matching Lamina version
const HEADER_MARKER = 123456789
const FOOTER_MARKER = 987654321

func isPrime(n int) bool {
	if n <= 1 {
		return false
	}
	if n <= 3 {
		return true
	}
	if n%2 == 0 {
		return false
	}

	for i := 3; i*i <= n; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func countPrimes(limit int) int64 {
	if limit < 2 {
		return 0
	}
	var count int64 = 0
	for i := 2; i <= limit; i++ {
		if isPrime(i) {
			count++
		}
	}
	return count
}

func main() {
	// Print header marker
	fmt.Println(HEADER_MARKER)

	// Count and print prime counts for different limits
	fmt.Println(countPrimes(LIMIT1))
	fmt.Println(countPrimes(LIMIT2))
	fmt.Println(countPrimes(LIMIT3))
	fmt.Println(countPrimes(LIMIT4))

	// Print footer marker
	fmt.Println(FOOTER_MARKER)
}
