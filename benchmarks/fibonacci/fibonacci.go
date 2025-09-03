// fibonacci_benchmark.go
// Go implementation of Fibonacci sequence benchmark
package main

import "fmt"

// Configuration
const N1 = 10
const N2 = 20
const N3 = 30
const N4 = 35

// Markers matching Lamina version
const HEADER_MARKER = 123456789
const FOOTER_MARKER = 987654321

func fibonacci_iterative(n int) int64 {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}

	var a int64 = 0
	var b int64 = 1
	for i := 2; i <= n; i++ {
		temp := a + b
		a = b
		b = temp
	}
	return b
}

func main() {
	// Print header marker
	fmt.Println(HEADER_MARKER)

	// Compute and print fibonacci numbers
	fmt.Println(fibonacci_iterative(N1))
	fmt.Println(fibonacci_iterative(N2))
	fmt.Println(fibonacci_iterative(N3))
	fmt.Println(fibonacci_iterative(N4))

	// Print footer marker
	fmt.Println(FOOTER_MARKER)
}
