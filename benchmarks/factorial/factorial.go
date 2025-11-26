// factorial.go
package main

import "fmt"

// Configuration
const N1 = 10
const N2 = 12
const N3 = 15
const N4 = 18

// Markers matching Lamina version
const HEADER_MARKER = 123456789
const FOOTER_MARKER = 987654321

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
    // Print header marker
    fmt.Println(HEADER_MARKER)

    // Compute and print factorial values
    fmt.Println(factorial_iterative(N1))
    fmt.Println(factorial_iterative(N2))
    fmt.Println(factorial_iterative(N3))
    fmt.Println(factorial_iterative(N4))

    // Print footer marker
    fmt.Println(FOOTER_MARKER)
}



