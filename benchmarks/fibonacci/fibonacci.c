#include <stdio.h>
#include <stdint.h>

// Configuration
#define N1 10
#define N2 20
#define N3 30
#define N4 35

// Markers matching Lamina version
#define HEADER_MARKER 123456789LL
#define FOOTER_MARKER 987654321LL

long long fibonacci_iterative(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main() {
    // Print header marker
    printf("%lld\n", HEADER_MARKER);

    // Compute and print fibonacci numbers
    printf("%lld\n", fibonacci_iterative(N1));
    printf("%lld\n", fibonacci_iterative(N2));
    printf("%lld\n", fibonacci_iterative(N3));
    printf("%lld\n", fibonacci_iterative(N4));

    // Print footer marker
    printf("%lld\n", FOOTER_MARKER);

    return 0;
}

