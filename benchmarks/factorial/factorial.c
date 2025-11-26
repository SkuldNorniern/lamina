#include <stdio.h>
#include <stdint.h>

// Configuration
#define N1 10
#define N2 12
#define N3 15
#define N4 18

// Markers matching Lamina version
#define HEADER_MARKER 123456789LL
#define FOOTER_MARKER 987654321LL

long long factorial_iterative(int n) {
    if (n == 0 || n == 1) {
        return 1;
    }

    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main() {
    // Print header marker
    printf("%lld\n", HEADER_MARKER);

    // Compute and print factorial values
    printf("%lld\n", factorial_iterative(N1));
    printf("%lld\n", factorial_iterative(N2));
    printf("%lld\n", factorial_iterative(N3));
    printf("%lld\n", factorial_iterative(N4));

    // Print footer marker
    printf("%lld\n", FOOTER_MARKER);

    return 0;
}



