#include <iostream>

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
    std::cout << HEADER_MARKER << std::endl;

    // Compute and print fibonacci numbers
    std::cout << fibonacci_iterative(N1) << std::endl;
    std::cout << fibonacci_iterative(N2) << std::endl;
    std::cout << fibonacci_iterative(N3) << std::endl;
    std::cout << fibonacci_iterative(N4) << std::endl;

    // Print footer marker
    std::cout << FOOTER_MARKER << std::endl;

    return 0;
}

