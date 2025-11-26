#include <iostream>

// Configuration
const int N1 = 10;
const int N2 = 12;
const int N3 = 15;
const int N4 = 18;

// Markers matching Lamina version
const long long HEADER_MARKER = 123456789LL;
const long long FOOTER_MARKER = 987654321LL;

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
    std::cout << HEADER_MARKER << std::endl;

    // Compute and print factorial values
    std::cout << factorial_iterative(N1) << std::endl;
    std::cout << factorial_iterative(N2) << std::endl;
    std::cout << factorial_iterative(N3) << std::endl;
    std::cout << factorial_iterative(N4) << std::endl;

    // Print footer marker
    std::cout << FOOTER_MARKER << std::endl;

    return 0;
}



