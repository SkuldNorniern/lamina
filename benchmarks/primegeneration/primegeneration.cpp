#include <iostream>

// Configuration - limits for counting
const int LIMIT1 = 100;
const int LIMIT2 = 1000;
const int LIMIT3 = 10000;
const int LIMIT4 = 50000;

// Markers matching Lamina version
const long long HEADER_MARKER = 123456789LL;
const long long FOOTER_MARKER = 987654321LL;

bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

long long count_primes(int limit) {
    if (limit < 2) return 0;
    long long count = 0;
    for (int i = 2; i <= limit; i++) {
        if (is_prime(i)) count++;
    }
    return count;
}

int main() {
    // Print header marker
    std::cout << HEADER_MARKER << std::endl;

    // Count and print prime counts for different limits
    std::cout << count_primes(LIMIT1) << std::endl;
    std::cout << count_primes(LIMIT2) << std::endl;
    std::cout << count_primes(LIMIT3) << std::endl;
    std::cout << count_primes(LIMIT4) << std::endl;

    // Print footer marker
    std::cout << FOOTER_MARKER << std::endl;

    return 0;
}
