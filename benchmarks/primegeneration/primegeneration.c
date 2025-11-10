#include <stdio.h>
#include <stdint.h>

// Configuration - limits for counting
#define LIMIT1 100
#define LIMIT2 1000
#define LIMIT3 10000
#define LIMIT4 50000

// Markers matching Lamina version
#define HEADER_MARKER 123456789LL
#define FOOTER_MARKER 987654321LL

int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0) return 0;

    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
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
    printf("%lld\n", HEADER_MARKER);

    // Count and print prime counts for different limits
    printf("%lld\n", count_primes(LIMIT1));
    printf("%lld\n", count_primes(LIMIT2));
    printf("%lld\n", count_primes(LIMIT3));
    printf("%lld\n", count_primes(LIMIT4));

    // Print footer marker
    printf("%lld\n", FOOTER_MARKER);

    return 0;
}
