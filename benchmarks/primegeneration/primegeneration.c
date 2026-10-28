#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

static int is_prime(int64_t n) {
    if (n <= 1) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0) return 0;
    for (int64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

static int64_t count_primes(int64_t limit) {
    int64_t count = 0;
    for (int64_t current = 2; current <= limit; current++)
        count += is_prime(current);
    return count;
}

int main(int argc, char **argv) {
    printf("123456789\n");
    for (int i = 1; i <= 4; i++)
        printf("%lld\n", (long long)count_primes(arg(argc, argv, i)));
    printf("987654321\n");
    return 0;
}
