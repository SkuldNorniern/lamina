#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static __attribute__((noinline)) int64_t arg(int argc, char **argv, int i, int64_t fallback) {
    if (i >= argc) return fallback;
    return strtoll(argv[i], NULL, 10);
}

static long long factorial_iterative(int n) {
    if (n == 0 || n == 1) return 1;
    long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

int main(int argc, char **argv) {
    const int n1 = (int)arg(argc, argv, 1, 10), n2 = (int)arg(argc, argv, 2, 12);
    const int n3 = (int)arg(argc, argv, 3, 15), n4 = (int)arg(argc, argv, 4, 18);
    const int64_t sweep_count = arg(argc, argv, 5, 50000000);

    printf("123456789\n");
    printf("%lld\n", factorial_iterative(n1));
    printf("%lld\n", factorial_iterative(n2));
    printf("%lld\n", factorial_iterative(n3));
    printf("%lld\n", factorial_iterative(n4));

    uint64_t acc = 0;
    for (int64_t s = 0; s < sweep_count; s++)
        acc += (uint64_t)factorial_iterative((int)(s % 20));
    printf("%llu\n", (unsigned long long)acc);
    printf("987654321\n");
    return 0;
}
