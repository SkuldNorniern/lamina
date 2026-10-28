#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

static uint64_t fibonacci_iterative(uint64_t n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    uint64_t a = 0, b = 1;
    for (uint64_t i = 2; i <= n; i++) {
        uint64_t temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main(int argc, char **argv) {
    uint64_t n1 = arg(argc, argv, 1), n2 = arg(argc, argv, 2);
    uint64_t n3 = arg(argc, argv, 3), n4 = arg(argc, argv, 4);
    int64_t sweep_count = arg(argc, argv, 5);

    printf("123456789\n");
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n1));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n2));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n3));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n4));

    uint64_t acc = 0;
    for (int64_t s = 0; s < sweep_count; s++)
        acc += fibonacci_iterative((uint64_t)s % 90);
    printf("%llu\n", (unsigned long long)acc);
    printf("987654321\n");
    return 0;
}
