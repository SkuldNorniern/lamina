#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

int main(int argc, char **argv) {
    int64_t iters = arg(argc, argv, 1);
    uint64_t x = (uint64_t)arg(argc, argv, 2);
    uint64_t y = 0, z = 0;
    for (int64_t i = 0; i < iters; i++) {
        uint64_t m = x * 1103515245u + 12345u;
        x = m & 2147483647u;
        uint64_t s = x >> 7;
        uint64_t s2 = y << 3;
        y = (s ^ s2) & 2147483647u;
        uint64_t w = y ^ x;
        z += w;
    }
    printf("%llu\n", (unsigned long long)x);
    printf("%llu\n", (unsigned long long)y);
    printf("%llu\n", (unsigned long long)z);
    return 0;
}
