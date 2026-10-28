#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

int main(int argc, char **argv) {
    int64_t n = arg(argc, argv, 1);
    int64_t passes = arg(argc, argv, 2);
    uint64_t *arr = malloc((size_t)n * sizeof *arr);
    if (!arr) return 1;

    for (int64_t i = 0; i < n; i++)
        arr[i] = (uint64_t)(i * 7 + 3);

    uint64_t total = 0;
    for (int64_t pass = 0; pass < passes; pass++) {
        uint64_t scale = (uint64_t)(pass * 3);
        for (int64_t k = 0; k < n; k++)
            total += arr[k] * scale;
    }
    printf("%llu\n", (unsigned long long)total);
    free(arr);
    return 0;
}
