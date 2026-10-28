#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t arg(int argc, char **argv, int i) {
    if (i >= argc) { fprintf(stderr, "missing argument %d\n", i); exit(2); }
    return strtoll(argv[i], NULL, 10);
}

typedef struct { int64_t x, y, vx, vy; } Particle;

int main(int argc, char **argv) {
    int64_t steps = arg(argc, argv, 1);
    Particle p = { 0, 0, arg(argc, argv, 2), arg(argc, argv, 3) };
    int64_t flips = 0;
    for (int64_t i = 0; i < steps; i++) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x * p.x > 1000000) { p.vx = -p.vx; flips++; }
        if (p.y * p.y > 1000000) { p.vy = -p.vy; flips++; }
    }
    printf("%lld\n", (long long)p.x);
    printf("%lld\n", (long long)p.y);
    printf("%lld\n", (long long)flips);
    return 0;
}
