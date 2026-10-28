#include <cstdint>
#include <cstdio>

static long long factorial_iterative(int n) {
    if (n == 0 || n == 1) return 1;
    long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

int main() {
    const long long n1 = 10, n2 = 12, n3 = 15, n4 = 18;
    const int64_t sweep_count = 4000000;

    printf("123456789\n");
    printf("%lld\n", factorial_iterative((int)n1));
    printf("%lld\n", factorial_iterative((int)n2));
    printf("%lld\n", factorial_iterative((int)n3));
    printf("%lld\n", factorial_iterative((int)n4));

    uint64_t acc = 0;
    for (int64_t s = 0; s < sweep_count; s++)
        acc += (uint64_t)factorial_iterative((int)(s % 20));
    printf("%llu\n", (unsigned long long)acc);
    printf("987654321\n");
    return 0;
}
