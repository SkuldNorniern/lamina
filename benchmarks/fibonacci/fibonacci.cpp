#include <cstdint>
#include <cstdio>

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

int main() {
    const uint64_t n1 = 10, n2 = 20, n3 = 30, n4 = 35;
    const uint64_t sweep_count = 200000;

    printf("123456789\n");
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n1));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n2));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n3));
    printf("%llu\n", (unsigned long long)fibonacci_iterative(n4));

    uint64_t acc = 0;
    for (uint64_t s = 0; s < sweep_count; s++)
        acc += fibonacci_iterative(s % 90);
    printf("%llu\n", (unsigned long long)acc);
    printf("987654321\n");
    return 0;
}
