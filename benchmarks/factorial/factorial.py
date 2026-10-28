N1 = 10
N2 = 12
N3 = 15
N4 = 18
SWEEP_COUNT = 4000000
MASK = (1 << 64) - 1

HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def factorial_iterative(n):
    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def main():
    print(HEADER_MARKER)
    print(factorial_iterative(N1))
    print(factorial_iterative(N2))
    print(factorial_iterative(N3))
    print(factorial_iterative(N4))

    acc = 0
    for s in range(SWEEP_COUNT):
        acc = (acc + factorial_iterative(s % 20)) & MASK
    print(acc)
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()
