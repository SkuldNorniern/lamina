N1 = 10
N2 = 20
N3 = 30
N4 = 35
SWEEP_COUNT = 200000
MASK = (1 << 64) - 1

HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def fibonacci_iterative(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    a = 0
    b = 1
    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp
    return b

def main():
    print(HEADER_MARKER)
    print(fibonacci_iterative(N1))
    print(fibonacci_iterative(N2))
    print(fibonacci_iterative(N3))
    print(fibonacci_iterative(N4))

    acc = 0
    for s in range(SWEEP_COUNT):
        acc = (acc + fibonacci_iterative(s % 90)) & MASK
    print(acc)
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()
