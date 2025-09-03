# fibonacci_benchmark.py
# Python implementation of Fibonacci sequence benchmark

# Configuration
N1 = 10
N2 = 20
N3 = 30
N4 = 35

# Markers matching Lamina version
HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def fibonacci_iterative(n):
    """Compute nth Fibonacci number iteratively."""
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
    # Print header marker
    print(HEADER_MARKER)

    # Compute and print fibonacci numbers
    print(fibonacci_iterative(N1))
    print(fibonacci_iterative(N2))
    print(fibonacci_iterative(N3))
    print(fibonacci_iterative(N4))

    # Print footer marker
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()
