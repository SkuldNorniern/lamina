# factorial.py
# Python implementation of factorial benchmark

# Configuration
N1 = 10
N2 = 12
N3 = 15
N4 = 18

# Markers matching Lamina version
HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def factorial_iterative(n):
    """Compute factorial iteratively: n! = n * (n-1) * ... * 2 * 1"""
    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def main():
    # Print header marker
    print(HEADER_MARKER)

    # Compute and print factorial values
    print(factorial_iterative(N1))
    print(factorial_iterative(N2))
    print(factorial_iterative(N3))
    print(factorial_iterative(N4))

    # Print footer marker
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()



