# primegeneration.py
# Python implementation of prime generation benchmark (simplified counting)

# Configuration - limits for counting
LIMIT1 = 100
LIMIT2 = 1000
LIMIT3 = 10000
LIMIT4 = 50000

# Markers matching Lamina version
HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def is_prime(n):
    """Check if a number is prime using trial division."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Check divisibility by odd numbers up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def count_primes(limit):
    """Count prime numbers from 2 to limit inclusive."""
    if limit < 2:
        return 0
    count = 0
    for i in range(2, limit + 1):
        if is_prime(i):
            count += 1
    return count

def main():
    # Print header marker
    print(HEADER_MARKER)

    # Count and print prime counts for different limits
    print(count_primes(LIMIT1))
    print(count_primes(LIMIT2))
    print(count_primes(LIMIT3))
    print(count_primes(LIMIT4))

    # Print footer marker
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()
