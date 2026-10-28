HEADER_MARKER = 123456789
FOOTER_MARKER = 987654321

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def count_primes(limit):
    count = 0
    for i in range(2, limit + 1):
        if is_prime(i):
            count += 1
    return count

def main():
    print(HEADER_MARKER)
    print(count_primes(100))
    print(count_primes(1000))
    print(count_primes(10000))
    print(count_primes(1000000))
    print(FOOTER_MARKER)

if __name__ == "__main__":
    main()
