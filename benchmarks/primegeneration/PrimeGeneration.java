public class PrimeGeneration {
    public static boolean isPrime(long n) {
        if (n <= 1) return false;
        if (n == 2 || n == 3) return true;
        if (n % 2 == 0) return false;
        for (long d = 3; d * d <= n; d += 2)
            if (n % d == 0) return false;
        return true;
    }

    public static long countPrimes(long limit) {
        long count = 0;
        for (long current = 2; current <= limit; current++)
            if (isPrime(current)) count++;
        return count;
    }

    public static void main(String[] args) {
        System.out.println(123456789L);
        System.out.println(countPrimes(100));
        System.out.println(countPrimes(1000));
        System.out.println(countPrimes(10000));
        System.out.println(countPrimes(1000000));
        System.out.println(987654321L);
    }
}
