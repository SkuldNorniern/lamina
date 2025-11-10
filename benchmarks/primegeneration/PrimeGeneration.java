public class PrimeGeneration {
    // Configuration - limits for counting
    private static final int LIMIT1 = 100;
    private static final int LIMIT2 = 1000;
    private static final int LIMIT3 = 10000;
    private static final int LIMIT4 = 50000;

    // Markers matching Lamina version
    private static final long HEADER_MARKER = 123456789L;
    private static final long FOOTER_MARKER = 987654321L;

    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0) return false;

        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public static long countPrimes(int limit) {
        if (limit < 2) return 0;
        long count = 0;
        for (int i = 2; i <= limit; i++) {
            if (isPrime(i)) count++;
        }
        return count;
    }

    public static void main(String[] args) {
        // Print header marker
        System.out.println(HEADER_MARKER);

        // Count and print prime counts for different limits
        System.out.println(countPrimes(LIMIT1));
        System.out.println(countPrimes(LIMIT2));
        System.out.println(countPrimes(LIMIT3));
        System.out.println(countPrimes(LIMIT4));

        // Print footer marker
        System.out.println(FOOTER_MARKER);
    }
}
