public class Fibonacci {
    // Configuration
    private static final int N1 = 10;
    private static final int N2 = 20;
    private static final int N3 = 30;
    private static final int N4 = 35;

    // Markers matching Lamina version
    private static final long HEADER_MARKER = 123456789L;
    private static final long FOOTER_MARKER = 987654321L;

    public static long fibonacciIterative(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    public static void main(String[] args) {
        // Print header marker
        System.out.println(HEADER_MARKER);

        // Compute and print fibonacci numbers
        System.out.println(fibonacciIterative(N1));
        System.out.println(fibonacciIterative(N2));
        System.out.println(fibonacciIterative(N3));
        System.out.println(fibonacciIterative(N4));

        // Print footer marker
        System.out.println(FOOTER_MARKER);
    }
}

