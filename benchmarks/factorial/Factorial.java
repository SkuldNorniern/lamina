public class Factorial {
    // Configuration
    private static final int N1 = 10;
    private static final int N2 = 12;
    private static final int N3 = 15;
    private static final int N4 = 18;

    // Markers matching Lamina version
    private static final long HEADER_MARKER = 123456789L;
    private static final long FOOTER_MARKER = 987654321L;

    public static long factorial_iterative(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }

        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    public static void main(String[] args) {
        // Print header marker
        System.out.println(HEADER_MARKER);

        // Compute and print factorial values
        System.out.println(factorial_iterative(N1));
        System.out.println(factorial_iterative(N2));
        System.out.println(factorial_iterative(N3));
        System.out.println(factorial_iterative(N4));

        // Print footer marker
        System.out.println(FOOTER_MARKER);
    }
}



