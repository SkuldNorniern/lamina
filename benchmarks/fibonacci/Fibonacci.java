public class Fibonacci {
    private static final int N1 = 10;
    private static final int N2 = 20;
    private static final int N3 = 30;
    private static final int N4 = 35;
    private static final long SWEEP_COUNT = 200000L;

    public static long fibonacciIterative(long n) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        long a = 0, b = 1;
        for (long i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    public static void main(String[] args) {
        System.out.println(123456789L);
        System.out.println(fibonacciIterative(N1));
        System.out.println(fibonacciIterative(N2));
        System.out.println(fibonacciIterative(N3));
        System.out.println(fibonacciIterative(N4));

        long acc = 0;
        for (long s = 0; s < SWEEP_COUNT; s++)
            acc += fibonacciIterative(s % 90);
        System.out.println(acc);
        System.out.println(987654321L);
    }
}
