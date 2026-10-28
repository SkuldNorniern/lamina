public class Factorial {
    private static final int N1 = 10;
    private static final int N2 = 12;
    private static final int N3 = 15;
    private static final int N4 = 18;
    private static final long SWEEP_COUNT = 4000000L;

    public static long factorialIterative(int n) {
        if (n == 0 || n == 1) return 1;

        long result = 1;
        for (int i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    public static void main(String[] args) {
        System.out.println(123456789L);
        System.out.println(factorialIterative(N1));
        System.out.println(factorialIterative(N2));
        System.out.println(factorialIterative(N3));
        System.out.println(factorialIterative(N4));

        long acc = 0;
        for (long s = 0; s < SWEEP_COUNT; s++)
            acc += factorialIterative((int)(s % 20));
        System.out.println(acc);
        System.out.println(987654321L);
    }
}
