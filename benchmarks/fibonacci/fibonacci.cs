using System;

class Fibonacci
{
    // Configuration
    private const int N1 = 10;
    private const int N2 = 20;
    private const int N3 = 30;
    private const int N4 = 35;

    // Markers matching Lamina version
    private const long HEADER_MARKER = 123456789L;
    private const long FOOTER_MARKER = 987654321L;

    static long FibonacciIterative(int n)
    {
        if (n == 0) return 0;
        if (n == 1) return 1;

        long a = 0, b = 1;
        for (int i = 2; i <= n; i++)
        {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    static void Main()
    {
        // Print header marker
        Console.WriteLine(HEADER_MARKER);

        // Compute and print fibonacci numbers
        Console.WriteLine(FibonacciIterative(N1));
        Console.WriteLine(FibonacciIterative(N2));
        Console.WriteLine(FibonacciIterative(N3));
        Console.WriteLine(FibonacciIterative(N4));

        // Print footer marker
        Console.WriteLine(FOOTER_MARKER);
    }
}