using System;

class Factorial
{
    // Configuration
    private const int N1 = 10;
    private const int N2 = 12;
    private const int N3 = 15;
    private const int N4 = 18;

    // Markers matching Lamina version
    private const long HEADER_MARKER = 123456789L;
    private const long FOOTER_MARKER = 987654321L;

    static long FactorialIterative(int n)
    {
        if (n == 0 || n == 1)
        {
            return 1;
        }

        long result = 1;
        for (int i = 2; i <= n; i++)
        {
            result *= i;
        }
        return result;
    }

    static void Main()
    {
        // Print header marker
        Console.WriteLine(HEADER_MARKER);

        // Compute and print factorial values
        Console.WriteLine(FactorialIterative(N1));
        Console.WriteLine(FactorialIterative(N2));
        Console.WriteLine(FactorialIterative(N3));
        Console.WriteLine(FactorialIterative(N4));

        // Print footer marker
        Console.WriteLine(FOOTER_MARKER);
    }
}



