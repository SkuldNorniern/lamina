using System;

class PrimeGeneration
{
    // Configuration - limits for counting
    private const int LIMIT1 = 100;
    private const int LIMIT2 = 1000;
    private const int LIMIT3 = 10000;
    private const int LIMIT4 = 50000;

    // Markers matching Lamina version
    private const long HEADER_MARKER = 123456789L;
    private const long FOOTER_MARKER = 987654321L;

    static bool IsPrime(int n)
    {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0) return false;

        for (int i = 3; i * i <= n; i += 2)
        {
            if (n % i == 0) return false;
        }
        return true;
    }

    static long CountPrimes(int limit)
    {
        if (limit < 2) return 0;
        long count = 0;
        for (int i = 2; i <= limit; i++)
        {
            if (IsPrime(i)) count++;
        }
        return count;
    }

    static void Main()
    {
        // Print header marker
        Console.WriteLine(HEADER_MARKER);

        // Count and print prime counts for different limits
        Console.WriteLine(CountPrimes(LIMIT1));
        Console.WriteLine(CountPrimes(LIMIT2));
        Console.WriteLine(CountPrimes(LIMIT3));
        Console.WriteLine(CountPrimes(LIMIT4));

        // Print footer marker
        Console.WriteLine(FOOTER_MARKER);
    }
}
