using System;

class PrimeGeneration
{
    static bool IsPrime(long n)
    {
        if (n <= 1) return false;
        if (n == 2 || n == 3) return true;
        if (n % 2 == 0) return false;
        for (long d = 3; d * d <= n; d += 2)
            if (n % d == 0) return false;
        return true;
    }

    static long CountPrimes(long limit)
    {
        long count = 0;
        for (long current = 2; current <= limit; current++)
            if (IsPrime(current)) count++;
        return count;
    }

    static void Main()
    {
        Console.WriteLine(123456789);
        Console.WriteLine(CountPrimes(100));
        Console.WriteLine(CountPrimes(1000));
        Console.WriteLine(CountPrimes(10000));
        Console.WriteLine(CountPrimes(1000000));
        Console.WriteLine(987654321);
    }
}
