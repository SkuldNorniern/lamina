using System;

class Factorial
{
    const int N1 = 10;
    const int N2 = 12;
    const int N3 = 15;
    const int N4 = 18;
    const long SweepCount = 4000000;

    static long FactorialIterative(int n)
    {
        if (n == 0 || n == 1) return 1;

        long result = 1;
        for (int i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    static void Main()
    {
        Console.WriteLine(123456789);
        Console.WriteLine(FactorialIterative(N1));
        Console.WriteLine(FactorialIterative(N2));
        Console.WriteLine(FactorialIterative(N3));
        Console.WriteLine(FactorialIterative(N4));

        ulong acc = 0;
        for (long s = 0; s < SweepCount; s++)
            acc += (ulong)FactorialIterative((int)(s % 20));
        Console.WriteLine(acc);
        Console.WriteLine(987654321);
    }
}
