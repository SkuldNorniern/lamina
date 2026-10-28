using System;

class Fibonacci
{
    const int N1 = 10;
    const int N2 = 20;
    const int N3 = 30;
    const int N4 = 35;
    const long SweepCount = 200000;

    static ulong FibonacciIterative(ulong n)
    {
        if (n == 0) return 0;
        if (n == 1) return 1;

        ulong a = 0, b = 1;
        for (ulong i = 2; i <= n; i++)
        {
            ulong temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    static void Main()
    {
        Console.WriteLine(123456789);
        Console.WriteLine(FibonacciIterative(N1));
        Console.WriteLine(FibonacciIterative(N2));
        Console.WriteLine(FibonacciIterative(N3));
        Console.WriteLine(FibonacciIterative(N4));

        ulong acc = 0;
        for (long s = 0; s < SweepCount; s++)
            acc += FibonacciIterative((ulong)s % 90);
        Console.WriteLine(acc);
        Console.WriteLine(987654321);
    }
}
