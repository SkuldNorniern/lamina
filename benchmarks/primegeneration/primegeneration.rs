use std::env;

fn arg(i: usize) -> i64 {
    env::args().nth(i).expect("missing argument").parse().expect("bad integer")
}

fn is_prime(n: i64) -> bool {
    if n <= 1 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn count_primes(limit: i64) -> i64 {
    (2..=limit).filter(|&c| is_prime(c)).count() as i64
}

fn main() {
    println!("123456789");
    for i in 1..=4 {
        println!("{}", count_primes(arg(i)));
    }
    println!("987654321");
}
