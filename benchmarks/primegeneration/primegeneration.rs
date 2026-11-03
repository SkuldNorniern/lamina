use std::env;

#[inline(never)]
fn arg(i: usize, fallback: i64) -> i64 {
    env::args().nth(i).map_or(fallback, |s| s.parse().expect("bad integer"))
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
    let defaults = [100, 1000, 10000, 2000000];
    println!("123456789");
    for i in 1..=4 {
        println!("{}", count_primes(arg(i, defaults[i - 1])));
    }
    println!("987654321");
}
