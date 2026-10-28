use std::env;

#[inline(never)]
fn arg(i: usize, fallback: u64) -> u64 {
    env::args().nth(i).map_or(fallback, |s| s.parse().expect("bad integer"))
}

fn factorial_iterative(n: u32) -> i64 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut result: i64 = 1;
    for i in 2..=n {
        result *= i as i64;
    }
    result
}

fn main() {
    let (n1, n2, n3, n4) = (arg(1, 10) as u32, arg(2, 12) as u32, arg(3, 15) as u32, arg(4, 18) as u32);
    let sweep_count = arg(5, 4000000);

    println!("123456789");
    println!("{}", factorial_iterative(n1));
    println!("{}", factorial_iterative(n2));
    println!("{}", factorial_iterative(n3));
    println!("{}", factorial_iterative(n4));

    let mut acc: u64 = 0;
    for s in 0..sweep_count {
        acc = acc.wrapping_add(factorial_iterative((s % 20) as u32) as u64);
    }
    println!("{}", acc);
    println!("987654321");
}
