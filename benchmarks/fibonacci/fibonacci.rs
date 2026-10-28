use std::env;

#[inline(never)]
fn arg(i: usize, fallback: u64) -> u64 {
    env::args().nth(i).map_or(fallback, |s| s.parse().expect("bad integer"))
}

fn fibonacci_iterative(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 2..=n {
        let temp = a.wrapping_add(b);
        a = b;
        b = temp;
    }
    b
}

fn main() {
    let (n1, n2, n3, n4) = (arg(1, 10), arg(2, 20), arg(3, 30), arg(4, 35));
    let sweep_count = arg(5, 200000);

    println!("123456789");
    println!("{}", fibonacci_iterative(n1));
    println!("{}", fibonacci_iterative(n2));
    println!("{}", fibonacci_iterative(n3));
    println!("{}", fibonacci_iterative(n4));

    let mut acc: u64 = 0;
    for s in 0..sweep_count {
        acc = acc.wrapping_add(fibonacci_iterative(s % 90));
    }
    println!("{}", acc);
    println!("987654321");
}
