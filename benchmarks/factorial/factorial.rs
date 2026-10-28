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
    println!("123456789");
    println!("{}", factorial_iterative(10));
    println!("{}", factorial_iterative(12));
    println!("{}", factorial_iterative(15));
    println!("{}", factorial_iterative(18));

    let mut acc: u64 = 0;
    for s in 0..4_000_000u64 {
        acc = acc.wrapping_add(factorial_iterative((s % 20) as u32) as u64);
    }
    println!("{}", acc);
    println!("987654321");
}
