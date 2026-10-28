use std::env;

#[inline(never)]
fn arg(i: usize, fallback: u64) -> u64 {
    env::args().nth(i).map_or(fallback, |s| s.parse().expect("bad integer"))
}

fn main() {
    let iters = arg(1, 10000000);
    let mut x = arg(2, 12345);
    let mut y: u64 = 0;
    let mut z: u64 = 0;
    for _ in 0..iters {
        let m = x.wrapping_mul(1103515245).wrapping_add(12345);
        x = m & 2147483647;
        let s = x >> 7;
        let s2 = y << 3;
        y = (s ^ s2) & 2147483647;
        let w = y ^ x;
        z = z.wrapping_add(w);
    }
    println!("{}", x);
    println!("{}", y);
    println!("{}", z);
}
