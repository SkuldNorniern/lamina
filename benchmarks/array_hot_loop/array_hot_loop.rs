use std::env;

fn arg(i: usize) -> usize {
    env::args().nth(i).expect("missing argument").parse().expect("bad integer")
}

fn main() {
    let n = arg(1);
    let passes = arg(2);
    let arr: Vec<u64> = (0..n).map(|i| (i as u64) * 7 + 3).collect();

    let mut total: u64 = 0;
    for pass in 0..passes {
        let scale = (pass as u64) * 3;
        for &x in &arr {
            total = total.wrapping_add(x.wrapping_mul(scale));
        }
    }
    println!("{}", total);
}
