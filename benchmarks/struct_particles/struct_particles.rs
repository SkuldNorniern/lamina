use std::env;

fn arg(i: usize) -> i64 {
    env::args().nth(i).expect("missing argument").parse().expect("bad integer")
}

struct Particle {
    x: i64,
    y: i64,
    vx: i64,
    vy: i64,
}

fn main() {
    let steps = arg(1);
    let mut p = Particle { x: 0, y: 0, vx: arg(2), vy: arg(3) };
    let mut flips: i64 = 0;
    for _ in 0..steps {
        p.x = p.x.wrapping_add(p.vx);
        p.y = p.y.wrapping_add(p.vy);
        if p.x * p.x > 1000000 {
            p.vx = -p.vx;
            flips += 1;
        }
        if p.y * p.y > 1000000 {
            p.vy = -p.vy;
            flips += 1;
        }
    }
    println!("{}", p.x);
    println!("{}", p.y);
    println!("{}", flips);
}
