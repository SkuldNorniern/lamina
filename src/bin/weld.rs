//! The `weld` linker, installed alongside `lamina` so that `cargo install lamina`
//! places the linker the default toolchain reaches for.

use weld::main as weld_main;

fn main() {
    weld_main();
}
