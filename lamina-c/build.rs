// build.rs — lamina-c
//
// cbindgen 0.27 does not yet recognise #[unsafe(no_mangle)] (Rust 2024 edition
// syntax), so auto-generating lamina.h from source is not reliable. The
// committed include/lamina.h is maintained by hand until cbindgen adds support.
//
// When cbindgen gains #[unsafe(no_mangle)] support, replace this file with:
//
//   cbindgen::Builder::new()
//       .with_crate(env!("CARGO_MANIFEST_DIR"))
//       .with_config(cbindgen::Config::from_file("cbindgen.toml").unwrap())
//       .generate().unwrap()
//       .write_to_file("include/lamina.h");

fn main() {
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
