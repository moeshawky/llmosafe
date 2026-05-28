fn main() {
    // Only generate C headers when the `ffi` feature is active.
    // During `cargo publish`, ffi is not enabled, so cbindgen won't
    // run and won't create files outside OUT_DIR.
    if std::env::var("CARGO_FEATURE_FFI").is_err() {
        return;
    }

    extern crate cbindgen;
    use std::env;
    use std::path::PathBuf;

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let package_name = env::var("CARGO_PKG_NAME").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join(format!("{package_name}.h"));

    std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();

    cbindgen::generate(crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file(output_file);
}
