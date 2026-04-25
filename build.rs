fn main() {
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
