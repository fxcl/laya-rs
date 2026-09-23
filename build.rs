fn main() {
    // pyo3 0.29 stopped emitting the extension-module linker flags from its own build
    // script (`pyo3_build_config::add_extension_module_link_args` is now the consumer's
    // job). Without this, building the extension module on macOS fails to link with
    // undefined Python symbols such as `_PyBool_Type`.
    //
    // Gated on the `python` feature so a plain `cargo build` / `cargo test` never needs
    // a Python toolchain, and on macOS because that is the only platform that needs it.
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
