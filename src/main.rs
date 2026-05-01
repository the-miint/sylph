// The sylph CLI binary. Gated behind the `cli` feature: when sylph is built
// as a static library for duckdb-miint, the binary disappears and only the
// FFI surface remains. tikv-jemallocator was removed with the rest of the
// musl-static-only allocator setup (see Cargo.toml comment).

#[cfg(feature = "cli")]
use clap::Parser;
#[cfg(feature = "cli")]
use sylph::cmdline::*;
#[cfg(feature = "cli")]
use sylph::sketch;
#[cfg(feature = "cli")]
use sylph::contain;
#[cfg(feature = "cli")]
use sylph::inspect;

#[cfg(feature = "cli")]
fn main() {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Sketch(sketch_args) => sketch::sketch(sketch_args),
        Mode::Query(contain_args) => contain::contain(contain_args, false),
        Mode::Profile(contain_args) => contain::contain(contain_args, true),
        Mode::Inspect(inspect_args) => inspect::inspect(inspect_args),
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    // No-op stub. The `cargo rustc --crate-type=staticlib` invocation used
    // by duckdb-miint doesn't build main.rs, but cargo's check / test paths
    // do, so a stub avoids spurious compile errors.
}
