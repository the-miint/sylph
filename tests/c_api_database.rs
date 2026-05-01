//! Integration test for the database-load FFI surface.
//!
//! Builds a `.syldb` from the bundled `test_files/` references in a temp
//! file, then loads it via the C FFI and checks num_genomes. Self-contained:
//! does not depend on any test data outside the submodule.

#![cfg(feature = "fastx")]

use std::ffi::CString;
use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use sylph::c_api::{
    sylph_database_free, sylph_database_load, sylph_database_num_genomes, sylph_get_last_error,
};
use sylph::sketch::sketch_genome;

const K: usize = 31;
const C: usize = 200;
const MIN_SPACING: usize = 30;

fn build_tiny_syldb() -> PathBuf {
    let refs = [
        "test_files/e.coli-EC590.fasta.gz",
        "test_files/e.coli-K12.fasta.gz",
        "test_files/e.coli-o157.fasta.gz",
    ];
    let genomes: Vec<sylph::types::GenomeSketch> = refs
        .iter()
        .map(|p| sketch_genome(C, K, p, MIN_SPACING, true).expect("genome sketch"))
        .collect();

    let dir = std::env::temp_dir();
    // Use a unique-enough filename for parallel test runs.
    let path = dir.join(format!(
        "sylph_c_api_test_{}.syldb",
        std::process::id(),
    ));
    let f = std::fs::File::create(&path).expect("create temp syldb");
    bincode::serialize_into(f, &genomes).expect("serialize syldb");
    path
}

#[test]
fn ffi_loads_real_syldb_and_reports_correct_num_genomes() {
    let path = build_tiny_syldb();
    let cpath = CString::new(path.to_string_lossy().as_bytes()).expect("cstring");
    unsafe {
        let db = sylph_database_load(cpath.as_ptr());
        if db.is_null() {
            let err = sylph_get_last_error();
            let msg = if err.is_null() {
                "<no error message>".to_string()
            } else {
                CStr::from_ptr(err).to_string_lossy().into_owned()
            };
            panic!("sylph_database_load returned NULL: {}", msg);
        }
        assert_eq!(
            sylph_database_num_genomes(db),
            3,
            "expected 3 genomes (E. coli EC590, K12, O157)"
        );
        sylph_database_free(db);

        // Loading again must work — i.e. no global state was clobbered.
        let db2 = sylph_database_load(cpath.as_ptr());
        assert!(!db2.is_null(), "second load should also succeed");
        assert_eq!(sylph_database_num_genomes(db2), 3);
        sylph_database_free(db2);

        // Free-after-NULL is documented as a no-op.
        sylph_database_free(ptr::null_mut());
    }

    let _ = std::fs::remove_file(&path);
}
