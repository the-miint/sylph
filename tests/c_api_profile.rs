//! Phase 2.4 integration test: full FFI profile round-trip.
//!
//! Builds a syldb in-memory, builds a paired-end sketch via the FFI builder,
//! calls sylph_profile, decodes the resulting Arrow C Data Interface batch,
//! and asserts the K12-only single-row golden expectation.

#![cfg(all(feature = "arrow-ffi", feature = "fastx"))]

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;

use sylph::c_api::{
    sylph_database_free, sylph_database_load, sylph_get_last_error, sylph_profile,
    sylph_sketch_builder_add_pair, sylph_sketch_builder_create, sylph_sketch_builder_finalize,
    sylph_sketch_free, SylphProfileParams,
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
    let path = dir.join(format!("sylph_c_api_profile_test_{}.syldb", std::process::id()));
    let f = std::fs::File::create(&path).expect("create temp syldb");
    bincode::serialize_into(f, &genomes).expect("serialize");
    path
}

fn slurp_fastq(path: &str) -> Vec<Vec<u8>> {
    use needletail::parse_fastx_file;
    let mut r = parse_fastx_file(path).expect("parse_fastx_file");
    let mut out = Vec::new();
    while let Some(rec) = r.next() {
        out.push(rec.expect("record").seq().into_owned());
    }
    out
}

fn err_msg() -> String {
    unsafe {
        let p = sylph_get_last_error();
        if p.is_null() {
            "<no error>".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[test]
fn ffi_profile_recovers_k12_at_full_abundance() {
    use arrow::array::{
        Array, Float64Array, LargeStringArray, RecordBatch, StructArray, UInt32Array, UInt64Array,
    };
    use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};

    // ---- Build inputs ----
    let syldb_path = build_tiny_syldb();
    let r1 = slurp_fastq("test_files/k12_R1.fq");
    let r2 = slurp_fastq("test_files/k12_R2.fq");
    assert_eq!(r1.len(), r2.len());

    unsafe {
        let cpath = CString::new(syldb_path.to_string_lossy().as_bytes()).unwrap();
        let db = sylph_database_load(cpath.as_ptr());
        assert!(!db.is_null(), "database load failed: {}", err_msg());

        let sketch = sylph_sketch_builder_create(ptr::null());
        assert!(!sketch.is_null());
        for (a, b) in r1.iter().zip(r2.iter()) {
            let rc = sylph_sketch_builder_add_pair(sketch, a.as_ptr(), a.len(), b.as_ptr(), b.len());
            assert_eq!(rc, 0, "add_pair: {}", err_msg());
        }
        assert_eq!(sylph_sketch_builder_finalize(sketch), 0);

        let params = SylphProfileParams::default();
        let mut out_array = std::mem::MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut out_schema = std::mem::MaybeUninit::<FFI_ArrowSchema>::uninit();

        let rc = sylph_profile(db, sketch, &params, out_array.as_mut_ptr(), out_schema.as_mut_ptr());
        assert_eq!(rc, 0, "sylph_profile: {}", err_msg());

        // Decode the FFI output into an arrow ArrayData / RecordBatch.
        let ffi_arr = out_array.assume_init();
        let ffi_sch = out_schema.assume_init();
        let array_data = from_ffi(ffi_arr, &ffi_sch).expect("from_ffi");
        let struct_array = StructArray::from(array_data);
        let batch = RecordBatch::from(&struct_array);

        assert_eq!(batch.num_columns(), 9);
        assert_eq!(
            batch.num_rows(),
            1,
            "expected exactly 1 row (K12 only); got {}",
            batch.num_rows()
        );

        let names = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "genome_index",
                "genome_name",
                "contig_name",
                "sequence_abundance",
                "taxonomic_abundance",
                "adjusted_ani",
                "eff_cov",
                "naive_ani",
                "kmers_reassigned",
            ]
        );

        let _idx_col = batch.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        let names_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        let seq_ab = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let tax_ab = batch
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let adj_ani = batch
            .column(5)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let kmers_reassigned = batch
            .column(8)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let _ = kmers_reassigned;

        assert!(names_col.value(0).contains("K12"), "expected K12, got {}", names_col.value(0));
        assert!(
            (seq_ab.value(0) - 100.0).abs() < 0.01,
            "sequence_abundance: expected ~100, got {}",
            seq_ab.value(0)
        );
        assert!(
            (tax_ab.value(0) - 100.0).abs() < 0.01,
            "taxonomic_abundance: expected ~100, got {}",
            tax_ab.value(0)
        );
        assert!(
            (adj_ani.value(0) * 100.0 - 98.89).abs() < 0.5,
            "adjusted_ani: expected ~98.89, got {}",
            adj_ani.value(0) * 100.0
        );

        sylph_sketch_free(sketch);
        sylph_database_free(db);
    }

    let _ = std::fs::remove_file(&syldb_path);
}
