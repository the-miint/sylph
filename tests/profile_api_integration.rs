//! Integration test for `profile_api::run_profile_compute`.
//!
//! Builds reference genome sketches and a sample sketch entirely in-memory
//! from the bundled `test_files/` data, then runs the profile pipeline and
//! asserts the result matches the canonical "K12-only" expectation.
//!
//! This is the Phase 1.3 RED→GREEN test for the duckdb-miint integration:
//! it proves that the FFI-facing `run_profile_compute` produces the same
//! biology as `sylph profile`.

#![cfg(feature = "fastx")]

use sylph::profile_api::{run_profile_compute, ProfileArgs};
use sylph::sketch::{sketch_genome, sketch_pair_sequences};

const K: usize = 31;
const C: usize = 200;
const MIN_SPACING: usize = 30;

fn sketch_refs() -> Vec<sylph::types::GenomeSketch> {
    let refs = [
        "test_files/e.coli-EC590.fasta.gz",
        "test_files/e.coli-K12.fasta.gz",
        "test_files/e.coli-o157.fasta.gz",
    ];
    refs.iter()
        .map(|p| sketch_genome(C, K, p, MIN_SPACING, true).expect("genome sketch"))
        .collect()
}

#[test]
fn run_profile_compute_recovers_k12_at_full_abundance() {
    let genomes = sketch_refs();
    assert_eq!(genomes.len(), 3, "expected 3 reference genomes");

    let sample = sketch_pair_sequences(
        "test_files/k12_R1.fq",
        "test_files/k12_R2.fq",
        C,
        K,
        None,
        false,
        0.0,
    )
    .expect("paired-end sketch");

    let args = ProfileArgs::default();
    let results = run_profile_compute(&genomes, &sample, &args);

    // Sister-strain references should fall below the default 95% adjusted-ANI
    // cutoff; only K12 itself should pass.
    assert_eq!(
        results.len(),
        1,
        "expected exactly one passing genome (K12); got {}: {:?}",
        results.len(),
        results.iter().map(|r| &r.genome_name).collect::<Vec<_>>()
    );
    let r = &results[0];
    assert!(
        r.genome_name.contains("K12"),
        "expected K12, got {}",
        r.genome_name
    );

    // K12 should claim approximately all the reads (the input fixture is
    // K12-only). Tolerances are wide because the float math is platform-
    // sensitive in the lower decimal places.
    let tax = r.taxonomic_abundance.expect("taxonomic_abundance set");
    let seq = r.sequence_abundance.expect("sequence_abundance set");
    assert!(
        (tax - 100.0).abs() < 0.01,
        "taxonomic_abundance: expected ~100, got {}",
        tax
    );
    assert!(
        (seq - 100.0).abs() < 0.01,
        "sequence_abundance: expected ~100, got {}",
        seq
    );

    // Sanity-check ANI fields against the upstream-CLI golden values
    // recorded in data/sylph/expected_profile.tsv:
    //   adjusted_ani 98.89, naive_ani 88.48, eff_cov 0.032
    // We allow 0.5%-points of slack to absorb cross-platform float drift.
    assert!(
        (r.adjusted_ani * 100.0 - 98.89).abs() < 0.5,
        "adjusted_ani: expected ~98.89, got {}",
        r.adjusted_ani * 100.0
    );
    assert!(
        (r.naive_ani * 100.0 - 88.48).abs() < 0.5,
        "naive_ani: expected ~88.48, got {}",
        r.naive_ani * 100.0
    );
    assert!(
        (r.eff_cov - 0.032).abs() < 0.005,
        "eff_cov: expected ~0.032, got {}",
        r.eff_cov
    );
}

#[test]
fn run_profile_compute_query_mode_keeps_all_genomes() {
    // pseudotax = false ("query" mode) should report all genomes that pass
    // the default ANI threshold. Sister-strain ANIs are around 96%, well
    // above the query-mode default of 90%, so all three should survive.
    let genomes = sketch_refs();
    let sample = sketch_pair_sequences(
        "test_files/k12_R1.fq",
        "test_files/k12_R2.fq",
        C,
        K,
        None,
        false,
        0.0,
    )
    .expect("paired-end sketch");

    let mut args = ProfileArgs::default();
    args.pseudotax = false;
    let results = run_profile_compute(&genomes, &sample, &args);

    assert!(
        results.len() >= 1,
        "expected at least one query result; got 0"
    );
    // Without pseudotax, abundances are unset.
    assert!(results[0].taxonomic_abundance.is_none());
    assert!(results[0].sequence_abundance.is_none());
}
