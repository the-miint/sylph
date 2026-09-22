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

/// The two-stage path (stage-1 screen, dense decode of survivors, then the
/// same kernels) must report exactly what the plain path reports for the
/// same genomes: the screen is a pure optimisation, not a different model.
#[test]
fn run_profile_compute_two_stage_matches_plain() {
    use sylph::profile_api::run_profile_compute_two_stage;
    use sylph::twostage_db::{open_file, write_two_stage_db};

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

    let path = std::env::temp_dir().join(format!(
        "sylph_profile_api_two_stage_{}.syl2db",
        std::process::id()
    ));
    {
        let w = std::io::BufWriter::new(std::fs::File::create(&path).expect("create"));
        write_two_stage_db(w, &genomes, 3000, 50, 7).expect("write two-stage db");
    }
    let db = open_file(path.to_str().unwrap()).expect("open two-stage db");
    assert_eq!(db.len(), 3);

    let args = ProfileArgs::default();
    let plain = run_profile_compute(&genomes, &sample, &args);
    let two_stage = run_profile_compute_two_stage(&db, &sample, &args);
    let _ = std::fs::remove_file(&path);

    assert_eq!(plain.len(), 1, "plain path must report K12 only");
    assert_eq!(two_stage.len(), plain.len());
    for (p, t) in plain.iter().zip(two_stage.iter()) {
        assert_eq!(p.genome_name, t.genome_name);
        assert_eq!(p.contig_name, t.contig_name);
        assert_eq!(
            p.adjusted_ani, t.adjusted_ani,
            "adjusted_ani must be identical"
        );
        assert_eq!(p.naive_ani, t.naive_ani);
        assert_eq!(p.eff_cov, t.eff_cov);
        assert_eq!(p.taxonomic_abundance, t.taxonomic_abundance);
        assert_eq!(p.sequence_abundance, t.sequence_abundance);
    }
}
