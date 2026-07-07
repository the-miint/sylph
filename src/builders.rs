//! In-memory, byte-slice sketch builders for embedding sylph (duckdb-miint).
//!
//! Upstream sylph sketches from files via needletail (the `fastx` feature).
//! These builders instead accept raw sequence bytes fed one read/contig at a
//! time, so a host (the duckdb-miint FFI) can sketch data streamed out of a
//! DuckDB table without the `fastx` feature or any file I/O. They live in this
//! separate module — rather than in `sketch.rs` — so the embedding surface adds
//! no lines to upstream files and merges of new sylph releases stay clean.
//!
//! Both builders reuse sylph's own compute primitives (`extract_markers`,
//! `extract_markers_positions`, `pair_kmer*`, `dup_removal_lsh_full*`) so the
//! byte-fed path is identical to the file-fed path; the tests below assert that
//! equivalence against the bundled fixtures.
//!
//! - [`SketchPairBuilder`] / [`sketch_pair_slices`] build a sample
//!   `SequencesSketch` from reads (backs `sylph_profile`).

use crate::sketch::{
    dup_removal_lsh_full, dup_removal_lsh_full_exact, extract_markers, pair_kmer, pair_kmer_single,
    Marker,
};
use crate::types::SequencesSketch;

use fxhash::{FxHashSet, FxHasher};
use scalable_cuckoo_filter::{ScalableCuckooFilter, ScalableCuckooFilterBuilder};

// ============================================================================
// Sample-read sketch builder (produces a SequencesSketch)
// ============================================================================

/// Streaming builder for paired-end FracMinHash sketches.
///
/// Equivalent to the inner loop of `sketch::sketch_pair_sequences` but driven
/// by the caller — feed `(r1, Some(r2))` pairs (or `(r1, None)` for single-end)
/// one at a time, then call `finalize` to obtain the `SequencesSketch`.
///
/// This is the direct primitive the FFI streaming sketch builder wraps.
pub struct SketchPairBuilder {
    sketch: SequencesSketch,
    no_dedup: bool,
    dedup_fpr: f64,
    kmer_pair_set_exact: FxHashSet<(u64, [Marker; 2])>,
    kmer_pair_set_approx: ScalableCuckooFilter<(u64, [Marker; 2]), FxHasher>,
    num_dup_removed: usize,
    mean_read_length: f64,
    counter: f64,
    paired: bool,
    paired_locked: bool,
    temp_vec1: Vec<u64>,
    temp_vec2: Vec<u64>,
}

impl SketchPairBuilder {
    /// Construct a fresh builder. `file_name` and `sample_name` are propagated
    /// to the resulting `SequencesSketch` for diagnostic logging only — they
    /// don't affect compute.
    pub fn new(
        file_name: String,
        sample_name: Option<String>,
        c: usize,
        k: usize,
        no_dedup: bool,
        dedup_fpr: f64,
    ) -> Self {
        let sketch = SequencesSketch::new(file_name, c, k, false, sample_name, 0.);
        let fpr = if dedup_fpr != 0.0 { dedup_fpr } else { 0.001 };
        let kmer_pair_set_approx = ScalableCuckooFilterBuilder::new()
            .initial_capacity(1_000_000_0)
            .false_positive_probability(fpr)
            .hasher(FxHasher::default())
            .finish();
        SketchPairBuilder {
            sketch,
            no_dedup,
            dedup_fpr,
            kmer_pair_set_exact: FxHashSet::default(),
            kmer_pair_set_approx,
            num_dup_removed: 0,
            mean_read_length: 0.0,
            counter: 0.0,
            paired: false,
            paired_locked: false,
            temp_vec1: Vec::new(),
            temp_vec2: Vec::new(),
        }
    }

    /// Add one read pair. `r2 = None` for single-end. Mixing single-end and
    /// paired-end input within one builder is an error (the resulting sketch
    /// would have ambiguous semantics), so the first call locks `paired` mode.
    pub fn add_pair(&mut self, r1: &[u8], r2: Option<&[u8]>) {
        let is_paired = r2.is_some();
        if !self.paired_locked {
            self.paired = is_paired;
            self.paired_locked = true;
        } else if self.paired != is_paired {
            // Refuse to silently mix; log and drop. Caller bug, not data bug.
            log::error!(
                "SketchPairBuilder: mixing single-end and paired-end reads within \
                 one builder is not supported (locked={}, this={})",
                self.paired,
                is_paired
            );
            return;
        }

        self.temp_vec1.clear();
        self.temp_vec2.clear();
        extract_markers(r1, &mut self.temp_vec1, self.sketch.c, self.sketch.k);
        let kmer_pair = match r2 {
            Some(r2_seq) => {
                extract_markers(r2_seq, &mut self.temp_vec2, self.sketch.c, self.sketch.k);
                pair_kmer(r1, r2_seq)
            }
            None => pair_kmer_single(r1),
        };

        // Moving-average read length; use r1 only, matching upstream behaviour.
        self.counter += 1.0;
        self.mean_read_length += ((r1.len() as f64) - self.mean_read_length) / self.counter;

        // Index-based iteration so we can call &mut self inside the loop;
        // u64 is Copy so this is allocation-free.
        let len1 = self.temp_vec1.len();
        for i in 0..len1 {
            let km = self.temp_vec1[i];
            self.dedup_one(&km, kmer_pair);
        }
        let len2 = self.temp_vec2.len();
        for i in 0..len2 {
            let km = self.temp_vec2[i];
            if self.temp_vec1.contains(&km) {
                continue;
            }
            self.dedup_one(&km, kmer_pair);
        }
    }

    fn dedup_one(&mut self, km: &u64, kmer_pair: Option<([Marker; 2], [Marker; 2])>) {
        if self.dedup_fpr == 0.0 {
            dup_removal_lsh_full_exact(
                &mut self.sketch.kmer_counts,
                &mut self.kmer_pair_set_exact,
                km,
                kmer_pair,
                &mut self.num_dup_removed,
                self.no_dedup,
                None,
            );
        } else {
            dup_removal_lsh_full(
                &mut self.sketch.kmer_counts,
                &mut self.kmer_pair_set_approx,
                km,
                kmer_pair,
                &mut self.num_dup_removed,
                self.no_dedup,
            );
        }
    }

    /// Consume the builder and return the finalized sketch. After this call
    /// the builder is gone; create a new one for additional samples.
    pub fn finalize(mut self) -> SequencesSketch {
        self.sketch.mean_read_length = self.mean_read_length;
        self.sketch.paired = self.paired;
        self.sketch
    }

    /// Diagnostic accessor for tests / debugging. Equivalent to upstream's
    /// debug-logged duplication count.
    pub fn num_dup_removed(&self) -> usize {
        self.num_dup_removed
    }
}

/// One-shot paired-end sketcher driven by an iterator of byte-slice pairs.
///
/// Convenience wrapper around `SketchPairBuilder` for callers that already
/// have all reads in hand (e.g. an Arrow array materialized in one call).
/// Iterating consumes `reads`. `r2 = None` indicates single-end.
pub fn sketch_pair_slices<'a, I>(
    reads: I,
    file_name: String,
    sample_name: Option<String>,
    c: usize,
    k: usize,
    no_dedup: bool,
    dedup_fpr: f64,
) -> SequencesSketch
where
    I: IntoIterator<Item = (&'a [u8], Option<&'a [u8]>)>,
{
    let mut builder = SketchPairBuilder::new(file_name, sample_name, c, k, no_dedup, dedup_fpr);
    for (r1, r2) in reads {
        builder.add_pair(r1, r2);
    }
    builder.finalize()
}

// The tests feed the builders the same fixtures as the file-based sketchers in
// `sketch.rs` and assert byte-for-byte equivalence, so any divergence between
// the byte-fed and file-fed paths fails a test. They read files, so they are
// gated behind `fastx` like the path-based sketchers they compare against.
#[cfg(test)]
#[cfg(feature = "fastx")]
mod tests {
    use super::*;
    use crate::sketch::sketch_pair_sequences;
    use needletail::parse_fastx_file;

    /// Read every record's sequence out of a FASTQ file into owned Vec<u8>s.
    fn slurp_fastq(path: &str) -> Vec<Vec<u8>> {
        let mut reader = parse_fastx_file(path).expect("parse_fastx_file");
        let mut seqs = Vec::new();
        while let Some(rec) = reader.next() {
            let rec = rec.expect("record");
            seqs.push(rec.seq().into_owned());
        }
        seqs
    }

    /// Sketch via the path-based primitive vs. the slice-based one and check
    /// that the resulting kmer_count maps are identical.
    #[test]
    fn slice_sketcher_matches_path_sketcher_paired() {
        let r1_path = "test_files/k12_R1.fq";
        let r2_path = "test_files/k12_R2.fq";

        // Path-based reference output. dedup_fpr=0 forces exact dedup so the
        // FxHashSet branch executes (deterministic; the cuckoo-filter branch
        // has tunable false-positive behaviour).
        let path_sketch = sketch_pair_sequences(r1_path, r2_path, 200, 31, None, false, 0.0)
            .expect("path sketcher");

        // Slice-based: same data, same params.
        let r1_seqs = slurp_fastq(r1_path);
        let r2_seqs = slurp_fastq(r2_path);
        assert_eq!(r1_seqs.len(), r2_seqs.len(), "paired length mismatch");
        let pairs: Vec<(&[u8], Option<&[u8]>)> = r1_seqs
            .iter()
            .zip(r2_seqs.iter())
            .map(|(r1, r2)| (r1.as_slice(), Some(r2.as_slice())))
            .collect();

        let slice_sketch = sketch_pair_slices(pairs, r1_path.to_string(), None, 200, 31, false, 0.0);

        assert_eq!(
            path_sketch.kmer_counts, slice_sketch.kmer_counts,
            "kmer_counts must be identical between path and slice sketchers"
        );
        assert_eq!(path_sketch.c, slice_sketch.c, "c parameter must match");
        assert_eq!(path_sketch.k, slice_sketch.k, "k parameter must match");
        // Mean read length is computed identically (moving average over r1).
        assert!(
            (path_sketch.mean_read_length - slice_sketch.mean_read_length).abs() < 1e-9,
            "mean_read_length divergence: path={}, slice={}",
            path_sketch.mean_read_length,
            slice_sketch.mean_read_length,
        );
    }

    /// Single-end variant: feed only r1, expect SE-flavoured pair_kmer_single
    /// usage and a non-paired output sketch.
    #[test]
    fn slice_sketcher_single_end() {
        let r1_path = "test_files/k12_R1.fq";
        let r1_seqs = slurp_fastq(r1_path);
        let pairs: Vec<(&[u8], Option<&[u8]>)> =
            r1_seqs.iter().map(|r1| (r1.as_slice(), None)).collect();

        let sketch = sketch_pair_slices(pairs, r1_path.to_string(), None, 200, 31, false, 0.0);
        assert!(!sketch.paired, "single-end builder must report paired=false");
        assert!(!sketch.kmer_counts.is_empty(), "expected non-empty sketch");
    }

    /// Streaming builder — same data, called incrementally — must produce the
    /// same sketch as the one-shot iterator wrapper.
    #[test]
    fn streaming_builder_matches_one_shot() {
        let r1_path = "test_files/k12_R1.fq";
        let r2_path = "test_files/k12_R2.fq";
        let r1_seqs = slurp_fastq(r1_path);
        let r2_seqs = slurp_fastq(r2_path);

        let one_shot = {
            let pairs: Vec<(&[u8], Option<&[u8]>)> = r1_seqs
                .iter()
                .zip(r2_seqs.iter())
                .map(|(r1, r2)| (r1.as_slice(), Some(r2.as_slice())))
                .collect();
            sketch_pair_slices(pairs, r1_path.to_string(), None, 200, 31, false, 0.0)
        };

        let mut builder = SketchPairBuilder::new(r1_path.to_string(), None, 200, 31, false, 0.0);
        for (r1, r2) in r1_seqs.iter().zip(r2_seqs.iter()) {
            builder.add_pair(r1, Some(r2));
        }
        let streamed = builder.finalize();

        assert_eq!(one_shot.kmer_counts, streamed.kmer_counts);
        assert_eq!(one_shot.paired, streamed.paired);
    }
}
