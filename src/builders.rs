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
//! - [`GenomeSketchBuilder`] builds a reference `GenomeSketch` from contigs
//!   (backs `sylph_index_create`).

use crate::sketch::{
    dup_removal_lsh_full, dup_removal_lsh_full_exact, extract_markers, extract_markers_positions,
    pair_kmer, pair_kmer_single, Marker,
};
use crate::types::{GenomeSketch, MMHashSet, SequencesSketch};

use fxhash::{FxHashSet, FxHasher};
use scalable_cuckoo_filter::{ScalableCuckooFilter, ScalableCuckooFilterBuilder};

// ============================================================================
// Reference-genome sketch builder (produces a GenomeSketch → .syldb entry)
// ============================================================================

/// Streaming, byte-slice reference-genome sketcher.
///
/// Produces a reference `GenomeSketch` — the unit stored in a `.syldb`. The
/// kmer accumulation and the sort / dedup / min-spacing / pseudotax
/// finalization mirror `sketch::sketch_genome` exactly (the test
/// `genome_builder_matches_path_sketcher` pins this parity), so a genome built
/// here is byte-identical to one from `sylph sketch` given the same contigs in
/// the same order.
pub struct GenomeSketchBuilder {
    c: usize,
    k: usize,
    min_spacing: usize,
    pseudotax: bool,
    file_name: String,
    first_contig_name: Option<String>,
    gn_size: usize,
    contig_number: usize,
    // (contig_number, position, kmer) triples, exactly as sketch_genome's `vec`.
    kmer_positions: Vec<(usize, usize, u64)>,
}

impl GenomeSketchBuilder {
    /// Construct a fresh builder for one reference genome. `file_name` is the
    /// genome's identity in the resulting `GenomeSketch` (the FASTA path under
    /// the CLI; the genome-key column value under the FFI host).
    pub fn new(c: usize, k: usize, min_spacing: usize, pseudotax: bool, file_name: String) -> Self {
        GenomeSketchBuilder {
            c,
            k,
            min_spacing,
            pseudotax,
            file_name,
            first_contig_name: None,
            gn_size: 0,
            contig_number: 0,
            kmer_positions: Vec::new(),
        }
    }

    /// Add one contig. `contig_name` names the record; only the first contig's
    /// name is retained (as `first_contig_name`, tabs replaced with spaces),
    /// matching `sketch_genome`. Contigs must be added in a deterministic order
    /// for the resulting sketch to be reproducible.
    pub fn add_contig(&mut self, contig_name: &str, seq: &[u8]) {
        if self.first_contig_name.is_none() {
            self.first_contig_name = Some(contig_name.replace('\t', " "));
        }
        self.gn_size += seq.len();
        extract_markers_positions(seq, &mut self.kmer_positions, self.c, self.k, self.contig_number);
        self.contig_number += 1;
    }

    /// Consume the builder and produce the finalized `GenomeSketch`. Applies
    /// the same sort → drop-duplicated-kmers → min-spacing thinning →
    /// pseudotax-tracking pipeline as `sketch_genome`.
    pub fn finalize(self) -> GenomeSketch {
        let mut return_genome_sketch = GenomeSketch::default();
        return_genome_sketch.c = self.c;
        return_genome_sketch.k = self.k;
        return_genome_sketch.file_name = self.file_name;
        return_genome_sketch.first_contig_name = self.first_contig_name.unwrap_or_default();
        return_genome_sketch.gn_size = self.gn_size;

        let mut vec = self.kmer_positions;
        let mut pseudotax_track_kmers = vec![];
        let mut kmer_set = MMHashSet::default();
        let mut duplicate_set = MMHashSet::default();
        let mut new_vec = Vec::with_capacity(vec.len());
        vec.sort();
        for (_, _, km) in vec.iter() {
            if !kmer_set.contains(&km) {
                kmer_set.insert(km);
            } else {
                duplicate_set.insert(km);
            }
        }

        let mut last_pos = 0;
        let mut last_contig = 0;
        for (contig, pos, km) in vec.iter() {
            if !duplicate_set.contains(&km) {
                if last_pos == 0 || last_contig != *contig || pos - last_pos > self.min_spacing {
                    new_vec.push(*km);
                    last_contig = *contig;
                    last_pos = *pos;
                } else if self.pseudotax {
                    pseudotax_track_kmers.push(*km);
                }
            }
        }
        return_genome_sketch.genome_kmers = new_vec;
        return_genome_sketch.min_spacing = self.min_spacing;
        if self.pseudotax {
            return_genome_sketch.pseudotax_tracked_nonused_kmers = Some(pseudotax_track_kmers);
        }
        return_genome_sketch
    }
}

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
    use crate::sketch::{sketch_genome, sketch_pair_sequences};
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

    /// Read every record's (id, sequence) out of a FASTA file.
    fn slurp_fasta(path: &str) -> Vec<(String, Vec<u8>)> {
        let mut reader = parse_fastx_file(path).expect("parse_fastx_file");
        let mut out = Vec::new();
        while let Some(rec) = reader.next() {
            let rec = rec.expect("record");
            let id = String::from_utf8_lossy(rec.id()).to_string();
            out.push((id, rec.seq().into_owned()));
        }
        out
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

    /// GenomeSketchBuilder must reproduce `sketch_genome` exactly for all three
    /// bundled refs, including the multi-contig o157 (per-contig min-spacing
    /// reset). `sketch_genome` is what the CLI's `sylph sketch` calls per genome,
    /// so this is CLI parity without the external binary (which
    /// miint_builder_matches_real_cli_syldb additionally checks when available).
    #[test]
    fn genome_builder_matches_path_sketcher() {
        let (c, k, min_spacing, pseudotax) = (200usize, 31usize, 30usize, true);
        for ref_file in [
            "test_files/e.coli-EC590.fasta.gz",
            "test_files/e.coli-K12.fasta.gz",
            "test_files/e.coli-o157.fasta.gz",
        ] {
            let path_sketch =
                sketch_genome(c, k, ref_file, min_spacing, pseudotax).expect("sketch_genome");

            let mut builder =
                GenomeSketchBuilder::new(c, k, min_spacing, pseudotax, ref_file.to_string());
            for (id, seq) in slurp_fasta(ref_file) {
                builder.add_contig(&id, &seq);
            }
            let slice_sketch = builder.finalize();

            assert_eq!(
                path_sketch, slice_sketch,
                "GenomeSketchBuilder must reproduce sketch_genome exactly for {}",
                ref_file
            );
            assert!(
                !slice_sketch.genome_kmers.is_empty(),
                "expected a non-empty genome sketch for {}",
                ref_file
            );
        }
    }

    /// Content parity against a REAL `sylph sketch` .syldb (gated on
    /// MIINT_SYLPH_CLI_DB; params via MIINT_SYLPH_K/C/MINSPACING). Rebuilds the
    /// same genomes via GenomeSketchBuilder and asserts every GenomeSketch is
    /// identical. Sorted by file_name first: sylph writes genomes in
    /// parallel-completion order, so file order (and bytes) vary run-to-run —
    /// order-independent content parity is the guarantee.
    #[test]
    fn miint_builder_matches_real_cli_syldb() {
        use crate::types::GenomeSketch;
        use std::io::BufReader;

        let cli_db = match std::env::var("MIINT_SYLPH_CLI_DB") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip miint_builder_matches_real_cli_syldb: MIINT_SYLPH_CLI_DB unset");
                return;
            }
        };
        // Params the CLI db was built with (default to sylph's defaults). Lets
        // one test cover k/c/min_spacing parity by pointing at CLI dbs built
        // with different flags.
        let envn = |k: &str, d: usize| -> usize {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let (c, k, min_spacing) = (
            envn("MIINT_SYLPH_C", 200),
            envn("MIINT_SYLPH_K", 31),
            envn("MIINT_SYLPH_MINSPACING", 30),
        );

        let reader = BufReader::new(std::fs::File::open(&cli_db).expect("open CLI .syldb"));
        let mut cli: Vec<GenomeSketch> =
            bincode::deserialize_from(reader).expect("deserialize CLI .syldb");

        let files = [
            "test_files/e.coli-EC590.fasta.gz",
            "test_files/e.coli-K12.fasta.gz",
            "test_files/e.coli-o157.fasta.gz",
        ];
        let mut mine: Vec<GenomeSketch> = files
            .iter()
            .map(|f| {
                let mut b = GenomeSketchBuilder::new(c, k, min_spacing, true, f.to_string());
                let mut r = parse_fastx_file(f).expect("open ref");
                while let Some(rec) = r.next() {
                    let rec = rec.expect("record");
                    let id = String::from_utf8_lossy(rec.id()).to_string();
                    b.add_contig(&id, &rec.seq());
                }
                b.finalize()
            })
            .collect();

        cli.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        mine.sort_by(|a, b| a.file_name.cmp(&b.file_name));

        assert_eq!(cli.len(), mine.len(), "genome count differs from CLI");
        for (c, m) in cli.iter().zip(mine.iter()) {
            assert_eq!(
                c, m,
                "GenomeSketch differs from CLI for {} (kmers {} vs {}, gn_size {} vs {}, \
                 first_contig {:?} vs {:?})",
                c.file_name,
                c.genome_kmers.len(),
                m.genome_kmers.len(),
                c.gn_size,
                m.gn_size,
                c.first_contig_name,
                m.first_contig_name
            );
        }
    }
}
