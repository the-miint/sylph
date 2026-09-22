//! Sketch builders fed from in-memory byte slices instead of files.
//!
//! [`GenomeSketchBuilder`] produces a `GenomeSketch` from contigs and
//! [`SketchPairBuilder`] a `SequencesSketch` from reads, using the same
//! primitives as `sketch_genome` / `sketch_pair_sequences`; the tests assert
//! the results are identical.

use crate::constants::DEFAULT_RNG_SEED;
use crate::sketch::{
    dup_removal_lsh_full, dup_removal_lsh_full_exact, extract_markers, extract_markers_positions,
    pair_kmer, pair_kmer_single,
};
use crate::types::{GenomeSketch, MMHashSet, SequencesSketch};

use fxhash::{FxHashSet, FxHasher};
use rand::{rngs::SmallRng, SeedableRng};
use scalable_cuckoo_filter::{ScalableCuckooFilter, ScalableCuckooFilterBuilder};

/// Same as the private `sketch::Marker` alias.
type Marker = u32;

/// Builds a reference `GenomeSketch` from contigs; mirrors `sketch_genome`.
pub struct GenomeSketchBuilder {
    c: usize,
    k: usize,
    min_spacing: usize,
    pseudotax: bool,
    file_name: String,
    first_contig_name: Option<String>,
    gn_size: usize,
    contig_number: usize,
    // (contig_number, position, kmer), as in sketch_genome.
    kmer_positions: Vec<(usize, usize, u64)>,
}

impl GenomeSketchBuilder {
    /// `file_name` becomes the genome's identity in the resulting sketch.
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

    /// Add one contig. Only the first contig's name is kept, as in `sketch_genome`.
    pub fn add_contig(&mut self, contig_name: &str, seq: &[u8]) {
        if self.first_contig_name.is_none() {
            self.first_contig_name = Some(contig_name.replace('\t', " "));
        }
        self.gn_size += seq.len();
        extract_markers_positions(
            seq,
            &mut self.kmer_positions,
            self.c,
            self.k,
            self.contig_number,
        );
        self.contig_number += 1;
    }

    /// Sort, drop duplicated k-mers, apply min-spacing and pseudotax tracking,
    /// as in `sketch_genome`.
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

/// Builds a sample `SequencesSketch` from reads fed one pair at a time
/// (`r2 = None` for single-end); mirrors the loop in `sketch_pair_sequences`.
pub struct SketchPairBuilder {
    sketch: SequencesSketch,
    no_dedup: bool,
    dedup_fpr: f64,
    kmer_pair_set_exact: FxHashSet<(u64, [Marker; 2])>,
    kmer_pair_set_approx: ScalableCuckooFilter<(u64, [Marker; 2]), FxHasher, SmallRng>,
    num_dup_removed: usize,
    mean_read_length: f64,
    counter: f64,
    paired: bool,
    paired_locked: bool,
    temp_vec1: Vec<u64>,
    temp_vec2: Vec<u64>,
}

impl SketchPairBuilder {
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
            .rng(SmallRng::seed_from_u64(DEFAULT_RNG_SEED))
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

    /// Add one read pair. The first call fixes single- vs paired-end mode;
    /// reads of the other kind are logged and dropped.
    pub fn add_pair(&mut self, r1: &[u8], r2: Option<&[u8]>) {
        let is_paired = r2.is_some();
        if !self.paired_locked {
            self.paired = is_paired;
            self.paired_locked = true;
        } else if self.paired != is_paired {
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

        // Running mean read length over r1, as in sketch_pair_sequences.
        self.counter += 1.0;
        self.mean_read_length += ((r1.len() as f64) - self.mean_read_length) / self.counter;

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

    pub fn finalize(mut self) -> SequencesSketch {
        self.sketch.mean_read_length = self.mean_read_length;
        self.sketch.paired = self.paired;
        self.sketch
    }

    pub fn num_dup_removed(&self) -> usize {
        self.num_dup_removed
    }
}

/// Sketch an iterator of `(r1, Option<r2>)` byte-slice pairs in one call.
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

// Parity against the file-based sketchers on the bundled fixtures.
#[cfg(test)]
#[cfg(feature = "fastx")]
mod tests {
    use super::*;
    use crate::sketch::{sketch_genome, sketch_pair_sequences};
    use needletail::parse_fastx_file;

    fn slurp_fastq(path: &str) -> Vec<Vec<u8>> {
        let mut reader = parse_fastx_file(path).expect("parse_fastx_file");
        let mut seqs = Vec::new();
        while let Some(rec) = reader.next() {
            let rec = rec.expect("record");
            seqs.push(rec.seq().into_owned());
        }
        seqs
    }

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

    #[test]
    fn slice_sketcher_matches_path_sketcher_paired() {
        let r1_path = "test_files/k12_R1.fq";
        let r2_path = "test_files/k12_R2.fq";

        // dedup_fpr = 0 selects the exact (deterministic) dedup path.
        let path_sketch = sketch_pair_sequences(r1_path, r2_path, 200, 31, None, false, 0.0)
            .expect("path sketcher");

        let r1_seqs = slurp_fastq(r1_path);
        let r2_seqs = slurp_fastq(r2_path);
        assert_eq!(r1_seqs.len(), r2_seqs.len(), "paired length mismatch");
        let pairs: Vec<(&[u8], Option<&[u8]>)> = r1_seqs
            .iter()
            .zip(r2_seqs.iter())
            .map(|(r1, r2)| (r1.as_slice(), Some(r2.as_slice())))
            .collect();

        let slice_sketch =
            sketch_pair_slices(pairs, r1_path.to_string(), None, 200, 31, false, 0.0);

        assert_eq!(
            path_sketch.kmer_counts, slice_sketch.kmer_counts,
            "kmer_counts must be identical between path and slice sketchers"
        );
        assert_eq!(path_sketch.c, slice_sketch.c, "c parameter must match");
        assert_eq!(path_sketch.k, slice_sketch.k, "k parameter must match");
        assert!(
            (path_sketch.mean_read_length - slice_sketch.mean_read_length).abs() < 1e-9,
            "mean_read_length divergence: path={}, slice={}",
            path_sketch.mean_read_length,
            slice_sketch.mean_read_length,
        );
    }

    #[test]
    fn slice_sketcher_single_end() {
        let r1_path = "test_files/k12_R1.fq";
        let r1_seqs = slurp_fastq(r1_path);
        let pairs: Vec<(&[u8], Option<&[u8]>)> =
            r1_seqs.iter().map(|r1| (r1.as_slice(), None)).collect();

        let sketch = sketch_pair_slices(pairs, r1_path.to_string(), None, 200, 31, false, 0.0);
        assert!(
            !sketch.paired,
            "single-end builder must report paired=false"
        );
        assert!(!sketch.kmer_counts.is_empty(), "expected non-empty sketch");
    }

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

    // Includes the multi-contig o157 genome (per-contig min-spacing reset).
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
}
