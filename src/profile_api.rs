//! Pure data-in / data-out profile entry point.
//!
//! `run_profile_compute` is the testable seam between the CLI orchestrator
//! (`contain::contain`) and the C/Arrow FFI (`c_api.rs`): it takes already-loaded
//! reference genomes and a sample sketch, runs the full sylph profiling
//! pipeline (containment-ANI estimation, optional pseudotax winner-take-all
//! reassignment, abundance computation), and returns owned, lifetime-free
//! results that can outlive the inputs.
//!
//! The compute kernel itself still lives in `contain.rs` (`get_stats`,
//! `bootstrap_interval`, `winner_table`, etc.) — this module only owns the
//! orchestration loop and the `ProfileArgs` / `OwnedAniResult` shapes.

use crate::cmdline::ContainArgs;
use crate::contain::{
    derep_if_reassign_threshold, estimate_covered_bases, estimate_true_cov, get_kmer_identity,
    get_stats, winner_table,
};
use crate::types::{AdjustStatus, AniResult, GenomeSketch, SequencesSketch};
use clap::Parser;
use rayon::prelude::*;
use std::sync::Mutex;

/// Lambda estimator selection. Sylph supports four; the default is `Ratio`,
/// matching upstream's `sylph profile --reads`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdaEstimator {
    Ratio,
    Mme,
    Nb,
    Mle,
}

impl Default for LambdaEstimator {
    fn default() -> Self {
        LambdaEstimator::Ratio
    }
}

/// User-facing profile parameters: the subset of `ContainArgs` (cmdline.rs)
/// that affects compute, with defaults matching `sylph profile`.
///
/// `pseudotax = true` is the "profile" mode (winner-take-all reassignment +
/// taxonomic/sequence abundance). Setting it to `false` produces "query"-mode
/// results (containment ANI only, no abundances). The duckdb-miint table
/// function always sets `pseudotax = true`.
///
/// The compute kernels in `contain.rs` take a `ContainArgs`; `to_contain_args`
/// builds one by running these values through sylph's own clap definitions,
/// so every other `ContainArgs` field carries exactly the CLI default and new
/// upstream knobs need no mirroring here unless we want to expose them.
#[derive(Debug, Clone)]
pub struct ProfileArgs {
    pub estimator: LambdaEstimator,
    pub min_count_correct: f64,
    pub min_number_kmers: f64,
    /// Minimum contained k-mers for a genome to count as a hit (`--min-contain`).
    pub min_contain: usize,
    /// Minimum adjusted ANI cutoff in percent (0..100). `None` uses the
    /// internal default (90% in query mode, 95% in profile mode).
    pub minimum_ani: Option<f64>,
    pub pseudotax: bool,
    pub estimate_unknown: bool,
    pub estimate_read_counts: bool,
    pub no_ci: bool,
    pub no_adj: bool,
    pub mean_coverage: bool,
    /// Optional override for sequence identity in percent (0..100). When
    /// `None`, identity is estimated from the sample sketch.
    pub seq_id: Option<f64>,
    pub redundant_ani: f64,
    pub log_reassignments: bool,
    /// Number of rayon worker threads to use for the per-genome inner loop.
    /// `0` means "let rayon decide" (uses the global pool).
    pub num_threads: usize,
}

/// clap wrapper so `ContainArgs` (an `Args` struct) can be parsed on its own.
#[derive(Parser)]
#[clap(name = "sylph")]
struct ContainArgsCli {
    #[clap(flatten)]
    args: ContainArgs,
}

impl Default for ProfileArgs {
    fn default() -> Self {
        // Numeric defaults are read from the clap definitions in cmdline.rs —
        // the same source of truth `sylph profile` uses.
        let d = ContainArgsCli::parse_from(["sylph"]).args;
        ProfileArgs {
            estimator: LambdaEstimator::Ratio,
            min_count_correct: d.min_count_correct,
            min_number_kmers: d.min_number_kmers,
            min_contain: d.min_contain,
            minimum_ani: None,
            pseudotax: true,
            estimate_unknown: false,
            estimate_read_counts: false,
            no_ci: false,
            no_adj: false,
            mean_coverage: false,
            seq_id: None,
            redundant_ani: d.redundant_ani,
            log_reassignments: false,
            num_threads: 0,
        }
    }
}

impl ProfileArgs {
    /// Materialise the `ContainArgs` the `contain.rs` kernels expect, applying
    /// the same normalisation `contain::contain` does (`--estimate-read-counts`
    /// implies `-u`).
    pub fn to_contain_args(&self) -> ContainArgs {
        let mut argv: Vec<String> = vec!["sylph".to_string()];
        let mut flag = |f: &str| argv.push(f.to_string());
        flag(match self.estimator {
            LambdaEstimator::Ratio => "--ratio",
            LambdaEstimator::Mme => "--mme",
            LambdaEstimator::Nb => "--nb",
            LambdaEstimator::Mle => "--mle",
        });
        if self.pseudotax {
            flag("--pseudotax");
        }
        if self.estimate_unknown {
            flag("--estimate-unknown");
        }
        if self.estimate_read_counts {
            flag("--estimate-read-counts");
        }
        if self.no_ci {
            flag("--no-ci");
        }
        if self.no_adj {
            flag("--no-adjust");
        }
        if self.mean_coverage {
            flag("--mean-coverage");
        }
        if self.log_reassignments {
            flag("--log-reassignments");
        }
        let mut opt = |f: &str, v: String| {
            argv.push(f.to_string());
            argv.push(v);
        };
        opt("--min-count-correct", self.min_count_correct.to_string());
        opt("--min-number-kmers", self.min_number_kmers.to_string());
        opt("--min-contain", self.min_contain.to_string());
        opt("--redundancy-threshold", self.redundant_ani.to_string());
        if let Some(a) = self.minimum_ani {
            opt("--minimum-ani", a.to_string());
        }
        if let Some(id) = self.seq_id {
            opt("--read-seq-id", id.to_string());
        }
        let mut args = ContainArgsCli::try_parse_from(&argv)
            .unwrap_or_else(|e| panic!("ProfileArgs -> ContainArgs: {e} (argv {argv:?})"))
            .args;
        if args.estimate_read_counts {
            args.estimate_unknown = true;
        }
        args
    }
}

/// Owned, lifetime-free profile result for one genome. Strings are copied
/// out of the input `GenomeSketch` so callers can drop the database while
/// still using these results.
#[derive(Debug, Clone)]
pub struct OwnedAniResult {
    pub genome_name: String,
    pub contig_name: String,
    pub seq_name: String,
    /// Coverage-corrected containment ANI in [0, 1].
    pub adjusted_ani: f64,
    /// Raw containment ANI in [0, 1] (no λ correction).
    pub naive_ani: f64,
    /// Effective coverage estimate.
    pub eff_cov: f64,
    /// Mean coverage over k-mers seen at least once.
    pub mean_cov: f64,
    /// Median coverage over contained k-mers.
    pub median_cov: f64,
    /// (matched_kmers, total_genome_kmers).
    pub containment_index: (usize, usize),
    pub lambda: AdjustStatus,
    pub ani_ci_low: Option<f64>,
    pub ani_ci_high: Option<f64>,
    pub lambda_ci_low: Option<f64>,
    pub lambda_ci_high: Option<f64>,
    pub kmers_lost: Option<usize>,
    /// Taxonomic abundance in percent (0..100). Sums to ≈100 across results.
    /// `None` when `pseudotax = false`.
    pub taxonomic_abundance: Option<f64>,
    /// Sequence abundance in percent (0..100). Sums to ≈100 across results.
    /// `None` when `pseudotax = false`.
    pub sequence_abundance: Option<f64>,
}

impl OwnedAniResult {
    fn from_borrowed(r: &AniResult) -> Self {
        OwnedAniResult {
            genome_name: r.gn_name.to_string(),
            contig_name: r.contig_name.to_string(),
            seq_name: r.seq_name.clone(),
            // Clamp to 1.0 (= 100% ANI) to match the CLI's print path
            // (`f64::min(... * 100., 100.)` in contain.rs). Lambda correction
            // can push final_est_ani slightly past 1.0 on borderline-detected
            // genomes; sylph CLI hides this in display, so the FFI does too
            // for output equivalence.
            adjusted_ani: r.final_est_ani.min(1.0),
            naive_ani: r.naive_ani,
            eff_cov: r.final_est_cov,
            mean_cov: r.mean_cov,
            median_cov: r.median_cov,
            containment_index: r.containment_index,
            lambda: r.lambda,
            ani_ci_low: r.ani_ci.0,
            ani_ci_high: r.ani_ci.1,
            lambda_ci_low: r.lambda_ci.0,
            lambda_ci_high: r.lambda_ci.1,
            kmers_lost: r.kmers_lost,
            // rel_abund is the taxonomic abundance; seq_abund is sequence abundance.
            taxonomic_abundance: r.rel_abund,
            sequence_abundance: r.seq_abund,
        }
    }
}

/// Run the full sylph profile compute pipeline against a single sample
/// sketch and a set of reference genome sketches.
///
/// This is the algorithmic content of `contain::contain`'s inner loop,
/// extracted so it can be called both from the CLI orchestrator and from
/// the FFI without going through file I/O or clap.
///
/// Order of returned rows:
/// - `pseudotax = true`: descending by `taxonomic_abundance`.
/// - `pseudotax = false`: descending by `adjusted_ani`.
pub fn run_profile_compute(
    genomes: &[GenomeSketch],
    sample: &SequencesSketch,
    args: &ProfileArgs,
) -> Vec<OwnedAniResult> {
    if genomes.is_empty() {
        return Vec::new();
    }

    // Optional rayon thread-pool override. The duckdb-miint table function
    // uses this to either give all cores to a single sample (one sample, N
    // threads) or one thread per sample when many samples run in parallel.
    let owned_pool = if args.num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.num_threads)
            .build()
            .ok()
    } else {
        None
    };

    let compute = || run_profile_compute_inner(genomes, sample, args);

    if let Some(pool) = owned_pool {
        pool.install(compute)
    } else {
        compute()
    }
}

fn run_profile_compute_inner(
    genomes: &[GenomeSketch],
    sample: &SequencesSketch,
    args: &ProfileArgs,
) -> Vec<OwnedAniResult> {
    // The kernels below take sylph's own ContainArgs; from here on `args`
    // is that (field names coincide with ProfileArgs for everything used).
    let contain_args = args.to_contain_args();
    let args = &contain_args;

    // Estimated sequence identity used by estimate_true_cov.
    let kmer_id_opt = match args.seq_id {
        Some(pct) => Some((pct / 100.0).powf(sample.k as f64)),
        None => get_kmer_identity(sample, args.estimate_unknown),
    };

    // First pass: per-genome containment without winner-take-all.
    let stats_pass1: Mutex<Vec<AniResult>> = Mutex::new(Vec::new());
    genomes.par_iter().for_each(|gs| {
        if let Some(res) = get_stats(args, gs, sample, None, args.log_reassignments) {
            stats_pass1.lock().unwrap().push(res);
        }
    });
    let mut stats: Vec<AniResult> = stats_pass1.into_inner().unwrap();
    estimate_true_cov(
        &mut stats,
        kmer_id_opt,
        args.estimate_unknown,
        sample.mean_read_length,
        sample.k,
    );

    if args.pseudotax {
        let winner_map = winner_table(&stats, args.log_reassignments);
        let remaining: Vec<&GenomeSketch> = stats.iter().map(|s| s.genome_sketch).collect();
        let stats_pass2: Mutex<Vec<AniResult>> = Mutex::new(Vec::new());
        remaining.into_par_iter().for_each(|gs| {
            if let Some(res) =
                get_stats(args, gs, sample, Some(&winner_map), args.log_reassignments)
            {
                stats_pass2.lock().unwrap().push(res);
            }
        });
        stats = derep_if_reassign_threshold(
            &stats,
            stats_pass2.into_inner().unwrap(),
            args.redundant_ani,
            sample.k,
        );
        estimate_true_cov(
            &mut stats,
            kmer_id_opt,
            args.estimate_unknown,
            sample.mean_read_length,
            sample.k,
        );

        let bases_explained = if args.estimate_unknown {
            estimate_covered_bases(&stats, sample, sample.mean_read_length, sample.k)
        } else {
            1.0
        };

        let total_cov: f64 = stats.iter().map(|x| x.final_est_cov).sum();
        let total_seq_cov: f64 = stats
            .iter()
            .map(|x| x.final_est_cov * x.genome_sketch.gn_size as f64)
            .sum();
        for s in stats.iter_mut() {
            s.rel_abund = if total_cov > 0.0 {
                Some(s.final_est_cov / total_cov * 100.0)
            } else {
                Some(0.0)
            };
        }
        for s in stats.iter_mut() {
            if args.estimate_read_counts {
                s.seq_abund = Some(
                    (s.final_est_cov * s.genome_sketch.gn_size as f64 / sample.mean_read_length
                        * bases_explained)
                        .round(),
                );
            } else if total_seq_cov > 0.0 {
                s.seq_abund = Some(
                    s.final_est_cov * s.genome_sketch.gn_size as f64 / total_seq_cov
                        * 100.0
                        * bases_explained,
                );
            } else {
                s.seq_abund = Some(0.0);
            }
        }

        stats.sort_by(|x, y| {
            y.rel_abund
                .unwrap_or(0.0)
                .partial_cmp(&x.rel_abund.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        stats.sort_by(|x, y| {
            y.final_est_ani
                .partial_cmp(&x.final_est_ani)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    stats.iter().map(OwnedAniResult::from_borrowed).collect()
}
