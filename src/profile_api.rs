//! Profile already-loaded sketches without the CLI: no files, logger, global
//! thread pool or `exit`. The kernels stay in `contain.rs`; this module owns
//! only the per-sample loop from `contain::contain` and the owned result type.

use crate::cmdline::ContainArgs;
use crate::contain::{
    compute_dense_survivors, derep_if_reassign_threshold, estimate_covered_bases,
    estimate_true_cov, get_kmer_identity, get_stats, winner_table,
};
use crate::twostage_db::TwoStageDb;
use crate::types::{AdjustStatus, AniResult, GenomeSketch, SequencesSketch};
use clap::Parser;
use rayon::prelude::*;
use std::sync::Mutex;

/// Coverage (lambda) estimator; `Ratio` is the CLI default.
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

/// The compute-relevant subset of `ContainArgs`. `pseudotax = true` is
/// `profile` mode (reassignment + abundances), `false` is `query` mode.
/// Defaults are read from the clap definitions in cmdline.rs.
#[derive(Debug, Clone)]
pub struct ProfileArgs {
    pub estimator: LambdaEstimator,
    pub min_count_correct: f64,
    pub min_number_kmers: f64,
    pub min_contain: usize,
    /// Percent; `None` = CLI default (90 for query, 95 for profile).
    pub minimum_ani: Option<f64>,
    pub pseudotax: bool,
    pub estimate_unknown: bool,
    pub estimate_read_counts: bool,
    pub no_ci: bool,
    pub no_adj: bool,
    pub mean_coverage: bool,
    /// Read identity in percent; `None` = estimate from the sample.
    pub seq_id: Option<f64>,
    pub redundant_ani: f64,
    pub log_reassignments: bool,
    /// `.syl2db` only: stage-1 screen ANI in percent; `None` = CLI default.
    pub screen_ani: Option<f64>,
    /// Rayon threads for this call; `0` = the global pool.
    pub num_threads: usize,
}

#[derive(Parser)]
#[clap(name = "sylph")]
struct ContainArgsCli {
    #[clap(flatten)]
    args: ContainArgs,
}

impl Default for ProfileArgs {
    fn default() -> Self {
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
            screen_ani: None,
            num_threads: 0,
        }
    }
}

impl ProfileArgs {
    /// Build the `ContainArgs` the kernels take, via clap so every other field
    /// keeps the CLI default; `--estimate-read-counts` implies `-u` as in
    /// `contain::contain`.
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
        if let Some(s) = self.screen_ani {
            opt("--screen-ani", s.to_string());
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

/// One genome's result, owning its strings so it can outlive the database.
#[derive(Debug, Clone)]
pub struct OwnedAniResult {
    pub genome_name: String,
    pub contig_name: String,
    pub seq_name: String,
    /// Coverage-adjusted ANI in [0, 1].
    pub adjusted_ani: f64,
    pub naive_ani: f64,
    pub eff_cov: f64,
    pub mean_cov: f64,
    pub median_cov: f64,
    /// (contained k-mers, genome k-mers).
    pub containment_index: (usize, usize),
    pub lambda: AdjustStatus,
    pub ani_ci_low: Option<f64>,
    pub ani_ci_high: Option<f64>,
    pub lambda_ci_low: Option<f64>,
    pub lambda_ci_high: Option<f64>,
    pub kmers_lost: Option<usize>,
    /// Percent; `None` in query mode.
    pub taxonomic_abundance: Option<f64>,
    /// Percent; `None` in query mode.
    pub sequence_abundance: Option<f64>,
}

impl OwnedAniResult {
    fn from_borrowed(r: &AniResult) -> Self {
        OwnedAniResult {
            genome_name: r.gn_name.to_string(),
            contig_name: r.contig_name.to_string(),
            seq_name: r.seq_name.clone(),
            // The CLI clamps to 100% when printing; do the same.
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
            taxonomic_abundance: r.rel_abund,
            sequence_abundance: r.seq_abund,
        }
    }
}

/// Profile one sample against loaded genome sketches: the per-sample body of
/// `contain::contain`. Sorted by taxonomic abundance (profile) or adjusted
/// ANI (query), descending.
pub fn run_profile_compute(
    genomes: &[GenomeSketch],
    sample: &SequencesSketch,
    args: &ProfileArgs,
) -> Vec<OwnedAniResult> {
    if genomes.is_empty() {
        return Vec::new();
    }
    with_thread_pool(args, || run_profile_compute_inner(genomes, sample, args))
}

/// [`run_profile_compute`] for a two-stage database: screen, decode the
/// survivors' dense sketches (`compute_dense_survivors`), then the same kernels.
pub fn run_profile_compute_two_stage(
    db: &TwoStageDb,
    sample: &SequencesSketch,
    args: &ProfileArgs,
) -> Vec<OwnedAniResult> {
    if db.is_empty() {
        return Vec::new();
    }
    with_thread_pool(args, || {
        let contain_args = args.to_contain_args();
        let survivors = compute_dense_survivors(&contain_args, db, sample);
        if survivors.is_empty() {
            return Vec::new();
        }
        run_profile_compute_inner(&survivors, sample, args)
    })
}

/// Run `compute` on a private rayon pool of `num_threads`, or the global pool if 0.
fn with_thread_pool<R: Send>(args: &ProfileArgs, compute: impl FnOnce() -> R + Send) -> R {
    let owned_pool = if args.num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.num_threads)
            .build()
            .ok()
    } else {
        None
    };
    match owned_pool {
        Some(pool) => pool.install(compute),
        None => compute(),
    }
}

fn run_profile_compute_inner(
    genomes: &[GenomeSketch],
    sample: &SequencesSketch,
    args: &ProfileArgs,
) -> Vec<OwnedAniResult> {
    let contain_args = args.to_contain_args();
    let args = &contain_args;

    let kmer_id_opt = match args.seq_id {
        Some(pct) => Some((pct / 100.0).powf(sample.k as f64)),
        None => get_kmer_identity(sample, args.estimate_unknown),
    };

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
