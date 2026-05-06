pub const EM_ABUND_CUTOFF: f64 = 0.01;
pub const PAIR_REGEX: &str = r"(.+)(_?1|_?2)(\..+)";
pub const CUTOFF_PVALUE:f64 = 0.9999999999;
pub const SAMPLE_SIZE_CUTOFF: usize = 25;
pub const MEDIAN_ANI_THRESHOLD: f64 = 2.;
pub const QUERY_FILE_SUFFIX: &str = ".syldb";
pub const SAMPLE_FILE_SUFFIX: &str = ".sylsp";
pub const QUERY_FILE_SUFFIX_VALID : [&str;2] = [QUERY_FILE_SUFFIX, ".sylqueries"];
pub const SAMPLE_FILE_SUFFIX_VALID : [&str;2] = [SAMPLE_FILE_SUFFIX, ".sylsample"];
pub const MIN_ANI_DEF: f64 = 0.9;
pub const MIN_ANI_P_DEF: f64 = 0.95;
pub const MAX_MEDIAN_FOR_MEAN_FINAL_EST: f64 = 15.;
// Redundancy threshold for `sylph profile`'s genome dereplication. Carried
// on the percentage scale (e.g. 99.0 ≈ 99% ANI) — `derep_if_reassign_threshold`
// in contain.rs divides by 100 internally. Must agree with the CLI default
// `default_value_t = 99.0` on `--redundancy-threshold` in cmdline.rs (clap 3
// won't let us reference a const there). Earlier value was 0.975, which
// yielded an effective threshold of ~0.975% inside the function — i.e. zero —
// and collapsed nearly every detected genome on the FFI path.
pub const DEREP_PROFILE_ANI: f64 = 99.0;
pub const MAX_DEDUP_COUNT: u32 = 4;
pub const MAX_DEDUP_LEN: usize = 10000000;
pub const DEFAULT_FPR: f64 = 0.0001;
pub const MED_KMER_FOR_ID_EST: f64 = 3.;
