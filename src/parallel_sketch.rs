//! Single-producer/multi-consumer read sketching.
//!
//! Today's `sketch_sequences_needle`/`sketch_pair_sequences` (`sketch.rs`) do
//! gzip inflate, FASTQ parsing, k-mer extraction, PCR-duplicate detection, and
//! `kmer_counts` accumulation on one thread, start to finish. That leaves the
//! bulk of `-t/--threads`' budget idle whenever there aren't many files to
//! parallelize across -- the common case for one or a few large samples.
//!
//! This module parallelizes *within* one file/pair:
//!   * **one producer thread** owns the needletail reader(s) (the inherent
//!     serial bottleneck -- gzip inflate can't itself be parallelized), copies
//!     each record's sequence into an owned buffer, and sends batches of
//!     records through a bounded channel;
//!   * **N consumer threads** pull batches and run the CPU-bound work
//!     (`extract_markers`, already AVX2-accelerated and pure/stateless per
//!     call) in parallel.
//!
//! PCR-duplicate detection is the one genuinely shared, order-sensitive piece
//! (it's keyed on `(kmer_hash, pair_signature)`, where `pair_signature` is
//! derived from a read/pair's own sequence -- true duplicates always produce
//! the same signature). To parallelize it without a single global lock (which
//! would serialize everything) or per-consumer-private dedup state (which
//! would silently miss duplicates whose two occurrences land on different
//! consumers), dedup+count state is split into `NUM_SHARDS` independent,
//! individually-locked shards, and every k-mer from a given read/pair is
//! routed to the SAME shard via `hash(pair_signature) % NUM_SHARDS` --
//! guaranteeing true duplicates always collide in the same shard regardless of
//! which consumer processed which read, or how many shards exist. A whole
//! batch's results are grouped by shard locally before any lock is taken, so
//! each touched shard is locked at most once per batch, not once per read.
//!
//! `--no-dedup` mode needs none of this: with no correctness dependency on
//! shard placement, each consumer just accumulates into a fully thread-local
//! map, merged at the end with zero locking.
//!
//! Callers (`sketch.rs::sketch()`, `contain.rs::get_seq_sketch`) decide
//! *whether* to use this pipeline (based on thread budget and file size,
//! falling back to the legacy sequential functions otherwise) -- this module
//! only implements *how*, and always assumes it's worth doing.

use crate::constants::*;
use crate::sketch::{
    dup_removal_lsh_full, dup_removal_lsh_full_exact, extract_markers, pair_kmer,
    pair_kmer_single, Marker,
};
use crate::types::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use fxhash::{FxHashMap, FxHashSet, FxHasher};
use log::*;
use needletail::parser::FastxReader;
use needletail::parse_fastx_file;
use rand::{rngs::SmallRng, SeedableRng};
use scalable_cuckoo_filter::{ScalableCuckooFilter, ScalableCuckooFilterBuilder};
use std::collections::VecDeque;
use std::sync::Mutex;

type PairSig = ([Marker; 2], [Marker; 2]);
type SeBatch = Vec<Vec<u8>>;
type PeBatch = Vec<(Vec<u8>, Vec<u8>)>;

/// Tunable knobs for the pipeline, threaded down from CLI flags.
pub struct PipelineParams {
    pub batch_records: usize,
    pub channel_depth: usize,
    pub num_shards: Option<usize>,
}

impl Default for PipelineParams {
    fn default() -> Self {
        PipelineParams {
            batch_records: DEFAULT_SKETCH_BATCH_SIZE,
            channel_depth: DEFAULT_SKETCH_CHANNEL_DEPTH,
            num_shards: None,
        }
    }
}

fn default_num_shards(num_consumers: usize) -> usize {
    (num_consumers.max(1) * 8).next_power_of_two().clamp(16, 128)
}

#[inline]
fn shard_of_sig(sig: &PairSig, num_shards: usize) -> usize {
    (fxhash::hash64(sig) as usize) % num_shards
}

#[inline]
fn shard_of_kmer(km: u64, num_shards: usize) -> usize {
    // `km` is already a Murmur-style hash (see seeding.rs), well-distributed
    // on its own -- used only for the (rare) items with no pair_sig to route
    // by, purely for load balance across shards.
    (km as usize) % num_shards
}

// --- shard state --------------------------------------------------------

enum Dedup {
    Exact(FxHashSet<(u64, [Marker; 2])>),
    // Boxed: keeps this arm from bloating the enum when Exact (single-end,
    // and paired-end at --fpr 0) is the common case.
    Approx(Box<ScalableCuckooFilter<(u64, [Marker; 2]), FxHasher, SmallRng>>),
}

struct ShardState {
    dedup: Dedup,
    kmer_counts: FxHashMap<Kmer, u32>,
    num_dup_removed: usize,
}

struct Shard(Mutex<ShardState>);

fn build_shards_exact(num_shards: usize) -> Vec<Shard> {
    (0..num_shards)
        .map(|_| {
            Shard(Mutex::new(ShardState {
                dedup: Dedup::Exact(FxHashSet::default()),
                kmer_counts: FxHashMap::default(),
                num_dup_removed: 0,
            }))
        })
        .collect()
}

fn build_shards_approx(num_shards: usize, dedup_fpr: f64) -> Vec<Shard> {
    let fpr = if dedup_fpr != 0. { dedup_fpr } else { 0.001 };
    // Today's single-filter code reserves a flat 10M initial capacity per
    // file; split that budget across shards so total reserved memory stays
    // comparable (the filter is scalable/grows past this regardless).
    let cap = (10_000_000usize / num_shards.max(1)).max(1000);
    (0..num_shards)
        .map(|_| {
            let filter = ScalableCuckooFilterBuilder::new()
                .initial_capacity(cap)
                .false_positive_probability(fpr)
                .hasher(FxHasher::default())
                .rng(SmallRng::from_entropy())
                .finish();
            Shard(Mutex::new(ShardState {
                dedup: Dedup::Approx(Box::new(filter)),
                kmer_counts: FxHashMap::default(),
                num_dup_removed: 0,
            }))
        })
        .collect()
}

/// Thin adapter over the existing (unchanged) `dup_removal_lsh_full[_exact]`
/// bodies, re-pointed at one shard's state instead of a single-threaded
/// local -- this is what keeps the shard-based dedup logic identical to the
/// sequential path rather than a parallel reimplementation that could drift.
fn dedup_and_count(
    state: &mut ShardState,
    km: u64,
    pair_sig: Option<PairSig>,
    no_dedup: bool,
    threshold: Option<u32>,
) {
    let ShardState {
        dedup,
        kmer_counts,
        num_dup_removed,
    } = state;
    match dedup {
        Dedup::Exact(set) => {
            dup_removal_lsh_full_exact(
                kmer_counts,
                set,
                &km,
                pair_sig,
                num_dup_removed,
                no_dedup,
                threshold,
            );
        }
        Dedup::Approx(filter) => {
            dup_removal_lsh_full(kmer_counts, filter, &km, pair_sig, num_dup_removed, no_dedup);
        }
    }
}

fn merge_shards(shards: Vec<Shard>) -> (FxHashMap<Kmer, u32>, usize) {
    let mut kmer_counts = FxHashMap::default();
    let mut num_dup_removed = 0usize;
    for shard in shards {
        let state = shard.0.into_inner().unwrap();
        for (km, v) in state.kmer_counts {
            *kmer_counts.entry(km).or_insert(0) += v;
        }
        num_dup_removed += state.num_dup_removed;
    }
    (kmer_counts, num_dup_removed)
}

// --- producers -----------------------------------------------------------

fn producer_se(mut reader: Box<dyn FastxReader>, tx: Sender<SeBatch>, batch_records: usize) {
    let mut batch: SeBatch = Vec::with_capacity(batch_records);
    while let Some(record) = reader.next() {
        match record {
            Ok(rec) => {
                batch.push(rec.seq().into_owned());
                if batch.len() >= batch_records {
                    let full = std::mem::replace(&mut batch, Vec::with_capacity(batch_records));
                    if tx.send(full).is_err() {
                        return; // consumers gone
                    }
                }
            }
            Err(_) => {
                warn!("Encountered an invalid record while sketching; skipping.");
            }
        }
    }
    if !batch.is_empty() {
        let _ = tx.send(batch);
    }
}

/// Returns `false` iff mate-1 had a parse error -- matches
/// `sketch_pair_sequences`'s existing behaviour of aborting the WHOLE sketch
/// (discarding everything accumulated so far) on a mate-1 error. A mate-2
/// error instead silently drops just that one pair (also matching today's
/// behaviour). Pairing stops as soon as EITHER reader is exhausted; today's
/// sequential code instead spins, silently re-`.next()`-ing the
/// still-live reader without processing anything once its mate has run dry,
/// until it too exhausts -- output-equivalent (no k-mers are ever processed
/// from those trailing reads either way) but wasted work, so stopping
/// immediately here is a free efficiency win, not a behaviour change.
fn producer_pe(
    mut r1: Box<dyn FastxReader>,
    mut r2: Box<dyn FastxReader>,
    tx: Sender<PeBatch>,
    batch_records: usize,
) -> bool {
    let mut batch: PeBatch = Vec::with_capacity(batch_records);
    loop {
        let (n1, n2) = (r1.next(), r2.next());
        let (rec1_o, rec2_o) = match (n1, n2) {
            (Some(a), Some(b)) => (a, b),
            _ => break,
        };
        let rec1 = match rec1_o {
            Ok(r) => r,
            Err(_) => return false,
        };
        let rec2 = match rec2_o {
            Ok(r) => r,
            Err(_) => continue,
        };
        batch.push((rec1.seq().into_owned(), rec2.seq().into_owned()));
        if batch.len() >= batch_records {
            let full = std::mem::replace(&mut batch, Vec::with_capacity(batch_records));
            if tx.send(full).is_err() {
                return true;
            }
        }
    }
    if !batch.is_empty() {
        let _ = tx.send(batch);
    }
    true
}

/// One mate's outcome for a single attempted record, as produced by
/// `producer_mate` -- kept distinct per mate (rather than collapsed to
/// `Option<Vec<u8>>`) so `zip_pe`'s abort-vs-skip decision, which depends on
/// *which* mate errored, is made correctly by the zipper.
enum MateItem {
    Ok(Vec<u8>),
    ParseErr,
}

/// Reads one mate file to completion, batching `Ok(seq)`/`ParseErr` outcomes
/// (in record order) to `tx`. Used in pairs (one thread per mate) so the two
/// mates' gzip inflate + FASTQ parsing -- otherwise the paired producer's
/// entire cost, confirmed empirically to dominate wall time for compressed
/// input -- run concurrently on separate cores instead of serialized on one.
fn producer_mate(mut reader: Box<dyn FastxReader>, tx: Sender<Vec<MateItem>>, batch_records: usize) {
    let mut batch: Vec<MateItem> = Vec::with_capacity(batch_records);
    while let Some(record) = reader.next() {
        batch.push(match record {
            Ok(rec) => MateItem::Ok(rec.seq().into_owned()),
            Err(_) => MateItem::ParseErr,
        });
        if batch.len() >= batch_records {
            let full = std::mem::replace(&mut batch, Vec::with_capacity(batch_records));
            if tx.send(full).is_err() {
                return;
            }
        }
    }
    if !batch.is_empty() {
        let _ = tx.send(batch);
    }
}

/// Merges matched per-mate batches from two `producer_mate` threads into the
/// same `PeBatch` stream `producer_pe` would have produced directly --
/// preserving its exact semantics (mate-1 error aborts everything, mate-2
/// error drops just that pair, stop as soon as either mate is exhausted).
/// Lightweight (just reshuffles already-copied `Vec<u8>`s, no I/O or parsing
/// of its own), so it costs a thread but not meaningful CPU time.
fn zip_pe(rx1: Receiver<Vec<MateItem>>, rx2: Receiver<Vec<MateItem>>, tx: Sender<PeBatch>) -> bool {
    loop {
        let (b1, b2) = match (rx1.recv(), rx2.recv()) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return true, // either mate's reader thread is done
        };
        let uneven = b1.len() != b2.len(); // one mate ran dry mid-batch
        let mut out: PeBatch = Vec::with_capacity(b1.len().min(b2.len()));
        let mut aborted = false;
        for pair in b1.into_iter().zip(b2.into_iter()) {
            match pair {
                (MateItem::ParseErr, _) => {
                    aborted = true;
                    break;
                }
                (MateItem::Ok(_), MateItem::ParseErr) => continue,
                (MateItem::Ok(s1), MateItem::Ok(s2)) => out.push((s1, s2)),
            }
        }
        if !out.is_empty() && tx.send(out).is_err() {
            return true;
        }
        if aborted {
            return false;
        }
        if uneven {
            return true;
        }
    }
}

/// Runs the paired-end intake stage, choosing between one thread doing
/// lockstep single-threaded reading of both mates (`producer_pe`, cheap on
/// threads, used when the budget is too tight to spare more) or two
/// independent per-mate reader threads plus a lightweight zipper
/// (`producer_mate` x2 + `zip_pe`, needs 2 extra threads but lets both
/// mates' gzip inflate + parsing run concurrently instead of serialized).
/// Returns `false` iff mate-1 had a parse error (matches `producer_pe`).
fn run_pe_intake(
    reader1: Box<dyn FastxReader>,
    reader2: Box<dyn FastxReader>,
    tx: Sender<PeBatch>,
    batch_records: usize,
    channel_depth: usize,
    use_dual_readers: bool,
) -> bool {
    if !use_dual_readers {
        return producer_pe(reader1, reader2, tx, batch_records);
    }
    let (tx1, rx1) = bounded::<Vec<MateItem>>(channel_depth);
    let (tx2, rx2) = bounded::<Vec<MateItem>>(channel_depth);
    std::thread::scope(|scope| {
        scope.spawn(move || producer_mate(reader1, tx1, batch_records));
        scope.spawn(move || producer_mate(reader2, tx2, batch_records));
        zip_pe(rx1, rx2, tx)
    })
}

// --- consumers -------------------------------------------------------------

fn consumer_se(
    rx: Receiver<SeBatch>,
    shards: &[Shard],
    c: usize,
    k: usize,
    no_dedup: bool,
) -> (u64, u64) {
    let mut scratch: Vec<u64> = Vec::new();
    let mut outbox: Vec<Vec<(u64, Option<PairSig>)>> = vec![Vec::new(); shards.len()];
    let (mut sum_len, mut count) = (0u64, 0u64);
    for batch in rx.iter() {
        for seq in &batch {
            count += 1;
            sum_len += seq.len() as u64;
            let pair_sig = if seq.len() <= 400 {
                pair_kmer_single(seq)
            } else {
                None
            };
            scratch.clear();
            extract_markers(seq, &mut scratch, c, k);
            for &km in scratch.iter() {
                let idx = match pair_sig {
                    Some(sig) => shard_of_sig(&sig, shards.len()),
                    None => shard_of_kmer(km, shards.len()),
                };
                outbox[idx].push((km, pair_sig));
            }
        }
        for (idx, items) in outbox.iter_mut().enumerate() {
            if items.is_empty() {
                continue;
            }
            let mut state = shards[idx].0.lock().unwrap();
            for (km, sig) in items.drain(..) {
                dedup_and_count(&mut state, km, sig, no_dedup, Some(MAX_DEDUP_COUNT));
            }
        }
    }
    (sum_len, count)
}

fn consumer_se_no_dedup(rx: Receiver<SeBatch>, c: usize, k: usize) -> (FxHashMap<Kmer, u32>, u64, u64) {
    let mut scratch: Vec<u64> = Vec::new();
    let mut counts: FxHashMap<Kmer, u32> = FxHashMap::default();
    let (mut sum_len, mut count) = (0u64, 0u64);
    for batch in rx.iter() {
        for seq in &batch {
            count += 1;
            sum_len += seq.len() as u64;
            scratch.clear();
            extract_markers(seq, &mut scratch, c, k);
            for &km in scratch.iter() {
                *counts.entry(km).or_insert(0) += 1;
            }
        }
    }
    (counts, sum_len, count)
}

fn consumer_pe(
    rx: Receiver<PeBatch>,
    shards: &[Shard],
    c: usize,
    k: usize,
    no_dedup: bool,
) -> (u64, u64) {
    let mut scratch1: Vec<u64> = Vec::new();
    let mut scratch2: Vec<u64> = Vec::new();
    let mut outbox: Vec<Vec<(u64, Option<PairSig>)>> = vec![Vec::new(); shards.len()];
    let (mut sum_len, mut count) = (0u64, 0u64);
    for batch in rx.iter() {
        for (seq1, seq2) in &batch {
            count += 1;
            sum_len += seq1.len() as u64; // matches sketch_pair_sequences: mate-1 length only
            let pair_sig = pair_kmer(seq1, seq2);
            scratch1.clear();
            scratch2.clear();
            extract_markers(seq1, &mut scratch1, c, k);
            extract_markers(seq2, &mut scratch2, c, k);
            for &km in scratch1.iter() {
                let idx = match pair_sig {
                    Some(sig) => shard_of_sig(&sig, shards.len()),
                    None => shard_of_kmer(km, shards.len()),
                };
                outbox[idx].push((km, pair_sig));
            }
            for &km in scratch2.iter() {
                if scratch1.contains(&km) {
                    continue; // cross-mate suppression, matches sketch.rs
                }
                let idx = match pair_sig {
                    Some(sig) => shard_of_sig(&sig, shards.len()),
                    None => shard_of_kmer(km, shards.len()),
                };
                outbox[idx].push((km, pair_sig));
            }
        }
        for (idx, items) in outbox.iter_mut().enumerate() {
            if items.is_empty() {
                continue;
            }
            let mut state = shards[idx].0.lock().unwrap();
            for (km, sig) in items.drain(..) {
                dedup_and_count(&mut state, km, sig, no_dedup, None);
            }
        }
    }
    (sum_len, count)
}

fn consumer_pe_no_dedup(rx: Receiver<PeBatch>, c: usize, k: usize) -> (FxHashMap<Kmer, u32>, u64, u64) {
    let mut scratch1: Vec<u64> = Vec::new();
    let mut scratch2: Vec<u64> = Vec::new();
    let mut counts: FxHashMap<Kmer, u32> = FxHashMap::default();
    let (mut sum_len, mut count) = (0u64, 0u64);
    for batch in rx.iter() {
        for (seq1, seq2) in &batch {
            count += 1;
            sum_len += seq1.len() as u64;
            scratch1.clear();
            scratch2.clear();
            extract_markers(seq1, &mut scratch1, c, k);
            extract_markers(seq2, &mut scratch2, c, k);
            for &km in scratch1.iter() {
                *counts.entry(km).or_insert(0) += 1;
            }
            for &km in scratch2.iter() {
                if scratch1.contains(&km) {
                    continue;
                }
                *counts.entry(km).or_insert(0) += 1;
            }
        }
    }
    (counts, sum_len, count)
}

// --- entry points ------------------------------------------------------

pub fn sketch_sequences_needle_parallel(
    read_file: &str,
    c: usize,
    k: usize,
    sample_name: Option<String>,
    no_dedup: bool,
    threads: usize,
    params: &PipelineParams,
) -> Option<SequencesSketch> {
    let reader = match parse_fastx_file(&read_file) {
        Ok(r) => r,
        Err(_) => {
            warn!("{} is not a valid fasta/fastq file; skipping.", read_file);
            return None;
        }
    };
    let num_consumers = threads.saturating_sub(1).max(1);
    let num_shards = params
        .num_shards
        .unwrap_or_else(|| default_num_shards(num_consumers));
    let (tx, rx) = bounded::<SeBatch>(params.channel_depth * num_consumers);
    let batch_records = params.batch_records;

    let (kmer_counts, num_dup_removed, sum_len, count) = if no_dedup {
        std::thread::scope(|scope| {
            scope.spawn(move || producer_se(reader, tx, batch_records));
            let handles: Vec<_> = (0..num_consumers)
                .map(|_| {
                    let rx = rx.clone();
                    scope.spawn(move || consumer_se_no_dedup(rx, c, k))
                })
                .collect();
            drop(rx);
            let mut kmer_counts = FxHashMap::default();
            let (mut sum_len, mut count) = (0u64, 0u64);
            for h in handles {
                let (local, sl, ct) = h.join().unwrap();
                for (km, v) in local {
                    *kmer_counts.entry(km).or_insert(0) += v;
                }
                sum_len += sl;
                count += ct;
            }
            (kmer_counts, 0usize, sum_len, count)
        })
    } else {
        let shards = build_shards_exact(num_shards);
        let (sum_len, count) = std::thread::scope(|scope| {
            scope.spawn(move || producer_se(reader, tx, batch_records));
            let handles: Vec<_> = (0..num_consumers)
                .map(|_| {
                    let rx = rx.clone();
                    let shards_ref = &shards;
                    scope.spawn(move || consumer_se(rx, shards_ref, c, k, no_dedup))
                })
                .collect();
            drop(rx);
            let (mut sum_len, mut count) = (0u64, 0u64);
            for h in handles {
                let (sl, ct) = h.join().unwrap();
                sum_len += sl;
                count += ct;
            }
            (sum_len, count)
        });
        let (kmer_counts, num_dup_removed) = merge_shards(shards);
        (kmer_counts, num_dup_removed, sum_len, count)
    };

    let mean_read_length = if count > 0 {
        sum_len as f64 / count as f64
    } else {
        0.0
    };
    log_dedup_stats(read_file, &kmer_counts, num_dup_removed);

    Some(SequencesSketch {
        kmer_counts,
        file_name: read_file.to_string(),
        c,
        k,
        paired: false,
        sample_name,
        mean_read_length,
    })
}

pub fn sketch_pair_sequences_parallel(
    read_file1: &str,
    read_file2: &str,
    c: usize,
    k: usize,
    sample_name: Option<String>,
    no_dedup: bool,
    dedup_fpr: f64,
    threads: usize,
    params: &PipelineParams,
) -> Option<SequencesSketch> {
    let r1o = parse_fastx_file(&read_file1);
    let r2o = parse_fastx_file(&read_file2);
    if r1o.is_err() || r2o.is_err() {
        log::error!("Paired end reading failed for '{}' and '{}'. Make sure the files are present or the sequences are valid.", read_file1, read_file2);
        std::process::exit(1);
    }
    let reader1 = r1o.unwrap();
    let reader2 = r2o.unwrap();

    // Dual mate-reader threads need 2 extra threads over the single-lockstep
    // producer (2 readers + a zipper vs. 1 combined reader) -- only worth it
    // when there's budget to spare beyond that plus at least one consumer.
    let use_dual_readers = threads >= 4;
    let num_consumers = threads.saturating_sub(if use_dual_readers { 3 } else { 1 }).max(1);
    let num_shards = params
        .num_shards
        .unwrap_or_else(|| default_num_shards(num_consumers));
    let (tx, rx) = bounded::<PeBatch>(params.channel_depth * num_consumers);
    let batch_records = params.batch_records;
    let channel_depth = params.channel_depth;
    let exact = dedup_fpr == 0.;

    let mut aborted = false;
    let (kmer_counts, num_dup_removed, sum_len, count) = if no_dedup {
        std::thread::scope(|scope| {
            let prod = scope.spawn(move || {
                run_pe_intake(reader1, reader2, tx, batch_records, channel_depth, use_dual_readers)
            });
            let handles: Vec<_> = (0..num_consumers)
                .map(|_| {
                    let rx = rx.clone();
                    scope.spawn(move || consumer_pe_no_dedup(rx, c, k))
                })
                .collect();
            drop(rx);
            let mut kmer_counts = FxHashMap::default();
            let (mut sum_len, mut count) = (0u64, 0u64);
            for h in handles {
                let (local, sl, ct) = h.join().unwrap();
                for (km, v) in local {
                    *kmer_counts.entry(km).or_insert(0) += v;
                }
                sum_len += sl;
                count += ct;
            }
            if !prod.join().unwrap() {
                aborted = true;
            }
            (kmer_counts, 0usize, sum_len, count)
        })
    } else {
        let shards = if exact {
            build_shards_exact(num_shards)
        } else {
            build_shards_approx(num_shards, dedup_fpr)
        };
        let (sum_len, count) = std::thread::scope(|scope| {
            let prod = scope.spawn(move || {
                run_pe_intake(reader1, reader2, tx, batch_records, channel_depth, use_dual_readers)
            });
            let handles: Vec<_> = (0..num_consumers)
                .map(|_| {
                    let rx = rx.clone();
                    let shards_ref = &shards;
                    scope.spawn(move || consumer_pe(rx, shards_ref, c, k, no_dedup))
                })
                .collect();
            drop(rx);
            let (mut sum_len, mut count) = (0u64, 0u64);
            for h in handles {
                let (sl, ct) = h.join().unwrap();
                sum_len += sl;
                count += ct;
            }
            if !prod.join().unwrap() {
                aborted = true;
            }
            (sum_len, count)
        });
        let (kmer_counts, num_dup_removed) = merge_shards(shards);
        (kmer_counts, num_dup_removed, sum_len, count)
    };

    if aborted {
        return None;
    }

    let mean_read_length = if count > 0 {
        sum_len as f64 / count as f64
    } else {
        0.0
    };
    log_dedup_stats(read_file1, &kmer_counts, num_dup_removed);

    Some(SequencesSketch {
        kmer_counts,
        file_name: read_file1.to_string(),
        c,
        k,
        paired: true,
        sample_name,
        mean_read_length,
    })
}

fn log_dedup_stats(file_name: &str, kmer_counts: &FxHashMap<Kmer, u32>, num_dup_removed: usize) {
    let percent = (num_dup_removed as f64)
        / ((kmer_counts.values().sum::<u32>() as f64) + num_dup_removed as f64)
        * 100.;
    log::debug!(
        "Number of sketched k-mers removed due to read duplication for {}: {}. Percentage: {:.2}%",
        file_name,
        num_dup_removed,
        percent,
    );
}

// --- outer file-level scheduling -----------------------------------------

/// Runs `per_file(file_index, threads_for_this_file)` for every file in
/// `0..num_files`, using `files_in_flight = min(num_files, total_threads)`
/// concurrent slots pulling from a shared work queue (so a slot that
/// finishes a small file quickly immediately grabs the next one, rather than
/// files being statically pre-partitioned). Any threads left over after
/// giving each slot a base share are rationed evenly across the first few
/// slots. Slots are plain `std::thread::scope` threads, not rayon tasks --
/// each slot's `per_file` call itself spawns and blocks on its own dedicated
/// producer+consumer threads, and nesting raw thread spawns inside a rayon
/// closure would leave rayon unable to account for them.
pub fn run_file_slots<T: Send>(
    num_files: usize,
    total_threads: usize,
    per_file: impl Fn(usize, usize) -> T + Sync,
) -> Vec<T> {
    if num_files == 0 {
        return Vec::new();
    }
    let files_in_flight = num_files.min(total_threads.max(1));
    let threads_per_file = (total_threads / files_in_flight).max(1);
    let remainder = total_threads.saturating_sub(threads_per_file * files_in_flight);
    let queue: Mutex<VecDeque<usize>> = Mutex::new((0..num_files).collect());
    let results: Vec<Mutex<Option<T>>> = (0..num_files).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|scope| {
        for slot in 0..files_in_flight {
            let extra = if slot < remainder { 1 } else { 0 };
            let slot_threads = threads_per_file + extra;
            let queue = &queue;
            let results = &results;
            let per_file = &per_file;
            scope.spawn(move || loop {
                let idx = queue.lock().unwrap().pop_front();
                let idx = match idx {
                    Some(i) => i,
                    None => break,
                };
                let r = per_file(idx, slot_threads);
                *results[idx].lock().unwrap() = Some(r);
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap().unwrap())
        .collect()
}

/// Whether it's worth using the parallel pipeline for one file, given the
/// thread budget assigned to it and a cheap size hint (compressed size for
/// `.gz` inputs, since that's what's readily available via `fs::metadata`
/// without opening/decompressing the file).
pub fn should_use_pipeline(no_sketch_pipeline: bool, threads_for_file: usize, path: &str) -> bool {
    if no_sketch_pipeline || threads_for_file < 2 {
        return false;
    }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    size >= MIN_BYTES_FOR_SKETCH_PIPELINE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{sketch_pair_sequences, sketch_sequences_needle};
    use std::io::Write;

    /// Gzips `src` into a fresh temp file, returning its path. Used to test
    /// the parallel pipeline against compressed input without committing new
    /// binary fixtures to the repo.
    fn gzip_to_temp(src: &str, tag: &str) -> String {
        let data = std::fs::read(src).unwrap();
        let dst = std::env::temp_dir().join(format!(
            "sylph_test_{}_{}.fq.gz",
            tag,
            std::process::id()
        ));
        let f = std::fs::File::create(&dst).unwrap();
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        enc.write_all(&data).unwrap();
        enc.finish().unwrap();
        dst.to_str().unwrap().to_string()
    }

    const K12_R1: &str = "test_files/k12_R1.fq";
    const K12_R2: &str = "test_files/k12_R2.fq";
    const C: usize = 5;
    const K: usize = 21;

    fn small_params(num_shards: usize) -> PipelineParams {
        // Small batch size so a 2500-read fixture actually exercises multiple
        // batches (and thus the batch-grouped-locking path) even at low
        // thread counts.
        PipelineParams {
            batch_records: 64,
            channel_depth: 2,
            num_shards: Some(num_shards),
        }
    }

    #[test]
    fn parallel_matches_sequential_pe_no_dedup() {
        let seq =
            sketch_pair_sequences(K12_R1, K12_R2, C, K, None, /*no_dedup=*/ true, 0.).unwrap();
        for &threads in &[2usize, 3, 4, 8] {
            for &shards in &[1usize, 4, 16] {
                let par = sketch_pair_sequences_parallel(
                    K12_R1,
                    K12_R2,
                    C,
                    K,
                    None,
                    true,
                    0.,
                    threads,
                    &small_params(shards),
                )
                .unwrap();
                assert_eq!(
                    seq.kmer_counts, par.kmer_counts,
                    "no_dedup mismatch at threads={} shards={}",
                    threads, shards
                );
                assert!((seq.mean_read_length - par.mean_read_length).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn parallel_matches_sequential_se_no_dedup() {
        let seq =
            sketch_sequences_needle(K12_R1, C, K, None, /*no_dedup=*/ true).unwrap();
        for &threads in &[2usize, 3, 4, 8] {
            for &shards in &[1usize, 4, 16] {
                let par = sketch_sequences_needle_parallel(
                    K12_R1,
                    C,
                    K,
                    None,
                    true,
                    threads,
                    &small_params(shards),
                )
                .unwrap();
                assert_eq!(
                    seq.kmer_counts, par.kmer_counts,
                    "no_dedup mismatch at threads={} shards={}",
                    threads, shards
                );
            }
        }
    }

    /// Paired-end, exact dedup (`dedup_fpr = 0.`) -- the primary target use
    /// case. Correctness here rests entirely on routing every k-mer from a
    /// pair by the SAME pair-signature hash, so true duplicates (which always
    /// share a signature) always land in the same shard regardless of thread
    /// or shard count.
    #[test]
    fn parallel_matches_sequential_pe_exact_dedup() {
        let seq = sketch_pair_sequences(K12_R1, K12_R2, C, K, None, false, 0.).unwrap();
        assert!(
            !seq.kmer_counts.is_empty(),
            "sanity: fixture should yield k-mers"
        );
        for &threads in &[2usize, 3, 4, 8] {
            for &shards in &[1usize, 4, 16, 64] {
                let par = sketch_pair_sequences_parallel(
                    K12_R1,
                    K12_R2,
                    C,
                    K,
                    None,
                    false,
                    0.,
                    threads,
                    &small_params(shards),
                )
                .unwrap();
                assert_eq!(
                    seq.kmer_counts, par.kmer_counts,
                    "exact-dedup mismatch at threads={} shards={}",
                    threads, shards
                );
            }
        }
    }

    /// Same as above but against gzipped copies of the same fixture, exactly
    /// exercising the code path a real compressed sample takes (needletail's
    /// internal `MultiGzDecoder`) rather than only the plain-file path.
    #[test]
    fn parallel_matches_sequential_pe_exact_dedup_gzipped() {
        let gz1 = gzip_to_temp(K12_R1, "r1");
        let gz2 = gzip_to_temp(K12_R2, "r2");
        let seq = sketch_pair_sequences(&gz1, &gz2, C, K, None, false, 0.).unwrap();
        for &threads in &[2usize, 4, 8] {
            let par = sketch_pair_sequences_parallel(
                &gz1,
                &gz2,
                C,
                K,
                None,
                false,
                0.,
                threads,
                &small_params(16),
            )
            .unwrap();
            assert_eq!(
                seq.kmer_counts, par.kmer_counts,
                "gzipped exact-dedup mismatch at threads={}",
                threads
            );
        }
        let _ = std::fs::remove_file(gz1);
        let _ = std::fs::remove_file(gz2);
    }

    /// Single-end exact dedup on a low-coverage fixture: every true k-mer
    /// multiplicity here is well under `MAX_DEDUP_COUNT`, so the accepted
    /// per-shard-vs-global `MAX_DEDUP_COUNT` divergence (see module docs)
    /// never engages, and equality should hold exactly.
    #[test]
    fn parallel_matches_sequential_se_exact_dedup_low_coverage() {
        let seq = sketch_sequences_needle(K12_R1, C, K, None, false).unwrap();
        assert!(
            *seq.kmer_counts.values().max().unwrap_or(&0) < MAX_DEDUP_COUNT,
            "sanity: fixture must be low-coverage for this test to be meaningful"
        );
        for &threads in &[2usize, 4, 8] {
            for &shards in &[1usize, 8, 32] {
                let par = sketch_sequences_needle_parallel(
                    K12_R1,
                    C,
                    K,
                    None,
                    false,
                    threads,
                    &small_params(shards),
                )
                .unwrap();
                assert_eq!(
                    seq.kmer_counts, par.kmer_counts,
                    "low-coverage exact-dedup mismatch at threads={} shards={}",
                    threads, shards
                );
            }
        }
    }

    /// Approximate (cuckoo-filter) paired dedup is already RNG/insertion-
    /// order-dependent even sequentially, so this checks statistical
    /// closeness rather than bit-for-bit equality.
    #[test]
    fn parallel_close_to_sequential_pe_approx_dedup() {
        let seq = sketch_pair_sequences(K12_R1, K12_R2, C, K, None, false, DEFAULT_FPR).unwrap();
        let par = sketch_pair_sequences_parallel(
            K12_R1,
            K12_R2,
            C,
            K,
            None,
            false,
            DEFAULT_FPR,
            4,
            &small_params(16),
        )
        .unwrap();
        let sum_seq: u64 = seq.kmer_counts.values().map(|&v| v as u64).sum();
        let sum_par: u64 = par.kmer_counts.values().map(|&v| v as u64).sum();
        let diff = (sum_seq as i64 - sum_par as i64).unsigned_abs();
        assert!(
            (diff as f64) <= 0.01 * (sum_seq.max(sum_par) as f64),
            "approximate-dedup totals diverged too much: sequential={}, parallel={}",
            sum_seq,
            sum_par
        );
    }

    #[test]
    fn run_file_slots_covers_every_file_exactly_once() {
        let seen: Vec<Mutex<u32>> = (0..7).map(|_| Mutex::new(0)).collect();
        let results = run_file_slots(7, 3, |idx, threads| {
            *seen[idx].lock().unwrap() += 1;
            assert!(threads >= 1);
            idx * 2
        });
        for s in &seen {
            assert_eq!(*s.lock().unwrap(), 1, "each file must be processed exactly once");
        }
        assert_eq!(results, (0..7).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn should_use_pipeline_respects_thread_and_size_floor() {
        assert!(!should_use_pipeline(false, 1, K12_R1)); // threads < 2
        assert!(!should_use_pipeline(true, 8, K12_R1)); // explicitly disabled
        assert!(!should_use_pipeline(false, 8, K12_R1)); // fixture is tiny (< MIN_BYTES)
    }
}
