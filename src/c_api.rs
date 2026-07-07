//! C/FFI surface for embedding sylph in non-Rust hosts (in particular,
//! the duckdb-miint DuckDB extension).
//!
//! The contract is documented in the hand-written `sylph.h` header at the
//! crate root. Memory ownership rules:
//!
//! - All `sylph_*_load` / `_create` / `_finalize` functions transfer ownership
//!   of the returned pointer to the caller. Free with the matching
//!   `sylph_*_free`.
//! - Strings returned by `sylph_get_last_error`, `sylph_version`, and
//!   `sylph_miint_fork_version` are static or thread-local and must NOT be
//!   freed.
//! - All entry points are panic-safe via `catch_unwind`. A panic surfaces as
//!   a NULL/0/-1 return + a message in `sylph_get_last_error`.

#![allow(clippy::missing_safety_doc)]

use crate::builders::{GenomeSketchBuilder, SketchPairBuilder};
use crate::types::{GenomeSketch, SequencesSketch};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

// ============================================================================
// Thread-local error string
// ============================================================================

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

/// Set the thread-local error message. Called by every entry point's error
/// path.
fn set_error<S: Into<Vec<u8>>>(msg: S) {
    let bytes = msg.into();
    // Replace any interior NULs to keep CString::new from failing on user
    // data (e.g. file paths with embedded NULs, which shouldn't happen but
    // we don't want to panic on either).
    let cleaned: Vec<u8> = bytes.into_iter().map(|b| if b == 0 { b'?' } else { b }).collect();
    let cstr = CString::new(cleaned).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(cstr));
}

/// Format-and-set helper.
macro_rules! set_error_fmt {
    ($($arg:tt)*) => { set_error(format!($($arg)*)) };
}

/// Caller-facing accessor for the last error on the calling thread.
/// Returned pointer is valid until the next sylph_* call on this thread.
#[no_mangle]
pub extern "C" fn sylph_get_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Run a closure inside `catch_unwind`, mapping panics into a thread-local
/// error and returning the supplied default on panic.
fn guarded<R, F: FnOnce() -> R>(default: R, f: F) -> R {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                format!("sylph panic: {}", s)
            } else if let Some(s) = payload.downcast_ref::<String>() {
                format!("sylph panic: {}", s)
            } else {
                "sylph panic: <opaque payload>".to_string()
            };
            set_error(msg);
            default
        }
    }
}

// ============================================================================
// Versioning
// ============================================================================

const VERSION_C: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
const FORK_VERSION_C: &str = concat!("v", env!("CARGO_PKG_VERSION"), "-miint\0");

/// Upstream sylph version embedded in this build.
#[no_mangle]
pub extern "C" fn sylph_version() -> *const c_char {
    VERSION_C.as_ptr() as *const c_char
}

/// Fork tag for diagnostics / source_hash debugging.
#[no_mangle]
pub extern "C" fn sylph_miint_fork_version() -> *const c_char {
    FORK_VERSION_C.as_ptr() as *const c_char
}

// ============================================================================
// Database (opaque .syldb handle)
// ============================================================================

/// Loaded `.syldb`. Immutable, freely shareable across threads for `profile`.
pub struct SylphDatabase {
    pub(crate) genomes: Vec<GenomeSketch>,
}

/// Load a `.syldb` from disk.
///
/// Returns NULL on error; call `sylph_get_last_error()` for details.
/// The file format is bincode-serialized `Vec<GenomeSketch>` (sylph 0.9.0).
///
/// # Safety
/// `path` must be a valid pointer to a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn sylph_database_load(path: *const c_char) -> *mut SylphDatabase {
    guarded(ptr::null_mut(), || {
        if path.is_null() {
            set_error("sylph_database_load: path is NULL");
            return ptr::null_mut();
        }
        let path_str = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_error("sylph_database_load: path is not valid UTF-8");
                return ptr::null_mut();
            }
        };

        let file = match File::open(path_str) {
            Ok(f) => f,
            Err(e) => {
                set_error_fmt!("sylph_database_load: failed to open '{}': {}", path_str, e);
                return ptr::null_mut();
            }
        };
        let reader = BufReader::with_capacity(10_000_000, file);
        let genomes: Vec<GenomeSketch> = match bincode::deserialize_from(reader) {
            Ok(g) => g,
            Err(e) => {
                set_error_fmt!(
                    "sylph_database_load: '{}' is not a valid .syldb (bincode error: {}). \
                     Possibly an older incompatible sylph version.",
                    path_str,
                    e
                );
                return ptr::null_mut();
            }
        };
        if genomes.is_empty() {
            set_error_fmt!("sylph_database_load: '{}' contains zero genomes", path_str);
            return ptr::null_mut();
        }
        Box::into_raw(Box::new(SylphDatabase { genomes }))
    })
}

/// Free a database. Safe to call with NULL.
///
/// # Safety
/// `db` must be a pointer previously returned by `sylph_database_load` and
/// not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sylph_database_free(db: *mut SylphDatabase) {
    if db.is_null() {
        return;
    }
    let _ = guarded((), || {
        drop(Box::from_raw(db));
    });
}

/// Number of reference genome sketches in the database. Returns 0 if `db` is
/// NULL (without setting an error — NULL-tolerance lets callers loop over an
/// optional handle).
///
/// # Safety
/// `db` must be a valid (or NULL) pointer from `sylph_database_load`.
#[no_mangle]
pub unsafe extern "C" fn sylph_database_num_genomes(db: *const SylphDatabase) -> usize {
    if db.is_null() {
        return 0;
    }
    guarded(0, || (*db).genomes.len())
}

// ============================================================================
// Sample sketch (streaming builder + finalized handle)
// ============================================================================

/// Streaming sketch parameters. Mirrors fields the host needs to control;
/// the FFI accepts a `*const SylphSketchParams` so we can extend the layout
/// non-breakingly later (passing 0 for new fields = "use default").
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SylphSketchParams {
    /// k-mer size. Must match the syldb. 0 = use the syldb's k.
    pub k: u8,
    /// FracMinHash subsampling rate. Must be ≤ syldb's c. 0 = use syldb's c.
    pub c: u16,
    /// 1 = sylph paired-read deduplication. 0 = no dedup.
    pub dedup: u8,
    /// Cuckoo-filter false-positive rate for approximate dedup. 0 = exact
    /// dedup via FxHashSet (deterministic but higher memory at large scale).
    pub dedup_fpr: f64,
    /// Reserved for future expansion; pass 0.
    pub _reserved0: u64,
}

impl Default for SylphSketchParams {
    fn default() -> Self {
        SylphSketchParams {
            k: 0,
            c: 0,
            dedup: 1,
            // Match sylph CLI's `--fpr` default (DEFAULT_FPR = 0.0001) so a
            // C/C++ caller using sylph_sketch_params_default() gets the same
            // approximate-cuckoo-filter behavior as `sylph profile`. Earlier
            // value was 0.0 (exact FxHashSet dedup) — deterministic but
            // disagreed with the CLI default and confused FFI consumers.
            dedup_fpr: crate::constants::DEFAULT_FPR,
            _reserved0: 0,
        }
    }
}

impl SylphSketchParams {
    /// Resolve effective (k, c) given a database. If the caller passed 0,
    /// inherit from the database; otherwise validate compatibility.
    fn resolve_kc(&self, db_k: usize, db_c: usize) -> Result<(usize, usize), String> {
        let k = if self.k == 0 { db_k } else { self.k as usize };
        let c = if self.c == 0 { db_c } else { self.c as usize };
        if k != db_k {
            return Err(format!(
                "sketch k ({}) must match database k ({})",
                k, db_k
            ));
        }
        if c < db_c {
            return Err(format!(
                "sketch c ({}) must be >= database c ({})",
                c, db_c
            ));
        }
        Ok((k, c))
    }
}

/// Two-state sketch handle: under construction, or finalized.
pub struct SylphSketch {
    inner: SylphSketchInner,
}

enum SylphSketchInner {
    Building(SketchPairBuilder),
    Finalized(Box<SequencesSketch>),
}

/// Construct an empty paired-end sketch builder. Caller must subsequently
/// call `sylph_sketch_builder_add_pair` zero or more times, then
/// `sylph_sketch_builder_finalize` to obtain a usable sketch.
///
/// `params` is optional (NULL → defaults). When NULL or when `k`/`c` are 0,
/// the sketch params remain unset; they are validated against the database
/// at `sylph_profile()` time. Until then we use placeholder defaults.
///
/// # Safety
/// `params` is borrowed for the duration of the call. The returned sketch
/// owns its accumulator state and must be released with `sylph_sketch_free`.
#[no_mangle]
pub unsafe extern "C" fn sylph_sketch_builder_create(
    params: *const SylphSketchParams,
) -> *mut SylphSketch {
    guarded(ptr::null_mut(), || {
        let p = if params.is_null() {
            SylphSketchParams::default()
        } else {
            *params
        };
        // If k/c are 0 we don't yet know them. Use sylph defaults (k=31, c=200)
        // as placeholders; the caller can override post-hoc by reconstructing,
        // and profile-time validation catches mismatches.
        let k = if p.k == 0 { 31 } else { p.k as usize };
        let c = if p.c == 0 { 200 } else { p.c as usize };
        let no_dedup = p.dedup == 0;
        let builder = SketchPairBuilder::new(
            String::from("<ffi>"),
            None,
            c,
            k,
            no_dedup,
            p.dedup_fpr,
        );
        Box::into_raw(Box::new(SylphSketch {
            inner: SylphSketchInner::Building(builder),
        }))
    })
}

/// Add one read (or read pair) to a sketch builder.
///
/// Returns 0 on success, non-zero on error (wrong handle state, NULL builder,
/// length mismatch with NULL pointer, etc.). On error call
/// `sylph_get_last_error()` for details.
///
/// `r2 == NULL` and `r2_len == 0` → single-end. Mixing single-end and
/// paired-end calls within the same builder is not supported and is logged
/// then dropped (returns 0 for the offending call to keep streaming
/// pipelines tolerant — the data is just skipped, not corrupted).
///
/// # Safety
/// `builder` must be a non-finalized sketch from `sylph_sketch_builder_create`.
/// `r1`/`r2` must point to at least `r1_len`/`r2_len` bytes if non-NULL.
#[no_mangle]
pub unsafe extern "C" fn sylph_sketch_builder_add_pair(
    builder: *mut SylphSketch,
    r1: *const u8,
    r1_len: usize,
    r2: *const u8,
    r2_len: usize,
) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_sketch_builder_add_pair: builder is NULL");
            return -1;
        }
        if r1.is_null() || r1_len == 0 {
            set_error("sylph_sketch_builder_add_pair: r1 must be a non-empty byte buffer");
            return -1;
        }
        if r2.is_null() != (r2_len == 0) {
            set_error(
                "sylph_sketch_builder_add_pair: r2 NULL/length mismatch \
                 (NULL must imply length 0)",
            );
            return -1;
        }
        let r1_slice = std::slice::from_raw_parts(r1, r1_len);
        let r2_slice = if r2.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(r2, r2_len))
        };
        match &mut (*builder).inner {
            SylphSketchInner::Building(b) => {
                b.add_pair(r1_slice, r2_slice);
                0
            }
            SylphSketchInner::Finalized(_) => {
                set_error(
                    "sylph_sketch_builder_add_pair: builder is already finalized; \
                     create a new sketch",
                );
                -1
            }
        }
    })
}

/// Finalize the builder. After this call the sketch transitions to
/// "Finalized" state and is usable in `sylph_profile`. Subsequent
/// `add_pair` calls on the same handle will fail.
///
/// Returns 0 on success, non-zero on error.
///
/// # Safety
/// `builder` must be a non-finalized sketch from `sylph_sketch_builder_create`.
#[no_mangle]
pub unsafe extern "C" fn sylph_sketch_builder_finalize(builder: *mut SylphSketch) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_sketch_builder_finalize: builder is NULL");
            return -1;
        }
        // We need to consume the SketchPairBuilder out of the inner enum.
        // std::mem::replace lets us take ownership without breaking exclusive
        // access: the temporary Finalized(empty) is overwritten immediately.
        let placeholder = SylphSketchInner::Finalized(Box::new(SequencesSketch::default()));
        let old = std::mem::replace(&mut (*builder).inner, placeholder);
        match old {
            SylphSketchInner::Building(b) => {
                let sketch = b.finalize();
                (*builder).inner = SylphSketchInner::Finalized(Box::new(sketch));
                0
            }
            SylphSketchInner::Finalized(_) => {
                // Restore so the handle remains usable; report the redundant call.
                (*builder).inner = old;
                set_error("sylph_sketch_builder_finalize: builder is already finalized");
                -1
            }
        }
    })
}

/// Free a sketch. Safe to call with NULL.
///
/// # Safety
/// `sketch` must be a pointer from `sylph_sketch_builder_create` (in either
/// state) and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sylph_sketch_free(sketch: *mut SylphSketch) {
    if sketch.is_null() {
        return;
    }
    let _ = guarded((), || {
        drop(Box::from_raw(sketch));
    });
}

/// (Internal, used by Phase 2.4's sylph_profile.) Borrow the finalized
/// SequencesSketch out of a `*const SylphSketch`. Returns None if the handle
/// is NULL or still in the Building state.
///
/// # Safety
/// Caller must guarantee the returned reference does not outlive the
/// SylphSketch pointer.
pub(crate) unsafe fn sketch_borrow_finalized<'a>(
    sketch: *const SylphSketch,
) -> Option<&'a SequencesSketch> {
    if sketch.is_null() {
        return None;
    }
    match &(*sketch).inner {
        SylphSketchInner::Finalized(s) => Some(&**s),
        SylphSketchInner::Building(_) => None,
    }
}

// ============================================================================
// Index builder (build + write a .syldb)
//
// The mirror image of the sample-sketch builder above: instead of sketching
// reads into a SequencesSketch, it sketches reference genomes into
// GenomeSketches and bincode-serializes the Vec<GenomeSketch> to a `.syldb`
// (the same on-disk format sylph_database_load consumes). Lifecycle:
//
//   create → { begin_genome → add_contig* → end_genome }* → write → free
//
// One genome is under construction at a time (the host groups its reference
// table by a genome-key column and feeds each genome's contigs contiguously).
// ============================================================================

/// Reference-genome sketch parameters. Layout-stable; trailing fields can be
/// added non-breakingly (pass 0 for "use default").
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SylphGenomeSketchParams {
    /// k-mer size. 0 = default 31. Only 21 and 31 are supported.
    pub k: u8,
    /// FracMinHash subsampling rate. 0 = default 200.
    pub c: u16,
    /// Minimum k-mer spacing (thins densely-seeded regions). 0 = default 30.
    pub min_spacing: u32,
    /// 1 = track min-spacing-dropped k-mers for pseudotax/profiling (default).
    /// 0 = do not (query-only databases).
    pub pseudotax: u8,
    /// Reserved; pass 0.
    pub _reserved0: u64,
}

impl Default for SylphGenomeSketchParams {
    fn default() -> Self {
        // Mirrors the `sylph sketch` CLI defaults (cmdline.rs: k=31, c=200,
        // min-spacing=30, pseudotax on unless --disable-profiling).
        SylphGenomeSketchParams {
            k: 0,
            c: 0,
            min_spacing: 0,
            pseudotax: 1,
            _reserved0: 0,
        }
    }
}

impl SylphGenomeSketchParams {
    /// Resolve (k, c, min_spacing) applying defaults for 0 fields, and validate
    /// k. Returns Err(message) on an unsupported k.
    fn resolve(&self) -> Result<(usize, usize, usize), String> {
        let k = if self.k == 0 { 31 } else { self.k as usize };
        let c = if self.c == 0 { 200 } else { self.c as usize };
        let min_spacing = if self.min_spacing == 0 { 30 } else { self.min_spacing as usize };
        if k != 21 && k != 31 {
            return Err(format!("k must be 21 or 31 (got {})", k));
        }
        Ok((k, c, min_spacing))
    }
}

/// Accumulates GenomeSketches and serializes them to a `.syldb`. Holds at most
/// one in-progress genome builder at a time.
pub struct SylphIndexBuilder {
    c: usize,
    k: usize,
    min_spacing: usize,
    pseudotax: bool,
    genomes: Vec<GenomeSketch>,
    current: Option<GenomeSketchBuilder>,
}

/// Populate `out` with the default reference-genome sketch parameters. As with
/// the other `*_params_default` helpers, seed a `SylphGenomeSketchParams` via
/// this rather than zero-initializing — `pseudotax = 1` is a non-zero default
/// that a zero-init would silently drop. Returns 0 on success, non-zero on a
/// NULL `out`.
#[no_mangle]
pub unsafe extern "C" fn sylph_genome_sketch_params_default(out: *mut SylphGenomeSketchParams) -> i32 {
    if out.is_null() {
        return 1;
    }
    *out = SylphGenomeSketchParams::default();
    0
}

/// Create an index builder. `params` may be NULL (uses defaults). Returns NULL
/// on error (e.g. an unsupported k); call sylph_get_last_error() for details.
/// Free with sylph_index_builder_free.
///
/// # Safety
/// `params` is borrowed for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_create(
    params: *const SylphGenomeSketchParams,
) -> *mut SylphIndexBuilder {
    guarded(ptr::null_mut(), || {
        let p = if params.is_null() {
            SylphGenomeSketchParams::default()
        } else {
            *params
        };
        let (k, c, min_spacing) = match p.resolve() {
            Ok(v) => v,
            Err(msg) => {
                set_error_fmt!("sylph_index_builder_create: {}", msg);
                return ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(SylphIndexBuilder {
            c,
            k,
            min_spacing,
            pseudotax: p.pseudotax != 0,
            genomes: Vec::new(),
            current: None,
        }))
    })
}

/// Begin a new reference genome. `file_name` is the genome's identity in the
/// resulting `.syldb` (the genome-key column value); it may be NULL (→ empty
/// string). Errors if a previous genome is still open (call end_genome first).
/// Returns 0 on success, non-zero on error.
///
/// # Safety
/// `builder` must be a live SylphIndexBuilder. `file_name`, if non-NULL, must
/// be a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_begin_genome(
    builder: *mut SylphIndexBuilder,
    file_name: *const c_char,
) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_index_builder_begin_genome: builder is NULL");
            return -1;
        }
        let b = &mut *builder;
        if b.current.is_some() {
            set_error(
                "sylph_index_builder_begin_genome: a genome is already open; \
                 call sylph_index_builder_end_genome first",
            );
            return -1;
        }
        let name = if file_name.is_null() {
            String::new()
        } else {
            match CStr::from_ptr(file_name).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    set_error("sylph_index_builder_begin_genome: file_name is not valid UTF-8");
                    return -1;
                }
            }
        };
        b.current = Some(GenomeSketchBuilder::new(b.c, b.k, b.min_spacing, b.pseudotax, name));
        0
    })
}

/// Add one contig to the open genome. `contig_name` may be NULL (→ empty); only
/// the first contig's name is retained as `first_contig_name`. `seq` must be a
/// non-NULL pointer to `seq_len` bytes (an empty contig — seq_len 0 — is
/// permitted). Errors if no genome is open. Returns 0 on success, non-zero on
/// error.
///
/// # Safety
/// `builder` must be a live SylphIndexBuilder with an open genome. `seq` must
/// point to at least `seq_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_add_contig(
    builder: *mut SylphIndexBuilder,
    contig_name: *const c_char,
    seq: *const u8,
    seq_len: usize,
) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_index_builder_add_contig: builder is NULL");
            return -1;
        }
        if seq.is_null() {
            set_error("sylph_index_builder_add_contig: seq is NULL");
            return -1;
        }
        let b = &mut *builder;
        let contig = match &mut b.current {
            Some(g) => g,
            None => {
                set_error(
                    "sylph_index_builder_add_contig: no genome open; call \
                     sylph_index_builder_begin_genome first",
                );
                return -1;
            }
        };
        let name = if contig_name.is_null() {
            String::new()
        } else {
            match CStr::from_ptr(contig_name).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    set_error("sylph_index_builder_add_contig: contig_name is not valid UTF-8");
                    return -1;
                }
            }
        };
        let seq_slice = std::slice::from_raw_parts(seq, seq_len);
        contig.add_contig(&name, seq_slice);
        0
    })
}

/// Finalize the open genome into a GenomeSketch and append it to the database.
/// Errors if no genome is open. Returns 0 on success, non-zero on error.
///
/// # Safety
/// `builder` must be a live SylphIndexBuilder with an open genome.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_end_genome(builder: *mut SylphIndexBuilder) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_index_builder_end_genome: builder is NULL");
            return -1;
        }
        let b = &mut *builder;
        match b.current.take() {
            Some(g) => {
                b.genomes.push(g.finalize());
                0
            }
            None => {
                set_error("sylph_index_builder_end_genome: no genome open");
                -1
            }
        }
    })
}

/// Number of completed (finalized) genomes. Returns 0 if `builder` is NULL. An
/// in-progress genome is not counted until end_genome.
///
/// # Safety
/// `builder` must be a valid (or NULL) SylphIndexBuilder pointer.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_num_genomes(builder: *const SylphIndexBuilder) -> usize {
    if builder.is_null() {
        return 0;
    }
    guarded(0, || (*builder).genomes.len())
}

/// Serialize the accumulated genomes to `path` as a `.syldb` (bincode
/// `Vec<GenomeSketch>`, the format sylph_database_load reads). Errors if a
/// genome is still open, if no genomes were added, or on I/O failure. Returns 0
/// on success, non-zero on error.
///
/// # Safety
/// `builder` must be a live SylphIndexBuilder. `path` must be a NUL-terminated
/// C string.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_write(
    builder: *mut SylphIndexBuilder,
    path: *const c_char,
) -> i32 {
    guarded(-1, || {
        if builder.is_null() {
            set_error("sylph_index_builder_write: builder is NULL");
            return -1;
        }
        if path.is_null() {
            set_error("sylph_index_builder_write: path is NULL");
            return -1;
        }
        let b = &*builder;
        if b.current.is_some() {
            set_error("sylph_index_builder_write: a genome is still open; call end_genome first");
            return -1;
        }
        if b.genomes.is_empty() {
            set_error("sylph_index_builder_write: no genomes added");
            return -1;
        }
        let path_str = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_error("sylph_index_builder_write: path is not valid UTF-8");
                return -1;
            }
        };
        let file = match File::create(path_str) {
            Ok(f) => f,
            Err(e) => {
                set_error_fmt!("sylph_index_builder_write: failed to create '{}': {}", path_str, e);
                return -1;
            }
        };
        let mut writer = BufWriter::new(file);
        // The on-disk format is a bincode-serialized Vec<GenomeSketch> (sylph
        // 0.9.0) — matches sketch::sketch's writer and sylph_database_load.
        match bincode::serialize_into(&mut writer, &b.genomes) {
            Ok(()) => 0,
            Err(e) => {
                set_error_fmt!("sylph_index_builder_write: bincode error: {}", e);
                -1
            }
        }
    })
}

/// Free an index builder. Safe to call with NULL.
///
/// # Safety
/// `builder` must be a pointer from sylph_index_builder_create, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sylph_index_builder_free(builder: *mut SylphIndexBuilder) {
    if builder.is_null() {
        return;
    }
    let _ = guarded((), || {
        drop(Box::from_raw(builder));
    });
}

// ============================================================================
// Profile (Arrow C Data Interface output) — gated behind `arrow-ffi` feature
// ============================================================================

/// Profile parameters. Mirrors `profile_api::ProfileArgs` with C-compatible
/// types. Layout-stable; new fields can be added at the end (callers passing
/// the older struct shape get the default for new fields).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SylphProfileParams {
    /// Lambda estimator selection: 0=Ratio (default), 1=MME, 2=NB, 3=MLE.
    pub estimator: u8,
    /// 1 = pseudotax / profile mode (default). 0 = query mode (no abundances).
    pub pseudotax: u8,
    /// 1 = renormalize to fraction-of-reads-explained.
    pub estimate_unknown: u8,
    /// 1 = report read counts instead of percent in seq_abund.
    pub estimate_read_counts: u8,
    /// 1 = skip 5/95 confidence interval bootstrap.
    pub no_ci: u8,
    /// 1 = skip lambda-based ANI adjustment.
    pub no_adj: u8,
    /// 1 = use mean coverage (instead of median) when median is high.
    pub mean_coverage: u8,
    /// 1 = log winner-take-all reassignments.
    pub log_reassignments: u8,
    pub min_count_correct: f64,
    pub min_number_kmers: f64,
    /// Minimum adjusted ANI (percent, 0..100). Negative = use sylph default.
    pub minimum_ani: f64,
    /// Sequence identity override (percent, 0..100). Negative = auto.
    pub seq_id: f64,
    /// Dereplication ANI for redundant-genome filtering.
    pub redundant_ani: f64,
    /// Number of rayon threads. 0 = use the global pool.
    pub num_threads: u32,
    /// Reserved for future expansion; pass 0.
    pub _reserved0: u32,
    pub _reserved1: u64,
}

impl Default for SylphProfileParams {
    fn default() -> Self {
        SylphProfileParams {
            estimator: 0,
            pseudotax: 1,
            estimate_unknown: 0,
            estimate_read_counts: 0,
            no_ci: 0,
            no_adj: 0,
            mean_coverage: 0,
            log_reassignments: 0,
            min_count_correct: 3.0,
            min_number_kmers: 50.0,
            minimum_ani: -1.0,
            seq_id: -1.0,
            redundant_ani: crate::constants::DEREP_PROFILE_ANI,
            num_threads: 0,
            _reserved0: 0,
            _reserved1: 0,
        }
    }
}

#[cfg(feature = "arrow-ffi")]
impl SylphProfileParams {
    fn to_profile_args(&self) -> crate::profile_api::ProfileArgs {
        use crate::profile_api::{LambdaEstimator, ProfileArgs};
        let estimator = match self.estimator {
            1 => LambdaEstimator::Mme,
            2 => LambdaEstimator::Nb,
            3 => LambdaEstimator::Mle,
            _ => LambdaEstimator::Ratio,
        };
        ProfileArgs {
            estimator,
            min_count_correct: self.min_count_correct,
            min_number_kmers: self.min_number_kmers,
            minimum_ani: if self.minimum_ani < 0.0 {
                None
            } else {
                Some(self.minimum_ani)
            },
            pseudotax: self.pseudotax != 0,
            estimate_unknown: self.estimate_unknown != 0,
            estimate_read_counts: self.estimate_read_counts != 0,
            no_ci: self.no_ci != 0,
            no_adj: self.no_adj != 0,
            mean_coverage: self.mean_coverage != 0,
            seq_id: if self.seq_id < 0.0 { None } else { Some(self.seq_id) },
            redundant_ani: self.redundant_ani,
            log_reassignments: self.log_reassignments != 0,
            num_threads: self.num_threads as usize,
        }
    }
}

/// Populate `out` with sylph's default profile parameters. C/C++ callers
/// should always seed a `SylphProfileParams` struct via this function rather
/// than zero-initializing — sylph has several non-zero defaults (e.g.
/// `pseudotax = 1`, `redundant_ani = 99.0`, `seq_id = -1.0`) that a fresh
/// zero-init struct would otherwise silently miss.
///
/// Returns 0 on success, non-zero on a NULL `out`.
#[no_mangle]
pub unsafe extern "C" fn sylph_profile_params_default(out: *mut SylphProfileParams) -> i32 {
    if out.is_null() {
        return 1;
    }
    *out = SylphProfileParams::default();
    0
}

/// Populate `out` with sylph's default sketch parameters. Same rationale as
/// `sylph_profile_params_default` — non-zero defaults (`dedup = 1`,
/// `dedup_fpr = DEFAULT_FPR (0.0001)`) get silently lost on a zero-init.
///
/// Returns 0 on success, non-zero on a NULL `out`.
#[no_mangle]
pub unsafe extern "C" fn sylph_sketch_params_default(out: *mut SylphSketchParams) -> i32 {
    if out.is_null() {
        return 1;
    }
    *out = SylphSketchParams::default();
    0
}

/// Run the sylph profile compute pipeline against the given database and
/// sample sketch, exporting the result as an Arrow RecordBatch over the
/// Arrow C Data Interface (FFI).
///
/// On success, `out_array` and `out_schema` are populated with caller-owned
/// pointers that must be released via the Arrow C Data Interface release
/// callbacks (the standard FFI_ArrowArray::release / FFI_ArrowSchema::release
/// fields on the structs themselves).
///
/// Output schema (9 columns, in order):
///   0. genome_index (UInt32, nullable false)
///   1. genome_name (LargeUtf8)
///   2. contig_name (LargeUtf8)
///   3. sequence_abundance (Float64, null when query mode)
///   4. taxonomic_abundance (Float64, null when query mode)
///   5. adjusted_ani (Float64)
///   6. eff_cov (Float64)
///   7. naive_ani (Float64)
///   8. kmers_reassigned (UInt64, null when no winner-take-all pass)
///
/// Returns 0 on success, non-zero on error. Call sylph_get_last_error() for
/// details. On error, out_array / out_schema are not populated.
///
/// # Safety
/// `db` must be a valid SylphDatabase pointer (or NULL → error).
/// `sample` must be a *finalized* SylphSketch (or NULL → error).
/// `params` is borrowed for the duration of the call (NULL = defaults).
/// `out_array` / `out_schema` must be valid uninitialized pointers to
/// `arrow::ffi::FFI_ArrowArray` / `arrow::ffi::FFI_ArrowSchema` slots
/// (typically via `MaybeUninit::uninit().as_mut_ptr()` in the caller).
#[cfg(feature = "arrow-ffi")]
#[no_mangle]
pub unsafe extern "C" fn sylph_profile(
    db: *const SylphDatabase,
    sample: *const SylphSketch,
    params: *const SylphProfileParams,
    out_array: *mut arrow::ffi::FFI_ArrowArray,
    out_schema: *mut arrow::ffi::FFI_ArrowSchema,
) -> i32 {
    guarded(-1, || {
        if db.is_null() {
            set_error("sylph_profile: db is NULL");
            return -1;
        }
        if sample.is_null() {
            set_error("sylph_profile: sample is NULL");
            return -1;
        }
        if out_array.is_null() || out_schema.is_null() {
            set_error("sylph_profile: out_array / out_schema must be non-NULL");
            return -1;
        }
        let sample_ref = match sketch_borrow_finalized(sample) {
            Some(s) => s,
            None => {
                set_error(
                    "sylph_profile: sample sketch is not finalized; call \
                     sylph_sketch_builder_finalize first",
                );
                return -1;
            }
        };
        let pa = if params.is_null() {
            SylphProfileParams::default().to_profile_args()
        } else {
            (*params).to_profile_args()
        };
        let results = crate::profile_api::run_profile_compute(
            &(*db).genomes,
            sample_ref,
            &pa,
        );
        match owned_results_to_ffi(&results, out_array, out_schema) {
            Ok(()) => 0,
            Err(msg) => {
                set_error(format!("sylph_profile: {}", msg));
                -1
            }
        }
    })
}

#[cfg(feature = "arrow-ffi")]
fn owned_results_to_ffi(
    results: &[crate::profile_api::OwnedAniResult],
    out_array: *mut arrow::ffi::FFI_ArrowArray,
    out_schema: *mut arrow::ffi::FFI_ArrowSchema,
) -> Result<(), String> {
    use arrow::array::{
        Array, ArrayRef, Float64Array, LargeStringArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow::datatypes::{DataType, Field};
    use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
    use std::sync::Arc;

    let n = results.len();

    let genome_index =
        UInt32Array::from_iter_values((0..n).map(|i| i as u32));
    let genome_name = LargeStringArray::from_iter_values(results.iter().map(|r| r.genome_name.as_str()));
    let contig_name = LargeStringArray::from_iter_values(results.iter().map(|r| r.contig_name.as_str()));
    let sequence_abundance = Float64Array::from_iter(results.iter().map(|r| r.sequence_abundance));
    let taxonomic_abundance = Float64Array::from_iter(results.iter().map(|r| r.taxonomic_abundance));
    let adjusted_ani = Float64Array::from_iter_values(results.iter().map(|r| r.adjusted_ani));
    let eff_cov = Float64Array::from_iter_values(results.iter().map(|r| r.eff_cov));
    let naive_ani = Float64Array::from_iter_values(results.iter().map(|r| r.naive_ani));
    let kmers_reassigned = UInt64Array::from_iter(
        results
            .iter()
            .map(|r| r.kmers_lost.map(|x| x as u64)),
    );

    let columns: Vec<(Arc<Field>, ArrayRef)> = vec![
        (
            Arc::new(Field::new("genome_index", DataType::UInt32, false)),
            Arc::new(genome_index) as ArrayRef,
        ),
        (
            Arc::new(Field::new("genome_name", DataType::LargeUtf8, false)),
            Arc::new(genome_name) as ArrayRef,
        ),
        (
            Arc::new(Field::new("contig_name", DataType::LargeUtf8, false)),
            Arc::new(contig_name) as ArrayRef,
        ),
        (
            Arc::new(Field::new("sequence_abundance", DataType::Float64, true)),
            Arc::new(sequence_abundance) as ArrayRef,
        ),
        (
            Arc::new(Field::new("taxonomic_abundance", DataType::Float64, true)),
            Arc::new(taxonomic_abundance) as ArrayRef,
        ),
        (
            Arc::new(Field::new("adjusted_ani", DataType::Float64, false)),
            Arc::new(adjusted_ani) as ArrayRef,
        ),
        (
            Arc::new(Field::new("eff_cov", DataType::Float64, false)),
            Arc::new(eff_cov) as ArrayRef,
        ),
        (
            Arc::new(Field::new("naive_ani", DataType::Float64, false)),
            Arc::new(naive_ani) as ArrayRef,
        ),
        (
            Arc::new(Field::new("kmers_reassigned", DataType::UInt64, true)),
            Arc::new(kmers_reassigned) as ArrayRef,
        ),
    ];

    let struct_array = StructArray::from(columns);
    let array_data = struct_array.into_data();

    let ffi_array = FFI_ArrowArray::new(&array_data);
    let ffi_schema = FFI_ArrowSchema::try_from(array_data.data_type())
        .map_err(|e| format!("FFI_ArrowSchema export: {}", e))?;

    // Move into caller-provided slots. Caller's slot is treated as
    // uninitialized memory — any pre-existing contents are overwritten.
    unsafe {
        std::ptr::write(out_array, ffi_array);
        std::ptr::write(out_schema, ffi_schema);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn version_string_is_null_terminated() {
        unsafe {
            let p = sylph_version();
            assert!(!p.is_null());
            // Length includes NUL via concat!(.., "\0"); from_ptr scans up to NUL.
            let s = CStr::from_ptr(p).to_str().unwrap();
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn fork_version_string_is_null_terminated() {
        unsafe {
            let p = sylph_miint_fork_version();
            let s = CStr::from_ptr(p).to_str().unwrap();
            assert!(s.contains("miint"), "fork version should mention miint, got {}", s);
        }
    }

    #[test]
    fn null_path_returns_null_with_error() {
        unsafe {
            let p = sylph_database_load(ptr::null());
            assert!(p.is_null());
            let err = sylph_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_str().unwrap();
            assert!(msg.contains("NULL"), "error message should mention NULL, got {}", msg);
        }
    }

    #[test]
    fn nonexistent_path_returns_null_with_error() {
        unsafe {
            let path = CString::new("/nonexistent/path/to.syldb").unwrap();
            let p = sylph_database_load(path.as_ptr());
            assert!(p.is_null());
            let err = sylph_get_last_error();
            let msg = CStr::from_ptr(err).to_str().unwrap();
            assert!(
                msg.contains("failed to open"),
                "expected 'failed to open' in error, got {}",
                msg
            );
        }
    }

    #[test]
    fn database_free_on_null_is_safe() {
        unsafe {
            sylph_database_free(ptr::null_mut());
        }
    }

    #[test]
    fn num_genomes_on_null_returns_zero() {
        unsafe {
            assert_eq!(sylph_database_num_genomes(ptr::null()), 0);
        }
    }

    // ----- Sketch builder FFI -----

    #[test]
    fn sketch_builder_create_with_null_params_uses_defaults() {
        unsafe {
            let s = sylph_sketch_builder_create(ptr::null());
            assert!(!s.is_null());
            // Cannot finalize without adding any reads — but the call should
            // still succeed (empty sketch is valid).
            let rc = sylph_sketch_builder_finalize(s);
            assert_eq!(rc, 0);
            // Already-finalized: second finalize should fail.
            let rc2 = sylph_sketch_builder_finalize(s);
            assert_eq!(rc2, -1);
            sylph_sketch_free(s);
        }
    }

    #[test]
    fn sketch_builder_add_pair_round_trip() {
        unsafe {
            let params = SylphSketchParams::default();
            let s = sylph_sketch_builder_create(&params);
            assert!(!s.is_null());

            // 70bp synthetic read pair (DNA only — passes the seeding alphabet).
            let r1: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC";
            let r2: &[u8] = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATG";
            let rc = sylph_sketch_builder_add_pair(s, r1.as_ptr(), r1.len(), r2.as_ptr(), r2.len());
            assert_eq!(rc, 0);

            let rc = sylph_sketch_builder_finalize(s);
            assert_eq!(rc, 0);

            // Cannot add after finalize.
            let rc = sylph_sketch_builder_add_pair(s, r1.as_ptr(), r1.len(), ptr::null(), 0);
            assert_eq!(rc, -1);

            // Internal accessor reflects finalized state.
            let borrow = sketch_borrow_finalized(s);
            assert!(borrow.is_some());

            sylph_sketch_free(s);
        }
    }

    #[test]
    fn sketch_builder_add_pair_rejects_null_r1() {
        unsafe {
            let s = sylph_sketch_builder_create(ptr::null());
            let rc = sylph_sketch_builder_add_pair(s, ptr::null(), 0, ptr::null(), 0);
            assert_eq!(rc, -1);
            let err = CStr::from_ptr(sylph_get_last_error()).to_str().unwrap();
            assert!(err.contains("non-empty"));
            sylph_sketch_free(s);
        }
    }

    #[test]
    fn sketch_builder_add_pair_rejects_null_r2_with_nonzero_len() {
        unsafe {
            let s = sylph_sketch_builder_create(ptr::null());
            let r1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
            // r2 NULL but r2_len > 0 → bug in caller.
            let rc = sylph_sketch_builder_add_pair(s, r1.as_ptr(), r1.len(), ptr::null(), 50);
            assert_eq!(rc, -1);
            sylph_sketch_free(s);
        }
    }

    #[test]
    fn sketch_free_on_null_is_safe() {
        unsafe {
            sylph_sketch_free(ptr::null_mut());
        }
    }

    /// Phase 2.3 round-trip: feeding the FFI builder the same reads as
    /// `sketch_pair_sequences` produces an identical kmer_counts map. Gated
    /// behind `fastx` because the reference sketcher reads files.
    #[cfg(feature = "fastx")]
    #[test]
    fn ffi_builder_matches_path_sketcher() {
        use crate::sketch::sketch_pair_sequences;
        use needletail::parse_fastx_file;

        let r1_path = "test_files/k12_R1.fq";
        let r2_path = "test_files/k12_R2.fq";
        let k = 31usize;
        let c = 200usize;

        // Reference: path-based sketcher with exact dedup (dedup_fpr = 0.0).
        let path_sketch = sketch_pair_sequences(r1_path, r2_path, c, k, None, false, 0.0)
            .expect("path sketcher");

        // Slurp paired reads into Vec<Vec<u8>> for FFI feeding.
        fn slurp(p: &str) -> Vec<Vec<u8>> {
            let mut r = parse_fastx_file(p).expect("parse_fastx_file");
            let mut out = Vec::new();
            while let Some(rec) = r.next() {
                out.push(rec.expect("record").seq().into_owned());
            }
            out
        }
        let r1s = slurp(r1_path);
        let r2s = slurp(r2_path);
        assert_eq!(r1s.len(), r2s.len());

        unsafe {
            let params = SylphSketchParams {
                k: 31,
                c: 200,
                dedup: 1,
                dedup_fpr: 0.0, // match the path sketcher's exact-dedup mode
                _reserved0: 0,
            };
            let s = sylph_sketch_builder_create(&params);
            assert!(!s.is_null());
            for (r1, r2) in r1s.iter().zip(r2s.iter()) {
                let rc = sylph_sketch_builder_add_pair(
                    s,
                    r1.as_ptr(),
                    r1.len(),
                    r2.as_ptr(),
                    r2.len(),
                );
                assert_eq!(rc, 0, "add_pair failed");
            }
            let rc = sylph_sketch_builder_finalize(s);
            assert_eq!(rc, 0, "finalize failed");

            let ffi_sketch = sketch_borrow_finalized(s).expect("finalized borrow");
            assert_eq!(
                path_sketch.kmer_counts, ffi_sketch.kmer_counts,
                "FFI sketch kmer_counts must match the path sketcher"
            );
            assert_eq!(path_sketch.k, ffi_sketch.k);
            assert_eq!(path_sketch.c, ffi_sketch.c);

            sylph_sketch_free(s);
        }
    }

    // ----- Index builder FFI -----

    /// A synthetic ~2 kb DNA contig with enough entropy to survive FracMinHash
    /// subsampling at c=200. Built without rng (unavailable in this crate's
    /// test env) via a simple LCG over the 4 bases.
    fn synthetic_contig(len: usize) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push(bases[((state >> 33) & 0b11) as usize]);
        }
        out
    }

    #[test]
    fn index_builder_round_trip_writes_loadable_syldb() {
        unsafe {
            let b = sylph_index_builder_create(ptr::null());
            assert!(!b.is_null());
            assert_eq!(sylph_index_builder_num_genomes(b), 0);

            let name = CString::new("genome-A").unwrap();
            assert_eq!(sylph_index_builder_begin_genome(b, name.as_ptr()), 0);
            let contig = CString::new("contig-1").unwrap();
            let seq = synthetic_contig(3000);
            assert_eq!(
                sylph_index_builder_add_contig(b, contig.as_ptr(), seq.as_ptr(), seq.len()),
                0
            );
            assert_eq!(sylph_index_builder_end_genome(b), 0);
            assert_eq!(sylph_index_builder_num_genomes(b), 1);

            let path = std::env::temp_dir().join("sylph_ffi_index_round_trip.syldb");
            let path_c = CString::new(path.to_str().unwrap()).unwrap();
            assert_eq!(sylph_index_builder_write(b, path_c.as_ptr()), 0);
            sylph_index_builder_free(b);

            // The written file must load back through the read-side FFI.
            let db = sylph_database_load(path_c.as_ptr());
            assert!(!db.is_null(), "written .syldb must be loadable");
            assert_eq!(sylph_database_num_genomes(db), 1);
            sylph_database_free(db);
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn index_builder_rejects_bad_k() {
        unsafe {
            let params = SylphGenomeSketchParams {
                k: 30,
                ..SylphGenomeSketchParams::default()
            };
            let b = sylph_index_builder_create(&params);
            assert!(b.is_null());
            let err = CStr::from_ptr(sylph_get_last_error()).to_str().unwrap();
            assert!(err.contains("k must be"), "got: {}", err);
        }
    }

    #[test]
    fn index_builder_add_without_begin_errors() {
        unsafe {
            let b = sylph_index_builder_create(ptr::null());
            let contig = CString::new("c").unwrap();
            let seq = synthetic_contig(100);
            let rc = sylph_index_builder_add_contig(b, contig.as_ptr(), seq.as_ptr(), seq.len());
            assert_eq!(rc, -1);
            sylph_index_builder_free(b);
        }
    }

    #[test]
    fn index_builder_double_begin_errors() {
        unsafe {
            let b = sylph_index_builder_create(ptr::null());
            let name = CString::new("g").unwrap();
            assert_eq!(sylph_index_builder_begin_genome(b, name.as_ptr()), 0);
            assert_eq!(sylph_index_builder_begin_genome(b, name.as_ptr()), -1);
            sylph_index_builder_free(b);
        }
    }

    #[test]
    fn index_builder_write_empty_or_open_errors() {
        unsafe {
            // No genomes at all.
            let b = sylph_index_builder_create(ptr::null());
            let path = std::env::temp_dir().join("sylph_ffi_index_empty.syldb");
            let path_c = CString::new(path.to_str().unwrap()).unwrap();
            assert_eq!(sylph_index_builder_write(b, path_c.as_ptr()), -1);

            // Genome still open.
            let name = CString::new("g").unwrap();
            assert_eq!(sylph_index_builder_begin_genome(b, name.as_ptr()), 0);
            assert_eq!(sylph_index_builder_write(b, path_c.as_ptr()), -1);
            sylph_index_builder_free(b);
        }
    }

    #[test]
    fn index_builder_free_on_null_is_safe() {
        unsafe {
            sylph_index_builder_free(ptr::null_mut());
            assert_eq!(sylph_index_builder_num_genomes(ptr::null()), 0);
        }
    }
}
