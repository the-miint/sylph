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

use crate::sketch::SketchPairBuilder;
use crate::types::{GenomeSketch, SequencesSketch};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::BufReader;
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
            dedup_fpr: 0.0,
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
}
