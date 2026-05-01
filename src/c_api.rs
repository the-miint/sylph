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

use crate::types::GenomeSketch;
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
}
