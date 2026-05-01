#ifndef SYLPH_H
#define SYLPH_H

/* sylph FFI — duckdb-miint integration surface.
 *
 * Memory ownership:
 *   - All sylph_*_load / _create / _finalize transfer ownership of the
 *     returned pointer to the caller. Free with the matching sylph_*_free.
 *   - Strings returned by sylph_get_last_error / sylph_version /
 *     sylph_miint_fork_version are owned by sylph; do NOT free them.
 *   - sylph_get_last_error returns a thread-local string valid until the
 *     next sylph_* call on the same thread.
 *
 * Thread safety:
 *   - Loading / freeing a SylphDatabase: NOT thread-safe.
 *   - Querying num_genomes / running sylph_profile against a loaded
 *     database from multiple threads concurrently: SAFE (read-only).
 *   - SylphSketch builders are NOT thread-safe — one builder per thread.
 *
 * Errors:
 *   - Functions returning a pointer return NULL on error.
 *   - Functions returning int return non-zero on error (typically -1).
 *   - Functions returning size_t return 0 on error (or on NULL input,
 *     without setting an error).
 *   - Call sylph_get_last_error() for a human-readable message.
 *
 * Panic safety:
 *   - Every entry point wraps the Rust body in catch_unwind. A Rust panic
 *     surfaces as a NULL/0/-1 return + a "sylph panic: ..." error message.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Versioning
 * ============================================================================ */

/* Upstream sylph crate version baked into this library. */
const char *sylph_version(void);

/* Fork tag (e.g. "v0.9.0-miint"). Useful for source_hash debugging. */
const char *sylph_miint_fork_version(void);

/* ============================================================================
 * Errors
 * ============================================================================ */

/* Last error on the calling thread, or NULL if there isn't one. Returned
 * pointer is valid until the next sylph_* call on this thread. */
const char *sylph_get_last_error(void);

/* ============================================================================
 * Database (.syldb)
 * ============================================================================ */

/* Opaque handle to a loaded sylph database. */
typedef struct SylphDatabase SylphDatabase;

/* Load a .syldb (bincode-serialized Vec<GenomeSketch>) from disk.
 * Returns NULL on error; call sylph_get_last_error() for details.
 *
 * The on-disk format is sylph 0.9.0 .syldb. Loading an older or future
 * format returns NULL with a "not a valid .syldb" error. */
SylphDatabase *sylph_database_load(const char *path);

/* Free a database. Safe to call with NULL. Must NOT be called while any
 * sylph_profile() invocations against this database are still running. */
void sylph_database_free(SylphDatabase *db);

/* Number of reference genomes. Returns 0 if db is NULL (without setting an
 * error — NULL-tolerant for ergonomic reasons). */
size_t sylph_database_num_genomes(const SylphDatabase *db);

/* (Phase 2.2+) sketch builders, one-shot Arrow sketcher, and sylph_profile
 * will be added here as those phases land. */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SYLPH_H */
