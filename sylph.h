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
#include <stdint.h>

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

/* ============================================================================
 * Sample sketch (streaming builder)
 * ============================================================================ */

/* Opaque handle to a paired-end sketch (under construction or finalized). */
typedef struct SylphSketch SylphSketch;

/* Streaming sketch parameters. Layout-stable for now; trailing fields can
 * be added non-breakingly (callers that don't know about a new field pass 0
 * and get the default). */
typedef struct {
    /* k-mer size. Must match the syldb. 0 = use the syldb's k. */
    uint8_t  k;
    /* FracMinHash subsampling rate. Must be <= syldb's c. 0 = use syldb's c. */
    uint16_t c;
    /* 1 = sylph paired-read deduplication (default). 0 = disable. */
    uint8_t  dedup;
    /* Cuckoo-filter false-positive rate for approximate dedup. 0 = exact
     * dedup via FxHashSet (deterministic but more memory at large scale). */
    double   dedup_fpr;
    /* Reserved; pass 0. */
    uint64_t _reserved0;
} SylphSketchParams;

/* Create an empty paired-end sketch builder. params may be NULL (uses
 * defaults). Free with sylph_sketch_free. Returns NULL on error. */
SylphSketch *sylph_sketch_builder_create(const SylphSketchParams *params);

/* Add one read (or read pair) to the builder. r2 = NULL and r2_len = 0 for
 * single-end. Mixing single-end and paired-end calls within the same builder
 * is not supported (subsequent calls in the wrong mode are silently dropped).
 * Returns 0 on success, non-zero on error. */
int sylph_sketch_builder_add_pair(
    SylphSketch *builder,
    const unsigned char *r1, size_t r1_len,
    const unsigned char *r2, size_t r2_len);

/* Finalize the builder. After this call the sketch is usable in
 * sylph_profile() and further sylph_sketch_builder_add_pair calls fail.
 * Returns 0 on success, non-zero on error. */
int sylph_sketch_builder_finalize(SylphSketch *builder);

/* Free a sketch (in either Building or Finalized state). Safe with NULL. */
void sylph_sketch_free(SylphSketch *sketch);

/* ============================================================================
 * Index builder (build + write a .syldb)
 *
 * The mirror image of the sample sketch builder: sketches reference genomes
 * into a database and serializes it to a `.syldb` (the format
 * sylph_database_load reads). Lifecycle:
 *
 *   create -> add_contig(genome_id, order, ...)* -> end_genome(genome_id) ... -> merge* -> write -> free
 *
 * Contigs are addressed by genome_id, so the builder keeps one in-progress sketch
 * per open genome id and different genomes' contigs may be interleaved. Add a
 * genome's contigs then call end_genome to finalize + free it. `order` fixes only
 * first_contig_name (the lowest-order contig), so results match `sylph sketch`.
 * merge combines per-thread builders before a single write.
 * NOT thread-safe — one builder per thread.
 * ============================================================================ */

/* Opaque handle to a database under construction. */
typedef struct SylphIndexBuilder SylphIndexBuilder;

/* Reference-genome sketch parameters. Layout-stable; trailing fields can be
 * added non-breakingly (pass 0 for "use default"). */
typedef struct {
    /* k-mer size. 0 = default 31. Only 21 and 31 are supported. */
    uint8_t  k;
    /* FracMinHash subsampling rate. 0 = default 200. */
    uint16_t c;
    /* Minimum k-mer spacing. 0 = default 30. */
    uint32_t min_spacing;
    /* 1 = track min-spacing-dropped k-mers for pseudotax/profiling (default).
     * 0 = do not (query-only databases). */
    uint8_t  pseudotax;
    /* Reserved; pass 0. */
    uint64_t _reserved0;
} SylphGenomeSketchParams;

/* Populate `out` with sylph's default reference-genome sketch parameters. Seed
 * a SylphGenomeSketchParams via this rather than zero-initializing — pseudotax
 * = 1 is a non-zero default. Returns 0 on success, non-zero if `out` is NULL. */
int sylph_genome_sketch_params_default(SylphGenomeSketchParams *out);

/* Create an index builder. params may be NULL (defaults). Returns NULL on
 * error (e.g. unsupported k); call sylph_get_last_error(). Free with
 * sylph_index_builder_free. */
SylphIndexBuilder *sylph_index_builder_create(const SylphGenomeSketchParams *params);

/* Add one contig to the genome identified by genome_id (created on first sight).
 * genome_id must be non-NULL — it is the genome's identity in the `.syldb`.
 * `order` orders contigs within the genome; the lowest-order contig supplies
 * first_contig_name (pass the source contig index / sequence_index to match
 * `sylph sketch`). contig_name may be NULL (-> empty). seq must be non-NULL
 * (seq_len 0 permitted). Contigs of different genomes may be interleaved.
 * Returns 0 on success, non-zero on error. */
int sylph_index_builder_add_contig(SylphIndexBuilder *builder, const char *genome_id, int64_t order,
                                   const char *contig_name, const unsigned char *seq, size_t seq_len);

/* Finalize the single in-progress genome `genome_id`, freeing its marker buffer
 * immediately. A host streaming a genome-id-clustered source calls this the
 * moment a genome's rows end, so peak memory stays ~the finalized sketches.
 * Errors if no open genome has that id. Returns 0 on success, non-zero on error. */
int sylph_index_builder_end_genome(SylphIndexBuilder *builder, const char *genome_id);

/* Number of completed genomes (in-progress genomes are not counted until
 * end_genome). Returns 0 if builder is NULL. */
size_t sylph_index_builder_num_genomes(const SylphIndexBuilder *builder);

/* Move all completed genomes from `src` into `dst` (src ends empty). For merging
 * per-thread builders before a single write. Both must have no open genome (call
 * end_genome for each first). Returns 0 on success, non-zero on error. */
int sylph_index_builder_merge(SylphIndexBuilder *dst, SylphIndexBuilder *src);

/* Serialize the accumulated genomes to `path` as a `.syldb`. Errors if any
 * genome is still open (call end_genome), if no genomes were added, or on I/O
 * failure. Returns 0 on success, non-zero on error. */
int sylph_index_builder_write(SylphIndexBuilder *builder, const char *path);

/* Free an index builder. Safe to call with NULL. */
void sylph_index_builder_free(SylphIndexBuilder *builder);

/* ============================================================================
 * Profile (Arrow C Data Interface output)
 *
 * sylph_profile takes a loaded database + finalized sample sketch + params
 * and returns a 9-column Arrow RecordBatch over the Arrow C Data Interface.
 * The caller provides storage for FFI_ArrowArray and FFI_ArrowSchema slots
 * (typically uninitialized stack/heap memory matching the Arrow C ABI
 * struct layout). On success, sylph populates them; the caller must
 * eventually invoke the structs' release callbacks.
 *
 * Output schema (in this column order):
 *   genome_index (UInt32)            — index into the syldb
 *   genome_name (LargeUtf8)          — GenomeSketch.file_name
 *   contig_name (LargeUtf8)          — GenomeSketch.first_contig_name
 *   sequence_abundance (Float64)     — fraction of effective coverage (sums to ~100), null in query mode
 *   taxonomic_abundance (Float64)    — fraction of reads winner-take-all-assigned (sums to ~100), null in query mode
 *   adjusted_ani (Float64)           — coverage-corrected containment ANI (0..1)
 *   eff_cov (Float64)                — effective coverage estimate
 *   naive_ani (Float64)              — raw containment ANI (0..1)
 *   kmers_reassigned (UInt64)        — k-mers winner-take-all-assigned, null when no winner pass
 *
 * Forward-declare the Arrow C ABI structs without pulling Arrow headers.
 * Layout matches the Arrow Columnar Format specification's Arrow C data
 * interface.
 * ============================================================================ */

struct ArrowArray;
struct ArrowSchema;

/* Profile parameters. Layout-stable; trailing fields can be added without
 * breaking older callers. Pass 0 / negative for "use default".
 *   estimator: 0 = ratio (default), 1 = mme, 2 = nb, 3 = mle.
 *   pseudotax: 1 = profile mode (default), 0 = query mode (no abundances).
 *   minimum_ani / seq_id: pass < 0 to use sylph defaults.
 */
typedef struct {
    uint8_t  estimator;
    uint8_t  pseudotax;
    uint8_t  estimate_unknown;
    uint8_t  estimate_read_counts;
    uint8_t  no_ci;
    uint8_t  no_adj;
    uint8_t  mean_coverage;
    uint8_t  log_reassignments;
    double   min_count_correct;
    double   min_number_kmers;
    double   minimum_ani;
    double   seq_id;
    double   redundant_ani;
    uint32_t num_threads;
    uint32_t _reserved0;
    uint64_t _reserved1;
} SylphProfileParams;

/* Populate `out` with sylph's default profile parameters. C/C++ callers
 * should always seed a SylphProfileParams via this function rather than
 * zero-initializing — sylph has several non-zero defaults (pseudotax = 1,
 * redundant_ani = 99.0, seq_id = -1.0, ...) that a zero-init struct would
 * silently miss. Returns 0 on success, non-zero if `out` is NULL. */
int sylph_profile_params_default(SylphProfileParams *out);

/* Populate `out` with sylph's default sketch parameters. Same rationale —
 * dedup = 1 and dedup_fpr = 0.0001 (matches sylph CLI's --fpr default) are
 * non-zero and would be silently lost on a zero-init. Returns 0 on success,
 * non-zero if `out` is NULL. */
int sylph_sketch_params_default(SylphSketchParams *out);

/* Run profile and write the result into caller-provided Arrow C Data
 * Interface slots. Returns 0 on success, non-zero on error. The sample
 * sketch must be finalized (sylph_sketch_builder_finalize). */
int sylph_profile(
    const SylphDatabase *db,
    const SylphSketch   *sample,
    const SylphProfileParams *params,
    struct ArrowArray  *out_array,
    struct ArrowSchema *out_schema);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SYLPH_H */
