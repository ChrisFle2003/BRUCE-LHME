#ifndef BRUCE_LHME_C_API_H
#define BRUCE_LHME_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lhme_engine lhme_engine;
typedef struct lhme_formula lhme_formula;
typedef struct lhme_result lhme_result;

/*
 * All three handle types are opaque and are owned by the caller after a
 * successful create/register/evaluate call. Release each non-null handle with
 * its matching *_destroy function exactly once. Destroy functions accept a
 * null pointer. Passing a pointer after its destroy function returned, or a
 * pointer not obtained from this ABI, is invalid.
 *
 * Input structs and their strings are borrowed only for the duration of their
 * call. Text returned by lhme_result_exact_decimal() or
 * lhme_result_inspect() belongs to the result handle and remains valid until
 * lhme_result_destroy() is called for that handle.
 */

typedef enum lhme_status {
  LHME_STATUS_CONFIRMED,
  LHME_STATUS_REJECTED,
  LHME_STATUS_BUDGET_EXCEEDED,
  LHME_STATUS_CANCELLED,
  LHME_STATUS_INVALID_ARGUMENT,
  LHME_STATUS_UNSUPPORTED_MODE,
  LHME_STATUS_INTERNAL_ERROR
} lhme_status;

typedef enum lhme_mode {
  LHME_MODE_SYMBOLIC,
  LHME_MODE_METRIC,
  LHME_MODE_MODULAR,
  LHME_MODE_EXACT
} lhme_mode;

typedef struct lhme_engine_config {
  size_t cache_max_entries;
  size_t cache_max_bytes;
  uint32_t worker_count;
  uint64_t tile_size;
} lhme_engine_config;

typedef struct lhme_power_offset_spec {
  const char* base_decimal;
  const char* scale_decimal;
  uint32_t version;
} lhme_power_offset_spec;

typedef struct lhme_evaluation_request {
  int64_t index;
  lhme_mode mode;
  uint64_t max_bit_length;
  const char* modulus_decimal;
} lhme_evaluation_request;

/*
 * Creates an engine owned by the caller. A null config selects
 * {cache_max_entries=0, cache_max_bytes=0, worker_count=1, tile_size=1}.
 * Returns null if construction fails.
 */
lhme_engine* lhme_engine_create(const lhme_engine_config* config);

/* Releases an engine created by lhme_engine_create(); null is accepted. */
void lhme_engine_destroy(lhme_engine* engine);

/*
 * Registers a PowerOffset specification and returns an owned formula handle,
 * or null for null arguments or a construction failure. base_decimal and
 * scale_decimal must be non-null for the call. A non-null handle does not
 * itself prove that the Engine accepted the specification: evaluating an
 * invalid or unregistered specification returns LHME_STATUS_INVALID_ARGUMENT.
 */
lhme_formula* lhme_formula_register(lhme_engine* engine,
                                    const lhme_power_offset_spec* spec);

/* Releases a formula handle created by lhme_formula_register(); null is accepted. */
void lhme_formula_destroy(lhme_formula* formula);

/*
 * Evaluates a live formula handle with a live engine. The caller owns every
 * non-null result and must release it with lhme_result_destroy(). Null
 * arguments produce an InternalError result when that result can be
 * allocated; allocation failure returns null. An unrecognised mode produces
 * an InvalidArgument result.
 */
lhme_result* lhme_evaluate(lhme_engine* engine, const lhme_formula* formula,
                           const lhme_evaluation_request* request);

/* Returns InternalError for a null result handle. */
lhme_status lhme_result_status(const lhme_result* result);

/*
 * Returns a result-owned decimal integer string when the result contains an
 * integer payload, otherwise null. The pointer is valid until that result is
 * destroyed.
 */
const char* lhme_result_exact_decimal(const lhme_result* result);

/*
 * Returns a result-owned decimal representation of the internal result status
 * value, otherwise null. It is not a trace or a diagnostic message. The
 * pointer is valid until that result is destroyed.
 */
const char* lhme_result_inspect(const lhme_result* result);

/* Releases a result returned by lhme_evaluate(); null is accepted. */
void lhme_result_destroy(lhme_result* result);

/*
 * Requests cancellation of evaluations currently active on this engine. A
 * null engine is ignored. Calling lhme_cancel() while the engine is idle does
 * not affect a later evaluation. Once the cancelled active evaluation group
 * has returned, the engine can be reused for another evaluation.
 */
void lhme_cancel(lhme_engine* engine);

#ifdef __cplusplus
}
#endif

#endif
