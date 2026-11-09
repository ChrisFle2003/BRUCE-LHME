# C ABI: lifecycle and cancellation

The public C interface is located in
[`include/bruce/lhme/c_api.h`](../include/bruce/lhme/c_api.h). It uses opaque
handles and has no C++ classes in its signatures.

## Handles and ownership

| Handle | Created by | Owner after success | Release |
| --- | --- | --- | --- |
| `lhme_engine*` | `lhme_engine_create` | Caller | `lhme_engine_destroy` |
| `lhme_formula*` | `lhme_formula_register` | Caller | `lhme_formula_destroy` |
| `lhme_result*` | `lhme_evaluate` | Caller | `lhme_result_destroy` |

Every non-null handle must be released exactly once with its corresponding
`*_destroy` function. All three destroy functions accept `NULL`. A pointer
that has already been released, belongs to another object, or is otherwise
invalid is not a valid handle; this behavior is not checked and cannot be
safely used through this interface.

`lhme_engine_config`, `lhme_power_offset_spec`, and
`lhme_evaluation_request`, as well as the input strings they contain, are read
only during the respective call. The API copies the values required for the
engine, formula, and evaluation. An `lhme_formula` handle contains a copy of
the specification, but no binding to a particular engine. During evaluation,
the supplied engine must have registered this specification; otherwise, the
result is `LHME_STATUS_INVALID_ARGUMENT`.

The strings returned by `lhme_result_exact_decimal` and `lhme_result_inspect`
belong to the result handle. They remain valid until
`lhme_result_destroy(result)` and must not be freed or modified by the caller.

## Creation, return values, and errors

`lhme_engine_create(NULL)` uses the configuration
`{cache_max_entries=0, cache_max_bytes=0, worker_count=1, tile_size=1}`. The
function returns `NULL` if construction fails.

`lhme_formula_register` returns `NULL` if the engine, specification,
`base_decimal`, or `scale_decimal` is null, or if construction fails. A
non-null formula handle confirms only that it was created: a semantically
invalid specification or one not registered with the engine becomes visible
only at `lhme_evaluate` as `LHME_STATUS_INVALID_ARGUMENT`.

`lhme_evaluate` normally returns a result handle for either a successful or a
semantically failed evaluation. An unknown `lhme_mode` yields
`LHME_STATUS_INVALID_ARGUMENT`. Null arguments and exceptions within the C ABI
are represented as `LHME_STATUS_INTERNAL_ERROR` in a result handle when that
handle can be created. If creating that result is also impossible,
`lhme_evaluate` returns `NULL`.

`lhme_result_status(NULL)` returns `LHME_STATUS_INTERNAL_ERROR`.
`lhme_result_exact_decimal` returns `NULL` if the result handle is null or the
result has no integer payload. For a valid handle, `lhme_result_inspect`
returns the decimal representation of the internal status value; it is not a
trace or error-text interface. For a null handle, it returns `NULL`.

The `lhme_status` values correspond to the evaluation result categories:
`CONFIRMED`, `REJECTED`, `BUDGET_EXCEEDED`, `CANCELLED`, `INVALID_ARGUMENT`,
`UNSUPPORTED_MODE`, and `INTERNAL_ERROR`.

## Cancellation and reuse

`Engine::cancel()` and the C ABI function `lhme_cancel(engine)` request
cancellation of evaluations currently active on that engine.
`lhme_cancel(NULL)` is a no-op. A cancellation request for an inactive engine
does not affect a later run.

For a new group of active evaluations, the engine creates a fresh cancellation
signal. After an individual successful, failed, or cancelled evaluation has
finished, the same engine can be reused for a new run provided no other
evaluation of that engine remains active. Cancellation does not force work to
stop; when a running evaluation observes the request depends on its evaluation
checkpoints.

Simultaneously active evaluations of the same engine share one cancellation
signal in the current code. A cancellation request therefore targets all
evaluations active on that engine at that point; the C ABI provides no
per-request cancellation.

## Threading boundary

The implementation synchronizes registry, cache, and cancellation state
internally. A broader public C ABI guarantee for arbitrary parallel calls or
for concurrent use and destruction of the same handle cannot be derived
unambiguously from the repository. In particular, callers must synchronize a
handle's lifetime against its concurrent use themselves.
