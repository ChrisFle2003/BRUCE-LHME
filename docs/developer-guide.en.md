# Developer Guide for BRUCE LHME

This guide describes the repository as implemented in version 0.2.0. It is
intended for developers integrating, maintaining, or extending the library.
Claims about runtime behaviour are based on the public headers, source files,
CMake configuration, examples, tests, and release notes in this repository.

## 1. Overview

BRUCE LHME is a C++20 static library for deterministic, lazy, and
memory-conscious evaluation of formula-defined values. It is useful when a
logical value space is large, but most values can be reconstructed from a
compact rule rather than stored individually in RAM.

The primary built-in formula is currently PowerOffset:

```text
value(index) = scale * (base^index - 1)
```

The library can return that value exactly as a `BigInt`, estimate its bit
length, calculate a modular residue, or return a symbolic representation. It
also supports bounded result caching, resource budgets, cancellation, sparse
value overrides, custom C++ kernels, hierarchical KEVIN grid filtering, and a
read-only GGUF V3 inspector and planner.

Use BRUCE LHME when deterministic reconstruction, explicit resource limits,
and C++ integration matter. It is a library, not an inference runtime, web
service, database, or general-purpose data store.

## 2. Project Status and Scope

**Current version:** `0.2.0` (`CMakeLists.txt`). The project is pre-1.0; the
release policy treats a new `0.y.0` minor release as a new C++ API/ABI
generation. C++ consumers must be rebuilt and relinked for a new minor
release. The public C ABI and `TechniqueKernel` vtable are documented as
unchanged for v0.2.0.

### Implemented

- CPU/RAM-only formula evaluation in Exact, Metric, Modular, and Symbolic
  modes.
- Built-in PowerOffset formula evaluation and `FibonacciKernel`/
  `PowerOffsetKernel` technique kernels.
- Formula and technique registration through `Engine`.
- Bounded LRU result caching for Exact and Metric results.
- Engine-managed cancellation groups and deterministic traces.
- Range evaluation and streaming evaluation with CPU workers and budgets.
- Sparse `FormulaOverrideStore` values and validated text snapshots.
- The modern KEVIN domain/projection/filter API and the legacy
  `KevinMemoryStore` compatibility API.
- A public C API for PowerOffset formula evaluation.
- An installable CMake package and independently built consumer examples.
- GGUF V3 inventory, storage planning, bounded pattern scanning, text reports,
  and JSON plans.

### Explicitly excluded from v0.2

- GPU execution or GPU backends.
- Model inference, quantization, dequantization, or GGUF rewriting.
- A runtime plug-in loader. Kernel SDK integrations are compiled with the
  consuming application.
- NanoCell runtime, semantic routing, AnswerPoints, and Experience Records.
- Database storage, server APIs, authentication, and network access.

### Planned or documented only

- A future controlled migration layer after
  `choose_storage_strategy(...)` selects a target storage strategy.
- A separately validated, optional lossless GGUF rewriter or FormulaRecipe
  sidecar.
- The v0.2 release note lists external release-evidence gates that are still
  unchecked. Whether they were completed later is not determinable from this
  repository snapshot.

### Legacy compatibility

`KevinMemoryStore`, `GridSpec`, and `KevinSnapshot` remain supported for
existing callers. New code should use `GridDomainSpec`, `GridAddress`, an
explicit `DomainProjection`, `FormulaEvaluator`, and `FormulaOverrideStore`.
The new formula-override snapshot loader intentionally does not accept legacy
KEVIN snapshots.

## 3. Technology Stack

| Area | Technology and repository evidence |
| --- | --- |
| Implementation language | C++20; C11 is enabled to compile the C header smoke test (`CMakeLists.txt`). |
| Compiler requirements | Any C++20 compiler supported by CMake; exact supported compiler versions are not specified. |
| Build system | CMake 3.25 or newer, with CMake presets. |
| Package manager | None. There is no Conan, vcpkg, npm, Cargo, or lockfile configuration. |
| Direct dependencies | `Threads` and Boost headers, including `boost/multiprecision/cpp_int.hpp`. No compiled Boost library is required. |
| Test system | CTest plus a small repository-local C++ test harness. |
| CI | GitHub Actions on Ubuntu and Windows for Debug/Release; Ubuntu AddressSanitizer and UndefinedBehaviorSanitizer jobs. |
| Packaging | CMake config package exporting `bruce_lhme::core`. |
| License | MIT, copyright 2026 Christopher Flegel. |
| Platforms | CI names Ubuntu and Windows. macOS support is not determinable from the repository. |

The code uses the C++ standard library for concurrency, filesystem access,
streams, containers, and futures. `BigInt` is an alias for
`boost::multiprecision::cpp_int`.

## 4. Repository Layout

Generated build products and local editor state are intentionally omitted.

```text
.
├── CMakeLists.txt                 build, test, package, and option definitions
├── CMakePresets.json              Debug, Release, GCC/Ninja, sanitizer presets
├── README.en.md / README.md       project entry documentation
├── LICENSE                        EUPL-1.2-or-later licence text
├── cmake/                         installed-package configuration template
├── include/bruce/lhme/            public C++ and C headers
├── src/
│   ├── core/                      formulas, traces, built-in kernels
│   ├── cache/                     bounded result cache
│   ├── cpu/                       range scheduler and streaming execution
│   ├── gguf/                      GGUF reader, planner, scan, and reports
│   ├── kevin/                     domain, projection, filters, legacy store
│   ├── engine.cpp                 orchestration, registries, cancellation
│   ├── c_api.cpp                  C ABI adapter
│   └── formula_override_*.cpp     sparse overrides and snapshots
├── tools/gguf_inspect/            optional command-line executable
├── examples/                      installed-package consumer projects
├── demos/lhme/                    demonstrations and benchmarks
├── tests/                         unit, integration, packaging, and contract tests
├── docs/                          English and German technical documentation
├── concepts/                      implemented-component overview
└── .github/workflows/             CI and release-readiness workflows
```

| Path | Purpose | Production / Test / Tooling / Generated | Important Dependencies |
| ---- | ------- | --------------------------------------- | ---------------------- |
| `include/bruce/lhme/` | Public contracts for C++, C, SDK, KEVIN, overrides, and GGUF | Production | Used by every consumer and source module |
| `src/core/` | Formula evaluation, traces, built-in kernels | Production | Boost Multiprecision |
| `src/engine.cpp` | Registration, cache orchestration, cancellation groups | Production | Core, cache, CPU range code |
| `src/cache/` | Thread-safe size-bounded LRU cache | Production | Formula/result models |
| `src/cpu/` | Parallel range and streaming evaluation | Production | Core evaluation, Threads |
| `src/kevin/` | Grid address system, projections, filters, compatibility store | Production / Legacy | Engine or `FormulaEvaluator` |
| `src/gguf/` | Local GGUF V3 validation, planning, scanning, reporting | Production | Filesystem and streams |
| `src/c_api.cpp` | Opaque-handle C adapter | Production | Engine |
| `tools/gguf_inspect/` | CLI parsing and report writing | Tooling | GGUF modules |
| `examples/` | External `find_package()` examples | Tooling / Integration test | Installed CMake package |
| `demos/lhme/` | Executable examples and benchmarks | Tooling | Public C++ API |
| `tests/` | Custom test harness and test programs | Test | Library and CMake |
| `cmake/` | Package config template | Tooling | CMake package helpers |
| `out/` | Existing local build outputs | Generated | Ignore; do not edit |

## 5. Build and Installation

The root CMake project builds the static target `bruce_lhme_core`. The named
presets set `BUILD_TESTING=ON`; optional tools, examples, and benchmarks remain
off unless explicitly enabled.

| Task | Command | Status |
| --- | --- | --- |
| Configure Debug | `cmake --preset debug` | Confirmed from repository |
| Build Debug | `cmake --build --preset debug` | Confirmed from repository |
| Test Debug | `ctest --preset debug` | Confirmed from repository |
| Configure Release | `cmake --preset release` | Confirmed from repository |
| Build Release | `cmake --build --preset release` | Confirmed from repository |
| Test Release | `ctest --preset release` | Confirmed from repository |
| Install package | `cmake --install <build-dir> --prefix <prefix>` | Confirmed from CMake install rules |
| Enable tools | `cmake --preset release -DBRUCE_BUILD_TOOLS=ON` | Derived from build configuration |
| Enable examples | `cmake --preset release -DBRUCE_BUILD_EXAMPLES=ON` | Derived from build configuration |
| Enable benchmarks | `cmake --preset release -DBRUCE_BUILD_BENCHMARKS=ON` | Derived from build configuration |
| Address sanitizer | `cmake --preset address && cmake --build --preset address && ctest --preset address` | Confirmed from repository presets |
| Undefined sanitizer | `cmake --preset undefined && cmake --build --preset undefined && ctest --preset undefined` | Confirmed from repository presets |
| Thread sanitizer | `cmake --preset thread && cmake --build --preset thread && ctest --preset thread` | Confirmed from repository presets |

The command separator `&&` above is shell syntax; run each command separately
if preferred.

### Installing and consuming the package

Install an already configured build tree:

```bash
cmake --install out/build/release --prefix /opt/bruce-lhme
```

Consume it from another project:

```cmake
find_package(bruce_lhme 0.2 CONFIG REQUIRED)

add_executable(my_program main.cpp)
target_link_libraries(my_program PRIVATE bruce_lhme::core)
```

The exported target propagates C++20, Threads, and Boost headers. See
`examples/external_consumer` and `examples/kernel_sdk_consumer`; CTest stages
an install tree and builds both independently.

### Optional executable caveat

The historical GGUF document said that `cmake --preset release` alone creates
`bruce_gguf_inspect`. That is inconsistent with the build configuration:
`BRUCE_BUILD_TOOLS` defaults to `OFF`. Enable it before targeting the tool:

```bash
cmake --preset release -DBRUCE_BUILD_TOOLS=ON
cmake --build --preset release --target bruce_gguf_inspect
```

## 6. Public API Guide

The C++ API is under `include/bruce/lhme/`; all types are in namespace
`bruce::lhme`.

| API | Location | Responsibility | Main Inputs | Outputs | Important Failure Modes |
| --- | --- | --- | --- | --- | --- |
| `Engine` | `engine.hpp` | Register formulas/kernels; evaluate; cache; cancel | `EngineConfig`, requests | `EvaluationResult`, `RangeResult` | Invalid registration/request, budget exceeded, cancellation |
| `FormulaSpec` | `formula.hpp` | Immutable-by-convention PowerOffset specification | base, scale, version, hash | valid/invalid spec | invalid base/scale/version/hash |
| `EvaluationRequest` | `formula.hpp` | A formula evaluation request | spec, address, mode, budget, modulus | passed to engine/core | unregistered spec at engine boundary |
| `EvaluationResult` | `types.hpp` | Typed calculation outcome | produced by evaluators | status and optional payload | `exact_integer()` throws if no integer payload |
| `Budget` | `types.hpp` | Per-request resource limits | bit length, operations, bytes, trace, candidates | rejection threshold | limits can cause `BudgetExceeded` |
| `TechniqueKernel` | `technique.hpp` | Custom deterministic calculation interface | `TechniqueRequest` | `EvaluationResult` | invalid identity, registration rejection, thrown kernel exception |
| `FormulaOverrideStore` | `formula_override_store.hpp` | Sparse formula/technique overrides | binding, address, optional override | `FormulaValue`, snapshots | invalid binding, engine-side invalid request |
| `ResultCache` | `cache.hpp` | Bounded LRU accelerator | cache key/result | cached result/statistics | disabled or too-large entries are not admitted |
| `DomainProjection` | `domain_projection.hpp` | Explicit GridAddress-to-FormulaAddress mapping | `GridAddress` | optional formula address | missing/invalid/unmappable projection |
| `FormulaEvaluator` | `formula_evaluator.hpp` | Port used by modern KEVIN evaluation | address, mode, budget, modulus | `EvaluationResult` | implemented by caller/adapter |
| `KevinMemoryStore` | `kevin_memory_store.hpp` | Legacy grid/technique/override store | grid, address, mode | `KevinValue` | unknown grid, depth/budget/modulus errors |
| GGUF APIs | `gguf_*.hpp` | inspect, plan, scan, report | local path, limits/config | structured result/report | malformed or oversized input |

### Small C++ example

This follows the supported formula registration path:

```cpp
#include "bruce/lhme/engine.hpp"

using namespace bruce::lhme;

int main() {
  auto engine = Engine::create({.cache_max_entries = 16,
                                .cache_max_bytes = 8192});
  const auto formula = engine.register_formula(make_power_offset_spec(2, 100, 1));

  const auto result = engine.evaluate({
      .spec = formula,
      .address = {.index = 10},
      .mode = EvaluationMode::Exact,
      .budget = Budget::unlimited(),
  });

  return result.status == Status::Confirmed &&
                 result.integer == BigInt{102300}
             ? 0
             : 1;
}
```

The external-consumer example additionally demonstrates a custom kernel and
`FormulaOverrideStore`.

## 7. Core Evaluation Model

### Formula specification and validation

`FormulaSpec` currently has one operation: `FormulaOperation::PowerOffset`.
`make_power_offset_spec(base, scale, version)` constructs a canonical spec and
its 32-byte FNV-derived specification hash. `valid_formula_spec(...)` requires:

- `base > 0`;
- `scale >= 0`;
- `nonnegative_index == true`;
- nonzero version;
- matching specification hash.

`Engine::register_formula(...)` stores a valid specification by its canonical
identity. An invalid spec is returned unchanged but not registered. A later
engine evaluation returns `InvalidArgument`.

### Addresses, modes, and budgets

`FormulaAddress` contains one signed `index`. PowerOffset requires a
non-negative index. `EvaluationMode` has four values:

| Mode | Confirmed result payload |
| --- | --- |
| `Exact` | `integer`: exact `BigInt` |
| `Metric` | `metric_bit_length`: estimated bit length |
| `Modular` | `integer`: residue modulo a positive modulus |
| `Symbolic` | `symbolic_expression`, such as `100*(2^10-1)` |

`Budget` limits exact expansion and published traces. `max_bit_length`,
`max_operations`, `max_temporary_bytes`, `max_trace_bytes`, and
`max_candidate_count` default to their largest `uint64_t` value in
`Budget::unlimited()`.

### Request-to-result flow

```text
EvaluationRequest
      |
      v
Engine checks cancellation and registered FormulaSpec
      |
      +-- Exact/Metric: result-cache lookup
      |
      v
Core evaluator validates address, mode, budget, and modulus
      |
      +-- Symbolic: construct expression
      +-- Metric: estimate result bit length
      +-- Modular: modular exponentiation
      +-- Exact: budget-check then BigInt exponentiation
      |
      v
Trace budget enforcement and optional cache admission
      |
      v
EvaluationResult { Status, optional payload, TraceRecord }
```

`TraceRecord` contains ordered `GateEvent` entries. Events identify branch,
tile, run, cache, publish, or projection decisions. If a trace exceeds
`max_trace_bytes`, `enforce_trace_budget(...)` replaces the result with an
untraced `BudgetExceeded` result; callers do not receive a partial successful
trace.

### Errors and cancellation

- Invalid specs, negative PowerOffset addresses, invalid modular requests, or
  an unregistered formula become `InvalidArgument`.
- Exact work estimated to exceed a caller budget becomes `BudgetExceeded`.
- A request-level or engine-level cancellation signal becomes `Cancelled` when
  observed at evaluation checkpoints.
- An unrecognised internal mode path is `InternalError`; no normal public mode
  reaches it.

## 8. Engine, Cache, and Cancellation

### Engine responsibilities

`Engine` owns:

- registered `FormulaSpec` objects;
- registered `TechniqueKernel` instances;
- immutable Exact formula/technique cache bindings;
- a `ResultCache`;
- in-flight cancellation groups;
- mutexes protecting registry and cancellation state.

The engine is created with `EngineConfig`:

```cpp
EngineConfig{
  .cache_max_entries = 0,
  .cache_max_bytes = 0,
  .worker_count = 1,
  .tile_size = 1,
};
```

Zero cache limits disable effective cache admission.

### Registration

- `register_formula(...)` accepts only valid formula specs.
- `register_technique(...)` rejects null kernels, invalid identities, duplicate
  identities, and exceptions from `identity()`.
- A `TechniqueIdentity` must have a 1–128 byte ASCII alphanumeric/`._-` ID and
  a version greater than zero.
- `has_technique(...)` checks a registered identity.

### Cache behaviour

Only Exact and Metric `Engine` evaluations are cacheable. Modular and Symbolic
results are evaluated each time. The cache stores only confirmed results with a
non-null trace. It uses recency ordering and evicts least-recently-used entries
until both entry and byte budgets are satisfied.

Cache hits are checked again against the caller's current budget and
cancellation signal. A cached large exact result can therefore be rejected for
a stricter caller without modifying the cache entry.

`Engine::evaluate_range(...)` and `evaluate_streaming_range(...)` invoke the
core evaluator for each address rather than routing each address through
`Engine::evaluate(...)`; individual range/streaming candidates do not use the
engine result cache. This is confirmed by the direct calls to
`bruce::lhme::evaluate(...)` in the scheduler and streaming implementation.

### Cross-API Exact cache equivalence

Implemented in v0.2: an engine may accept an
`ExactFormulaTechniqueCacheBinding` linking one registered technique identity
and exact parameter string to one registered formula in the *same* engine.

- Binding parameters are compared byte-for-byte.
- Only Exact requests share a formula cache key.
- Metric, Modular, and Symbolic calls remain API-separated.
- Re-registering an identical binding succeeds; redirecting it to another
  formula is rejected.
- A cross-API hit returns a deliberately neutral trace containing
  `cross-api exact cache hit`, rather than exposing the producer trace.
- The binding does not make concurrent misses single-flight; duplicate work is
  still possible.

See `docs/kernel-sdk.en.md` and `examples/kernel_sdk_consumer`.

### Cancellation and reuse

An active evaluation acquires an engine cancellation-group lease. Calls that
resolve to a compatible in-flight calculation can share a signal. `cancel()`
sets the signal for active groups only; an idle engine is not poisoned for the
next calculation. Once all leases leave a group, the group is removed and the
engine can be reused.

Confirmed limitations:

- Cancellation is cooperative, not a forced thread termination mechanism.
- Simultaneously active compatible requests can share one signal, so an engine
  cancellation can affect more than one active request.
- There is no public per-request engine cancellation API.
- Registry, cache, and cancellation state are synchronized internally.
- A comprehensive public guarantee for arbitrary parallel calls through the C
  ABI, especially use concurrent with destruction of the same handle, is not
  established by the repository. Synchronize handle lifetime externally.

## 9. Extending the Project with Custom Kernels

Implement `TechniqueKernel`, register it in an `Engine`, and submit a matching
`TechniqueRequest`. A kernel is a compile-time C++ extension; BRUCE does not
load plug-ins at runtime.

Requirements for a robust kernel:

1. Return a stable, valid `TechniqueIdentity`.
2. Validate identity, canonical parameters, address, requested mode, and
   modulus before computing.
3. Be deterministic for equivalent inputs.
4. Check `cancellation_requested(request)` before and during expensive work.
5. Enforce budgets inside potentially expensive custom calculations. The engine
   also verifies confirmed published values and trace limits, but it cannot
   prevent a kernel from doing excessive work before returning.
6. Return meaningful traces. SDK helpers create standard traces for exact,
   invalid, and cancelled outcomes.
7. Add direct kernel tests, engine integration tests, cancellation tests, and
   an installed-package consumer test if the public SDK contract changes.

Minimal kernel based on the repository SDK conventions:

```cpp
#include <memory>

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/kernel_sdk.hpp"

class AffineKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "example.affine", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    using namespace bruce::lhme;
    if (cancellation_requested(request)) {
      return cancelled_technique_result(request.mode, "kernel cancellation requested");
    }
    if (request.identity != identity() ||
        request.parameters != "factor=1:2;offset=1:3" ||
        request.address.index < 0 ||
        request.mode != EvaluationMode::Exact) {
      return invalid_technique_request_result(request.mode,
                                              "expected exact affine request");
    }
    return confirmed_exact_result(BigInt{request.address.index} * 2 + 3,
                                  make_trace({}));
  }
};

// Registration:
auto kernel = std::make_shared<AffineKernel>();
auto engine = bruce::lhme::Engine::create({});
if (!engine.register_technique(kernel)) {
  // Invalid identity, duplicate registration, or an exception.
}
```

By default, an Exact or Metric technique result uses a technique-specific cache
key. A technique can share an Exact formula cache entry only through an
explicit engine-owned `ExactFormulaTechniqueCacheBinding`; overriding
`canonical_computation_identity(...)` does not authorize cache equivalence.

## 10. Range and Streaming Evaluation

### Range evaluation

`Engine::evaluate_range(request, AddressRange)` validates a non-empty inclusive
range, a nonzero worker count, a nonzero tile size, and
`max_candidate_count`. It divides the range into tiles and uses `std::async`
for up to `worker_count` tiles per batch. Results are aggregated in tile order,
which preserves deterministic candidate ordering for equal input.

`RangeResult` contains a total status, `GlobalSummary`, and confirmed
candidates. Invalid per-address evaluations increment
`invalid_argument_count`; non-confirmed outcomes other than invalid arguments
are not retained as candidates. Cancellation clears candidates and returns
`Cancelled`.

### Streaming evaluation

`evaluate_streaming_range(...)` accepts only an Exact base request and creates
a temporary `CpuWorkerPool`. It supports:

| Retention mode | Behaviour |
| --- | --- |
| `SummaryOnly` | retain no values; only counts are returned |
| `FilteredCandidates` | retain candidates while count and memory budgets allow |
| `TopK` | retain the earliest surviving candidates within limits; there is no score-based ranking |
| `Adaptive` | start retaining candidates; switch to `TopK` mode after a limit prevents retention |

`StreamingOptions` sets worker count, tile size, retention count/byte limits,
and an optional `CandidateFilter`. `FilterMode::ModularEquals` first evaluates
the modular residue and performs exact evaluation only for matching addresses.
`AcceptAll` performs exact work for every address.

Important behaviour:

- Zero retention count or zero retention bytes forces `SummaryOnly`.
- Candidate count limits are checked before work begins.
- Retained-result order is stable. When retention is enabled, tiles are
  submitted one at a time to avoid worker-order effects on retention.
- Cancellation clears retained candidates.
- Candidate memory accounting is an internal estimate, not a general process
  memory measurement.
- Core evaluation is used directly; the engine cache is not used here.

## 11. Override Storage and Snapshots

### Formula overrides

`FormulaOverrideStore` associates a `(FormulaBinding, FormulaAddress)` with an
explicit `BigInt` value. `FormulaBinding` is either:

- a `FormulaSpec`; or
- a `TechniqueBinding` containing `TechniqueIdentity`, canonical parameter
  text, and a default budget.

Overrides take precedence over computed values. The result reports
`FormulaValueSource::Override` or `FormulaValueSource::Computed`.

Override calculation still respects caller-facing mode and relevant limits:

- Exact checks bit and temporary-byte limits.
- Metric reports override bit length.
- Modular requires a positive modulus.
- Symbolic returns `override(<value>)`.

For a technique binding, the effective budget is the stricter value of caller
and binding default budgets. A computed formula path calls `Engine::evaluate`;
a computed technique path calls `Engine::evaluate_technique`.

### Thread safety

The override map uses a shared mutex: reads/snapshots use shared locks and
mutation/replacement uses exclusive locks. `computed_exact_evaluation_count`
is atomic. The store holds an `Engine&`; engine lifetime must outlive the
store.

### Formula-override snapshot format

`save_formula_override_snapshot(...)` writes an ASCII/binary-stream text format
starting with:

```text
BRUCE-FORMULA-OVERRIDE-1
```

It serializes a record count, formula or technique records, length-prefixed
strings, and a trailing `CHECKSUM <decimal>` line. The loader enforces a 16 MiB
file limit, at most 100,000 records, up to 1 MiB per stored string, canonical
integer text, unique override keys, matching checksum, known magic, and no
trailing data.

Save behaviour is replace-on-success:

1. Write a uniquely named temporary file beside the requested destination.
2. Reopen it to append the checksum.
3. Replace the destination through `std::filesystem::rename` on non-Windows or
   `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)` on
   Windows.

This avoids overwriting the target when writing fails. The repository does not
claim crash durability across filesystems or call `fsync`, so power-loss
durability is not guaranteed by the code.

The FNV-based checksum detects accidental corruption; it is not authentication
and does not protect against an attacker who can rewrite both payload and
checksum.

## 12. KEVIN Domain System

### Modern KEVIN API

The modern architecture deliberately separates a hierarchical grid coordinate
from the source of a value:

```text
GridDomainSpec + GridAddress
          |
          v
explicit DomainProjection
          |
          v
FormulaAddress
          |
          v
FormulaEvaluator
          |
          v
EvaluationResult
```

`GridAddress` is a sequence of `(x, y)` steps where each component is 0–7.
The implementation allows a maximum depth of 10. `GridDomainSpec` contains a
validated identity, `max_depth`, and a default budget.

`DomainProjection` is an interface. `DepthPathProjection` maps the full
hierarchical path to a non-negative `FormulaAddress`; the depth is encoded so
paths at different depths do not collide. A grid address has no implicit value
without a projection.

`FormulaEvaluator` is another interface through which the KEVIN domain invokes
the actual formula/technique implementation. The library does not expose a
general built-in `Engine`-to-`FormulaEvaluator` adapter; the KEVIN demo and
benchmark provide local adapters.

### Tile filtering

A parent grid address expands to an 8×8 tile of 64 child addresses. Bit `y * 8
+ x` identifies `(x, y)`. The domain filter can:

1. evaluate modular residues without exact materialization;
2. evaluate metric bit lengths without exact materialization;
3. intersect the two candidate masks;
4. run Exact evaluation only for survivors, bounded by a caller limit.

`evaluate_domain_address(...)` appends domain, projection, and reached formula
address information to the trace. Missing, invalid, out-of-range, or unmappable
projections become branch-local `InvalidArgument` results.

`KevinTileCpuExecutor` maintains a worker pool for repeated legacy-store tile
filtering. It returns candidates in supplied parent order. A worker count of
zero is invalid.

### Legacy KevinMemoryStore

`KevinMemoryStore` couples a `GridSpec` to a technique binding and uses
`DepthPathProjection` internally. It stores registered grids and sparse
overrides, forwarding normal requests to `Engine::evaluate_technique`.

For new code:

1. Replace `GridSpec` with `GridDomainSpec`.
2. Choose an explicit `DomainProjection` such as `DepthPathProjection`.
3. Keep formula/technique binding independently in `FormulaOverrideStore`.
4. Pass an explicit `FormulaEvaluator` adapter into the domain context.

Legacy snapshots use `BRUCE-KEVIN-2` when written. The loader can also read
`BRUCE-KEVIN-1`; V1 omitted grid budgets, which load as `Budget::unlimited()`.

## 13. GGUF Inspector and CLI

The GGUF inspector processes local GGUF version 3 files only. It performs no
inference, downloads no models, opens no network connections, and never writes
or reorders model weights.

### Supported layouts

Supported GGML tensor layouts are F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_K. Unknown GGUF versions and
unknown tensor types are rejected rather than guessed.

### Inspection limits and validation

`GgufInspectionLimits` defaults to a 1 GiB maximum file, one million metadata
entries, one million tensors, 32 dimensions per tensor, and 1 MiB strings. The
reader validates magic, version, counts, metadata types, alignment, tensor
rank/dimensions, checked arithmetic, tensor block layout, offsets, overlap, and
file bounds. Failures return structured `GgufInspection` values with a status,
validation error, diagnostic, and trace.

### Plans and pattern scans

`make_gguf_storage_plan(...)` splits tensors into chunks and assigns one of:

- `Resident` — loaded within the memory budget;
- `OnDemand` — not resident in a budgeted plan;
- `Deferred` — metadata-only plan;
- `Rejected` — all-resident plan cannot fit or planning rejected the item.

Policies are `AllResident`, `Budgeted`, and `MetadataOnly`. Budgeted planning
uses a supplied tensor-use profile first, then stable file-offset/tensor/chunk
ordering.

`scan_gguf_pattern_candidates(...)` accepts only validated plan chunks, checks
an aggregate payload budget, and reports bounded candidates for constant bytes,
repeated byte blocks, and bit-exact F32 arithmetic progressions. A candidate is
not a proven formula and is always reported with `exactly_verified == false`.

Text and JSON report functions provide stable, escaped representations of the
inspection, plan, optional scan, and trace data.

### CLI

```text
bruce_gguf_inspect MODEL.gguf
  --memory-budget BYTES
  --chunk-size BYTES
  --policy all-resident|budgeted|metadata-only
  [--use-profile FILE]
  [--json-output FILE]
  [--scan]
```

The three first options are required. `--use-profile` reads one tensor name per
line. `--json-output` writes a report to the caller-selected path. `--scan`
uses the CLI memory budget as the scan payload limit and a hard-coded maximum
of 128 candidates.

| Exit code | Meaning |
| ---: | --- |
| 0 | confirmed inspection/plan/optional scan |
| 2 | argument, profile, planning, or scan-input error |
| 3 | inspection or scan failure |
| 4 | file, plan, or scan budget exceeded |
| 5 | JSON output could not be written |

**Build caveat:** set `BRUCE_BUILD_TOOLS=ON` before requesting the target; see
section 5.

## 14. C API

The public C header is `include/bruce/lhme/c_api.h`. It exposes opaque
`lhme_engine`, `lhme_formula`, and `lhme_result` handles. The C API supports
PowerOffset formulas, not arbitrary C++ `TechniqueKernel` registration.

### Ownership and lifecycle

| Handle | Create | Release | Notes |
| --- | --- | --- | --- |
| `lhme_engine*` | `lhme_engine_create` | `lhme_engine_destroy` | `NULL` config selects default config |
| `lhme_formula*` | `lhme_formula_register` | `lhme_formula_destroy` | contains a copied formula spec |
| `lhme_result*` | `lhme_evaluate` | `lhme_result_destroy` | owns returned result strings |

Destroy functions accept `NULL`. Every non-null valid handle must be destroyed
once with its matching function. Returned strings remain valid until their
result handle is destroyed; callers must not free or modify them.

### Evaluation and statuses

`lhme_evaluate` accepts index, mode, `max_bit_length`, and an optional decimal
modulus string. Remaining C++ budget fields retain their `Budget` defaults.
The C API maps C++ result statuses to:

`LHME_STATUS_CONFIRMED`, `REJECTED`, `BUDGET_EXCEEDED`, `CANCELLED`,
`INVALID_ARGUMENT`, `UNSUPPORTED_MODE`, and `INTERNAL_ERROR`.

`lhme_result_exact_decimal` returns `NULL` when a result has no integer payload.
`lhme_result_inspect` returns the decimal numeric representation of the
internal status value; it is not a trace or diagnostic interface.

### Minimal C example

```c
#include <stdint.h>
#include "bruce/lhme/c_api.h"

int main(void) {
  lhme_engine_config config = {16, 8192, 1, 1};
  lhme_engine *engine = lhme_engine_create(&config);
  if (engine == NULL) return 1;

  const lhme_power_offset_spec spec = {"2", "100", 1};
  lhme_formula *formula = lhme_formula_register(engine, &spec);
  if (formula == NULL) {
    lhme_engine_destroy(engine);
    return 1;
  }

  const lhme_evaluation_request request = {
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = NULL,
  };
  lhme_result *result = lhme_evaluate(engine, formula, &request);
  const int ok = result != NULL &&
                 lhme_result_status(result) == LHME_STATUS_CONFIRMED;

  lhme_result_destroy(result);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
  return ok ? 0 : 1;
}
```

`lhme_cancel(engine)` requests cancellation for active engine evaluations. It
does nothing for `NULL` or an idle engine. The same C API threading/lifetime
limitation stated in section 8 applies: callers must synchronize a handle's
destruction against concurrent use.

## 15. Tests and Quality Assurance

The CMake project registers 44 CTest programs. The repository contains unit,
integration, contract, packaging, documentation-contract, and concurrency
coverage, including extensive deterministic cancellation interleavings in
`tests/engine_tests.cpp`.

| Area | Representative tests | What is covered |
| --- | --- | --- |
| Core formula and traces | `core_formula_tests`, `core_budget_trace_tests` | Formula modes, budgets, trace limits |
| Engine and cache | `engine_tests`, `cache_tests`, `exact_cache_equivalence_tests` | registry, cache, cancellation, cross-API binding |
| CPU execution | `cpu_scheduler_tests`, `streaming_tests` | tiling, workers, retention, determinism |
| C API | `c_api_tests`, `c_api_header_smoke` | ABI surface, ownership, cancellation |
| Overrides/snapshots | `formula_override_*_tests` | precedence, locking, validation, replacement |
| KEVIN | `tests/kevin/*` | addresses, projections, filters, legacy snapshots/store |
| GGUF | `gguf_*_tests` | malformed input, plans, scans, reports, command parsing |
| Packaging | `lhme_package_layout`, external/kernel consumer tests | install/export/`find_package()` contracts |
| Benchmarks/demos | demo and benchmark CTest registrations | expected executable behaviour |

The project also provides AddressSanitizer, UndefinedBehaviorSanitizer, and
ThreadSanitizer presets. CI runs Debug and Release on Ubuntu/Windows, plus
AddressSanitizer and UndefinedBehaviorSanitizer on Ubuntu. ThreadSanitizer is
documented as an additional local check, not hosted-CI coverage.

### Strong coverage

- Cancellation races around cache hits, misses, ranges, and cross-API groups.
- Snapshot rejection without mutating existing state.
- GGUF malformed-input bounds checking.
- Installed-package consumption independent of the source include path.

### Important gaps

- There is no dedicated CTest target proving `BRUCE_BUILD_TOOLS=ON` builds and
  runs the GGUF executable.
- CI does not include ThreadSanitizer.
- No fuzzing, code-coverage threshold, dependency vulnerability scan, or secret
  scan is configured.
- Passing tests show that covered scenarios passed; they do not prove the
  project is bug-free or all platforms are supported.

### Adding a test

1. Put a focused `TEST(...)` case in the relevant test source, or create a new
   source under `tests/`.
2. Register a corresponding executable and `add_test(...)` in
   `CMakeLists.txt` when creating a new test program.
3. Use `tests/support/test_main.cpp` and `tests/support/test.hpp` for new test
   executables.
4. Run the matching test with `ctest --preset debug -R '<name>'` and then the
   appropriate broader suite.

## 16. Documentation Consistency Review

| Documentation Claim | Implementation Evidence | Status | Recommended Action |
| ------------------- | ----------------------- | ------ | ------------------ |
| C++20 library with Boost headers and Threads | `CMakeLists.txt` requires C++20, Boost, Threads | Confirmed | Keep requirements current. |
| Exact, Metric, Modular, Symbolic evaluation | `types.hpp`, `core/formula.cpp`, kernel tests | Confirmed | Keep API examples mode-specific. |
| Exact formula/technique cache equivalence | Engine binding map, v0.2 tests, SDK example | Confirmed | Preserve contract tests for future releases. |
| KEVIN requires explicit projection in new paths | `DomainProjection`, `FormulaEvaluator`, domain-filter code | Confirmed | Prefer this path in new examples. |
| Legacy KEVIN compatibility remains | `KevinMemoryStore`, snapshots, tests | Confirmed | Mark it clearly as legacy. |
| GGUF inspector is read-only and performs no inference | GGUF reader/scan code and CLI | Confirmed | Keep security boundary explicit. |
| `cmake --preset release` directly builds `bruce_gguf_inspect` | Tools are gated by default-off `BRUCE_BUILD_TOOLS` | Contradictory | Enable the option in instructions; this guide and GGUF docs do so. |
| GPU, NanoCell, semantic routing, AnswerPoints | Excluded by README/scope; no production code found | Planned Only | Avoid presenting these as shipped capabilities. |
| Storage migration after strategy choice | Documentation says future; selector returns only a decision | Planned Only | Design separately with migration/rollback tests. |
| v0.2 external release gates completed | Release note leaves gates unchecked | Not Verifiable | Record external CI/compiler evidence when available. |

## 17. Security and Reliability Notes

| Finding | Evidence | Risk Level | Classification | Recommendation |
| ------- | -------- | ---------- | -------------- | -------------- |
| Formula and GGUF input validation | Validity checks, bounded reader, checked arithmetic, supported-layout table | Low | Confirmed Issue prevented by design | Maintain rejection tests and add fuzzing for parsers. |
| Caller-controlled resource limits | `Budget`, GGUF limits, scan payload configuration | Medium | Likely Risk | Set deployment-specific ceilings; do not blindly pass untrusted limits. |
| FNV snapshot checksums are not authentication | Snapshot checksum implementations and documentation | Medium | Confirmed Issue | Sign or MAC snapshots when attackers can modify files. |
| Snapshot replacement lacks explicit fsync | Temporary-write-and-rename code | Low–Medium | Theoretical Risk | Add durable-write semantics if power-loss durability is required. |
| Caller-selected JSON path | GGUF CLI `--json-output` | Low | Theoretical Risk | Restrict output locations if invoked in a privileged environment. |
| C ABI lifetime concurrency | C ABI documentation explicitly limits guarantee | Medium | Missing Information | Synchronize destruction/use; specify a stronger API contract if needed. |
| Cooperative cancellation | Atomic signal checks throughout evaluators | Low | Confirmed behaviour | Kernels must check signals during long work; do not assume forced stop. |
| Error information exposure | C API returns status only; C++ traces include controlled reasons | Low | Confirmed behaviour | Avoid adding secrets to traces or diagnostics. |
| Network exposure | No sockets, HTTP clients, or server entry points found | Low | Missing Information | Preserve local-only design unless an authenticated service boundary is added. |
| Secrets | No likely secret file names or common secret patterns found in the scanned repository files | Low | Missing Information | Add CI secret scanning; do not commit credentials. |
| Dependency version pinning | No lockfile or explicit Boost version | Medium | Likely Risk | Document tested compiler/Boost matrix or pin dependencies in a supported package workflow. |

No secrets are exposed by this guide. The secret finding is limited to a
repository scan result and does not establish that no secret can ever exist.

## 18. Common Development Tasks

### 1. Add a formula operation

| Item | Guidance |
| --- | --- |
| Affected files | `formula.hpp`, `src/core/formula.cpp`, cache identity/key handling if required, engine tests, C API if exposed |
| Work | Add enum member, canonical identity/hash rules, validation, every evaluation mode, budget estimates, and traces |
| Tests | valid/invalid specs, all modes, overflow/budget/cancellation/cache tests |
| Risks | Breaking canonical identities, cache collisions, underestimating Exact resource cost |
| Docs | README scope, this guide, C API docs, release notes if public API changes |

### 2. Add a custom kernel

| Item | Guidance |
| --- | --- |
| Affected files | New kernel header/source or consumer project; optional examples/tests |
| Work | Implement `TechniqueKernel`, validate canonical inputs, return traces, check cancellation/budgets |
| Tests | direct kernel tests, engine registration, cache, cancellation, installed consumer |
| Risks | Non-deterministic output, malformed parameters, uncancellable long work |
| Docs | `docs/kernel-sdk.en.md`, example or API note |

### 3. Add a C API function

| Item | Guidance |
| --- | --- |
| Affected files | `c_api.h`, `src/c_api.cpp`, C API docs, C tests |
| Work | Define C-safe structs/handles/status behaviour and exception boundary |
| Tests | header smoke, ownership/null/invalid input/cancellation tests |
| Risks | ABI compatibility, borrowed-vs-owned memory confusion, race-prone lifetime |
| Docs | C ABI lifecycle document and release notes |

### 4. Add a CLI option

| Item | Guidance |
| --- | --- |
| Affected files | `tools/gguf_inspect/command.cpp`, GGUF tests, CLI documentation |
| Work | Parse strictly, validate bounds, map failure to a stable exit code |
| Tests | missing value, invalid value, conflict/required-option behaviour, output |
| Risks | unbounded resource use or unsafe caller-provided file paths |
| Docs | `docs/gguf-inspector.en.md`, usage text |

### 5. Add a snapshot field

| Item | Guidance |
| --- | --- |
| Affected files | Store model, snapshot save/load source, tests, version docs |
| Work | Decide whether format version must change; validate canonical encoding; preserve fail-before-replace semantics |
| Tests | round trip, old format compatibility, malformed/trailing/oversized input, write failure |
| Risks | Breaking old readers, silent defaulting, checksum ambiguity |
| Docs | Snapshot sections and release compatibility note |

### 6. Add a GGUF validation rule

| Item | Guidance |
| --- | --- |
| Affected files | `gguf_inspector.hpp`, `src/gguf/reader.cpp`, GGUF tests/docs |
| Work | Add an explicit error category and fail-closed diagnostic/trace |
| Tests | valid boundary, invalid input, arithmetic/resource boundary |
| Risks | accepting malformed data or rejecting valid V3 files accidentally |
| Docs | Supported-layout/limits section |

### 7. Add a test

| Item | Guidance |
| --- | --- |
| Affected files | relevant `tests/*.cpp`, possibly `CMakeLists.txt` |
| Work | Use repository test helpers; create a new executable only when needed |
| Tests | Run targeted CTest then affected preset |
| Risks | Race tests that depend on timing rather than provided barriers |
| Docs | Update contracts/examples when behaviour changes |

### 8. Update package versioning and release notes

| Item | Guidance |
| --- | --- |
| Affected files | `CMakeLists.txt`, release notes, README, package consumer tests |
| Work | Apply the pre-1.0 minor/patch ABI policy; review `SameMinorVersion` package compatibility |
| Tests | Debug/Release/sanitizer suites; stage-install and external consumer tests |
| Risks | Shipping C++ ABI changes without a minor bump; stale consumer constraints |
| Docs | `docs/release-v0.2.en.md` successor and compatibility matrix |

## 19. Architecture Diagrams

### Overall system architecture

```text
C++ consumer / C consumer / optional GGUF CLI
                  |
                  +------------------+
                  |                  |
                  v                  v
              Engine             GGUF modules
        registry/cache/cancel   reader -> planner -> report
          |            |
          v            v
     Formula core   TechniqueKernel
          |
          +--> Overrides / snapshots
          +--> Range / streaming CPU paths
```

### Evaluation request lifecycle

```text
FormulaSpec + FormulaAddress + Mode + Budget
                    |
                    v
        Engine registration/cancellation check
                    |
           Exact/Metric cache lookup
              |             |
            hit            miss
              |             |
              +------> core or kernel evaluation
                              |
                   budget + trace enforcement
                              |
                      optional cache admission
                              |
                              v
                     EvaluationResult + Trace
```

### Module dependencies

```text
public headers
  ├── Engine ──> cache, core, range scheduler
  ├── OverrideStore ──> Engine
  ├── KEVIN modern ──> DomainProjection + FormulaEvaluator
  ├── KEVIN legacy ──> Engine technique path
  ├── C API ──> Engine
  └── GGUF CLI ──> GGUF reader/planner/scan/report
```

### KEVIN projection flow

```text
GridDomainSpec + GridAddress
             |
             v
      DomainProjection
             |
             v
       FormulaAddress
             |
             v
      FormulaEvaluator
             |
             v
  Metric/Modular tile mask -> Exact survivors
```

### GGUF inspection flow

```text
local GGUF V3 path
       |
       v
file-size + bounded-header/metadata/tensor validation
       |
       v
GgufInventory
       |
       +--> storage policy + RAM budget --> GgufStoragePlan
       |
       +--> selected plan chunks + scan budget --> pattern candidates
       |
       v
stable text report and optional JSON plan
```

### Build/package-consumer flow

```text
CMake preset/configure
        |
        v
bruce_lhme_core static library
        |
        v
cmake --install
        |
        v
bruce_lhmeConfig.cmake + exported target
        |
        v
consumer: find_package(bruce_lhme 0.2 CONFIG REQUIRED)
        |
        v
target_link_libraries(... bruce_lhme::core)
```

## 20. Maintenance Assessment

| Area | Score | Evidence-based justification |
| --- | ---: | --- |
| Modularity | 4/5 | Core, cache, CPU, KEVIN, GGUF, C adapter, and overrides have clear source boundaries. |
| Coupling | 3/5 | `Engine` is deliberately central; public KEVIN filtering mixes modern and legacy overloads. |
| Cohesion | 4/5 | Core formula/cache/reader modules have focused responsibilities; `engine.cpp` concentrates complex cancellation logic. |
| Testability | 5/5 | 44 CTest programs, explicit barriers for race tests, external package consumers, snapshot failure tests. |
| Documentation | 4/5 | Architecture, C ABI, SDK, KEVIN, GGUF, benchmarks, and release notes are detailed; the historical GGUF build command was inconsistent. |
| Error handling | 4/5 | Structured statuses/traces, input validation, budgets, and C ABI exception boundary are consistent; logging is intentionally absent. |
| Security | 4/5 | Local-only design, bounded parsers, strict validation, no secrets found; snapshots lack cryptographic authenticity. |
| Performance awareness | 5/5 | Lazy computation, exact/metric cache, resource budgets, tiling, retention limits, and benchmark suites are first-class concerns. |
| Extensibility | 4/5 | C++ kernel SDK, projection/evaluator interfaces, package consumers, and explicit bindings support extension; runtime plugins are not supported. |
| Observability | 3/5 | Deterministic structured traces and statistics exist, but there is no configurable logging, metrics exporter, or tracing backend. |
| Reproducibility | 4/5 | Presets, deterministic ordering, tests, staged package consumers, and CI improve reproducibility; exact external toolchain versions are not pinned. |

Overall, BRUCE LHME is a well-tested, deliberately bounded C++ library with a
strong deterministic evaluation core. The highest-value maintenance work is to
keep the public API/ABI policy disciplined, test optional tool builds in CI,
maintain parser hardening, and keep modern KEVIN paths distinct from legacy
compatibility code.
