# BRUCE LHME Benchmark Report — 2026-11-11

## 1. Executive Summary

This report records a fresh local Release-build measurement of BRUCE LHME
v0.2.0. The checkout supplied no usable Git metadata, so an exact source
revision is **not determinable from the execution environment**. Results apply
to this checkout and machine only; they are not a portable performance promise.

Measured: a deterministic CSV harness executed 38 validated cases with one
warm-up and 11 timed samples each (418 raw measurements), and the existing
storage, streaming, KEVIN, hybrid-storage, and stress executables were run.
The canonical raw data is
[`benchmark-raw-2026-11-11-v3.csv`](benchmark-raw-2026-11-11-v3.csv).

- The registered `PowerOffset` request at index 1,024 had a 2,601.660 ns/op
  Exact-cache-hit median and a 3,745.016 ns/op distinct-address-miss median.
  These are different workloads and remain separate.
- A 512-address materialized `Engine::evaluate_range` request had medians of
  609.134 µs with one worker and 398.783 µs with two workers. The 1,024-address
  SummaryOnly streaming request did not show a median improvement at two
  workers and had much higher variation.
- Modern KEVIN filtering validated 4,096 checked cells, 512 survivors, and
  zero Exact materializations for every worker configuration. Legacy
  `KevinMemoryStore` lookup was measured separately.

All 15 focused correctness tests passed, all 38 harness cases validated their
outputs, and every manually run existing benchmark reported `PASS`. This is
evidence for the executed contracts only; it does not prove the project is
defect-free or performance-regression-free.

## 2. Scope and Non-Goals

Measured scope includes direct `PowerOffset` modes; Engine registration and
cache paths; built-in Fibonacci/PowerOffset kernels; range and streaming modes;
formula override retrieval and snapshots; modern and legacy KEVIN paths; and
GGUF V3 inspection, planning, reporting, and scanning. Existing synthetic
storage, isolated peak-RSS, and stress workloads were also executed.

The following were not measured: external projects; downloaded models; GPU,
network, database, or cryptographic workloads; C API throughput; installation
or package-consumer throughput; actual CLI parsing/output-file throughput;
CPU time, hardware counters, energy, or new-harness RSS; invalid-GGUF timing;
and concurrent cancellation races. Sanitized binaries were not used for
timing. The `--full` fair-storage matrix was not run; quick mode covers 1,024
synthetic values, while the table-memory workload covers one million virtual
addresses.

No files matching common private-key, AWS access-key, GitHub-token, or `sk-`
token signatures were found outside generated directories. This is a limited
signature scan, not a complete secret-management audit.

## 3. Repository Version and Environment

| Property | Observed value | Classification |
| --- | --- | --- |
| Project version | `0.2.0` from `CMakeLists.txt` | Derived from build configuration |
| Source revision | Not determinable: `git rev-parse` reported no Git worktree | Environment observation |
| Operating system | Debian GNU/Linux 13, Linux 6.12.116, x86_64 | Measured environment |
| CPU | AMD Ryzen 7 3800X; 8 cores / 16 logical CPUs | Measured environment |
| RAM visible | 65,775,372 KiB | Measured environment |
| Compiler | GCC 14.2.0 (`/usr/bin/c++`) | Measured environment |
| CMake | 3.31.6 | Measured environment |
| Generator | Unix Makefiles | Measured build configuration |
| Build type / sanitizer | Release / `none` | Measured build configuration |
| CPU affinity | logical CPUs `0-15` | Measured environment |
| Governor | `schedutil` via `acpi-cpufreq` | Measured environment |
| Load at capture | 0.94 / 0.74 / 0.61 (1/5/15 min) | Measured environment |

No CPU pinning, governor lock, thermal stabilization, exclusive-machine
reservation, or priority change was applied. The machine was observable under
light load, so variance is expected.

## 4. Build Configuration

The fresh named build directory was `out/build/benchmark-report`.

```bash
cmake -S . -B out/build/benchmark-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBRUCE_BUILD_BENCHMARKS=ON \
  -DBRUCE_BUILD_TOOLS=ON
```

This executed successfully and is confirmed. It enables tests, benchmark
targets, and `bruce_gguf_inspect`; no packages were installed. The project
requires CMake 3.25, C++20, Threads, and Boost headers
(`CMakeLists.txt:3-10,42-45`). Presets also define Debug, AddressSanitizer,
UndefinedBehaviorSanitizer, ThreadSanitizer, and GCC+Ninja variants, but only
the unsanitized Release build was timed here.

## 5. Benchmark Inventory

| Benchmark / demo | Source | Built when | Workload / metrics | Correctness check | Status |
| --- | --- | --- | --- | --- | --- |
| `lhme_end_to_end_benchmark` | `demos/lhme/benchmarks/end_to_end_benchmark.cpp` | testing or benchmark option | 38 API cases; raw CSV and repeat statistics | Per-operation status/value/counter checks | Added for this report; not CTest |
| `lhme_streaming_benchmark` | `demos/lhme/benchmarks/streaming_benchmark.cpp` | same | 31-address eager/lazy/prefilter vs streaming | Counts, retention, ordering | Implemented; CTest |
| `lhme_fair_storage_benchmark` | `demos/lhme/benchmarks/fair/fair_storage_benchmark.cpp` | same | Synthetic modular storage; 11 lookup samples/case, optional CSV | Seeded workload/checksum | Implemented; Linux standalone quick CTest and Linux CSV E2E CTest |
| `lhme_table_memory_benchmark` | `demos/lhme/benchmarks/table_memory/table_memory_benchmark.cpp` | same | Repeated one-million-address isolated construction/RSS, optional CSV | Probe lookup/wire validation per child | Implemented; standalone quick CTest and Linux CSV E2E CTest |
| `lhme_kevin_cpu_benchmark` | `demos/lhme/benchmarks/kevin_cpu/kevin_cpu_benchmark.cpp` | same | Modern KEVIN, repeated 64-tile reductions, optional CSV | Same survivors/order; zero Exact work per sample | Implemented; standalone quick CTest and CSV E2E CTest |
| `lhme_hybrid_storage_benchmark` | `demos/lhme/benchmarks/hybrid_storage/hybrid_storage_benchmark.cpp` | same | Repeated override insertion and snapshot-size observations, optional CSV | Override/fallback/count checks per sample | Implemented; standalone quick CTest and CSV E2E CTest |
| `lhme_controlled_stress` | `demos/lhme/benchmarks/stress/controlled_stress.cpp` | same | Repeated isolated scans and bounded retention, optional CSV | Fixed counters and child report per sample | Implemented; Linux CSV E2E CTest; no standalone benchmark CTest |

The new harness exercises public product APIs. Dense, sorted-pair, and hash
representations in fair/table benchmarks are demonstration-only comparison
implementations, not BRUCE persistence backends. Do not read them as a general
product ranking. On Linux, `BUILD_TESTING=ON` registers five existing benchmark
executables: streaming, fair storage, table memory, KEVIN CPU, and hybrid
storage. On other platforms, the fair-storage executable is excluded from
standalone CTest because its required isolated peak-RSS measurement is Linux-only.

## 6. Methodology and Fairness Controls

The new harness uses `std::chrono::steady_clock`, one unscored warm-up, and 11 timed samples per case.
It writes one normalized value per sample: nanoseconds per operation or registration, or microseconds per request.
Statistics are median, mean, minimum, maximum, and sample standard deviation (`n - 1`).
The raw CSV has 419 lines (header plus 418 measurements) and had SHA-256
`d0dfeb0a3837a300c607f5340addf993cd3d832321dfa49c4d433b0bfa78fb7b` at capture.

Setup is excluded from every timed sample; CSV notes identify case-specific setup
exclusions such as Engine construction or cache fills. Cache hits have a separate
unscored fill; misses use distinct addresses. Budget rejection and pre-cancellation
are separate from success. The CSV value `validation=confirmed` means the harness's
expected status/value/counter check passed; it does not mean every measured product
request returned `Status::Confirmed`. Every timed operation validates an observable
result. The optional post-batch `verify` callback is also inside the timed interval:
`ns/op` values amortize it across their repetitions, while `µs/request` cases include
it once per request. Streaming validates 1,024 checked,
896 filtered, 128 Exact/confirmed, and its selected retention count per sample.
The generated GGUF fixture is local, deterministic, and removed afterwards.

The `-v3` data remains the canonical capture above; no new measurements are
merged into it. After that capture, the existing fair-storage, table-memory,
modern-KEVIN, hybrid-storage, and stress executables gained opt-in `--csv FILE`
output with a fixed schema and mean/median/minimum/maximum/sample-standard-
deviation summaries. Fair storage keeps 11 deterministic lookup samples. Table,
hybrid, and stress default to three independent samples (`--quick` uses one);
modern KEVIN defaults to 11 reductions (`--quick` uses three) after an unscored
warm-up. Linux peak RSS is `getrusage(...).ru_maxrss * 1024` in a child process
and includes runtime/allocator overhead. Throughput below is derived from rounded
median latency and labelled derived.

The post-capture CSV schema, case identifiers, input generation, and validation
checks are deterministic; elapsed-time values are deliberately not expected to
be byte-for-byte repeatable across runs or machines. The CSV writes each value
with nine fixed decimal places so its displayed summaries can be recomputed from
the individual validated observations; each executable also prints that summary.
Under `BUILD_TESTING`, `benchmark_csv_quick_e2e_tests` runs the quick modes in
a CTest-owned directory, checks exact CSV schema/quoting, case and iteration
contracts, confirmed validation, fixed-nine values, and recomputed statistics
against the emitted summaries. Fair storage, table-memory, and stress execution
are included only on Linux because each requires an isolated process/RSS
measurement; other platforms validate modern KEVIN and hybrid storage and emit
an explicit skip notice for those three paths. A second CTest forces
`LINUX_SUPPORTED=0` on Linux to regress that path: it validates only KEVIN and
hybrid CSVs, requires the skip notice, and rejects Fair/Table/Stress output
artifacts before the marker-owned directory is removed.

Future archived `--csv` captures are non-canonical sidecar pairs: a new
`*.capture.json` file generated by `cmake/benchmark_capture_metadata.cmake`
uses schema `bruce.lhme.benchmark-capture`, version `1` (format contract:
[`benchmark-capture-metadata-v1.en.md`](benchmark-capture-metadata-v1.en.md)). It records the CSV
path and SHA-256, UTC capture time, build configuration and selected CMake
options, compiler/CMake/generator details, operating-system/CPU/RAM/affinity
observations where available, iteration and warm-up protocol, platform limits,
and a revision identifier or the literal `not available`. Passing a configured
`CMakeCache.txt` lets the script collect the relevant BRUCE options and C++
toolchain details without a new dependency. The script refuses an existing
sidecar and refuses the canonical `benchmark-raw-2026-11-11-v3.csv`, so it
cannot reclassify or overwrite the canonical evidence. `benchmark_capture_metadata_tests`
uses a deterministic fixture to validate this schema, SHA reference, explicit
missing-revision representation, platform label, and no-overwrite behavior.

## 7. Workloads and Input Data

| Area | Deterministic input | Requested output / limit | Classification |
| --- | --- | --- | --- |
| Core formula | `PowerOffset(2,100,1)` at 32, 4,096, 1,000,000 | Exact, Metric, Modular 65,537, Symbolic, budget/cancel | Measured |
| Engine/cache | index 1,024 or 64 distinct indexes; capacities 0/1/8/256 | Miss/hit/eviction/cross-API/budget | Measured |
| Kernels | Fibonacci 2,048; PowerOffset 1,024 | Exact or unregistered status | Measured |
| Range / streaming | Exact `0..511` or `0..1023`; tile 64 or 4 | Range, SummaryOnly, FilteredCandidates, TopK, Adaptive | Measured |
| Overrides | `-42` at 1,024; 100 snapshot records | Formula/technique/fallback/save-load | Measured |
| Modern KEVIN | 64 parent tiles × 64 cells | modulus 17; bits 7..5,000; filter only | Existing benchmark |
| Legacy KEVIN | depth-1 PowerOffset grid plus override | computed or override retrieval | Measured |
| GGUF | generated V3 F32, 4,096 elements, 16 KiB | inspect/plan/reports/bounded scan | Measured |
| Storage/RSS | one million virtual addresses | construction, bytes, child RSS | Existing benchmark |

All data is synthetic and local. The GGUF fixture covers parser, plan, report, and scan behavior.
It does not model real-model I/O, quantized layouts, or inference.

## 8. Correctness Verification

Executed focused Release verification:

```bash
ctest --test-dir out/build/benchmark-report \
  -R '^(core_formula_tests|engine_tests|cache_tests|exact_cache_equivalence_tests|formula_override_store_tests|formula_override_snapshot_tests|streaming_tests|kevin_domain_tests|kevin_filter_tests|kevin_memory_store_tests|gguf_reader_tests|gguf_storage_plan_tests|gguf_pattern_scan_tests|gguf_report_tests|benchmark_statistics_tests)$' \
  --output-on-failure
```

Result: **15/15 passed, 0 failed, 0 skipped** in 0.09 s. The final Debug smoke run completed
38 validated cases with seven timed samples and one warm-up. The final Release run completed
38/38 cases with 11 timed samples and one warm-up.

`lhme_streaming_benchmark`, `lhme_fair_storage_benchmark --quick`,
`lhme_table_memory_benchmark`, `lhme_kevin_cpu_benchmark`,
`lhme_hybrid_storage_benchmark`, and `lhme_controlled_stress` each returned exit code 0 and `PASS`.

The complete CTest suite, a full fair-storage matrix, sanitizer suites, and a cross-platform build
were not rerun for this report.

## 9. Results

Full raw values for the new harness are in
[`benchmark-raw-2026-11-11-v3.csv`](benchmark-raw-2026-11-11-v3.csv).
All statistics in Sections 10–15 are recomputed from that file.

| Result family | Samples | Primary unit | Validation outcome | Interpretation boundary |
| --- | ---: | --- | --- | --- |
| Core/engine/override/legacy KEVIN | 11/case | ns/op | Expected value or status | Microbenchmark only |
| Range/streaming/GGUF | 11/case | µs/request | Counters, retention, fixture outputs | Bounded inputs |
| Fair storage quick | 11 lookup samples/case | µs/257 lookups | Workload/checksum/RSS | Synthetic 1,024-domain |
| Table/hybrid/stress | 1/configuration | ms or bytes | Executable `PASS` | No repeat distribution |
| Modern KEVIN CPU | 4 internal reductions/configuration | µs/reduction | Deterministic survivors | One reported average |

No result families are collapsed into a single ranking. Cold/hot, materializing/non-materializing,
and success/error paths are intentionally kept apart.

## 10. Core Formula Evaluation Analysis

All rows are direct `evaluate(...)` calls, not `Engine` calls. Successful rows validate their
result shape; the final two validate the intended non-success status.

| Workload | Input | Iterations | Median (ns/op) | Mean (ns/op) | Min (ns/op) | Max (ns/op) | Std. Dev. (ns/op) | Validation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exact small | index 32 | 11 | 241.797 | 246.440 | 241.723 | 269.297 | 9.748 | Exact integer |
| Exact medium | index 4,096 | 11 | 1,961.875 | 1,960.244 | 1,952.500 | 1,964.375 | 3.369 | Large Exact integer |
| Metric large | index 1,000,000 | 11 | 132.070 | 132.050 | 131.992 | 132.074 | 0.027 | Bit length present |
| Modular large | index 1,000,000; modulus 65,537 | 11 | 637.660 | 642.334 | 633.012 | 654.496 | 7.066 | Result in `[0, modulus)` |
| Symbolic large | index 1,000,000 | 11 | 291.328 | 290.872 | 289.496 | 291.371 | 0.662 | Nonempty expression |
| Budget rejection | index 4,096; max bits 1 | 11 | 199.531 | 215.726 | 198.598 | 378.598 | 54.019 | `BudgetExceeded` |
| Pre-cancelled | index 4,096 | 11 | 148.750 | 148.733 | 147.422 | 151.250 | 1.307 | `Cancelled` |

The rejection/cancellation rows are controlled early exits. They are not equivalent work to Exact success
and do not establish the cost of concurrent cancellation or arbitrary budget enforcement.

## 11. Engine and Cache Analysis

Engine construction is excluded from registration rows. Cache-hit rows include an unscored fill;
distinct-index miss rows validate 64 resident entries, while eviction validates one entry.

| Registration workload | Iterations | Median (ns/registration) | Mean (ns/registration) | Min (ns/registration) | Max (ns/registration) | Std. Dev. (ns/registration) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 distinct FormulaSpecs/sample | 11 | 3,245.656 | 3,245.219 | 3,231.562 | 3,271.875 | 10.699 |
| 32 distinct identity-only TechniqueKernels/sample | 11 | 248.469 | 253.128 | 247.500 | 275.938 | 8.513 |

| Workload | Variant | Iterations | Median (ns/op) | Mean (ns/op) | Min (ns/op) | Max (ns/op) | Std. Dev. (ns/op) | Validation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exact formula | 64 distinct misses | 11 | 3,745.016 | 3,769.216 | 3,731.594 | 3,857.828 | 42.280 | Confirmed; 64 entries |
| Metric formula | 64 distinct misses | 11 | 3,348.312 | 3,353.844 | 3,337.688 | 3,420.797 | 23.079 | Confirmed; 64 entries |
| Exact formula | Same-address hit | 11 | 2,601.660 | 2,620.629 | 2,592.596 | 2,815.742 | 65.235 | Confirmed; 1 entry |
| Metric formula | Same-address hit | 11 | 2,497.906 | 2,504.866 | 2,485.406 | 2,599.373 | 32.178 | Confirmed; 1 entry |
| Exact formula | Cache disabled | 11 | 3,198.375 | 3,203.040 | 3,190.492 | 3,269.320 | 22.309 | Confirmed; 0 entries |
| Exact formula | One-entry eviction churn | 11 | 3,658.895 | 3,739.784 | 3,621.512 | 4,290.770 | 202.361 | Confirmed; 1 entry |
| Formula → Technique | Explicit cross-API Exact hit | 11 | 3,056.018 | 3,054.732 | 3,037.014 | 3,086.662 | 13.240 | Confirmed; shared entry |
| Cached Exact | Stricter caller budget | 11 | 2,809.434 | 2,808.933 | 2,792.754 | 2,827.051 | 11.012 | `BudgetExceeded` |

On this input, Exact hit median was 1,143.356 ns/op below Exact miss median. This is a
cold-versus-hot observation, not a claim that the paths do equivalent work.

| Kernel path | Iterations | Median (ns/op) | Mean (ns/op) | Min (ns/op) | Max (ns/op) | Std. Dev. (ns/op) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FibonacciKernel` Exact, index 2,048 | 11 | 3,057.516 | 3,068.644 | 3,039.078 | 3,172.219 | 38.124 | Confirmed integer |
| `PowerOffsetKernel` Exact, index 1,024 | 11 | 4,794.797 | 4,791.795 | 4,748.234 | 4,864.258 | 41.976 | Confirmed integer |
| Unregistered PowerOffset technique | 11 | 236.055 | 236.141 | 233.750 | 238.984 | 1.448 | `InvalidArgument` |

Code-derived behavior (`src/engine.cpp`, `src/cache/result_cache.cpp`): only Exact and Metric
are cacheable; admission requires a confirmed result with a trace; zero limits disable admission;
LRU eviction enforces entry/byte bounds; and Formula/Technique Exact equivalence is explicit opt-in.

## 12. Range and Streaming Analysis

`Engine::evaluate_range` materializes all successful candidates. Throughput is derived as
address count divided by rounded median latency, not separately timed.

| Workload | Workers | Checked / retained | Iterations | Median (µs/request) | Mean (µs/request) | Min (µs/request) | Max (µs/request) | Std. Dev. (µs/request) | Derived addresses/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact range `0..511` | 1 | 512 / 512 | 11 | 609.134 | 609.815 | 583.694 | 652.584 | 22.174 | 840,538 |
| Exact range `0..511` | 2 | 512 / 512 | 11 | 398.783 | 406.201 | 394.133 | 433.033 | 13.505 | 1,283,906 |

Streaming validates 1,024 checked inputs. The modular filter has 128 Exact survivors and
896 misses; FilteredCandidates retains four values and TopK/Adaptive retain three.

| Streaming variant | Workers | Exact / filtered / retained | Iterations | Median (µs/request) | Mean (µs/request) | Min (µs/request) | Max (µs/request) | Std. Dev. (µs/request) | Derived scan addresses/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SummaryOnly + AcceptAll | 1 | 1,024 / 0 / 0 | 11 | 1,841.653 | 1,845.321 | 1,825.762 | 1,865.592 | 12.770 | 556,022 |
| SummaryOnly + AcceptAll | 2 | 1,024 / 0 / 0 | 11 | 1,842.772 | 1,647.983 | 1,016.048 | 2,128.564 | 443.048 | 555,685 |
| FilteredCandidates + ModularEquals | 1 | 128 / 896 / 4 | 11 | 3,506.024 | 3,444.012 | 2,770.319 | 3,559.925 | 224.603 | 292,069 |
| TopK + ModularEquals | 1 | 128 / 896 / 3 | 11 | 4,412.021 | 4,296.356 | 3,396.674 | 5,526.818 | 755.399 | 232,093 |
| Adaptive + ModularEquals | 1 | 128 / 896 / 3 | 11 | 3,515.855 | 3,358.264 | 2,583.688 | 3,571.065 | 355.318 | 291,252 |

The two-worker range median is lower for this small request. The two-worker streaming SummaryOnly
result is much noisier and has essentially equal median. The data proves the fixed work reduction,
not general parallel scaling. Retention bytes are an internal estimate; no process RSS is inferred.

## 13. Override and Snapshot Analysis

| Workload | Iterations | Median (ns/op) | Mean (ns/op) | Min (ns/op) | Max (ns/op) | Std. Dev. (ns/op) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FormulaBinding sparse override hit | 11 | 1,402.568 | 1,402.293 | 1,390.107 | 1,424.951 | 10.881 | Source `Override`, value `-42` |
| FormulaBinding computed Metric fallback | 11 | 3,810.062 | 3,806.099 | 3,762.957 | 3,856.977 | 33.522 | Source `Computed`, metric present |
| TechniqueBinding sparse override hit | 11 | 366.740 | 369.490 | 364.748 | 393.676 | 8.202 | Source `Override`, value `-42` |

| Snapshot workload | Iterations | Median (µs/request) | Mean (µs/request) | Min (µs/request) | Max (µs/request) | Std. Dev. (µs/request) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Save then load 100 FormulaBinding overrides | 11 | 389.493 | 391.685 | 381.862 | 418.203 | 10.477 | Load confirmed; count 100 |

The timing includes save and load, excludes construction of the source store, and deletes the temporary file.
It does not record the 100-record file size and therefore makes no RAM or size claim.

Existing hybrid output records current ASCII snapshot disk size, not RAM:

| Override share | Records | Snapshot size (bytes) | Insert time (ms, single run) |
| --- | ---: | ---: | ---: |
| 0% | 0 | 56 | 0 |
| 0.1% | 1,000 | 27,848 | 1 |
| 1% | 10,000 | 278,013 | 16 |
| 10% | 100,000 | 2,779,671 | 158 |

Code-derived: Formula and Technique bindings have different identities and overrides take precedence.
`choose_storage_strategy(...)` selects FormulaOnly, FormulaWithOverrides, DenseTable, or HashTable;
its selection latency was not measured.

## 14. KEVIN Analysis

Modern and legacy KEVIN are separate paths. The modern benchmark uses `GridDomainSpec`,
`DepthPathProjection`, and an Engine-backed `FormulaEvaluator`; it performs filter-only work.

| Modern filter | Internal reductions averaged | Checked cells | Survivors | Exact evaluations | Observed time (µs/reduction) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 worker | 4 | 4,096 | 512 | 0 | 29,718 | Deterministic reference |
| 2 workers | 4 | 4,096 | 512 | 0 | 23,891 | Same candidates |
| 4 workers | 4 | 4,096 | 512 | 0 | 23,851 | Same candidates |
| 8 workers | 4 | 4,096 | 512 | 0 | 25,452 | Same candidates |

The executable uses bounded `std::async` batches. These are one reported averages per worker count,
not repeat distributions or a stable scalability curve.

| Legacy `KevinMemoryStore::get` | Iterations | Median (ns/op) | Mean (ns/op) | Min (ns/op) | Max (ns/op) | Std. Dev. (ns/op) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Registered-grid computed PowerOffset fallback | 11 | 4,638.156 | 4,649.236 | 4,606.672 | 4,705.500 | 31.658 | Source `Computed` |
| Registered-grid sparse override hit | 11 | 399.242 | 396.421 | 384.477 | 407.775 | 8.312 | Source `Override`, `-42` |

Do not compare legacy lookup medians to modern filter timings: one returns a compatibility-store value,
the other filters 4,096 coordinates without Exact materialization. New code should prefer modern
`GridDomainSpec` + `DomainProjection` + `FormulaOverrideStore` composition.

## 15. GGUF Inspector Analysis

The fixture is local GGUF V3, one F32 tensor, and a 16 KiB payload. It is not a downloaded model.

| Workload | Iterations | Median (µs/request) | Mean (µs/request) | Min (µs/request) | Max (µs/request) | Std. Dev. (µs/request) | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Valid V3 inspection | 11 | 55.440 | 55.803 | 55.140 | 57.710 | 0.934 | One inventory tensor |
| Budgeted storage plan | 11 | 1.300 | 1.430 | 1.220 | 2.820 | 0.465 | Four chunks |
| Text report rendering | 11 | 2.170 | 2.195 | 2.130 | 2.450 | 0.089 | Nonempty report |
| JSON plan rendering | 11 | 4.800 | 4.815 | 4.740 | 5.050 | 0.083 | Input fingerprint field |
| Bounded constant-byte scan | 11 | 44.060 | 47.659 | 43.600 | 79.191 | 10.604 | At least one candidate |

`src/gguf/reader.cpp` accepts V3 only and checks supported layouts, alignment, offsets, counts, and limits.
The scan row is a bounded successful path; invalid-input rejection timing is intentionally not primary.
The CLI was built but CLI parsing/output-file I/O was not timed. Scanner tests also verify read-only input behavior.

## 16. Memory and Resource-Budget Analysis

The existing table-memory benchmark creates a fresh Linux child for each representation.
Peak RSS includes executable, allocator, and runtime overhead; container bytes are not RSS.

| Representation | Overrides | Build time (ms, single run) | Container bytes | Isolated peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Dense vector | 0 | 3 | 8,000,000 | 11,202,560 |
| Sorted pairs | 0 | 4 | 16,000,000 | 19,214,336 |
| Hash table | 0 | 44 | 24,450,584 | 43,618,304 |
| Formula only | 0 | 0 | 144 | 3,407,872 |
| Formula + overrides | 0 | 0 | 624 | 3,497,984 |
| Formula + overrides | 10,000 (1%) | 16 | 624 + 80,000 metadata | 6,672,384 |
| Formula + overrides | 100,000 (10%) | 165 | 624 + 800,000 metadata | 35,241,984 |
| Formula + overrides | 1,000,000 (100%) | 1,676 | 624 + 8,000,000 metadata | 323,117,056 |

This is scoped evidence that sparse formula representation used less observed process memory than fully
materialized synthetic tables on this machine, while BigInt override density changes the result substantially.
It is not a general RAM claim or a semantic equivalence proof for arbitrary data.

| Stress scenario | Workers | Time (ms, single run) | Peak RSS (bytes) | Checked | Exact | Filtered | Retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Million scan | 1 | 709 | 3,371,008 | 1,000,001 | 0 | 1,000,001 | 0 |
| Million scan | 8 | 149 | 3,346,432 | 1,000,001 | 0 | 1,000,001 | 0 |
| Adaptive hits | scenario-defined | 76 | 3,489,792 | 10,001 | 1,250 | 8,751 | 3 |

The million scans demonstrate bounded retention for all-filtered input; single timings do not prove speedup.
Budget enforcement is also covered by core/cache error paths, GGUF limits, and focused snapshot tests.

## 17. Parallelism and Determinism Analysis

For the 512-candidate range, observed median latency was 609.134 µs with one worker and 398.783 µs
with two, about 1.53× by rounded medians; both returned 512 candidates. SummaryOnly streaming medians
were 1,841.653 µs and 1,842.772 µs for one/two workers, while the two-worker standard deviation was
443.048 µs. That does not support a streaming parallel-speedup claim.

Modern KEVIN produced the same 512 survivors and candidate order at 1/2/4/8 workers with zero Exact work.
Streaming sorts completed tiles by tile ID before merge, and KEVIN CPU tests compare parallel to stable
sequential reduction. These are determinism evidence, not timing evidence. Cancellation measurement was
limited to safe pre-cancelled paths; cancellation-group reuse is tested but was not timed as a race workload.

## 18. Existing Benchmark Comparison

The 2026-10-21 review is historical documentation, not a baseline here: its revision cannot be verified
in this checkout, its date differs, and it contains single-run values. No cross-date speedup/regression is justified.

| Claim | Source | Code / measurement evidence | Status | Action |
| --- | --- | --- | --- | --- |
| Benchmark targets are conditionally available | `CMakeLists.txt:232-469` | Fresh Release build created all targets | Confirmed | Keep conditions documented |
| Four CTest-registered benchmarks ran | 2026-10-21 review | The archived review now labels its four-target result as date-specific; current CMake registers five existing benchmarks, including fair storage | Confirmed | Keep historical scope explicit |
| KEVIN CPU averages with a persistent worker pool | `kevin-memory-store.en.md` | Documentation now distinguishes the persistent legacy `KevinTileCpuExecutor` from the benchmark's transient bounded `std::async` batches | Confirmed | Keep the paths separate |
| Exact cache equivalence rechecks budget/cancellation | `kernel-sdk.en.md` | Engine code, tests, cross-API and strict-budget rows agree | Confirmed | Preserve coverage |
| KEVIN filtering avoids Exact materialization | README/architecture | Modern benchmark measured zero Exact work | Confirmed | Keep counter visible |
| GGUF scan is read-only | `gguf-inspector.en.md` | Input-stream implementation and read-only tests | Confirmed | Keep regression tests |
| Historical memory/speed applies to this checkout | 2026-10-21 review | No revision evidence and a different run date | Not Verifiable | Treat as archive only |

The `-v3` CSV is canonical. Unversioned and `-v2` CSV outputs are retained preliminary data because
the harness refuses overwriting existing raw output; they are not inputs to this report.

## 19. Limitations and Threats to Validity

- Source revision is unavailable. A source archive or future revision identifier is needed for exact replay.
- `schedutil`, unrestricted 16-CPU affinity, background load, cache warmth, temperature, and allocator state affect results.
- The canonical `-v3` report still contains the earlier single-observation or
  internally averaged existing-benchmark values. New `--csv` runs are separate
  artifacts and must not be compared with or substituted for the canonical data.
- Small ns-level operations include API, trace, locking, and timer noise. The budget-rejection maximum was 378.598 ns/op,
  so its median is more robust than its mean.
- Range/streaming inputs are deliberately bounded; larger ranges, tiles, retention, and cache matrices remain unmeasured.
- A fixed-affinity Range/Streaming or persistent-KEVIN executor benchmark was
  not added. `Engine::evaluate_range` has no affinity setting, and
  `evaluate_streaming_range` constructs a request-local `CpuWorkerPool`
  (`src/cpu/streaming.cpp`), so it exposes no reusable streaming executor.
  `KevinTileCpuExecutor` is persistent, but it accepts the legacy
  `KevinMemoryStore` rather than the modern `DomainEvaluationContext` used by
  `lhme_kevin_cpu_benchmark`. The repository provides no cross-platform
  affinity abstraction. Adding platform APIs or a new persistent modern executor
  would require production changes and would not be a comparable extension of
  the existing measurement.
- Fair/table comparisons use synthetic modular `UInt64` values while formula overrides can use BigInt values.
- The 16 KiB F32 GGUF fixture does not model large I/O, quantized layouts, filesystem cache effects, or inference.
- Peak RSS is Linux-specific and includes process overhead. Snapshot bytes are on-disk bytes, not heap use.
- Snapshot FNV checksums detect accidental corruption; they are not cryptographic authentication.

## 20. Reproducibility Instructions

Use a clean or clearly identified build directory and a new CSV filename; the harness refuses to overwrite one.

```bash
cmake -S . -B out/build/benchmark-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBRUCE_BUILD_BENCHMARKS=ON \
  -DBRUCE_BUILD_TOOLS=ON

cmake --build out/build/benchmark-report --target \
  lhme_end_to_end_benchmark lhme_streaming_benchmark lhme_controlled_stress \
  lhme_fair_storage_benchmark lhme_table_memory_benchmark \
  lhme_kevin_cpu_benchmark lhme_hybrid_storage_benchmark \
  core_formula_tests engine_tests cache_tests exact_cache_equivalence_tests \
  formula_override_store_tests formula_override_snapshot_tests streaming_tests \
  kevin_domain_tests kevin_filter_tests kevin_memory_store_tests \
  gguf_reader_tests gguf_storage_plan_tests gguf_pattern_scan_tests \
  gguf_report_tests benchmark_statistics_tests benchmark_output_tests \
  benchmark_csv_e2e_tests canonical_benchmark_report_tests -j2

ctest --test-dir out/build/benchmark-report \
  -R '^(core_formula_tests|engine_tests|cache_tests|exact_cache_equivalence_tests|formula_override_store_tests|formula_override_snapshot_tests|streaming_tests|kevin_domain_tests|kevin_filter_tests|kevin_memory_store_tests|gguf_reader_tests|gguf_storage_plan_tests|gguf_pattern_scan_tests|gguf_report_tests|benchmark_statistics_tests|benchmark_output_tests|benchmark_csv_quick_e2e_tests|benchmark_csv_forced_non_linux_skip_tests|canonical_benchmark_report_tests|benchmark_capture_metadata_tests|fair_benchmark_docs_test)$' \
  --output-on-failure

./out/build/benchmark-report/lhme_end_to_end_benchmark \
  --csv docs/benchmark-raw-YYYY-MM-DD-local.csv

./out/build/benchmark-report/lhme_streaming_benchmark
./out/build/benchmark-report/lhme_fair_storage_benchmark --quick \
  --csv out/build/benchmark-report/fair-storage-local.csv
./out/build/benchmark-report/lhme_table_memory_benchmark --iterations 3 \
  --csv out/build/benchmark-report/table-memory-local.csv
./out/build/benchmark-report/lhme_kevin_cpu_benchmark --iterations 11 \
  --csv out/build/benchmark-report/kevin-cpu-local.csv
./out/build/benchmark-report/lhme_hybrid_storage_benchmark --iterations 3 \
  --csv out/build/benchmark-report/hybrid-storage-local.csv
./out/build/benchmark-report/lhme_controlled_stress --iterations 3 \
  --csv out/build/benchmark-report/stress-local.csv

cmake \
  -DCAPTURE_CSV=out/build/benchmark-report/fair-storage-local.csv \
  -DCAPTURE_OUTPUT=out/build/benchmark-report/fair-storage-local.capture.json \
  -DCMAKE_CACHE_FILE=out/build/benchmark-report/CMakeCache.txt \
  -DITERATION_PROTOCOL='11 timed lookup samples per case' \
  -DWARMUP_PROTOCOL='1 unscored warm-up before each case' \
  -DPLATFORM_LIMITS_JSON='["Linux-only isolated process/RSS measurement"]' \
  -DREVISION_IDENTIFIER='not available' \
  -P cmake/benchmark_capture_metadata.cmake
```

`--quick` is a correctness smoke mode: fair storage keeps 11 samples, modern
KEVIN uses three, and table, hybrid, and stress use one. A one-sample quick
run reports its standard deviation as unavailable; use the documented default
or `--iterations N` for a statistical summary.
For an archive, retain the raw CSV together with its generated non-canonical
sidecar. Replace `not available` with an immutable source revision identifier
when the source is a VCS checkout; this source archive has none. The sidecar
does not infer CPU governor, system load, or power mode, so record those
separately when they materially affect the comparison. Do not place a
non-canonical sidecar next to the canonical `-v3` CSV.
Do not compare cold/hot or success/rejection rows as equivalent work.

## 21. Conclusions

1. The checked core, Engine, override, legacy KEVIN, modern KEVIN-filter, and GGUF paths are runnable in a fresh
   Release configuration and produced validated results. Evidence: 15 focused tests, six existing executable runs,
   and 38 new benchmark cases.
2. On this machine, cache state is observable in the bounded Engine workload: Exact hit median 2,601.660 ns/op
   versus distinct-address miss median 3,745.016 ns/op. This includes Engine/cache mechanics and is not portable.
3. Resource controls show their behavioral role: modular streaming avoided 896 of 1,024 Exact evaluations,
   modern KEVIN filtered 4,096 cells with zero Exact materializations, and SummaryOnly stress scans retained none.

## 22. Recommended Next Steps

1. Archive future `--csv` outputs with the machine metadata and revision identifier; keep them separate from `-v3`
   until a new controlled report is deliberately published.
2. Do not add a fixed-affinity benchmark without a documented cross-platform affinity abstraction: the repository has
   none, and platform APIs would make such a benchmark non-portable.
3. Add larger controlled Range/Streaming matrices only after defining a stable workload and validation contract. The
   current public APIs construct their worker pools per request, so there is no reusable streaming executor to measure.
4. Measure `KevinTileCpuExecutor` only as a separate legacy-`KevinMemoryStore` workload. It is not semantically
   equivalent to the modern `DomainEvaluationContext` path used by `lhme_kevin_cpu_benchmark`.
5. Add dedicated C API overhead and CLI GGUF I/O measurements before treating those paths as regression thresholds.
