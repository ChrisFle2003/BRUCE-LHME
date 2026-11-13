# BRUCE LHME Benchmark Review — 2026-10-21

## Summary

`FormulaRecipe` is very memory-efficient for large value spaces that are
**regularly computable**. In the measurement, the million-value table required
a process peak of 3.38 MB as a formula base alone, instead of 11.19 MB for a
dense `UInt64` vector or 43.56 MB for a `HashTable`.

The advantage does not apply indiscriminately to every data set: real `BigInt`
overrides are deliberately stored individually. At 10% overrides, their
process peak is already 35.53 MB; at 100%, it is 323.08 MB. Once most values
are exceptions to the formula, a materialized table is the better
representation.

In its current state, KEVIN demonstrates a different property: it is an 8×8
domain and filtering layer, not a value store. The pipeline filtered 4,096
virtual cells with zero Exact materializations and remained deterministic with
1, 2, 4, and 8 workers.

## Fair comparison: Track A

Track A answers only the storage question: how do five representations of the
**same modular scalars** behave? The recipe is
`h(i) = 100 * (2^i - 1) mod M`, with `M = 2,305,843,009,213,693,951`.
`FormulaRecipe`, `FormulaRecipeWithOverrides`, `DenseVector`, `SortedPairs`,
and `HashTable` were measured. All use the same deterministic addresses, the
same seed `284849161029`, and a disabled recipe-cache configuration.

The complete release matrix ran with exit status 0 across domains of 1,024,
65,536, and 1,000,000; the requested override densities of 0%, 0.1%, 1%, 10%,
and 100%; and sequential, pseudo-random, and repeat-heavy access. Each case
has an unscored warm-up run followed by **eleven samples**; the report contains
minimum, median, and maximum. Small domains cannot represent requested
densities exactly as integers: the output then gives the actual share, for
example 1 of 1,024 as 0.0977%.

Selected raw release values for the 1,024-address sequential case:

| Representation | Construction | Lookup min / median / max | Isolated peak RAM |
|---|---:|---:|---:|
| FormulaRecipe | 16 µs | 186 / 186 / 236 µs | 1.482.752 B |
| FormulaRecipeWithOverrides, 0 % | 3 µs | 481 / 495 / 726 µs | 1.470.464 B |
| DenseVector, 0 % | 635 µs | 4 / 4 / 4 µs | 1.290.240 B |
| SortedPairs, 0 % | 441 µs | 4 / 4 / 7 µs | 1.298.432 B |
| HashTable, 0 % | 597 µs | 2 / 2 / 2 µs | 1.515.520 B |

This is **not a global ranking**: FormulaRecipe avoids materialization and is
therefore a storage representation for regular data. Dense and hash
representations pay construction and RAM costs but receive direct `UInt64`
access in return. With many exceptions, the decision must therefore depend on
the data shape and access pattern.

## Fair comparison: Track B

Track B is an archived measurement of a bounded publicly generated BSGS
teaching example with `p = 1019`, generator `2`, and known exponent `777`. It
measures a linear reference scan and Baby-Step Giant-Step with eleven runs each:

| Solver | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Linear scan | 2 µs | 2 µs | 2 µs |
| Baby-Step Giant-Step | 3 µs | 3 µs | 9 µs |

The result is not comparable with Track A (`not comparable to Track A`): it
concerns a different algebraic task and a tiny fixed group, not storing one
million formula values. No passwords, secrets, or production cryptographic
parameters are processed. The associated executable toy implementation is not
shipped; this section preserves only the result and its security boundaries.

## Fair comparison: Track C

Track C is `PrecomputedChainToy`: a **synthetic non-security toy** over a
public 16-bit state space. It is explicitly not a rainbow table and is not
intended for security or credential applications.

| Parameter / result | Value |
|---|---:|
| Deterministic chains | 128 |
| Chain length | 64 |
| Stored endpoints | 256 B |
| Observed coverage | 128 / 128 generated tokens |
| Verification work | 1.067.790 steps |
| Endpoint candidates | 142 |
| Replay-verified resolutions | 128 |

An identical endpoint is only a candidate. Only replaying the corresponding
origin chain may confirm a resolution. Track C is also `not comparable to
Track A`: it does not assess a RAM representation of the same values, but
verification of a synthetic chain model. The executable toy implementation is
not shipped; only this result is retained.

## Limitations

- The complete matrix is a single release run on this host. It does not
  replace a distribution analysis across CPU clock, temperature, allocator, or
  multiple machines.
- The documented raw values for 1,024 are a selected readable view; they are
  not a summary or overall ranking.
- Isolated peak RAM is a Linux child-process value and includes runtime and
  allocator overhead. On unsupported platforms, the runner reports
  `unavailable`.
- Track B and Track C are only bounded public learning experiments. Their
  timings or hit counts imply nothing about cryptography, password searching,
  or real attacks.

## Measurement setup

| Property | Value |
|---|---|
| Date | 2026-10-21 |
| Source revision | `630bdf26bbbf4af66928a13327451f9be9ae63c5` |
| Build | CMake release preset |
| Compiler | GNU C++ 14.2.0 |
| Operating system | Linux 6.12.113+deb13-amd64, x86_64 |
| CPU | AMD Ryzen 7 3800X, 16 logical CPUs visible |
| Main memory | 65,775,380 KiB visible |
| Process peak | Isolated Linux child process; includes allocator and runtime overhead |

The times are single local measurements with `std::chrono::steady_clock`.
They are useful for comparisons on this machine but are not a portable
performance promise. There was no CPU pinning, frequency, temperature, or
repeat analysis. The KEVIN test averages four reductions per worker
configuration; the other timing values are single runs.

## Verification

At the time of this historical capture, four CTest-registered benchmarks were
freshly executed in the release profile and passed: `lhme_streaming_benchmark`,
`lhme_table_memory_benchmark`, `lhme_kevin_cpu_benchmark`, and
`lhme_hybrid_storage_benchmark` (4/4, 2.44 s). In addition, the hybrid test
with `--full-overrides` and `lhme_controlled_stress` were executed directly;
both passed. Current CMake additionally registers `lhme_fair_storage_benchmark`;
this archived count is not a statement about current CI coverage.

## 1. One million virtual values: storage form

All variants represent a virtual space of 1,000,000 addresses. The formula
base is `h(x) = 100 * (2^x - 1)`; it stores the rule, not one million
materialized giant values.

| Representation | Process-peak RAM | Construction time | Analytical container size | Classification |
|---|---:|---:|---:|---|
| Dense `UInt64` table | 11,190,272 B | 2 ms | 8,000,000 B | Compact for fully materialized 64-bit values |
| Sorted pairs | 19,243,008 B | 3 ms | 16,000,000 B | Values plus address metadata |
| `HashTable` | 43,556,864 B | 46 ms | 24,450,584 B | Fast direct access, but high container overhead |
| `FormulaOnly` | 3,379,200 B | 0 ms | 144 B | Recipe instead of table contents |
| Formula + 0 % overrides | 3,432,448 B | 0 ms | 496 B | Empty exception store |
| Formula + 1 % overrides | 6,574,080 B | 18 ms | 496 B + 80,000 B address metadata | Still below dense table |
| Formula + 10 % overrides | 35,528,704 B | 167 ms | 496 B + 800,000 B address metadata | `BigInt` exceptions dominate the actual peak |
| Formula + 100 % overrides | 323,076,096 B | 1,714 ms | 496 B + 8,000,000 B address metadata | No longer a sparse special case |

### What the figures mean

- In this process-measurement run, `FormulaOnly` is about **70% below** the
  dense vector and about **92% below** the `HashTable`.
- With **1% genuine deviations**, the formula-plus-override variant is still
  about **41% below** the dense vector.
- With **10% BigInt deviations**, it is already about **3.2×** as large as the
  dense `UInt64` vector. This is not a formula-base fault, but the expected
  cost of a hash structure addressable across one million values with BigInt
  values.
- With **100% overrides**, the structure effectively contains a table again—
  only with heavier values and additional hash/allocator structure. This is
  when `StoragePolicy` should switch to dense or hash storage.

The analytical container size is only a lower bound. For real RAM decisions,
the isolated measured process peak is more meaningful here.

## 2. Persistence of the exception layer

The hybrid benchmark stores the formula base once and serializes only genuine
exceptions in the current ASCII snapshot format.

| Override share | Overrides | Snapshot size | Insertion time |
|---|---:|---:|---:|
| 0 % | 0 | 56 B | 0 ms |
| 0.1 % | 1,000 | 27,848 B | 1 ms |
| 1 % | 10,000 | 278,013 B | 15 ms |
| 10 % | 100,000 | 2,779,671 B | 154 ms |
| 100 % | 1,000,000 | 27,796,163 B | 1,535 ms |

Here, snapshot size grows approximately linearly with the number of
exceptions. It describes the current text format on disk, **not** RAM
consumption or a general upper bound for future binary serialization.

## 3. KEVIN: domain projection and filtering

The benchmark processes 64 parent tiles × 64 subcells = 4,096 virtual cells.
Every run produced exactly 512 survivors and **0** Exact evaluations.

| Workers | Average time | Speed-up against 1 worker | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 12,968 µs | 1.00× | 100 % |
| 2 | 8,252 µs | 1.57× | 78.6 % |
| 4 | 4,664 µs | 2.78× | 69.5 % |
| 8 | 3,818 µs | 3.40× | 42.5 % |

Decreasing efficiency is plausible for only 4,096 small tasks: creating and
synchronizing asynchronous tasks becomes relatively expensive. The test thus
demonstrates correct parallel reduction and useful acceleration on this host,
but no guarantee of linear scaling. Further measurement series should follow
for larger tile packages or a persistent thread pool.

## 4. Streaming and hierarchical reduction

The small comparison across 31 addresses showed:

| Variant | Exact evaluations | Filtered | Retained | Time |
|---|---:|---:|---:|---:|
| Eager reference | 31 | 0 | 31 | 17 µs |
| Lazy reference | 31 | 0 | 0 | 8 µs |
| Reference with modular prefilter | 4 | 27 | 4 | 12 µs |
| V2 Summary, 1 worker | 31 | 0 | 0 | 219 µs |
| V2 Summary, 4 workers | 31 | 0 | 0 | 252 µs |
| V2 Filter + adaptive retention | 4 | 27 | 3 | 135 µs |

The result primarily shows **work reduction**: the modular filter prevents 27
of 31 Exact evaluations. For this tiny task, the timings must not be read as a
throughput ranking; parallel runtime overhead exceeds the benefit. A real
performance decision requires larger ranges and repeated measurements.

## 5. Isolated million-scan stress test

| Scenario | Time | Process-peak RAM | Checked | Exact | Filtered | Retained |
|---|---:|---:|---:|---:|---:|---:|
| Million scan, 1 worker | 663 ms | 3,497,984 B | 1,000,001 | 0 | 1,000,001 | 0 |
| Million scan, 8 workers | 127 ms | 3,371,008 B | 1,000,001 | 0 | 1,000,001 | 0 |
| Adaptive hits | 76 ms | 3,715,072 B | 10,001 | 1,250 | 8,751 | 3 |

On this host, the eight-worker scan was about **5.2×** faster than the
single-worker scan, with a comparable process peak. The storage contract is
decisive: the million scan filtered out every candidate, performed no Exact
evaluation, and retained no result set in RAM.

## Review: which representation when?

| Data shape and access | Recommended strategy |
|---|---|
| Values stably follow a shared formula, with few exceptions | `FormulaOnly` |
| Formula base with few explicit special values | `FormulaWithOverrides` |
| High exception share, predominantly sequential access | Dense table |
| High exception share, predominantly random access | `HashTable` |
| Hierarchical search, dimensions, or candidate filtering | KEVIN domain plus explicit projection; values still via `FormulaRecipe` |

## Next benchmark steps

1. Repeat every main measurement several times and record median, variation,
   and hardware clock conditions.
2. Measure the `BigInt` override store against a more compact sorted sparse
   representation so the switching threshold becomes data-based instead of
   merely policy-based.
3. Measure larger KEVIN tile packages and a persistent executor; the current
   test deliberately also measures the cost of short-lived asynchronous tasks.
4. Test the storage decision with real production data, not only synthetic
   `UInt64` tables and small override BigInts.
