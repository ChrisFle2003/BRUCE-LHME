# KEVIN Memory Store (Legacy Compatibility)

`KevinMemoryStore` remains available for existing programs, but it is not the
architecture for new data paths. Historically, it couples a grid rule, a
Technique binding, and infrequent overrides. Its semantics and snapshot format
remain unchanged for this release line; an old KEVIN snapshot is never silently
read as a new FormulaOverrideStore snapshot.

New programs separate the roles:

```text
GridDomainSpec + GridAddress + DomainProjection  -> Domain/filter coordinate
FormulaSpec or TechniqueBinding + FormulaAddress -> Value generator
FormulaOverrideStore                              -> Infrequent deviations
```

For migration, replace `GridSpec` with a `GridDomainSpec`, choose an explicit
projection (V1: `DepthPathProjection`), and store the former Technique binding
as a `TechniqueBinding` in `FormulaOverrideStore`. Direct FormulaRecipe
evaluation does not require a KEVIN object.

KEVIN stores a grid rule, 8×8 address, and infrequent overrides, not every
regular BigInt. `get()` lazily forwards ordinary requests to the existing
engine. Deterministic reconstruction requires the grid and kernel versions,
canonical parameters, and the complete path.

```cpp
KevinMemoryStore store{engine};
store.register_grid(spec);
auto value = store.get(spec.identity, address, EvaluationMode::Metric,
                       Budget::unlimited());
store.set_override(spec.identity, address, BigInt{42});
```

The legacy store contains only GridSpecs and overrides; calculated values exist
only in the bounded engine cache. The cache is an accelerator, never the source
of stored truth.

## Snapshot formats and input limits

New KEVIN snapshots use `BRUCE-KEVIN-2`. In addition to identity, Technique,
canonical parameters, and `max_depth`, V2 contains all five fields of
`GridSpec::default_budget`. This keeps a grid reproducible with the same
resource limits after saving and loading.

The loader continues to accept `BRUCE-KEVIN-1`. This older format contained no
budget fields; loading it therefore necessarily uses `Budget::unlimited()`.
An originally different V1 budget cannot be reconstructed.
`save_snapshot()` always writes V2. Older programs cannot read new V2 files and
must remain on an older library version to export to an old program.

Both snapshot loaders limit input files to 16 MiB; they fully validate format,
numbers, the checksum line, and trailing data, and replace the target store
only after complete parsing succeeds. The FNV-based checksum detects
unintentional corruption; it is not cryptographic authentication and does not
protect against an attacker who can change the file and checksum together.

## 64-candidate tile filters (V1.6)

For a parent path, KEVIN considers a single 8×8 tile as exactly 64 possible
child addresses. The bit position is stably defined as `bit = y * 8 + x`: bit
0 represents `(0,0)` and bit 63 represents `(7,7)`.

```text
64 addresses
  -> Modular filter (without BigInt materialization)
  -> Magnitude/bit-length filter (without BigInt materialization)
  -> AND mask of survivors
  -> optional: Exact only for an explicitly bounded number of candidates
```

The mask is local to this tile. A non-matching candidate neither removes other
tiles nor stops the engine. `lhme_kevin_grid_filter_demo` shows the flow and
checks that the filter phase does not trigger an exact calculation.

`filter_tiles(...)` combines several parent tiles in the supplied order. The
64-bit masks and their rejections remain local to each tile; only surviving
addresses are combined in a shared deterministic candidate list.
`evaluate_reduced_exact(...)` then applies one global candidate limit. For
example, a CPU worker can first prefilter many tiles inexpensively and
materialize only the most important survivors.

For CPU parallelism, `filter_tiles_cpu(..., worker_count)` invokes no more
tiles concurrently than workers allow. Results are then always combined in the
original tile order, so the response remains deterministic for the same input.
`worker_count = 0` is invalid. Store tables, the engine registry, cache, and
Exact statistics are protected for parallel read evaluation. Configuration
changes such as grid registration or snapshot replacement deliberately remain
write operations protected exclusively.

For many consecutive jobs, `KevinTileCpuExecutor` uses the same persistent
pool. Workers start when the executor is created and stop only when it is
destroyed. `executor.filter(...)` continues to receive a fixed input order and
returns candidates in exactly that order.

## Comparable benchmarks

`lhme_table_memory_benchmark` compares the storage form of one million
materialized `UInt64` values with a symbolic formula recipe.
`lhme_kevin_cpu_benchmark` separately measures the KEVIN filter pipeline for
64 tiles (4,096 virtual cells) with 1, 2, 4, and 8 CPU workers. It verifies the
same survivors and zero Exact materializations for each worker count and
averages four runs per persistent worker pool. Timings depend on the machine;
the deterministic equality of candidates and counters is the fixed contract.

`lhme_hybrid_storage_benchmark` extends the comparison to a virtual address
space of one million values. It stores the formula base once and adds only real
deviations as BigInt overrides. By default, it measures 0%, 0.1%, 1%, and 10%
overrides against the container size of a one-million-value hash map;
`--full-overrides` additionally enables the 100% countercheck. The reported
snapshot size is the size of the current ASCII snapshot format on disk, not the
universal size of a future binary serialization. It therefore demonstrates the
behavior of this implementation, not a general RAM limit.

`lhme_table_memory_benchmark` also measures peak RAM in isolated Linux child
processes. In addition to the dense table, sorted pairs, and hash map, it
measures `FormulaOnly` and `FormulaWithOverrides` with 0%, 1%, 10%, and 100%
explicit deviations. All variants cover the same virtual space of one million
formula addresses. KEVIN is not a value representation in this comparison.
The process measurement therefore also includes allocator and runtime overhead;
it is more informative for practical RAM decisions than analytical container
sizes. The measurement defines no universal limit because values depend on the
platform, compiler, and data values.

## Automatic storage decision

`choose_storage_strategy(...)` is the public deterministic decision core for
calling code. With `StoragePolicy`, the sparse threshold is configurable in
basis points (default: 500 = 5%). The decision uses no floating-point values:

```text
0 overrides                   -> FormulaOnly
up to the sparse threshold    -> FormulaWithOverrides
above it, sequential access   -> DenseTable
above it, random access       -> HashTable
```

Invalid input—an empty address space, more overrides than addresses, or a
threshold above 100%—is rejected with `Status::InvalidArgument`. The policy
currently selects the target strategy; a future migration layer will move
stored data into this target representation in a controlled manner.
