# Implemented architecture

The current C++ core includes `FormulaSpec`/Technique kernels, BigInt/metric/
modular/symbolic evaluation, a bounded `ResultCache`, CPU streaming, and two
clearly separated layers:

```text
FormulaRecipe + FormulaAddress
  -> FormulaOverrideStore (only real deviations)
  -> Engine
  -> ResultCache

KEVIN GridDomainSpec + GridAddress
  -> explicit DomainProjection
  -> FormulaAddress
  -> FormulaEvaluator
```

The formula is the value generator. KEVIN is solely a hierarchical 8×8 domain,
filtering and process space: the same grid address can point to different
formula addresses through two projections. No value can be derived from a grid
address without a projection. 64-bit tile filters reduce locally by modular
and magnitude metrics; a filter miss discards only the affected candidate. The
cache remains a bounded accelerator, never the source of truth.

Every successful domain-based evaluation augments the trace with
`Domain-ID@Version`, `Projection-ID@Version`, and the reached
`FormulaAddress`. Missing, invalid, or unrepresentable projections are
recorded as a branch-local `InvalidArgument` trace. This keeps visible which
dimension and mapping rule determined a formula value.

`KevinMemoryStore` and its snapshots remain as a documented compatibility
boundary for older callers. New programs use `GridDomainSpec`/
`DomainProjection` plus `FormulaOverrideStore` instead. The new override
snapshot has its own format and does not silently accept old KEVIN snapshots.
It validates format, canonical numbers, checksums, and unexpected trailing
data fail-closed. Snapshot loaders limit input to 16 MiB and replace the
target store only after parsing has completed. The FNV-based checksums verify
integrity against accidental corruption; they are not cryptographic
authentication.

## Stable verification

In addition to the normal CTest suite, the library can be built with a runtime
sanitizer. Configuration deliberately remains explicit so normal release
binaries do not incur debug runtime costs:

```text
cmake -S . -B build-address -DBRUCE_SANITIZER=address
cmake --build build-address -j2
ctest --test-dir build-address --output-on-failure
```

Permitted values are `address`, `thread`, and `undefined`; `none` is the
default. The override store additionally has a parallel read/write stress test
that checks for consistent responses during concurrent override updates.

For reproducible debug, release, and sanitizer builds, identically named CMake
presets are available in `CMakePresets.json`. The active v0.2 release contract,
ABI policy, and externally pending release gates are documented in
[`release-v0.2.en.md`](release-v0.2.en.md); [`release-v0.1.en.md`](release-v0.1.en.md)
is historical.

If a monitored development environment explicitly reports LeakSanitizer as
incompatible with `ptrace`, the AddressSanitizer test run may be repeated
locally with `ASAN_OPTIONS=detect_leaks=0`. This disables only the leak check
that cannot run in that environment; regular CI runs must keep it active.

NanoCell, semantic routing, AnswerPoints, Experience Records, and GPU backends
remain concepts or later extensions.
