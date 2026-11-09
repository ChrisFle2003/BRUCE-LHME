# BRUCE LHME v0.2.0 — Release Notes and Version/ABI Policy

## Release note

Version 0.2.0 releases engine-owned Exact cache equivalence between a
registered Formula and an explicitly bound Technique. It adds
`ExactFormulaTechniqueCacheBinding` and
`Engine::register_exact_formula_cache_binding()` to the public C++ API.

The release is exclusively for `EvaluationMode::Exact`: a binding shares the
Formula Exact cache key only for byte-exact matching Technique parameters.
`Metric`, `Symbolic`, and `Modular` remain API-separated.
`ComputationIdentity` remains limited to inflight groups, leases, and
cancellation. Cross-API Exact hits receive the neutral trace
`cross-api exact cache hit`.

## Compatibility in v0.2.0

| Boundary | Status | Consequence |
|---|---|---|
| Public C++ source code | Additive | Source consumers can use the new API after rebuilding. |
| Public C++ binary ABI | Incompatible | `Engine` has a changed object layout; all C++ binary consumers must be rebuilt and relinked. |
| `TechniqueKernel` VTable | Unchanged | Existing kernels require no new virtual method. |
| C ABI | Unchanged | There is no new C binding API and no changed C structures or symbols. |

BRUCE LHME currently exports a static CMake library. A `find_package()`
consumer is built from source against the installed archive and must be
reconfigured, rebuilt, and relinked for every minor version anyway. Existing
prebuilt C++ artifacts must not continue to be used against `0.2.0`.

## Binding pre-1.0 versioning rule

- A `0.y.0` release denotes a new public C++ API or C++ ABI generation.
  Additive public C++ APIs therefore belong in the next minor release just as
  changes to public object layouts, VTables, or semantic contracts do.
- A `0.y.z` patch must not change the public C++ API or public C++ binary ABI.
  It is reserved for compatible bug fixes, implementation changes without a
  public effect, and documentation.
- Every public minor release requires release notes that state compatibility
  for C++ source code, C++ binary ABI, C ABI, and the `TechniqueKernel` VTable.
- The installed package contract uses `SameMinorVersion`. Therefore,
  `find_package(bruce_lhme 0.1 CONFIG REQUIRED)` cannot resolve a `0.2.x`
  package; v0.2 consumers request `bruce_lhme 0.2`.

This policy applies until an explicitly documented replacement at `1.0.0`.

## Release verification

```text
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug --output-on-failure -j 4
```

The suite includes the stage-install/`find_package()` consumer and the
Exact-cache-equivalence tests.

## v0.2.0 Release Candidate gates

### Confirmed local foundation

- [x] Exact-cache equivalence, the cancellation-group correction, v0.2.0, and
  `SameMinorVersion` are implemented and documented.
- [x] The Debug and Release suites, AddressSanitizer with leak checking
  disabled locally, ThreadSanitizer, and UndefinedBehaviorSanitizer were run
  successfully locally.
- [x] The stage-install/`find_package()` consumer and the v0.1-to-v0.2 package
  version boundary are covered by CTest.
- [x] Release owner: `ChrisFle2003`.
- [x] Distribution: a source-only GitHub Release without a package registry
  and without prebuilt binary artifacts.

### External gates still requiring evidence

- [ ] Run the complete AddressSanitizer suite with LeakSanitizer enabled on a
  host without `ptrace` restrictions and record the revision/run evidence.
- [ ] Build and test with an independently installed supported C++20 compiler;
  record the compiler and result.
- [ ] Record a successful GitHub Actions run for `windows-latest` with Debug
  and Release on the release-candidate revision.
- [ ] Record successful GitHub Actions runs for Ubuntu Debug/Release and Linux
  AddressSanitizer/UndefinedBehaviorSanitizer on the same revision.

ThreadSanitizer remains a deliberate local additional check. There is currently
no robustly justified hosted-CI integration for it; the readiness workflow
therefore does not claim TSan CI coverage.

### Manual tag and publish sequence after gates close

1. `ChrisFle2003` reviews the external evidence and release-candidate
   revision.
2. Only then is the annotated tag `v0.2.0` created and pushed for exactly that
   revision.
3. A GitHub Release is then created with these release notes. Distribution
   consists solely of GitHub-provided source archives; no binary artifacts or
   package-registry publications are produced.
4. The owner, tag, release URL, and associated CI run URLs are recorded in the
   release record.

The manual readiness workflow in
[`.github/workflows/release-readiness.yml`](../.github/workflows/release-readiness.yml)
performs none of these actions. It has only `contents: read` and performs only
configuration, build, and CTest.
