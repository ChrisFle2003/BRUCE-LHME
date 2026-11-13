# BRUCE LHME

BRUCE LHME is a C++20 library for deterministic, lazy and memory-conscious
formula evaluation. It is designed for cases where a large logical value space
is better represented by a compact rule than by materializing every value in
RAM.

Source repository: [ChrisFle2003/bruce-lhme](https://github.com/ChrisFle2003/bruce-lhme).

## The core idea

```text
FormulaRecipe + FormulaAddress
  -> FormulaOverrideStore (only real deviations)
  -> Engine + bounded ResultCache

KEVIN GridDomain + GridAddress
  -> explicit DomainProjection
  -> FormulaEvaluator
```

`FormulaRecipe` is the source of a value. `KEVIN` is a separate hierarchical
8x8 domain, dimension and filtering coordinate. It never silently chooses a
formula: a `DomainProjection` explicitly maps a grid address to a formula
address. This preserves deterministic reconstruction while keeping stored data
sparse.

## v0.2 scope

- CPU and RAM execution only
- exact `BigInt`, modular, metric and symbolic formula evaluation
- bounded result cache and sparse formula overrides
- deterministic traces, resource budgets and safe cancellation
- engine-owned Exact Formula/Technique cache equivalence with explicit bindings
- KEVIN domain projections and candidate-local filtering
- installable CMake package and extension-kernel SDK

GPU backends, NanoCell runtime, semantic routing, AnswerPoints and Experience
Records are future concepts, not shipped v0.2 features.

## Requirements

- CMake 3.25 or newer
- A C++20 compiler
- C and C++ standard library threading support
- Boost development headers, including `boost/multiprecision/cpp_int.hpp`,
  installed and discoverable by CMake (no compiled Boost library is required)

## Build and test

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cmake --preset release
cmake --build --preset release
ctest --preset release
```

### Optional: GCC + Ninja

If Ninja is installed, GCC and G++ remain the compilers; Ninja only replaces
the build generator. The profiles use separate build directories:

```bash
cmake --preset gcc-ninja-debug
cmake --build --preset gcc-ninja-debug --parallel 2
ctest --preset gcc-ninja-debug

cmake --preset gcc-ninja-release
cmake --build --preset gcc-ninja-release --parallel 2
ctest --preset gcc-ninja-release
```

Additional diagnostic profiles are `address`, `undefined` and `thread`.
They instrument the runtime and are intended for development, not release
binaries.

## Use as a CMake package

Install BRUCE LHME, then consume it from another project:

```cmake
find_package(bruce_lhme 0.2 CONFIG REQUIRED)

add_executable(my_program main.cpp)
target_link_libraries(my_program PRIVATE bruce_lhme::core)
```

The exported target resolves Boost and thread support transitively; consumers
need not link either dependency separately.

Minor releases before 1.0 are distinct public C++ API/ABI generations.
Reconfigure, rebuild and relink C++ consumers when moving to a new minor
version. The C ABI and `TechniqueKernel` VTable compatibility for v0.2 are
documented in the release note.

The checked external example lives in
[`examples/external_consumer`](examples/external_consumer).

## C ABI

The C-compatible public header is
[`include/bruce/lhme/c_api.h`](include/bruce/lhme/c_api.h). It uses opaque
engine, formula and result handles. Their ownership, nullable return values,
result strings and cancellation behaviour are specified in
[C ABI lifecycle and cancellation](docs/c-api.en.md).

## Documentation

- [Developer Guide](docs/developer-guide.en.md)
- [Implemented architecture](docs/architecture.en.md)
- [C ABI lifecycle and cancellation](docs/c-api.en.md)
- [Benchmark review — 2026-10-21](docs/benchmark-review-2026-10-21.en.md)
  - [Fair comparison: separate Tracks A, B and C](docs/benchmark-review-2026-10-21.en.md#fair-comparison-track-a)
- [v0.2 release notes and version/ABI policy](docs/release-v0.2.en.md)
- [v0.1 release contract (historical)](docs/release-v0.1.en.md)
- [Implemented component overview](concepts/README.en.md)
- [Kernel SDK](docs/kernel-sdk.en.md)
  - [Exact Formula/Technique cache equivalence](docs/kernel-sdk.en.md#5-exact-cache-equivalence-between-formula-and-technique)
- [GGUF Inspector V1](docs/gguf-inspector.en.md)

## License

BRUCE LHME is released under the
[European Union Public Licence v. 1.2 or later (EUPL-1.2-or-later)](LICENSE).
See [CONTRIBUTING.md](CONTRIBUTING.md) for the Developer Certificate of Origin
used for new contributions.
