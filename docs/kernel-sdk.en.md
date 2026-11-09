# BRUCE Kernel SDK

With the Kernel SDK, a C++ project can register its own deterministic formula
as a `TechniqueKernel`. V1.3 is a compile-time integration: the kernel is
built together with your application. There is no runtime plug-in loader.

## 1. Install and include the package locally

If you use BRUCE from the source tree, first create a local installation:

```bash
cmake -S /path/to/BRUCE -B /path/to/BRUCE/build
cmake --build /path/to/BRUCE/build
cmake --install /path/to/BRUCE/build --prefix /path/to/BRUCE/build/install
```

Then configure and build your own project against this package:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/BRUCE/build/install
cmake --build build
```

Your `CMakeLists.txt` contains:

```cmake
find_package(bruce_lhme 0.2 CONFIG REQUIRED)

add_executable(my_kernel main.cpp)
target_link_libraries(my_kernel PRIVATE bruce_lhme::core)
```

The exported target supplies the public BRUCE headers, C++20, and the required
Threads and Boost header dependencies.

## 2. Write a kernel

Implement `TechniqueKernel::identity()` and `TechniqueKernel::evaluate()`. An
identity has a length of `1–128`, uses only `A–Z`, `a–z`, `0–9`, `.`, `_`, or
`-`, and has a version greater than zero. Parameters are canonical strings;
validate them exactly before calculating.

```cpp
#include <memory>

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/kernel_sdk.hpp"

class AffineKernel final : public bruce::lhme::TechniqueKernel {
 public:
  bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "example.affine", .version = 1};
  }

  bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (bruce::lhme::cancellation_requested(request)) {
      return bruce::lhme::cancelled_technique_result(
          request.mode, "kernel cancellation requested");
    }
    if (request.identity != identity() ||
        request.parameters != "factor=1:2;offset=1:3" ||
        request.address.index < 0 ||
        request.mode != bruce::lhme::EvaluationMode::Exact) {
      return bruce::lhme::invalid_technique_request_result(
          request.mode, "expected exact affine request");
    }
    return bruce::lhme::confirmed_exact_result(
        bruce::lhme::BigInt{request.address.index} * 2 + 3,
        bruce::lhme::make_trace({}));
  }
};
```

The example uses the formula `2 × index + 3`. For `index = 21`, the confirmed
BigInt result is `45`.

## 3. Rules for robust kernels

1. Validate identity, parameters, address, and mode before calculation.
2. Check `cancellation_requested(request)` before every potentially lengthy
   calculation step.
3. Observe the resource limits in `request.budget`; `Engine` additionally
   checks the published result, including the hard `max_trace_bytes` limit for
   kernel traces.
4. Use `confirmed_exact_result` for exact integers. The helper requires a
   trace; it creates an empty base trace for a null trace.
5. Use `invalid_technique_request_result` for invalid input and
   `cancelled_technique_result` for cancellation. Both return an appropriate
   branch or run trace.
6. Register the kernel before evaluation:

```cpp
auto engine = bruce::lhme::Engine::create({});
const auto kernel = std::make_shared<AffineKernel>();
if (!engine.register_technique(kernel)) {
  // Identity ist ungültig oder bereits registriert.
}
```

## 4. Verify the SDK

In the BRUCE source tree, these commands verify the public contract and the
fully independent consumer:

```bash
ctest --test-dir build -R kernel_sdk_contract_tests --output-on-failure
ctest --test-dir build -R lhme_kernel_sdk_consumer --output-on-failure
```

The consumer example is located at `examples/kernel_sdk_consumer/`. It uses
only `find_package` and `bruce_lhme::core`; it is therefore the template for
an independent project.

## 5. Exact cache equivalence between Formula and Technique

A Technique may deliberately use the same result-cache entry as a Formula
**only for `EvaluationMode::Exact`**. This is not behavior a kernel can claim
itself or force through its `ComputationIdentity`: the owning `Engine` accepts
an immutable `ExactFormulaTechniqueCacheBinding` for this purpose.

The following complete sequence is also contained in the executable
[`lhme_technique_demo`](../demos/lhme/technique_demo.cpp). The example populates
the cache through the Formula API and then reads the exact same value through
the Technique API.

```cpp
#include <memory>

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/trace.hpp"

int main() {
  using namespace bruce::lhme;

  auto engine = Engine::create({.cache_max_entries = 8, .cache_max_bytes = 4096});

  // 1. Die äquivalente Formula in genau dieser Engine registrieren.
  const PowerOffsetKernel identity_source;
  const auto version = identity_source.identity().version;
  const auto formula = engine.register_formula(make_power_offset_spec(2, 100, version));

  // 2. Die Technique in derselben Engine registrieren.
  const auto kernel = std::make_shared<PowerOffsetKernel>();
  if (!engine.register_technique(kernel)) return 1;

  // 3. Die Engine-eigene, genaue Gleichheitsbehauptung registrieren.
  // Der Helper erzeugt die kanonischen Formula- und Parameterwerte.
  if (!engine.register_exact_formula_cache_binding(
          make_power_offset_exact_cache_binding(2, 100, version))) {
    return 1;
  }

  // 4. Eine Exact-Formula-Anfrage erzeugt den gemeinsamen Cache-Eintrag.
  const auto formula_result = engine.evaluate({
      .spec = formula,
      .address = {.index = 10},
      .mode = EvaluationMode::Exact,
      .budget = Budget::unlimited(),
  });

  // 5. Dieselbe Exact-Technique-Anfrage verwendet den Formula-Cache-Key.
  const auto technique_result = engine.evaluate_technique({
      .identity = kernel->identity(),
      .parameters = encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = EvaluationMode::Exact,
      .budget = Budget::unlimited(),
  });

  if (formula_result.status != Status::Confirmed ||
      technique_result.status != Status::Confirmed ||
      formula_result.exact_integer() != technique_result.exact_integer()) {
    return 1;
  }

  // Der API-übergreifende Treffer hat bewusst keine Producer-Trace.
  return technique_result.trace != nullptr &&
                 technique_result.trace->events.size() == 1 &&
                 technique_result.trace->events.front().reason ==
                     "cross-api exact cache hit"
             ? 0
             : 1;
}
```

`make_power_offset_exact_cache_binding(base, scale, version)` is a small,
explicit helper for the built-in `PowerOffsetKernel`. `version` must exactly
equal `PowerOffsetKernel::identity().version`. For any other version, the
helper throws `std::invalid_argument`; it does not create a potentially
incorrect equivalence declaration.

### Contract and boundaries

- A binding is accepted only if both its `FormulaSpec` and its
  `TechniqueIdentity` are already registered in the **same** engine. The
  Technique identity, version, and parameter string together form the
  immutable binding key.
- `accepted_parameters` are compared byte-for-byte with
  `TechniqueRequest::parameters`. The engine does not interpret or parse this
  string. Empty strings and strings longer than 4096 bytes are rejected at
  registration.
- An identical repeated registration is idempotent and successful. A binding
  with the same Technique ID, Technique version, and parameter string that
  points to a different Formula is rejected.
- Only Exact requests share a cache key: the Formula uses
  `CacheKey::exact(formula, index)` and a matching bound Technique uses the
  same key. `Metric`, `Symbolic`, and `Modular` are not shared across APIs.
- `ComputationIdentity` continues to control only inflight groups, leases, and
  cancellation. It authorizes no binding and produces no cache key.
- For a cross-API hit, the engine always returns the neutral trace with reason
  `cross-api exact cache hit`; the trace of the original producer is not
  disclosed. Same-API Formula or Technique hits retain their normal cache
  trace.
- Budget and cancellation are checked again on **every** cache hit. A
  cancelled call neither changes nor removes the existing cache entry. The
  binding introduces no singleflight: concurrent cache misses may still
  calculate in parallel.

The `TechniqueKernel` VTable and the C ABI remain unchanged. The binding is a
C++ `Engine` API; there is no new C ABI function. Because `Engine` was
extended as a public C++ class, C++ binary consumers must be rebuilt against
the new library version.
