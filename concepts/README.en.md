# Implemented components

This directory contains no unpublished architecture drafts. Public
documentation describes only components present in the current C++ source code
and its associated tests.

| Component | Repository evidence | Documentation |
| --- | --- | --- |
| Formula evaluation and Technique kernels | `include/bruce/lhme/formula.hpp`, `include/bruce/lhme/technique.hpp`, `src/core/` | [Architecture](../docs/architecture.en.md), [Kernel SDK](../docs/kernel-sdk.en.md) |
| Engine, result cache, and cancellation | `include/bruce/lhme/engine.hpp`, `include/bruce/lhme/cache.hpp`, `src/engine.cpp`, `src/cache/` | [Architecture](../docs/architecture.en.md) |
| Sparse Formula overrides and snapshots | `include/bruce/lhme/formula_override_store.hpp`, `include/bruce/lhme/formula_override_snapshot.hpp` | [Architecture](../docs/architecture.en.md) |
| KEVIN domain projection and filtering | `include/bruce/lhme/kevin_domain.hpp`, `include/bruce/lhme/domain_projection.hpp`, `include/bruce/lhme/kevin_filter.hpp`, `src/kevin/` | [Architecture](../docs/architecture.en.md), [KEVIN Memory Store](../docs/kevin-memory-store.en.md) |
| CPU streaming and scheduler | `include/bruce/lhme/streaming.hpp`, `src/cpu/` | [Architecture](../docs/architecture.en.md) |
| GGUF Inspector | `include/bruce/lhme/gguf_inspector.hpp`, `src/gguf/` | [GGUF Inspector V1](../docs/gguf-inspector.en.md) |
| Public C and C++ interfaces | `include/bruce/lhme/c_api.h`, `include/bruce/lhme/kernel_sdk.hpp` | [C ABI](../docs/c-api.en.md), [Kernel SDK](../docs/kernel-sdk.en.md) |

NanoCell, semantic routing, AnswerPoints, Experience Records, GPU backends,
and other unimplemented research or product designs are not part of this public
repository.
