# Umgesetzte Komponenten

Dieses Verzeichnis enthält keine unveröffentlichten Architekturentwürfe. Die
öffentliche Dokumentation beschreibt ausschließlich Komponenten, die im
aktuellen C++-Quellcode und den zugehörigen Tests vorhanden sind.

| Komponente | Nachweis im Repository | Dokumentation |
| --- | --- | --- |
| Formula-Auswertung und Technique-Kernels | `include/bruce/lhme/formula.hpp`, `include/bruce/lhme/technique.hpp`, `src/core/` | [Architektur](../docs/architecture.md), [Kernel SDK](../docs/kernel-sdk.md) |
| Engine, Result-Cache und Cancellation | `include/bruce/lhme/engine.hpp`, `include/bruce/lhme/cache.hpp`, `src/engine.cpp`, `src/cache/` | [Architektur](../docs/architecture.md) |
| Sparse Formula-Overrides und Snapshots | `include/bruce/lhme/formula_override_store.hpp`, `include/bruce/lhme/formula_override_snapshot.hpp` | [Architektur](../docs/architecture.md) |
| KEVIN-Domainprojektion und Filter | `include/bruce/lhme/kevin_domain.hpp`, `include/bruce/lhme/domain_projection.hpp`, `include/bruce/lhme/kevin_filter.hpp`, `src/kevin/` | [Architektur](../docs/architecture.md), [KEVIN Memory Store](../docs/kevin-memory-store.md) |
| CPU-Streaming und Scheduler | `include/bruce/lhme/streaming.hpp`, `src/cpu/` | [Architektur](../docs/architecture.md) |
| GGUF Inspector | `include/bruce/lhme/gguf_inspector.hpp`, `src/gguf/` | [GGUF Inspector V1](../docs/gguf-inspector.md) |
| Öffentliche C- und C++-Schnittstellen | `include/bruce/lhme/c_api.h`, `include/bruce/lhme/kernel_sdk.hpp` | [C ABI](../docs/c-api.md), [Kernel SDK](../docs/kernel-sdk.md) |

NanoCell, semantisches Routing, AnswerPoints, Experience Records, GPU-Backends
und weitere nicht implementierte Forschungs- oder Produktentwürfe sind nicht
Teil dieses öffentlichen Repositories.
