# BRUCE Kernel SDK

Mit dem Kernel SDK kann ein C++-Projekt eine eigene deterministische Formel
als `TechniqueKernel` registrieren. V1.3 ist eine Compile-Time-Integration:
Der Kernel wird zusammen mit deiner Anwendung gebaut. Es gibt keinen
Runtime-Plug-in-Loader.

## 1. Paket lokal installieren und einbinden

Wenn du BRUCE aus dem Quellbaum verwendest, erstelle zuerst eine lokale
Installation:

```bash
cmake -S /pfad/zu/BRUCE -B /pfad/zu/BRUCE/build
cmake --build /pfad/zu/BRUCE/build
cmake --install /pfad/zu/BRUCE/build --prefix /pfad/zu/BRUCE/build/install
```

Danach konfigurierst und baust du dein eigenes Projekt gegen dieses Paket:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/pfad/zu/BRUCE/build/install
cmake --build build
```

Die eigene `CMakeLists.txt` enthält:

```cmake
find_package(bruce_lhme 0.2 CONFIG REQUIRED)

add_executable(my_kernel main.cpp)
target_link_libraries(my_kernel PRIVATE bruce_lhme::core)
```

Das exportierte Target liefert die öffentlichen BRUCE-Header, C++20 und die
benötigten Threads- und Boost-Header-Abhängigkeiten.

## 2. Kernel schreiben

Implementiere `TechniqueKernel::identity()` und `TechniqueKernel::evaluate()`.
Eine Identity hat eine Länge von `1–128`, verwendet nur `A–Z`, `a–z`, `0–9`,
`.`, `_` oder `-` und besitzt eine Version größer als null. Parameter sind
kanonische Strings; prüfe sie exakt, bevor gerechnet wird.

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

Das Beispiel verwendet die Formel `2 × index + 3`. Bei `index = 21` ist das
bestätigte BigInt-Ergebnis `45`.

## 3. Regeln für robuste Kernel

1. Prüfe Identity, Parameter, Adresse und Modus vor der Berechnung.
2. Prüfe `cancellation_requested(request)` vor jedem potenziell langen
   Rechenschritt.
3. Beachte die Ressourcenlimits aus `request.budget`; `Engine` prüft das
   veröffentlichte Ergebnis zusätzlich, einschließlich der harten
   `max_trace_bytes`-Grenze für Kernel-Traces.
4. Nutze `confirmed_exact_result` für exakte Ganzzahlen. Der Helper verlangt
   einen Trace; bei einem Null-Trace wird ein leerer Basis-Trace erzeugt.
5. Nutze `invalid_technique_request_result` für ungültige Eingaben und
   `cancelled_technique_result` für Abbruch. Beide liefern einen passenden
   Branch- beziehungsweise Run-Trace.
6. Registriere den Kernel vor der Auswertung:

```cpp
auto engine = bruce::lhme::Engine::create({});
const auto kernel = std::make_shared<AffineKernel>();
if (!engine.register_technique(kernel)) {
  // Identity ist ungültig oder bereits registriert.
}
```

## 4. SDK prüfen

Im BRUCE-Quellbaum prüfen diese Befehle den öffentlichen Vertrag und den
komplett getrennten Consumer:

```bash
ctest --test-dir build -R kernel_sdk_contract_tests --output-on-failure
ctest --test-dir build -R lhme_kernel_sdk_consumer --output-on-failure
```

Das Consumer-Beispiel liegt unter `examples/kernel_sdk_consumer/`. Es nutzt
nur `find_package` und `bruce_lhme::core`; damit ist es die Vorlage für ein
unabhängiges Projekt.

## 5. Exakte Cache-Äquivalenz zwischen Formula und Technique

Eine Technique kann **nur für `EvaluationMode::Exact`** bewusst mit einer
Formula denselben Result-Cache-Eintrag verwenden. Das ist kein Verhalten, das
ein Kernel selbst behaupten oder über seine `ComputationIdentity` erzwingen
kann: Die besitzende `Engine` akzeptiert dafür eine unveränderliche
`ExactFormulaTechniqueCacheBinding`.

Die folgende, vollständige Reihenfolge ist auch im ausführbaren
[`lhme_technique_demo`](../demos/lhme/technique_demo.cpp) enthalten. Das
Beispiel befüllt den Cache über die Formula-API und liest den exakt selben Wert
anschließend über die Technique-API.

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

`make_power_offset_exact_cache_binding(base, scale, version)` ist ein kleiner,
expliziter Helper für den eingebauten `PowerOffsetKernel`. `version` muss
exakt `PowerOffsetKernel::identity().version` sein. Bei jeder anderen Version
wirft der Helper `std::invalid_argument`; er erzeugt keine möglicherweise
falsche Äquivalenzdeklaration.

### Vertrag und Grenzen

- Eine Binding wird nur akzeptiert, wenn sowohl ihre `FormulaSpec` als auch
  ihre `TechniqueIdentity` bereits in **derselben** Engine registriert sind.
  Die Technique-Identity, Version und der Parameterstring bilden gemeinsam den
  unveränderlichen Binding-Schlüssel.
- `accepted_parameters` werden bytegenau mit
  `TechniqueRequest::parameters` verglichen. Die Engine interpretiert oder
  parst diesen String nicht. Leere Strings und Strings über 4096 Bytes werden
  bei der Registrierung abgelehnt.
- Eine identische erneute Registrierung ist idempotent und erfolgreich. Eine
  Binding mit derselben Technique-ID, Technique-Version und demselben
  Parameterstring, die auf eine andere Formula zeigt, wird abgelehnt.
- Nur Exact-Anfragen teilen einen Cache-Key: Die Formula verwendet
  `CacheKey::exact(formula, index)` und eine passend gebundene Technique
  verwendet denselben Key. `Metric`, `Symbolic` und `Modular` werden nicht
  API-übergreifend geteilt.
- `ComputationIdentity` steuert weiterhin ausschließlich Inflight-Gruppen,
  Leases und Cancellation. Sie autorisiert keine Binding und erzeugt keinen
  Cache-Key.
- Bei einem API-übergreifenden Treffer liefert die Engine immer die neutrale
  Trace mit dem Grund `cross-api exact cache hit`; die Trace des ursprünglichen
  Producers wird nicht offengelegt. Gleichartige Formula- oder Technique-Hits
  behalten ihre normale Cache-Trace.
- Budget und Cancellation werden bei **jedem** Cache-Hit erneut geprüft. Ein
  gecancelter Aufruf verändert oder entfernt den vorhandenen Cache-Eintrag
  nicht. Die Binding führt kein Singleflight ein: Gleichzeitige Cache-Misses
  dürfen weiterhin parallel berechnen.

Die `TechniqueKernel`-VTable und die C-ABI bleiben dabei unverändert. Die
Binding ist eine C++-`Engine`-API; für die C-ABI gibt es keine neue Funktion.
Weil `Engine` als öffentliche C++-Klasse erweitert wurde, müssen C++-
Binary-Consumer gegen die neue Bibliotheksversion neu gebaut werden.
