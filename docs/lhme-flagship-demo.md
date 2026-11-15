# LHME-Flagship-Demo — virtuelle Regelwerte mit Sparse Overrides

Die textbasierte Flagship-Demo zeigt den zentralen LHME-Fall: Ein großer,
logisch adressierbarer Wertebereich wird durch eine deterministische Regel
erzeugt. Nur echte lokale Abweichungen werden gespeichert. Eine begrenzte
Streaming-Abfrage filtert zuerst modular und führt die teure Exact-Auswertung
nur für die verbleibenden Adressen aus.

Die Demo richtet sich an Personen, die den Unterschied zwischen einem
virtuellen Wertebereich und einer materialisierten Tabelle nachvollziehen
möchten. Sie verwendet ausschließlich die vorhandene `PowerOffset`-Formula,
den `FormulaOverrideStore` und dessen Store-gebundene Streaming-Abfrage.

## Voraussetzungen

- CMake 3.25 oder neuer
- ein C++20-Compiler mit Thread-Unterstützung
- Boost-Headers einschließlich `boost/multiprecision/cpp_int.hpp`

Die folgenden Befehle werden im Repository-Wurzelverzeichnis ausgeführt. Sie
bauen nur das Demo-Target; Tests, Tools und Benchmarks werden dafür nicht
benötigt.

```bash
cmake -S . -B out/build/lhme-flagship-demo -DBRUCE_BUILD_EXAMPLES=ON
cmake --build out/build/lhme-flagship-demo --target lhme_flagship_demo --parallel 2
./out/build/lhme-flagship-demo/lhme_flagship_demo \
  --snapshot /tmp/lhme-flagship-demo.snapshot
```

`--snapshot` ist verpflichtend. Verwenden Sie einen beschreibbaren Pfad, dessen
Elternverzeichnis bereits existiert. Die Demo schreibt dort den Snapshot und
liest ihn anschließend wieder ein.

## Was die Demo zeigt

### 1. Regel statt Volltabelle

Die registrierte Regel ist:

```text
PowerOffset(base=2, scale=100, version=1)
value(index) = 100 * (2^index - 1)
```

Damit ist beispielsweise die Adresse `1_000_000` logisch gültig, ohne dass
eine Million Werte gespeichert werden müssen. Die Demo fragt diese Adresse im
`Metric`-Modus ab. Das Ergebnis zeigt die Herkunft `Computed` und die
Bitlänge `1_000_007`; der exakte `BigInt` wird dafür nicht materialisiert.

### 2. Drei persistierbare lokale Ausnahmen

Die Demo legt genau diese Overrides an:

| Adresse | Effektiver Wert | Herkunft |
| ---: | ---: | --- |
| 10 | -40 | `Override` |
| 777 | 17 | `Override` |
| 999 | 0 | `Override` |

Eine Punktabfrage an Adresse 10 gibt deshalb `Override -40` aus. Adresse 11
hat keine Ausnahme und liefert `Computed 204700`. Die Herkunft ist Teil jedes
Store-Ergebnisses und jeder retained Streaming-Kandidatenzeile.

### 3. Begrenztes Streaming mit Modularfilter vor Exact

Danach fragt die Demo den Bereich `0..1023` ab. Die Optionen sind:

| Einstellung | Wert |
| --- | --- |
| Filter | effektiver Wert `mod 17 == 11` |
| Retention | `TopK` |
| Maximale Kandidaten | 4 |
| Retention-Speicher | 4 KiB |
| Adressbudget | 1.024 |

Das Retention-Speicherbudget ist eine deterministische Schätzung der bewusst
behaltenen Kandidaten einschließlich ihrer Auswertung. Es ist **keine** harte
Prozessspeichergrenze und beschränkt weder Thread-Stacks noch temporäre
Allokationen während einer Auswertung.

Der Modularfilter arbeitet auf dem **effektiven** Wert, also einschließlich
eines möglichen Overrides. Nur Filtertreffer werden anschließend exakt
ausgewertet. Die erwarteten Zähler sind:

```text
stream checked  | 1024
stream filtered | 896
stream exact    | 128
stream retained | 4
computed exact delta | 127
```

Der Delta-Wert ist kleiner als die Exact-Zahl, weil Adresse 10 als Override
keine berechnete Exact-Auswertung benötigt. Die vier retained Kandidaten
machen die Herkunft sichtbar:

```text
candidate 2  | Computed 300
candidate 10 | Override -40
candidate 18 | Computed 26214300
candidate 26 | Computed 6710886300
```

Die Demo stellt außerdem bewusst einen Bereich mit 1.025 Adressen. Das
Adressbudget lehnt ihn vor Exact-Arbeit mit `BudgetExceeded` ab.

## Override-Kapazität: klare Ablehnung, keine Eviktion

`FormulaOverrideStoreConfig` kann die Anzahl der Korrekturen, deren gesamten
deterministisch abgerechneten Payload und die Größe eines einzelnen
Override-Werts begrenzen. Der Payload besteht aus dem kanonischen gespeicherten
Override-Schlüssel und den Magnitudenbytes des `BigInt`; der Wert `0` zählt als
ein Wertbyte. Das ist eine Store-Kapazitätszusage, keine Prozessspeichermessung.

Die Standardkonfiguration bleibt aus Kompatibilitätsgründen unbegrenzt. Bei
einem konfigurierten Limit wird ein neues oder ersetztes Override mit
`BudgetExceeded` abgewiesen; ein vorhandenes Override bleibt unverändert. Es
gibt absichtlich keine Eviktion von fachlichen Korrekturen. Ein gültiger
Snapshot, der nicht in den Ziel-Store passt, liefert `CapacityExceeded` und
belässt dessen bisherigen Inhalt unverändert.

## Snapshot: Regelbindung und Ausnahmen, keine Volltabelle

Der Snapshot enthält die sparse Override-Datensätze mit ihrer Formula- oder
Technique-Bindung, Adresse und ihrem Wert sowie eine Integritätsprüfung. Er
enthält keine berechneten Werte für die übrigen logischen Adressen und keine
materialisierte Tabelle.

Für den Reload registriert die Demo dieselbe `PowerOffset`-Formula erneut,
lädt den Snapshot in einen neuen Store und prüft die Ausnahme an Adresse 10.
Die Ausgabe enthält deshalb sowohl die Zahl der Overrides, die Snapshot-Größe
als auch `snapshot reload | Override -40`. Die Bytezahl ist eine beobachtete
Ausgabe der konkreten Snapshot-Datei, keine Zusage für eine feste
Formatgröße.

## Kooperative Cancellation im Store-Streaming

Die Store-gebundene Streaming-Abfrage unterstützt ein request-lokales
Cancellation-Flag oder -Signal. Cancellation ist bewusst **kooperativ** und
wird an Stream-Grenzen beobachtet:

- vor dem Dispatch eines Tiles,
- zwischen Adressen beziehungsweise Tiles und
- vor Veröffentlichung retained Ergebnisse.

Wird Cancellation an einer solchen Grenze beobachtet, endet die Abfrage mit
`Cancelled` und `retained_candidates` ist leer. Eine bereits laufende
Formula- oder Technique-Auswertung wird dadurch ausdrücklich **nicht**
präemptiv unterbrochen; ihre Laufzeit begrenzt somit die
Cancellation-Reaktionszeit.

## Nicht-Ziele

Diese Demo ist absichtlich kein:

- Datenbanksystem oder persistenter Tabellen-Store,
- Netzwerkdienst oder Remote-API,
- GUI,
- GGUF-Importer, -Rewriter oder -Integration,
- Modell-Inferenz- oder LLM-Runtime.

Sie demonstriert stattdessen eine lokale, deterministische und begrenzte
Rekonstruktion von Regelwerten mit wenigen klar nachvollziehbaren Ausnahmen.
