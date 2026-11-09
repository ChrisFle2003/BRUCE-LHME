# BRUCE LHME v0.2.0 — Release Notes und Versions-/ABI-Policy

## Release-Notiz

Version 0.2.0 veröffentlicht die engine-besessene Exact-Cache-Äquivalenz
zwischen einer registrierten Formula und einer explizit gebundenen Technique.
Sie ergänzt die öffentliche C++-API um
`ExactFormulaTechniqueCacheBinding` und
`Engine::register_exact_formula_cache_binding()`.

Die Freigabe ist ausschließlich für `EvaluationMode::Exact`: Eine Binding
teilt den Formula-Exact-Cache-Key nur bei bytegenau passenden Technique-
Parametern. `Metric`, `Symbolic` und `Modular` bleiben API-getrennt.
`ComputationIdentity` bleibt auf Inflight-Gruppen, Leases und Cancellation
beschränkt. API-übergreifende Exact-Hits erhalten die neutrale Trace
`cross-api exact cache hit`.

## Kompatibilität in v0.2.0

| Grenze | Status | Konsequenz |
|---|---|---|
| Öffentlicher C++-Quellcode | Additiv | Source-Consumer können die neue API nach einem Neuaufbau verwenden. |
| Öffentlicher C++-Binary-ABI | Inkompatibel | `Engine` hat ein geändertes Objektlayout; alle C++-Binary-Consumer müssen neu gebaut und gelinkt werden. |
| `TechniqueKernel`-VTable | Unverändert | Bestehende Kernel benötigen keine neue virtuelle Methode. |
| C-ABI | Unverändert | Es gibt keine neue C-Binding-API und keine geänderten C-Strukturen oder Symbole. |

BRUCE LHME exportiert derzeit eine statische CMake-Bibliothek. Ein
`find_package()`-Consumer wird aus seinem Quellcode gegen das installierte
Archiv gebaut und muss bei jeder Minor-Version ohnehin neu konfiguriert,
gebaut und gelinkt werden. Vorhandene vorgebaute C++-Artefakte dürfen nicht
gegen `0.2.0` weiterverwendet werden.

## Verbindliche Pre-1.0-Versionierungsregel

- Ein Release `0.y.0` kennzeichnet eine neue öffentliche C++-API- oder
  C++-ABI-Generation. Additive öffentliche C++-APIs gehören deshalb ebenso in
  den nächsten Minor-Release wie Änderungen an öffentlichen Objektlayouts,
  VTables oder semantischen Verträgen.
- Ein Patch `0.y.z` darf keine öffentliche C++-API und keine öffentliche
  C++-Binary-ABI ändern. Er ist für kompatible Fehlerkorrekturen,
  Implementierungsänderungen ohne öffentliche Auswirkung und Dokumentation
  reserviert.
- Jeder öffentliche Minor-Release braucht eine Release-Notiz mit einer
  Aussage zu C++-Quellcode, C++-Binary-ABI, C-ABI und
  `TechniqueKernel`-VTable.
- Der installierte Paketvertrag verwendet `SameMinorVersion`. Damit kann
  `find_package(bruce_lhme 0.1 CONFIG REQUIRED)` kein `0.2.x`-Paket
  auflösen; v0.2-Consumer fordern `bruce_lhme 0.2` an.

Diese Policy gilt bis zu einer ausdrücklich dokumentierten Ablösung bei
`1.0.0`.

## Release-Verifikation

```text
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug --output-on-failure -j 4
```

Die Suite enthält den Stage-Install-/`find_package()`-Consumer sowie die
Exact-Cache-Äquivalenztests.

## v0.2.0 Release-Candidate-Gates

### Bestätigte lokale Grundlage

- [x] Exact-Cache-Äquivalenz, die Cancellation-Gruppenkorrektur, v0.2.0 und
  `SameMinorVersion` sind implementiert und dokumentiert.
- [x] Debug- und Release-Suite, AddressSanitizer mit lokal deaktivierter
  Leak-Prüfung, ThreadSanitizer und UndefinedBehaviorSanitizer wurden lokal
  erfolgreich ausgeführt.
- [x] Stage-Install-/`find_package()`-Consumer und die v0.1-zu-v0.2-
  Paketversionsgrenze sind durch CTest abgedeckt.
- [x] Release Owner: `ChrisFle2003`.
- [x] Distribution: ein source-only GitHub Release ohne Paketregistry und
  ohne vorgebaute Binärartefakte.

### Noch extern zu belegende Gates

- [ ] Den vollständigen AddressSanitizer-Lauf mit aktivierter LeakSanitizer-
  Prüfung auf einem Host ohne ptrace-Einschränkung durchführen und den
  Revision-/Run-Nachweis festhalten.
- [ ] Mit einem unabhängig installierten, unterstützten C++20-Compiler bauen
  und testen; Compiler und Ergebnis festhalten.
- [ ] Einen erfolgreichen GitHub-Actions-Run für `windows-latest` mit Debug
  und Release auf der Release-Candidate-Revision festhalten.
- [ ] Die erfolgreichen GitHub-Actions-Runs für Ubuntu Debug/Release sowie
  Linux AddressSanitizer/UndefinedBehaviorSanitizer auf derselben Revision
  festhalten.

ThreadSanitizer bleibt eine bewusste lokale Zusatzprüfung. Es gibt derzeit
keine belastbar begründete Hosted-CI-Integration dafür; der Readiness-Workflow
behauptet daher keine TSan-CI-Abdeckung.

### Manueller Tag- und Publish-Ablauf nach geschlossenen Gates

1. `ChrisFle2003` prüft die externen Nachweise und die Release-Candidate-
   Revision.
2. Erst danach wird der annotierte Tag `v0.2.0` auf genau diese Revision
   erstellt und gepusht.
3. Danach wird ein GitHub Release mit dieser Release-Notiz erstellt. Die
   Distribution besteht ausschließlich aus den von GitHub bereitgestellten
   Source-Archiven; es werden keine Binärartefakte und keine Paketregistry-
   Veröffentlichungen erzeugt.
4. Owner, Tag, Release-URL und die zugehörigen CI-Run-URLs werden im
   Release-Record festgehalten.

Der manuelle Readiness-Workflow in
`.github/workflows/release-readiness.yml` führt keine dieser Aktionen aus. Er
besitzt ausschließlich `contents: read` und führt nur Konfiguration, Build und
CTest aus.
