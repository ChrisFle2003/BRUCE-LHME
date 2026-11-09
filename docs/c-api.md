# C ABI: Lebenszyklus und Cancellation

Die öffentliche C-Schnittstelle liegt in
[`include/bruce/lhme/c_api.h`](../include/bruce/lhme/c_api.h). Sie verwendet
opaque Handles und hat keine C++-Klassen in ihrer Signatur.

## Handles und Ownership

| Handle | Erzeugt durch | Eigentümer nach Erfolg | Freigabe |
| --- | --- | --- | --- |
| `lhme_engine*` | `lhme_engine_create` | Aufrufer | `lhme_engine_destroy` |
| `lhme_formula*` | `lhme_formula_register` | Aufrufer | `lhme_formula_destroy` |
| `lhme_result*` | `lhme_evaluate` | Aufrufer | `lhme_result_destroy` |

Jeder nicht-null Handle ist genau einmal mit seiner zugehörigen
`*_destroy`-Funktion freizugeben. Alle drei Destroy-Funktionen akzeptieren
`NULL`. Ein bereits freigegebener, fremder oder anderweitig ungültiger Pointer
ist kein gültiger Handle; dieses Verhalten wird nicht geprüft und ist aus
dieser Schnittstelle nicht sicher verwendbar.

`lhme_engine_config`, `lhme_power_offset_spec` und
`lhme_evaluation_request` sowie die darin enthaltenen Eingabezeichenketten
werden nur während des jeweiligen Aufrufs gelesen. Die API kopiert die für
Engine, Formel und Auswertung benötigten Werte. Ein `lhme_formula`-Handle
enthält eine Spezifikationskopie, aber keine Bindung an eine bestimmte Engine.
Bei der Auswertung muss die übergebene Engine diese Spezifikation registriert
haben; andernfalls lautet das Ergebnis `LHME_STATUS_INVALID_ARGUMENT`.

Die von `lhme_result_exact_decimal` und `lhme_result_inspect` gelieferten
Zeichenketten gehören zum Result-Handle. Sie bleiben bis zu
`lhme_result_destroy(result)` gültig und dürfen nicht vom Aufrufer freigegeben
oder verändert werden.

## Erzeugung, Rückgabewerte und Fehler

`lhme_engine_create(NULL)` verwendet die Konfiguration
`{cache_max_entries=0, cache_max_bytes=0, worker_count=1, tile_size=1}`. Bei
einem Konstruktionsfehler liefert die Funktion `NULL`.

`lhme_formula_register` liefert `NULL`, wenn Engine, Spezifikation,
`base_decimal` oder `scale_decimal` null sind oder die Konstruktion scheitert.
Ein nicht-null Formula-Handle bestätigt nur seine Erzeugung: Eine fachlich
ungültige oder von der Engine nicht registrierte Spezifikation wird erst bei
`lhme_evaluate` als `LHME_STATUS_INVALID_ARGUMENT` sichtbar.

`lhme_evaluate` gibt für eine erfolgreiche oder fachlich fehlgeschlagene
Auswertung normalerweise einen Result-Handle zurück. Ein unbekannter
`lhme_mode` erzeugt `LHME_STATUS_INVALID_ARGUMENT`. Null-Argumente und
Ausnahmen innerhalb der C-ABI werden als `LHME_STATUS_INTERNAL_ERROR` in
einem Result-Handle repräsentiert, sofern dieser angelegt werden kann. Ist
auch das Anlegen dieses Resultats nicht möglich, liefert `lhme_evaluate`
`NULL`.

`lhme_result_status(NULL)` liefert `LHME_STATUS_INTERNAL_ERROR`.
`lhme_result_exact_decimal` liefert `NULL`, falls der Result-Handle null ist
oder das Ergebnis keinen Integer-Payload besitzt. `lhme_result_inspect`
liefert für einen gültigen Handle die dezimale Repräsentation des internen
Statuswerts; sie ist keine Trace- oder Fehlertext-Schnittstelle. Für einen
null Handle liefert sie `NULL`.

Die `lhme_status`-Werte entsprechen den Ergebniskategorien der Auswertung:
`CONFIRMED`, `REJECTED`, `BUDGET_EXCEEDED`, `CANCELLED`, `INVALID_ARGUMENT`,
`UNSUPPORTED_MODE` und `INTERNAL_ERROR`.

## Cancellation und Wiederverwendung

`Engine::cancel()` und die C-ABI-Funktion `lhme_cancel(engine)` fordern den
Abbruch von Auswertungen an, die auf dieser Engine gerade aktiv sind.
`lhme_cancel(NULL)` ist ein No-op. Ein Cancel-Aufruf auf einer inaktiven Engine
beeinflusst keinen späteren Lauf.

Die Engine erzeugt für eine neue Gruppe aktiver Auswertungen ein frisches
Cancellation-Signal. Nach dem Abschluss einer einzelnen erfolgreichen,
fehlerhaften oder abgebrochenen Auswertung ist dieselbe Engine für einen neuen
Lauf wiederverwendbar, sofern keine andere Auswertung dieser Engine noch aktiv
ist. Cancellation beendet keine Arbeit zwangsweise; wann eine laufende
Auswertung die Anfrage beobachtet, hängt von den Prüfstellen der Auswertung
ab.

Gleichzeitig aktive Auswertungen derselben Engine teilen im aktuellen Code
ein Cancellation-Signal. Ein Cancel-Aufruf richtet sich daher gegen alle zu
diesem Zeitpunkt aktiven Auswertungen dieser Engine; die C-ABI bietet keine
per-Request-Cancellation.

## Threading-Grenze

Die Implementierung synchronisiert Registry-, Cache- und Cancellation-Zustand
intern. Eine weitergehende, öffentliche C-ABI-Garantie für beliebige parallele
Aufrufe oder für gleichzeitige Nutzung und Freigabe desselben Handles ist nicht
eindeutig aus dem Repository ableitbar. Insbesondere müssen Aufrufer die
Lebensdauer eines Handles gegen dessen gleichzeitige Nutzung selbst
synchronisieren.
