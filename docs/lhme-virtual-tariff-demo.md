# Virtuelle Tarifzonen — eine LHME-Anwendungsdemo

`lhme_virtual_tariff_demo` übersetzt den LHME-Flagship-Pfad in einen
verständlich fachlichen Fall: Millionen logisch adressierbare Tarifzonen
besitzen einen deterministisch berechneten Preis. Nur verhandelte lokale
Sonderpreise werden gespeichert.

Die Tarifregel ist absichtlich ein kleines Beispiel für einen externen
`TechniqueKernel`, keine echte Preisempfehlung. Sie zeigt, dass eine Anwendung
ihre eigene Formel liefern kann, ohne den LHME-Kern zu verändern.

## Bauen und ausführen

```bash
cmake -S . -B out/build/lhme-virtual-tariff-demo -DBRUCE_BUILD_EXAMPLES=ON
cmake --build out/build/lhme-virtual-tariff-demo \
  --target lhme_virtual_tariff_demo --parallel 2
./out/build/lhme-virtual-tariff-demo/lhme_virtual_tariff_demo \
  --snapshot /tmp/lhme-virtual-tariff-demo.snapshot
```

`--snapshot` ist verpflichtend. Sein Elternverzeichnis muss existieren und
beschreibbar sein.

## Die Geschichte

Der Kernel erzeugt für die logischen Zonen `0..10_000_000` einen Preis zwischen
7,99 und 19,98 EUR. Es gibt keine Tabelle mit zehn Millionen Zeilen. Eine
Metrik-Anfrage an Zone `10_000_000` bestätigt, dass selbst diese Adresse ohne
Exact-Preis materialisiert werden kann.

Die Demo hinterlegt drei lokale Korrekturen:

| Zone | Effektiver Preis | Quelle |
| ---: | ---: | --- |
| 10 | 12,99 EUR | verhandelte Ausnahme |
| 777 | 15,00 EUR | verhandelte Ausnahme |
| 999 | 9,99 EUR | verhandelte Ausnahme |

Danach durchsucht sie die Zonen `0..9_999` nach effektiven Preisen, die auf
`.99` enden. Der Modularfilter arbeitet vor der Exact-Auswertung, auch für
Overrides. Die bestätigte Demoausgabe zeigt daher:

```text
stream checked          | 10000
prices ending in .99    | 102
retained price offers   | 4
computed exact prices   | 100
```

Von 10.000 Zonen werden 9.898 bereits über den günstigen Modularvergleich
verworfen. Zwei der 102 Treffer sind lokale Korrekturen; deshalb werden nur 100
Preise durch den Kernel exakt berechnet. `TopK` behält lediglich die ersten vier
Angebote. Das Retention-Budget ist dabei eine deterministische Schätzung für
bewusst behaltene Ergebnisse, keine harte Prozessspeichergrenze.

Die Demo speichert die drei Korrekturen in einem Snapshot, lädt sie in einen
neuen Store zurück und prüft den Preis von Zone 10 erneut. Ein Bereich mit
10.001 Zonen wird vor jeder Exact-Arbeit über das Adressbudget abgewiesen.

## Was die Demo nicht behauptet

Sie ist kein Preisfindungssystem, keine Datenbank und kein Ersatz für
fachliche Tarifregeln. Sie zeigt einen kleinen, reproduzierbaren
Anwendungszuschnitt für virtuelle Werte, lokale Korrekturen, gefilterte
Abfragen und Snapshot-Recovery.
