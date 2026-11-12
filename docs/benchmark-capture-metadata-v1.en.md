# BRUCE LHME Non-Canonical Benchmark Capture Metadata, Version 1

`cmake/benchmark_capture_metadata.cmake` creates a write-once JSON sidecar
for a future archived `--csv` benchmark run. It does not create, modify, or
validate benchmark measurements. The sidecar binds one already-written CSV to
the capture context observed when the script runs.

This format is intentionally separate from the canonical evidence in
`benchmark-raw-2026-11-11-v3.csv`. The script rejects that exact canonical CSV
and writes `"capture_class": "non-canonical"` unconditionally. A sidecar is
therefore not a way to promote a later local measurement into the canonical
report.

## Format contract

The JSON document has these fixed top-level fields:

```json
{
  "schema": "bruce.lhme.benchmark-capture",
  "schema_version": 1,
  "capture_class": "non-canonical",
  "captured_at_utc": "2026-11-11T12:34:56Z",
  "csv": {
    "path": "out/build/release/fair-storage-local.csv",
    "sha256": "<64 lowercase hexadecimal characters>"
  },
  "revision": {
    "identifier": "not available",
    "availability": "not available"
  },
  "build": {
    "configuration": "Release",
    "cmake_options": ["BUILD_TESTING=ON"],
    "compiler": {
      "id": "GNU",
      "path": "/path/to/c++",
      "version": "<compiler version>"
    },
    "cmake_version": "<CMake version>",
    "generator": "<CMake generator>"
  },
  "environment": {
    "platform_label": "Linux-x86_64",
    "operating_system": "Linux <kernel version>",
    "cpu_model": "<CPU model or not available>",
    "logical_cpu_count": "<count or not available>",
    "physical_memory_bytes": "<bytes or not available>",
    "cpu_affinity": "<CPU set or not available>"
  },
  "protocol": {
    "iterations": "<exact timed-iteration protocol>",
    "warmup": "<exact warm-up protocol>"
  },
  "platform_limits": ["<applicable limitation or not available>"]
}
```

`schema`, `schema_version`, and `capture_class` are format discriminators;
consumers must reject documents that do not match all three values. All other
scalar values are JSON strings. The literal `not available` is the required
representation when a value cannot be determined. In particular, an archive
without an immutable source revision must retain both
`revision.identifier` and `revision.availability` as `not available`; do not
invent a revision from a date, directory name, or compiler version.

`csv.sha256` is a SHA-256 over the referenced CSV bytes at capture time. It is
an integrity reference, not a signature or proof that the CSV reflects a fair
benchmark. Store the CSV and its sidecar together. Recompute the SHA-256 before
using an archived result if the CSV has been moved or copied.

## Creating a sidecar

Run benchmarks first, using a fresh CSV output path. Then generate a new
sidecar path; the script refuses to overwrite either a pre-existing sidecar or
the canonical `-v3` CSV.

```bash
cmake \
  -DCAPTURE_CSV=out/build/release/fair-storage-local.csv \
  -DCAPTURE_OUTPUT=out/build/release/fair-storage-local.capture.json \
  -DCMAKE_CACHE_FILE=out/build/release/CMakeCache.txt \
  -DITERATION_PROTOCOL='11 timed lookup samples per case' \
  -DWARMUP_PROTOCOL='1 unscored warm-up before each case' \
  -DPLATFORM_LIMITS_JSON='["Linux-only isolated process/RSS measurement"]' \
  -DREVISION_IDENTIFIER='not available' \
  -P cmake/benchmark_capture_metadata.cmake
```

`CAPTURE_CSV` and `CAPTURE_OUTPUT` are required. `CAPTURED_AT_UTC` is optional:
when supplied it must use `YYYY-MM-DDTHH:MM:SSZ`; otherwise the script writes
the current UTC time at second precision. `CMAKE_CACHE_FILE` is optional but
recommended. For a normal single-config build it supplies build type, generator,
C++ compiler path/ID/version, and the relevant BRUCE options
(`BUILD_TESTING`, `BRUCE_BUILD_BENCHMARKS`, `BRUCE_BUILD_TOOLS`,
`BRUCE_BUILD_EXAMPLES`, and `BRUCE_SANITIZER`).

`CMAKE_OPTIONS_JSON` may replace the cache-derived option list; it must be a
JSON array of strings. `PLATFORM_LIMITS_JSON` likewise must be a JSON array.
Explicit scalar overrides are available for constrained or cross-compiled
captures: `BUILD_CONFIGURATION`, `GENERATOR`, `COMPILER_ID`, `COMPILER_PATH`,
`COMPILER_VERSION`, `CMAKE_VERSION_OVERRIDE`, `OPERATING_SYSTEM`,
`PLATFORM_LABEL`, `CPU_MODEL`, `LOGICAL_CPU_COUNT`,
`PHYSICAL_MEMORY_BYTES`, and `CPU_AFFINITY`.

On Linux, absent overrides are populated from CMake host information and the
current process's `/proc` view where available: CPU model, logical CPU count,
`MemTotal` converted from KiB to bytes, and `Cpus_allowed_list`. The affinity
is the script process's inherited CPU set, not a guarantee that the benchmark
process was pinned. On other systems, or when a source is unavailable, the
field is `not available` unless an operator supplies a truthful override.
The script does not determine CPU governor, power mode, temperature, or
background load; record those separately when they matter to a comparison.

## Regression coverage

With `BUILD_TESTING=ON`, CTest registers
`benchmark_capture_metadata_tests`. It uses only CMake and a marker-owned
temporary directory. The deterministic fixture checks the complete Version 1
shape, required values, a recomputed SHA-256 reference, the explicit
`not available` revision state, platform labeling, cache-derived build fields,
and rejection of an output overwrite. The test removes only a directory whose
exact ownership marker it created.
