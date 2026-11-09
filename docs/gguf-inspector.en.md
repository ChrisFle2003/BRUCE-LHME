# BRUCE GGUF Inspector V1

The BRUCE GGUF Inspector examines local GGUF-V3 model files and produces a
deterministic CPU/RAM loading plan. It replaces neither GGUF, ggml, nor
llama.cpp, and performs no inference.

## What V1 does

- Validates headers, metadata, tensor offsets, alignment, and data ranges
  fail-closed.
- Inventories tensor name, shape, GGML type, stored bytes, and estimated
  F32-decoded RAM.
- Plans tensor chunks as `resident`, `on_demand`, `deferred`, or `rejected`.
- Produces a stable text report and optionally a JSON plan.
- Can report candidates for constant bytes, repeated byte blocks, or bit-exact
  F32 sequences in explicitly selected chunks.

## What V1 explicitly does not do

- Does not write, quantize, dequantize, or reorder model weights.
- Does not alter tensor axes or scalar coordinates.
- Does not rewrite GGUF files or implement inference.
- Does not use a formula representation automatically. A pattern candidate is
  **not** a proven `FormulaRecipe` and always has `exactly_verified=false`.

A tensor's stored ordering is part of its model semantics. BRUCE sorts only
inventory and plan descriptions to produce reproducible reports; it never
sorts individual weight values. Quantized, trained weights are not
automatically formula-compressible.

## Supported container and tensor layouts

V1 accepts only local **GGUF version 3** files. Unknown versions and GGML types
are rejected rather than guessed. Size calculation supports:

| Type | Elements/block | Bytes/block |
|---|---:|---:|
| F32 | 1 | 4 |
| F16 | 1 | 2 |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | 32 | 18, 20, 22, 24, 34, 40 |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | 256 | 84, 110, 144, 176, 210, 292 |

The RAM value is a conservative estimate for F32 decoding:
`logical_element_count × 4`. It is not a prediction for a particular
inference runtime.

## Build and use

```bash
cmake --preset release -DBRUCE_BUILD_TOOLS=ON
cmake --build --preset release --target bruce_gguf_inspect

./out/build/release/bruce_gguf_inspect model.gguf \
  --memory-budget 8589934592 \
  --chunk-size 67108864 \
  --policy budgeted \
  --json-output plan.json
```

The three loading policies are:

- `all-resident`: All chunks must fit jointly within the RAM budget.
  Otherwise, the entire plan is rejected.
- `budgeted`: An optional use profile receives resident RAM first; stable order
  by file offset, tensor name, and chunk index follows. Chunks that do not fit
  remain `on_demand`.
- `metadata-only`: All chunks remain `deferred`; only inventory and plan are
  produced.

A use profile is a local text file with one tensor name per line:

```bash
./out/build/release/bruce_gguf_inspect model.gguf \
  --memory-budget 8589934592 \
  --chunk-size 67108864 \
  --policy budgeted \
  --use-profile first-use.txt \
  --scan
```

`--scan` is bounded and read-only. It does not modify the source file. The
inspector neither downloads models from the network nor opens network
connections.

## Determinism and boundaries

The same GGUF bytes and configuration yield the same inventory order, plan,
and trace. The input fingerprint is a deterministic FNV-derived identifier for
comparison and tracing, not cryptographic security evidence.

V1 is deliberately an inspector and planner. A later optional lossless
rewriter or FormulaRecipe sidecar requires separate compatibility and byte
validation and is not part of this version.
