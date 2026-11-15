# Virtual tariff zones — an LHME domain demo

`lhme_virtual_tariff_demo` turns the LHME flagship path into a readable domain
case: millions of logically addressable tariff zones have a deterministically
computed price, while only negotiated local exceptions are stored.

The tariff rule is deliberately a small external `TechniqueKernel` example,
not a real pricing recommendation. It shows that an application can supply its
own rule without changing the LHME core.

## Build and run

```bash
cmake -S . -B out/build/lhme-virtual-tariff-demo -DBRUCE_BUILD_EXAMPLES=ON
cmake --build out/build/lhme-virtual-tariff-demo \
  --target lhme_virtual_tariff_demo --parallel 2
./out/build/lhme-virtual-tariff-demo/lhme_virtual_tariff_demo \
  --snapshot /tmp/lhme-virtual-tariff-demo.snapshot
```

`--snapshot` is required. Its parent directory must exist and be writable.

## The scenario

The kernel generates a price from 7.99 to 19.98 EUR for logical zones
`0..10_000_000`. There is no ten-million-row table. A Metric request for zone
`10_000_000` confirms that even this address can be evaluated without
materializing an Exact price.

The demo stores three local corrections:

| Zone | Effective price | Source |
| ---: | ---: | --- |
| 10 | 12.99 EUR | negotiated exception |
| 777 | 15.00 EUR | negotiated exception |
| 999 | 9.99 EUR | negotiated exception |

It then scans zones `0..9_999` for effective prices ending in `.99`. The
modular filter runs before Exact evaluation, including for overrides. The
verified demo output is:

```text
stream checked          | 10000
prices ending in .99    | 102
retained price offers   | 4
computed exact prices   | 100
```

Of 10,000 zones, 9,898 are discarded by the inexpensive modular comparison.
Two of the 102 matches are local corrections, so the kernel computes only 100
exact prices. `TopK` retains just the first four offers. Its retention budget
is a deterministic estimate for deliberately retained results, not a hard
process-memory limit.

The demo saves the three corrections to a snapshot, reloads them into a new
Store, and checks zone 10 again. A range of 10,001 zones is rejected by the
address budget before any Exact work begins.

## What the demo does not claim

It is not a pricing system, database, or substitute for real tariff policy. It
is a small, reproducible application slice for virtual values, local
corrections, filtered queries, and snapshot recovery.
