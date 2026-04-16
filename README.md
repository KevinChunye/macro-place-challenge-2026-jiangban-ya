# Min-Displacement Legalizer — Macro Placement Challenge 2026

## Team
- Kevin Wang (wanghaochuankevin@gmail.com, GitHub: KevinChunye)

## Method: Min-Displacement Legalizer

A legalization-based macro placer that starts from the benchmark's initial
placement and resolves all hard macro overlaps through minimum-displacement
shifts while preserving placement quality.

### Key Ideas

1. **Start from initial placement** — The provided hand-crafted placements
   already have excellent wirelength/density/congestion quality. Rather than
   placing from scratch, we treat this as a legalization problem: resolve
   overlaps with minimal perturbation.

2. **Multi-pass greedy overlap resolution** — Iteratively identifies overlapping
   macro pairs and shifts them apart using minimum-displacement separation
   vectors. Larger macros get priority.

3. **Proxy-aware spiral search** — When greedy shifting creates new overlaps,
   a spiral search finds the nearest non-overlapping position that minimizes
   incremental HPWL (half-perimeter wirelength) delta. Only nets touching the
   moved macro are re-evaluated (O(nets_per_macro) per candidate).

4. **Make-room for large macros** — Before placing a large macro, temporarily
   displaces blocking smaller macros, places the large macro near its original
   position, then re-legalizes the smaller macros.

5. **Swap fallback** — If overlap resolution stalls, attempts position swaps
   between overlapping macro pairs.

### Architecture

```
placer.py          Main entry point (MinDispLegalizer class)
legalizer_v3.py    Core legalization algorithm (911 lines)
```

## How to Run

Place both `placer.py` and `legalizer_v3.py` in your submission directory
(e.g., `submissions/our_team/`), then:

```bash
# Single benchmark
uv run evaluate submissions/our_team/placer.py -b ibm01

# All 17 benchmarks
uv run evaluate submissions/our_team/placer.py --all
```

## Results

Average proxy cost across 17 IBM benchmarks: **1.4944**
(Proxy = 1.0 * Wirelength + 0.5 * Density + 0.5 * Congestion)

| Benchmark | Proxy  | WL    | Density | Congestion | Time (s) |
|-----------|--------|-------|---------|------------|----------|
| ibm01     | 1.0891 | 0.067 | 0.858   | 1.186      |    19.9  |
| ibm02     | 1.5892 | 0.076 | 0.756   | 2.270      |  1269.1  |
| ibm03     | 1.3304 | 0.080 | 0.742   | 1.759      |    18.7  |
| ibm04     | 1.4576 | 0.078 | 0.983   | 1.775      |    37.2  |
| ibm06     | 1.7029 | 0.064 | 0.785   | 2.494      |   622.8  |
| ibm07     | 1.4826 | 0.065 | 0.834   | 2.001      |     7.5  |
| ibm08     | 1.5547 | 0.069 | 0.890   | 2.081      |    13.7  |
| ibm09     | 1.1527 | 0.058 | 0.886   | 1.304      |    20.4  |
| ibm10     | 1.3768 | 0.071 | 0.694   | 1.919      |    12.4  |
| ibm11     | 1.2319 | 0.054 | 0.885   | 1.470      |    23.7  |
| ibm12     | 1.8099 | 0.061 | 0.936   | 2.562      |    91.6  |
| ibm13     | 1.3909 | 0.054 | 0.898   | 1.776      |   102.6  |
| ibm14     | 1.5957 | 0.052 | 0.967   | 2.120      |   120.5  |
| ibm15     | 1.6033 | 0.058 | 0.932   | 2.158      |    94.8  |
| ibm16     | 1.4995 | 0.049 | 0.836   | 2.064      |    40.1  |
| ibm17     | 1.7411 | 0.054 | 0.949   | 2.426      |   667.3  |
| ibm18     | 1.7957 | 0.053 | 1.048   | 2.436      |    40.9  |
| **AVG**   |**1.4944**| 0.063 | 0.876 | 1.988      |   188.4  |

All placements are valid (zero hard macro overlaps, all macros within canvas bounds).
Average runtime: 188.4s per benchmark. Max: 1269s (ibm02).

## Dependencies

Beyond the base challenge repository:
- numpy
- torch
- (standard library: itertools, collections, pathlib, sys)

## License

Apache License 2.0
