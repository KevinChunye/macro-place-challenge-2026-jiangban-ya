# Min-Displacement Legalizer — Macro Placement Challenge 2026

**Team Jiangban Ya** 

Most teams treated the challenge as "optimize, then legalize." We inverted the pipeline: the IBM benchmarks' hand-crafted initial layouts already have strong wirelength, density, and congestion — so re-optimizing from scratch risks destroying quality that's already there. Our placer never moves a macro unless it has to.

<p align="center">
  <img src="assets/mindisp_ibm01.gif" alt="Min-Displacement Legalizer on ibm01" width="620"><br>
  <em>ibm01 — resolving 85 hard-macro overlaps with minimum-displacement shifts. Avg proxy cost 1.0891, ~20s runtime.</em>
</p>

**What you're seeing:**
- **Blue — hard macros:** big fixed-size blocks (SRAMs, IPs). The *only* things our algorithm moves.
- **Orange — just moved:** a hard macro that shifted in this step.
- **Pink — soft clusters:** groups of standard-cell logic the benchmark already placed. Not empty space — it's occupied. Putting a macro on top is legal, but hurts density and routing cost.
- **Right plot:** how many hard-macro pairs still overlap. We stop at zero.

**How it works:** start from the benchmark's hand-crafted layout (which is already good) and only nudge the macros that overlap. Four passes — greedy min-displacement shifts, a proxy-aware spiral search, a "make-room" move for large macros, and a swap fallback — fix everything with the smallest possible perturbation.

**Result:** avg proxy **1.4944** across 17 IBM benchmarks (~2.5% better than RePlAce), zero overlaps, ~188s avg — pure NumPy, no learning, no external solver.

### Scope and Assumption

This submission is framed as a **legalizer**, not a from-scratch placer. It assumes the input has an initial placement with exploitable structure — which is exactly what the IBM and NG45 benchmarks provide (hand-crafted for IBM; OpenROAD analytical/grouping output for NG45). On a cold, random initialization the algorithm would still produce a zero-overlap layout in bounded time, but it would inherit the random layout's wirelength and congestion — "garbage in, legal garbage out."

The natural extension, for pipelines where no useful prior exists, is to drop any fast global placer (force-directed, analytical, DREAMPlace) in front as a seeding step and then run this legalizer to finish. That's strictly better than the traditional "optimize-then-legalize" pipeline because legalization here is explicitly *displacement-bounded* against a reference, not rediscovering topology from scratch. We measure the prior-sensitivity quantitatively in the repo's noise-sweep results (`results/noise_sweep_summary.md`).

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
