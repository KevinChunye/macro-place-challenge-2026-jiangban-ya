"""
Min-Displacement Legalizer — Macro Placement Challenge 2026
============================================================

Author: Kevin Wang (KevinChunye)

Pipeline:
  1. Start from initial hand-crafted placement
  2. Multi-pass legalizer: greedy overlap resolution + proxy-aware spiral
     search + make-room for large macros + swap fallback
  3. Return legalized placement with zero overlaps

Usage (from challenge repo root):
    uv run evaluate submissions/our_team/placer.py
    uv run evaluate submissions/our_team/placer.py --all
    uv run evaluate submissions/our_team/placer.py -b ibm01
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project root and this file's directory are on sys.path
_this_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
for _p in [_this_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from macro_place.benchmark import Benchmark
from legalizer_v3 import legalize


def _load_plc(benchmark_name: str):
    """Load PlacementCost for net connectivity data."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(benchmark_name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d / "netlist" / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


class MinDispLegalizer:
    """
    Minimum-displacement legalizer.

    Starts from the initial hand-crafted placement and resolves overlaps
    with minimal perturbation, preserving wirelength/density/congestion quality.
    """

    def __init__(self, max_iters=50, stall_patience=3, verbose=False):
        self.max_iters = max_iters
        self.stall_patience = stall_patience
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)
        n_hard = benchmark.num_hard_macros

        legalized_hard, log = legalize(
            benchmark, plc=plc,
            max_iters=self.max_iters,
            stall_patience=self.stall_patience,
            verbose=self.verbose,
        )

        result = benchmark.macro_positions.clone()
        result[:n_hard] = torch.tensor(legalized_hard, dtype=torch.float32)

        fixed = benchmark.macro_fixed
        if fixed.any():
            result[fixed] = benchmark.macro_positions[fixed]

        return result
