"""
Spectral-Seed + Adaptive Legalizer — Macro Placement Challenge 2026
====================================================================

Author: Kevin Wang (KevinChunye)

Pipeline:
  1. Probe initial placement quality via HPWL ratio (per-net HPWL / canvas perimeter)
       - Low HPWL (< 0.20): expert IBM prior — macros are correctly positioned,
         just overlapping. Trust the prior; legalize only.
       - High HPWL (> 0.40): noisy/random init — macros far from optimal.
         Build connectivity Laplacian, compute spectral layout (Fiedler vectors),
         blend with initial, force-spread, then legalize.
  2. Spectral layout (Kennings & Markov, TCAD 2004; Hall, Management Science 1970):
       Minimizes a relaxation of quadratic wirelength. Init-independent.
  3. Multi-pass legalizer: greedy overlap resolution + proxy-aware spiral
     search + make-room for large macros + swap fallback.

Usage (from challenge repo root):
    uv run evaluate submission_clean/placer.py
    uv run evaluate submission_clean/placer.py --all
    uv run evaluate submission_clean/placer.py --ng45
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_this_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
for _p in [_this_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from macro_place.benchmark import Benchmark
from legalizer_v3 import find_overlapping_pairs, force_spread_pass, legalize


# ── Benchmark loader ──────────────────────────────────────────────────────────

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


# ── Quality diagnostics ───────────────────────────────────────────────────────

def _count_overlaps(pos, sizes, n_hard):
    c = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            sx = (sizes[i, 0] + sizes[j, 0]) / 2
            sy = (sizes[i, 1] + sizes[j, 1]) / 2
            if dx < sx and dy < sy:
                c += 1
    return c


def _density_imbalance(pos, sizes, n_hard, cw, ch, grid=4):
    bins = np.zeros((grid, grid))
    for i in range(n_hard):
        bx = int(np.clip(pos[i, 0] / cw * grid, 0, grid - 1))
        by = int(np.clip(pos[i, 1] / ch * grid, 0, grid - 1))
        bins[by, bx] += sizes[i, 0] * sizes[i, 1]
    total = bins.sum()
    if total < 1e-9:
        return 0.0
    bins /= total
    return float(bins.max() / max(bins.mean(), 1e-9))


def _initial_hpwl_ratio(benchmark, plc, n_hard):
    """Per-net HPWL / (cw+ch) — primary discriminator between IBM expert
    placements (low HPWL despite overlaps) and random inits (high HPWL)."""
    if plc is None:
        return None
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    pos = benchmark.macro_positions[:n_hard].numpy()

    name_to_hidx = {}
    for hidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        name_to_hidx[plc.modules_w_pins[plc_idx].get_name()] = hidx

    total_hpwl = 0.0
    n_nets = 0
    for driver, sinks in plc.nets.items():
        pins = [driver] + list(sinks)
        xs, ys = [], []
        for pin in pins:
            parent = pin.split("/")[0]
            if parent in name_to_hidx:
                hidx = name_to_hidx[parent]
                xs.append(float(pos[hidx, 0]))
                ys.append(float(pos[hidx, 1]))
        if len(xs) >= 2:
            total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
            n_nets += 1

    if n_nets == 0:
        return None
    return (total_hpwl / n_nets) / (cw + ch)


def _probe_quality(benchmark, plc=None):
    n_hard = benchmark.num_hard_macros
    pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    n_ov = _count_overlaps(pos, sizes, n_hard)
    overlap_frac = n_ov / max(n_hard, 1)
    perimeter = 2 * (cw + ch)
    span_x = pos[:, 0].max() - pos[:, 0].min() if n_hard else 0
    span_y = pos[:, 1].max() - pos[:, 1].min() if n_hard else 0
    spread = (span_x + span_y) / max(perimeter / 2, 1e-9)
    imbalance = _density_imbalance(pos, sizes, n_hard, cw, ch)
    hpwl_ratio = _initial_hpwl_ratio(benchmark, plc, n_hard)

    return {
        "n_hard": n_hard,
        "overlap_count": n_ov,
        "overlap_frac": overlap_frac,
        "spread": spread,
        "density_imbalance": imbalance,
        "hpwl_ratio": hpwl_ratio,
    }


# ── Connectivity Laplacian ────────────────────────────────────────────────────

def _build_laplacian(benchmark, plc, n_hard):
    name_to_hidx = {}
    for hidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        name = plc.modules_w_pins[plc_idx].get_name()
        name_to_hidx[name] = hidx

    A = np.zeros((n_hard, n_hard), dtype=np.float64)

    for driver, sinks in plc.nets.items():
        pins = [driver] + list(sinks)
        net_macros = []
        for pin_name in pins:
            parent = pin_name.split("/")[0]
            if parent in name_to_hidx:
                hidx = name_to_hidx[parent]
                if hidx not in net_macros:
                    net_macros.append(hidx)
        k = len(net_macros)
        if k < 2:
            continue
        w = 1.0 / (k - 1)
        for i in range(k):
            for j in range(i + 1, k):
                mi, mj = net_macros[i], net_macros[j]
                A[mi, mj] += w
                A[mj, mi] += w

    D = np.diag(A.sum(axis=1))
    return D - A


# ── Spectral layout ───────────────────────────────────────────────────────────

def _spectral_layout(L, n_hard, movable, pos_orig, cw, ch, sizes):
    if n_hard < 3:
        return pos_orig.copy()

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L + 1e-6 * np.eye(n_hard))
    except np.linalg.LinAlgError:
        return pos_orig.copy()

    v1 = eigenvectors[:, 1]
    v2 = eigenvectors[:, 2] if eigenvectors.shape[1] > 2 else np.random.randn(n_hard)

    def scale_vec(v, lo, hi):
        vmin, vmax = v.min(), v.max()
        if abs(vmax - vmin) < 1e-9:
            return np.full_like(v, (lo + hi) / 2)
        return (v - vmin) / (vmax - vmin) * (hi - lo) + lo

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    x_new = scale_vec(v1, half_w.mean(), cw - half_w.mean())
    y_new = scale_vec(v2, half_h.mean(), ch - half_h.mean())

    pos_new = pos_orig.copy()
    for i in range(n_hard):
        if movable[i]:
            pos_new[i, 0] = x_new[i]
            pos_new[i, 1] = y_new[i]
    return pos_new


def _uniform_grid_layout(pos_orig, sizes, movable, cw, ch):
    n = len(pos_orig)
    pos_new = pos_orig.copy()
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    step_x = cw / (cols + 1)
    step_y = ch / (rows + 1)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            if movable[idx]:
                pos_new[idx, 0] = (c + 1) * step_x
                pos_new[idx, 1] = (r + 1) * step_y
            idx += 1
    return pos_new


# ── Alpha computation ─────────────────────────────────────────────────────────

_FORCE_MODE = os.environ.get("PLACERV9_MODE", "auto").lower()

# Calibrated from noise sweep:
#   IBM clean priors:    hpwl_ratio ≈ 0.05-0.12  → alpha = 0
#   gauss_10 (σ=0.10):  hpwl_ratio ≈ 0.12-0.15  → handled by legalizer (alpha≈0)
#   uniform random:     hpwl_ratio ≈ 0.35+       → alpha → 1
HPWL_GOOD_THRESHOLD = 0.20
HPWL_BAD_THRESHOLD  = 0.40
IMBALANCE_TRIGGER   = 5.0


def _compute_alpha(quality):
    if _FORCE_MODE == "spectral":
        return 1.0
    if _FORCE_MODE == "legalize":
        return 0.0

    hpwl = quality.get("hpwl_ratio")
    if hpwl is not None:
        return float(np.clip(
            (hpwl - HPWL_GOOD_THRESHOLD) / (HPWL_BAD_THRESHOLD - HPWL_GOOD_THRESHOLD),
            0.0, 1.0
        ))

    imbalance = quality["density_imbalance"]
    return float(np.clip((imbalance - IMBALANCE_TRIGGER) / 5.0, 0.0, 1.0))


# ── Main placer ───────────────────────────────────────────────────────────────

class SpectralPlacer:
    """Spectral-Seed + Adaptive Legalization.

    Automatically detects initial placement quality (via HPWL ratio) and
    chooses the appropriate strategy:
      - Good prior (IBM expert): direct legalization, preserving the prior.
      - Bad prior (random/noisy): spectral global layout from netlist
        connectivity, then legalize. Fully init-independent.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable = ~benchmark.macro_fixed[:n_hard].numpy()

        # Phase 0: load netlist + probe quality
        try:
            plc = _load_plc(benchmark.name)
        except Exception:
            plc = None

        quality = _probe_quality(benchmark, plc=plc)
        alpha = _compute_alpha(quality)

        hpwl_str = f"{quality['hpwl_ratio']:.3f}" if quality['hpwl_ratio'] is not None else "N/A"
        print(f"[placer] overlap_frac={quality['overlap_frac']:.3f}  "
              f"hpwl_ratio={hpwl_str}  "
              f"imbalance={quality['density_imbalance']:.2f}  "
              f"→ spectral_alpha={alpha:.2f}")

        pos_orig = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

        if alpha < 0.01:
            # Fast path: good prior, legalize only
            print("[placer] Good prior — direct legalize")
            pos_final, _ = legalize(benchmark, plc=plc, verbose=False)

        else:
            # Phase 1: build Laplacian
            L = None
            if plc is not None:
                try:
                    L = _build_laplacian(benchmark, plc, n_hard)
                except Exception as e:
                    print(f"[placer] Laplacian failed ({e}), using grid fallback")

            # Phase 2: spectral (or grid) layout
            if L is not None and n_hard >= 3:
                pos_spectral = _spectral_layout(L, n_hard, movable, pos_orig, cw, ch, sizes)
            else:
                pos_spectral = _uniform_grid_layout(pos_orig, sizes, movable, cw, ch)

            # Phase 3: blend spectral with original prior
            pos_blended = (1 - alpha) * pos_orig + alpha * pos_spectral

            # Clamp to canvas
            for i in range(n_hard):
                hw = sizes[i, 0] / 2;  hh = sizes[i, 1] / 2
                pos_blended[i, 0] = np.clip(pos_blended[i, 0], hw, cw - hw)
                pos_blended[i, 1] = np.clip(pos_blended[i, 1], hh, ch - hh)

            # Phase 4: force-spread to clear overlap bulk
            pairs_before = find_overlapping_pairs(pos_blended, sizes, n_hard)
            if pairs_before:
                print(f"[placer] Force-spread from {len(pairs_before)} overlaps")
                pos_blended = force_spread_pass(
                    pos_blended, sizes, movable, n_hard, cw, ch, gap=0.001, n_passes=60)
                pairs_after = find_overlapping_pairs(pos_blended, sizes, n_hard)
                print(f"[placer] After spread: {len(pairs_after)} overlaps")

            new_positions = benchmark.macro_positions.clone()
            new_positions[:n_hard] = torch.tensor(pos_blended, dtype=torch.float32)
            benchmark.macro_positions = new_positions

            # Phase 5: fine-grained legalization
            print("[placer] Legalizing...")
            pos_final, _ = legalize(benchmark, plc=plc, verbose=False)

        result = benchmark.macro_positions.clone()
        result[:n_hard] = torch.tensor(pos_final, dtype=torch.float32)

        fixed = benchmark.macro_fixed
        if fixed.any():
            result[fixed] = benchmark.macro_positions[fixed]

        return result


# Keep MinDispLegalizer as an alias for backwards compatibility with placer_v8
class MinDispLegalizer:
    def __init__(self, max_iters=50, stall_patience=3, verbose=False):
        self.max_iters = max_iters
        self.stall_patience = stall_patience
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)
        n_hard = benchmark.num_hard_macros

        legalized_hard, _ = legalize(
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
