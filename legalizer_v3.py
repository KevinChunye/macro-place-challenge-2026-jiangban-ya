"""
Legalizer v3: Proxy-aware permutation search + make-room for large macros.

Improvements over v2:
  Problem 1 — Proxy-aware ordering: Evaluate incremental HPWL delta (only
    nets touching component macros) for each permutation candidate. O(nets_in_component)
    per candidate, not O(all_nets).

  Problem 2 — Make-room for large macros: Before spiral-searching a large
    macro, temporarily displace blocking small macros, place the large macro
    near its original position, then re-legalize the small macros.
"""

import numpy as np
from itertools import permutations
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

# ── Core overlap detection and greedy resolution (inlined from legalizer_v1) ─

def count_overlaps_for_macro(idx, pos, sizes, n_hard, gap=0.0):
    """Count how many other hard macros overlap with macro idx."""
    count = 0
    for j in range(n_hard):
        if j == idx:
            continue
        dx = abs(pos[idx, 0] - pos[j, 0])
        dy = abs(pos[idx, 1] - pos[j, 1])
        sep_x = (sizes[idx, 0] + sizes[j, 0]) / 2 + gap
        sep_y = (sizes[idx, 1] + sizes[j, 1]) / 2 + gap
        if dx < sep_x and dy < sep_y:
            count += 1
    return count


def find_overlapping_pairs(pos, sizes, n_hard, gap=0.0):
    """Find all overlapping hard macro pairs, sorted by overlap area descending."""
    pairs = []
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            sep_x = (sizes[i, 0] + sizes[j, 0]) / 2
            sep_y = (sizes[i, 1] + sizes[j, 1]) / 2
            ov_x = sep_x - dx
            ov_y = sep_y - dy
            if ov_x > 1e-9 and ov_y > 1e-9:
                pairs.append((i, j, ov_x, ov_y, ov_x * ov_y))
    pairs.sort(key=lambda p: -p[4])
    return pairs


def macro_has_any_overlap(idx, pos, sizes, n_hard, gap=0.0):
    """Check if macro idx overlaps any other hard macro."""
    for j in range(n_hard):
        if j == idx:
            continue
        dx = abs(pos[idx, 0] - pos[j, 0])
        dy = abs(pos[idx, 1] - pos[j, 1])
        sep_x = (sizes[idx, 0] + sizes[j, 0]) / 2 + gap
        sep_y = (sizes[idx, 1] + sizes[j, 1]) / 2 + gap
        if dx < sep_x and dy < sep_y:
            return True
    return False


def build_net_neighbors(benchmark, plc):
    """Build macro-to-macro adjacency from net connectivity."""
    n_hard = benchmark.num_hard_macros
    name_to_tidx = {}
    for tidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        name_to_tidx[plc.modules_w_pins[plc_idx].get_name()] = tidx
    neighbors = defaultdict(set)
    for driver, sinks in plc.nets.items():
        pins = [driver] + list(sinks)
        macros_in_net = set()
        for pin_name in pins:
            parent = pin_name.split("/")[0]
            if parent in name_to_tidx:
                macros_in_net.add(name_to_tidx[parent])
        for a in macros_in_net:
            for b in macros_in_net:
                if a != b:
                    neighbors[a].add(b)
    return {k: sorted(v) for k, v in neighbors.items()}


def resolve_pair(i, j, pos, sizes, movable, n_hard, cw, ch, neighbors, gap=0.001):
    """Resolve overlap between macros i and j by minimum displacement.
    Moves the smaller macro. Only accepts moves that reduce overlap count."""
    area_i = sizes[i, 0] * sizes[i, 1]
    area_j = sizes[j, 0] * sizes[j, 1]
    if not movable[i] and not movable[j]:
        return None
    if not movable[i]:
        mover, anchor = j, i
    elif not movable[j]:
        mover, anchor = i, j
    elif area_i <= area_j:
        mover, anchor = i, j
    else:
        mover, anchor = j, i

    half_w = sizes[mover, 0] / 2
    half_h = sizes[mover, 1] / 2
    sep_x = (sizes[mover, 0] + sizes[anchor, 0]) / 2 + gap
    sep_y = (sizes[mover, 1] + sizes[anchor, 1]) / 2 + gap
    old_x, old_y = pos[mover, 0], pos[mover, 1]

    candidates = []
    nx = pos[anchor, 0] + sep_x
    if nx + half_w <= cw and nx - half_w >= 0:
        candidates.append((abs(nx - old_x), nx, old_y))
    nx = pos[anchor, 0] - sep_x
    if nx + half_w <= cw and nx - half_w >= 0:
        candidates.append((abs(nx - old_x), nx, old_y))
    ny = pos[anchor, 1] + sep_y
    if ny + half_h <= ch and ny - half_h >= 0:
        candidates.append((abs(ny - old_y), old_x, ny))
    ny = pos[anchor, 1] - sep_y
    if ny + half_h <= ch and ny - half_h >= 0:
        candidates.append((abs(ny - old_y), old_x, ny))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])

    net_nbrs = neighbors.get(mover, [])
    if net_nbrs:
        hard_nbrs = [n for n in net_nbrs if n < n_hard]
        if hard_nbrs:
            centroid_x = np.mean([pos[n, 0] for n in hard_nbrs])
            centroid_y = np.mean([pos[n, 1] for n in hard_nbrs])
            best_dist = candidates[0][0]
            threshold = best_dist * 1.2 + 0.01
            near_best = [c for c in candidates if c[0] <= threshold]
            near_best.sort(key=lambda c: abs(c[1] - centroid_x) + abs(c[2] - centroid_y))
            rest = [c for c in candidates if c not in near_best]
            candidates = near_best + rest

    overlaps_before = count_overlaps_for_macro(mover, pos, sizes, n_hard)
    for _, nx, ny in candidates:
        pos[mover, 0] = nx
        pos[mover, 1] = ny
        if count_overlaps_for_macro(mover, pos, sizes, n_hard) < overlaps_before:
            return mover, (old_x, old_y), (nx, ny)
        pos[mover, 0] = old_x
        pos[mover, 1] = old_y
    return None


# ── Net data structures for incremental HPWL ────────────────────────────────

class NetIndex:
    """Precomputed net data for fast incremental HPWL evaluation."""

    def __init__(self, benchmark, plc):
        n_hard = benchmark.num_hard_macros
        name_to_tidx = {}
        for tidx, plc_idx in enumerate(benchmark.hard_macro_indices):
            name_to_tidx[plc.modules_w_pins[plc_idx].get_name()] = tidx
        for i, plc_idx in enumerate(benchmark.soft_macro_indices):
            name_to_tidx[plc.modules_w_pins[plc_idx].get_name()] = n_hard + i

        port_pos = {}
        for plc_idx in plc.port_indices:
            node = plc.modules_w_pins[plc_idx]
            port_pos[node.get_name()] = node.get_pos()

        # net_list[i] = list of (is_hard_macro, tidx, fixed_x, fixed_y)
        self.net_list = []
        # macro_to_nets[tidx] = set of net indices containing this macro
        self.macro_to_nets = defaultdict(set)

        net_idx = 0
        for driver, sinks in plc.nets.items():
            pins = []
            seen = set()
            for pin_name in [driver] + list(sinks):
                parent = pin_name.split("/")[0]
                if parent in name_to_tidx:
                    tidx = name_to_tidx[parent]
                    if tidx not in seen:
                        seen.add(tidx)
                        pins.append((True, tidx, 0.0, 0.0))
                elif parent in port_pos:
                    x, y = port_pos[parent]
                    pins.append((False, -1, x, y))
            if len(pins) >= 2:
                self.net_list.append(pins)
                for is_m, tidx, _, _ in pins:
                    if is_m:
                        self.macro_to_nets[tidx].add(net_idx)
                net_idx += 1

    def hpwl_for_nets(self, net_indices, pos_hard, pos_all, n_hard):
        """Compute HPWL for a subset of nets."""
        total = 0.0
        for ni in net_indices:
            pins = self.net_list[ni]
            min_x = float('inf'); max_x = float('-inf')
            min_y = float('inf'); max_y = float('-inf')
            for is_m, idx, fx, fy in pins:
                if is_m:
                    if idx < n_hard:
                        x, y = pos_hard[idx, 0], pos_hard[idx, 1]
                    else:
                        x, y = pos_all[idx, 0], pos_all[idx, 1]
                else:
                    x, y = fx, fy
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
            total += (max_x - min_x) + (max_y - min_y)
        return total

    def nets_for_macros(self, macro_set):
        """Get all net indices touching any macro in macro_set."""
        nets = set()
        for m in macro_set:
            nets.update(self.macro_to_nets.get(m, set()))
        return nets


# ── Conflict graph ───────────────────────────────────────────────────────────

def build_conflict_components(pairs, n_hard):
    adj = defaultdict(list)
    nodes = set()
    for i, j, *_ in pairs:
        nodes.add(i); nodes.add(j)
        if j not in adj[i]: adj[i].append(j)
        if i not in adj[j]: adj[j].append(i)
    # Sort adjacency lists for deterministic BFS order
    for k in adj:
        adj[k].sort()
    visited = set()
    components = []
    for node in sorted(nodes):
        if node in visited:
            continue
        comp = []
        queue = [node]
        visited.add(node)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for nb in adj[curr]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(sorted(comp))
    return components


# ── Spiral search ────────────────────────────────────────────────────────────

def spiral_search_single(idx, pos, sizes, movable, n_hard, cw, ch, gap=0.001):
    if not movable[idx]:
        return False, 0.0
    orig_x, orig_y = pos[idx, 0], pos[idx, 1]
    half_w = sizes[idx, 0] / 2
    half_h = sizes[idx, 1] / 2
    step = max(sizes[idx, 0], sizes[idx, 1]) * 0.15
    best_pos = None
    best_dist = float("inf")
    for r in range(1, 300):
        found = False
        for dxm in range(-r, r + 1):
            for dym in range(-r, r + 1):
                if abs(dxm) != r and abs(dym) != r:
                    continue
                cx = np.clip(orig_x + dxm * step, half_w, cw - half_w)
                cy = np.clip(orig_y + dym * step, half_h, ch - half_h)
                pos[idx, 0] = cx; pos[idx, 1] = cy
                if not macro_has_any_overlap(idx, pos, sizes, n_hard, gap=gap):
                    dist = abs(cx - orig_x) + abs(cy - orig_y)
                    if dist < best_dist:
                        best_dist = dist; best_pos = (cx, cy); found = True
        if found:
            pos[idx, 0] = best_pos[0]; pos[idx, 1] = best_pos[1]
            return True, best_dist
    pos[idx, 0] = orig_x; pos[idx, 1] = orig_y
    return False, 0.0


# ── Make-room for large macros ───────────────────────────────────────────────

def make_room_and_place(large_idx, pos, sizes, movable, n_hard, cw, ch, gap=0.001):
    """Temporarily displace blocking small macros, place large macro, re-legalize."""
    orig_large = (pos[large_idx, 0], pos[large_idx, 1])
    large_area = sizes[large_idx, 0] * sizes[large_idx, 1]

    # Find small overlapping macros
    blockers = []
    for j in range(n_hard):
        if j == large_idx or not movable[j]:
            continue
        dx = abs(pos[large_idx, 0] - pos[j, 0])
        dy = abs(pos[large_idx, 1] - pos[j, 1])
        sep_x = (sizes[large_idx, 0] + sizes[j, 0]) / 2 + gap
        sep_y = (sizes[large_idx, 1] + sizes[j, 1]) / 2 + gap
        if dx < sep_x and dy < sep_y:
            j_area = sizes[j, 0] * sizes[j, 1]
            if j_area < large_area:
                blockers.append(j)

    if not blockers:
        ok, disp = spiral_search_single(large_idx, pos, sizes, movable, n_hard, cw, ch, gap)
        return ok, disp, []

    blocker_orig = {b: (pos[b, 0], pos[b, 1]) for b in blockers}

    # Temporarily push blockers to far corners
    for b in blockers:
        half_w = sizes[b, 0] / 2; half_h = sizes[b, 1] / 2
        corners = [(half_w, half_h), (cw - half_w, half_h),
                    (half_w, ch - half_h), (cw - half_w, ch - half_h)]
        farthest = max(corners, key=lambda c: abs(c[0] - orig_large[0]) + abs(c[1] - orig_large[1]))
        pos[b, 0], pos[b, 1] = farthest

    # Place large macro
    pos[large_idx, 0], pos[large_idx, 1] = orig_large
    if not macro_has_any_overlap(large_idx, pos, sizes, n_hard, gap=gap):
        large_disp = 0.0; large_ok = True
    else:
        pos[large_idx, 0], pos[large_idx, 1] = orig_large
        large_ok, large_disp = spiral_search_single(
            large_idx, pos, sizes, movable, n_hard, cw, ch, gap)

    if not large_ok:
        pos[large_idx, 0], pos[large_idx, 1] = orig_large
        for b in blockers:
            pos[b, 0], pos[b, 1] = blocker_orig[b]
        return False, 0.0, []

    # Re-legalize blockers
    small_results = []
    for b in blockers:
        pos[b, 0], pos[b, 1] = blocker_orig[b]
        if not macro_has_any_overlap(b, pos, sizes, n_hard, gap=gap):
            small_results.append((b, 0.0))
        else:
            ok, disp = spiral_search_single(b, pos, sizes, movable, n_hard, cw, ch, gap)
            small_results.append((b, disp if ok else 0.0))

    return True, large_disp, small_results


# ── Proxy-aware component resolution ────────────────────────────────────────

def resolve_component(
    component, pos, sizes, movable, n_hard, cw, ch,
    net_index, pos_all, gap=0.001, max_perm=120
):
    """Try permutations; pick ordering minimizing incremental HPWL delta."""
    if not component:
        return []

    orig_positions = {m: (pos[m, 0], pos[m, 1]) for m in component}
    active = [m for m in component
              if movable[m] and macro_has_any_overlap(m, pos, sizes, n_hard, gap=gap)]
    if not active:
        return []

    areas = {m: sizes[m, 0] * sizes[m, 1] for m in active}
    median_area = np.median(list(areas.values())) if areas else 1.0
    large_macros = set(m for m in active if areas[m] > 4 * median_area)

    # Get nets touching this component (for incremental HPWL)
    relevant_nets = net_index.nets_for_macros(set(active)) if net_index else set()

    # Generate permutations
    if len(active) > 5:
        perms_to_try = [
            tuple(sorted(active, key=lambda m: -areas[m])),
            tuple(sorted(active, key=lambda m: areas[m])),
        ]
    else:
        perms_to_try = list(permutations(active))
        if len(perms_to_try) > max_perm:
            import random
            rng = random.Random(42)
            perms_to_try = list(rng.sample(perms_to_try, max_perm - 2))
        perms_to_try.append(tuple(sorted(active, key=lambda m: -areas[m])))
        perms_to_try.append(tuple(sorted(active, key=lambda m: areas[m])))

    best_result = None
    best_wl = float("inf")
    best_positions = None

    for perm in perms_to_try:
        for m in component:
            pos[m, 0], pos[m, 1] = orig_positions[m]

        result = []
        all_ok = True

        for midx in perm:
            if not macro_has_any_overlap(midx, pos, sizes, n_hard, gap=gap):
                result.append((midx, 0.0))
                continue

            if midx in large_macros:
                ok, disp, small_results = make_room_and_place(
                    midx, pos, sizes, movable, n_hard, cw, ch, gap)
                result.append((midx, disp))
                for si, sd in small_results:
                    if sd > 0:
                        result.append((si, sd))
                if not ok:
                    all_ok = False; break
            else:
                ok, disp = spiral_search_single(
                    midx, pos, sizes, movable, n_hard, cw, ch, gap)
                result.append((midx, disp))
                if not ok:
                    all_ok = False; break

        if not all_ok:
            continue

        # Evaluate incremental HPWL for nets touching this component
        if relevant_nets:
            wl = net_index.hpwl_for_nets(relevant_nets, pos[:n_hard], pos_all, n_hard)
        else:
            wl = sum(d for _, d in result)  # fallback: minimize displacement

        if wl < best_wl:
            best_wl = wl
            best_result = result
            best_positions = {m: (pos[m, 0], pos[m, 1]) for m in range(n_hard)}

    if best_result is not None:
        for m in range(n_hard):
            if m in best_positions:
                pos[m, 0], pos[m, 1] = best_positions[m]
        return best_result
    else:
        for m in component:
            pos[m, 0], pos[m, 1] = orig_positions[m]
        return [(m, 0.0) for m in active]


# ── Post-legalization displacement reduction ─────────────────────────────────

def reduce_displacement(
    pos, orig_pos, sizes, movable, n_hard, cw, ch,
    gap=0.001, n_passes=3, top_k=50, verbose=True,
):
    """Shift displaced macros back toward their original positions.

    For each of the top_k most-displaced macros, try shifting it 50%, 30%,
    20%, 10% of the way back toward its original position. Accept the largest
    shift that doesn't create any overlaps. Run for n_passes.
    """
    fractions = [0.50, 0.30, 0.20, 0.10]

    total_before = np.abs(pos - orig_pos).sum()
    if verbose:
        print(f"  Displacement reduction: total_disp_before={total_before:.4f}um")

    for pass_idx in range(n_passes):
        # Compute per-macro displacement
        per_disp = np.abs(pos - orig_pos).sum(axis=1)
        # Get top-k most displaced movable macros
        candidates = [(per_disp[i], i) for i in range(n_hard) if movable[i] and per_disp[i] > 0.01]
        candidates.sort(reverse=True)
        candidates = candidates[:top_k]

        improvements = 0
        for _, idx in candidates:
            ox, oy = pos[idx, 0], pos[idx, 1]
            target_x, target_y = orig_pos[idx, 0], orig_pos[idx, 1]
            half_w = sizes[idx, 0] / 2
            half_h = sizes[idx, 1] / 2

            for frac in fractions:
                nx = ox + frac * (target_x - ox)
                ny = oy + frac * (target_y - oy)
                # Clip to canvas
                nx = np.clip(nx, half_w, cw - half_w)
                ny = np.clip(ny, half_h, ch - half_h)

                # Check if this reduces displacement
                old_disp = abs(ox - target_x) + abs(oy - target_y)
                new_disp = abs(nx - target_x) + abs(ny - target_y)
                if new_disp >= old_disp - 0.001:
                    continue  # no improvement

                pos[idx, 0] = nx
                pos[idx, 1] = ny
                if not macro_has_any_overlap(idx, pos, sizes, n_hard, gap=gap):
                    improvements += 1
                    break  # accept this shift
                # Revert
                pos[idx, 0] = ox
                pos[idx, 1] = oy

        if verbose:
            total_after = np.abs(pos - orig_pos).sum()
            print(f"    pass {pass_idx}: {improvements} macros shifted back, "
                  f"total_disp={total_after:.4f}um")

        if improvements == 0:
            break

    total_after = np.abs(pos - orig_pos).sum()
    if verbose:
        reduction = total_before - total_after
        print(f"  Displacement reduction: {reduction:.4f}um reduced "
              f"({reduction/max(total_before, 1e-9)*100:.1f}%)")

    return pos


# ── Proxy-evaluated swap refinement ──────────────────────────────────────────

def _eval_proxy(pos, benchmark, plc, n_hard):
    """Evaluate proxy cost by writing positions into plc."""
    import torch
    from macro_place.objective import compute_proxy_cost
    placement = benchmark.macro_positions.clone()
    placement[:n_hard] = torch.tensor(pos, dtype=torch.float32)
    fixed = benchmark.macro_fixed
    if fixed.any():
        placement[fixed] = benchmark.macro_positions[fixed]
    costs = compute_proxy_cost(placement, benchmark, plc)
    return costs["proxy_cost"]


def swap_refine(
    pos, sizes, movable, n_hard, cw, ch, displaced_indices,
    benchmark, plc, gap=0.001, n_passes=2, verbose=True,
):
    """Proxy-evaluated swap refinement between displaced macros.

    For each pair of displaced macros with similar sizes and nearby positions,
    try swapping their positions. Accept if proxy cost improves and zero overlaps.
    Also try 3-way rotations among displaced triples.
    """
    if verbose:
        print(f"  Swap refine: {len(displaced_indices)} displaced macros")

    areas = sizes[:, 0] * sizes[:, 1]
    displaced_set = set(displaced_indices)

    # Build candidate swap pairs: similar size (within 20% on both dims) + nearby
    swap_candidates = []
    for ii, i in enumerate(displaced_indices):
        for jj in range(ii + 1, len(displaced_indices)):
            j = displaced_indices[jj]
            # Size similarity: both dimensions within 20%
            w_ratio = sizes[i, 0] / max(sizes[j, 0], 1e-9)
            h_ratio = sizes[i, 1] / max(sizes[j, 1], 1e-9)
            if not (0.8 <= w_ratio <= 1.25 and 0.8 <= h_ratio <= 1.25):
                continue
            # Proximity: within 3x the larger macro's width
            dist = abs(pos[i, 0] - pos[j, 0]) + abs(pos[i, 1] - pos[j, 1])
            max_dim = max(sizes[i, 0], sizes[j, 0], sizes[i, 1], sizes[j, 1])
            if dist > 3 * max_dim:
                continue
            swap_candidates.append((i, j))

    if verbose:
        print(f"  Swap candidates: {len(swap_candidates)} pairs")

    if not swap_candidates:
        return pos

    # Need a fresh plc for each eval — reuse by resetting positions
    from macro_place.loader import load_benchmark_from_dir
    bdir = f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}"
    _, plc_eval = load_benchmark_from_dir(bdir)

    current_proxy = _eval_proxy(pos, benchmark, plc_eval, n_hard)
    if verbose:
        print(f"  Proxy before swaps: {current_proxy:.4f}")

    total_accepted = 0
    for pass_idx in range(n_passes):
        accepted = 0
        for i, j in swap_candidates:
            # Save
            oi = (pos[i, 0], pos[i, 1])
            oj = (pos[j, 0], pos[j, 1])

            # Swap positions (clip for size differences)
            hw_i, hh_i = sizes[i, 0] / 2, sizes[i, 1] / 2
            hw_j, hh_j = sizes[j, 0] / 2, sizes[j, 1] / 2
            pos[i, 0] = np.clip(oj[0], hw_i, cw - hw_i)
            pos[i, 1] = np.clip(oj[1], hh_i, ch - hh_i)
            pos[j, 0] = np.clip(oi[0], hw_j, cw - hw_j)
            pos[j, 1] = np.clip(oi[1], hh_j, ch - hh_j)

            # Check overlaps
            if (macro_has_any_overlap(i, pos, sizes, n_hard, gap=gap) or
                macro_has_any_overlap(j, pos, sizes, n_hard, gap=gap)):
                pos[i, 0], pos[i, 1] = oi
                pos[j, 0], pos[j, 1] = oj
                continue

            # Evaluate proxy
            _, plc_eval2 = load_benchmark_from_dir(bdir)
            new_proxy = _eval_proxy(pos, benchmark, plc_eval2, n_hard)

            if new_proxy < current_proxy - 1e-6:
                current_proxy = new_proxy
                accepted += 1
                if verbose:
                    print(f"    Swap {benchmark.macro_names[i][:10]} <-> "
                          f"{benchmark.macro_names[j][:10]}: "
                          f"proxy={new_proxy:.4f} ({new_proxy - current_proxy + (current_proxy - new_proxy):+.4f})")
            else:
                # Revert
                pos[i, 0], pos[i, 1] = oi
                pos[j, 0], pos[j, 1] = oj

        total_accepted += accepted
        if verbose:
            print(f"    pass {pass_idx}: {accepted} swaps accepted, "
                  f"proxy={current_proxy:.4f}")
        if accepted == 0:
            break

    # Try 3-way rotations among displaced triples with similar sizes
    if len(displaced_indices) >= 3 and total_accepted == 0:
        if verbose:
            print(f"  Trying 3-way rotations...")
        # Build triples from swap candidates sharing a node
        from collections import defaultdict
        adj = defaultdict(set)
        for i, j in swap_candidates:
            adj[i].add(j)
            adj[j].add(i)

        triples_tried = 0
        for a in displaced_indices:
            if triples_tried > 200:
                break
            for b in adj.get(a, set()):
                for c in adj.get(b, set()):
                    if c == a or c not in adj.get(a, set()):
                        continue
                    triples_tried += 1
                    if triples_tried > 200:
                        break
                    # Try rotation: A->B, B->C, C->A
                    oa = (pos[a, 0], pos[a, 1])
                    ob = (pos[b, 0], pos[b, 1])
                    oc = (pos[c, 0], pos[c, 1])

                    hw_a, hh_a = sizes[a, 0]/2, sizes[a, 1]/2
                    hw_b, hh_b = sizes[b, 0]/2, sizes[b, 1]/2
                    hw_c, hh_c = sizes[c, 0]/2, sizes[c, 1]/2

                    pos[a, 0] = np.clip(ob[0], hw_a, cw-hw_a)
                    pos[a, 1] = np.clip(ob[1], hh_a, ch-hh_a)
                    pos[b, 0] = np.clip(oc[0], hw_b, cw-hw_b)
                    pos[b, 1] = np.clip(oc[1], hh_b, ch-hh_b)
                    pos[c, 0] = np.clip(oa[0], hw_c, cw-hw_c)
                    pos[c, 1] = np.clip(oa[1], hh_c, ch-hh_c)

                    if (macro_has_any_overlap(a, pos, sizes, n_hard, gap=gap) or
                        macro_has_any_overlap(b, pos, sizes, n_hard, gap=gap) or
                        macro_has_any_overlap(c, pos, sizes, n_hard, gap=gap)):
                        pos[a, 0], pos[a, 1] = oa
                        pos[b, 0], pos[b, 1] = ob
                        pos[c, 0], pos[c, 1] = oc
                        continue

                    _, plc_eval3 = load_benchmark_from_dir(bdir)
                    new_proxy = _eval_proxy(pos, benchmark, plc_eval3, n_hard)
                    if new_proxy < current_proxy - 1e-6:
                        current_proxy = new_proxy
                        total_accepted += 1
                        if verbose:
                            print(f"    3-way: {benchmark.macro_names[a][:8]}->"
                                  f"{benchmark.macro_names[b][:8]}->"
                                  f"{benchmark.macro_names[c][:8]}: "
                                  f"proxy={new_proxy:.4f}")
                    else:
                        pos[a, 0], pos[a, 1] = oa
                        pos[b, 0], pos[b, 1] = ob
                        pos[c, 0], pos[c, 1] = oc

        if verbose:
            print(f"  3-way rotations: {total_accepted} total accepted")

    if verbose:
        print(f"  Swap refine done: {total_accepted} total accepted, "
              f"proxy={current_proxy:.4f}")

    return pos


# ── Main legalizer v3 ────────────────────────────────────────────────────────

def legalize(
    benchmark,
    plc=None,
    max_iters=50,
    stall_patience=3,
    gap=0.001,
    verbose=True,
):
    n_hard = benchmark.num_hard_macros
    pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    movable = ~benchmark.macro_fixed[:n_hard].numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    pos_all = benchmark.macro_positions.numpy().copy().astype(np.float64)

    neighbors = {}
    net_index = None
    if plc is not None:
        neighbors = build_net_neighbors(benchmark, plc)
        net_index = NetIndex(benchmark, plc)

    log = []
    best_overlap_count = float("inf")
    stall_count = 0

    for iteration in range(max_iters):
        pairs = find_overlapping_pairs(pos, sizes, n_hard)
        n_overlaps = len(pairs)

        if verbose:
            total_ov_area = sum(p[4] for p in pairs)
            print(f"  iter {iteration:3d}: overlaps={n_overlaps:4d}  "
                  f"total_ov_area={total_ov_area:.6f}")

        log.append({"iteration": iteration, "overlap_count": n_overlaps,
                     "total_overlap_area": sum(p[4] for p in pairs)})

        if n_overlaps == 0:
            if verbose:
                print(f"  Converged: zero overlaps at iteration {iteration}")
            break

        if n_overlaps < best_overlap_count:
            best_overlap_count = n_overlaps
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= stall_patience:
            if verbose:
                print(f"  Greedy stalled at {n_overlaps} overlaps. "
                      f"Proxy-aware spiral + make-room...")

            components = build_conflict_components(pairs, n_hard)
            # Sort: components with large macros first
            def comp_max_area(comp):
                return max(sizes[m, 0] * sizes[m, 1] for m in comp)
            components.sort(key=comp_max_area, reverse=True)

            if verbose:
                print(f"  {len(components)} components "
                      f"(sizes: {[len(c) for c in components[:10]]}...)")

            total_resolved = 0
            spiral_disps = []
            for ci, comp in enumerate(components):
                result = resolve_component(
                    comp, pos, sizes, movable, n_hard, cw, ch,
                    net_index, pos_all, gap
                )
                for midx, disp in result:
                    if disp > 0:
                        total_resolved += 1
                        spiral_disps.append((midx, disp))
                        if verbose and disp > 2.0:
                            print(f"    comp {ci}: {benchmark.macro_names[midx]:<12} "
                                  f"disp={disp:.4f}um "
                                  f"size=({sizes[midx, 0]:.2f}x{sizes[midx, 1]:.2f})")

            pairs_after = find_overlapping_pairs(pos, sizes, n_hard)
            if verbose:
                print(f"  Resolved {total_resolved} macros. "
                      f"Remaining: {len(pairs_after)}")
                if spiral_disps:
                    print(f"  Displacement: max={max(d for _, d in spiral_disps):.4f}um  "
                          f"avg={np.mean([d for _, d in spiral_disps]):.4f}um")

            log.append({"iteration": iteration + 0.5,
                         "overlap_count": len(pairs_after),
                         "total_overlap_area": sum(p[4] for p in pairs_after),
                         "spiral": True})

            if len(pairs_after) == 0:
                break
            best_overlap_count = len(pairs_after)
            stall_count = 0
            continue

        # Greedy pass
        moved = set()
        for i, j, ov_x, ov_y, ov_area in pairs:
            if i in moved or j in moved:
                continue
            result = resolve_pair(i, j, pos, sizes, movable, n_hard, cw, ch, neighbors, gap)
            if result is not None:
                moved.add(result[0])

    # Final cleanup: if any overlaps remain, try swap-based fallback
    final_pairs = find_overlapping_pairs(pos, sizes, n_hard)
    if final_pairs:
        if verbose:
            print(f"  {len(final_pairs)} overlaps remain after main loop. "
                  f"Attempting swap fallback...")
        for i, j, ov_x, ov_y, ov_area in final_pairs:
            # Try swapping each overlapping macro with any non-overlapping
            # macro of similar size that's far from the conflict
            for mover in [i, j]:
                if not movable[mover]:
                    continue
                anchor = j if mover == i else i
                mover_area = sizes[mover, 0] * sizes[mover, 1]
                best_swap = None
                best_disp = float("inf")

                for k in range(n_hard):
                    if k == mover or k == anchor or not movable[k]:
                        continue
                    k_area = sizes[k, 0] * sizes[k, 1]
                    ratio = mover_area / max(k_area, 1e-9)
                    if not (0.5 <= ratio <= 2.0):
                        continue
                    # Check k is not currently overlapping anything
                    if macro_has_any_overlap(k, pos, sizes, n_hard, gap=gap):
                        continue

                    # Try swap: mover goes to k's position, k goes to mover's
                    om = (pos[mover, 0], pos[mover, 1])
                    ok_ = (pos[k, 0], pos[k, 1])
                    hw_m, hh_m = sizes[mover, 0]/2, sizes[mover, 1]/2
                    hw_k, hh_k = sizes[k, 0]/2, sizes[k, 1]/2

                    pos[mover, 0] = np.clip(ok_[0], hw_m, cw - hw_m)
                    pos[mover, 1] = np.clip(ok_[1], hh_m, ch - hh_m)
                    pos[k, 0] = np.clip(om[0], hw_k, cw - hw_k)
                    pos[k, 1] = np.clip(om[1], hh_k, ch - hh_k)

                    if (not macro_has_any_overlap(mover, pos, sizes, n_hard, gap=gap) and
                        not macro_has_any_overlap(k, pos, sizes, n_hard, gap=gap)):
                        disp = (abs(pos[mover, 0] - om[0]) + abs(pos[mover, 1] - om[1]) +
                                abs(pos[k, 0] - ok_[0]) + abs(pos[k, 1] - ok_[1]))
                        if disp < best_disp:
                            best_disp = disp
                            best_swap = (k, pos[mover, 0], pos[mover, 1],
                                         pos[k, 0], pos[k, 1])

                    # Revert
                    pos[mover, 0], pos[mover, 1] = om
                    pos[k, 0], pos[k, 1] = ok_

                if best_swap is not None:
                    k, mx, my, kx, ky = best_swap
                    pos[mover, 0], pos[mover, 1] = mx, my
                    pos[k, 0], pos[k, 1] = kx, ky
                    if verbose:
                        print(f"    Swap: {benchmark.macro_names[mover]} <-> "
                              f"{benchmark.macro_names[k]} (disp={best_disp:.4f}um)")
                    break  # resolved this pair
            else:
                # Neither macro in the pair could be swapped — try pushing anchor
                for mover, anchor in [(i, j), (j, i)]:
                    if not movable[anchor]:
                        continue
                    # Push anchor just enough to clear overlap
                    oa = (pos[anchor, 0], pos[anchor, 1])
                    sep_x = (sizes[mover, 0] + sizes[anchor, 0]) / 2 + gap
                    # Try pushing anchor left
                    new_x = pos[mover, 0] - sep_x
                    hw_a = sizes[anchor, 0] / 2
                    if new_x - hw_a >= 0:
                        pos[anchor, 0] = new_x
                        if not macro_has_any_overlap(anchor, pos, sizes, n_hard, gap=gap):
                            if verbose:
                                print(f"    Push: {benchmark.macro_names[anchor]} "
                                      f"left by {abs(new_x - oa[0]):.4f}um")
                            break
                        pos[anchor, 0] = oa[0]
                    # Try pushing anchor right
                    new_x = pos[mover, 0] + sep_x
                    if new_x + hw_a <= cw:
                        pos[anchor, 0] = new_x
                        if not macro_has_any_overlap(anchor, pos, sizes, n_hard, gap=gap):
                            if verbose:
                                print(f"    Push: {benchmark.macro_names[anchor]} "
                                      f"right by {abs(new_x - oa[0]):.4f}um")
                            break
                        pos[anchor, 0] = oa[0]

    final_pairs = find_overlapping_pairs(pos, sizes, n_hard)
    if len(final_pairs) == 0:
        pos = reduce_displacement(
            pos, benchmark.macro_positions[:n_hard].numpy().astype(np.float64),
            sizes, movable, n_hard, cw, ch, gap=gap,
            n_passes=3, top_k=50, verbose=verbose)

    # Swap refinement disabled — empirically <0.001 proxy improvement
    # for 40-330s additional runtime. Not worth the cost.

    final_pairs = find_overlapping_pairs(pos, sizes, n_hard)
    if verbose:
        print(f"  Final: {len(final_pairs)} overlaps remaining")
    return pos, log
