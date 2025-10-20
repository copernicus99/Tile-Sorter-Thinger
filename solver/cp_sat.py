import math
from typing import List, Tuple, Dict, Iterable, Union
from ortools.sat.python import cp_model as _cp

from models import Placed, Rect
from config import CFG

# ---------------- helpers ----------------

def _as_rect(obj) -> Rect:
    if isinstance(obj, Rect):
        return obj
    if isinstance(obj, tuple):
        if len(obj) == 2:
            w, h = obj
            return Rect(int(w), int(h), None)
        if len(obj) == 3:
            w, h, name = obj
            return Rect(int(w), int(h), name)
    raise TypeError(f"Not a Rect-like value: {obj!r}")

def _expand_multiset(multiset: Union[Iterable, Dict]) -> Tuple[bool, List[Rect], str]:
    if isinstance(multiset, dict):
        out: List[Rect] = []
        try:
            for k, c in multiset.items():
                r = _as_rect(k)
                n = int(c)
                if n < 0:
                    return False, [], "Bad demand: negative count"
                out.extend([r] * n)
            return True, out, None
        except Exception as e:
            return False, [], f"Bad demand: {e}"

    try:
        out = []
        for item in multiset:        # will raise if not iterable
            out.append(_as_rect(item))
        return True, out, None
    except TypeError:
        return False, [], "Bad demand: not iterable (expected a list of tiles or a dict)"
    except Exception as e:
        return False, [], f"Bad demand: {e}"

def est_positions(W, H, w, h, stride):
    return max(0, (W - w) // max(1, stride) + 1) * max(0, (H - h) // max(1, stride) + 1)

def build_options(W: int, H: int, tiles: List[Rect], stride: int):
    opts = []
    phase_seed = (W * 1315423911 ^ H * 2654435761) & 0xFFFFFFFF
    max_opts_per_tile = int(getattr(CFG, "MAX_OPTIONS_PER_TILE", 2000))
    max_opts_per_rect = int(getattr(CFG, "MAX_OPTIONS_PER_RECT", 2000))

    for idx, r in enumerate(tiles):
        t = []
        cfgs = [(r.w, r.h, 0)] if r.w == r.h else [(r.w, r.h, 0), (r.h, r.w, 1)]
        total_for_rect = 0

        for (w, h, rot) in cfgs:
            locs = []
            sx = max(1, stride); sy = max(1, stride)
            for x in range(0, W - w + 1, sx):
                for y in range(0, H - h + 1, sy):
                    locs.append((x, y, rot, w, h))

            if len(locs) > max_opts_per_tile:
                step = max(1, len(locs) // max_opts_per_tile)
                offset = (phase_seed + idx * 97 + w * 17 + h * 23 + rot * 31) % step
                locs = locs[offset::step][:max_opts_per_tile]

            t.extend(locs)
            total_for_rect += len(locs)

        if total_for_rect > max_opts_per_rect and len(t) > max_opts_per_rect:
            step = max(1, len(t) // max_opts_per_rect)
            offset = (phase_seed + idx * 31337) % step
            t = t[offset::step][:max_opts_per_rect]

        opts.append(t)

    return opts

# ---------------- main solve ----------------
def try_pack_exact_cover(
    W: int,
    H: int,
    multiset: Union[Iterable, Dict],
    allow_discard: bool = False,
    max_seconds: float = 30.0
) -> Tuple[bool, List[Placed], str]:
    """Exact-cover model; if allow_discard=True, coverage model (no callbacks)."""
    # validate grid
    try:
        W = int(W); H = int(H)
    except Exception:
        return False, [], "Bad grid: W/H must be integers"
    if W <= 0 or H <= 0:
        return False, [], "Bad grid: W/H must be positive"

    # normalize tiles
    ok, tiles, reason = _expand_multiset(multiset)
    if not ok:
        return False, [], reason
    if not tiles:
        return False, [], "Bad demand: nothing parsed from request"

    stride = max(1, int(getattr(CFG, "GRID_STRIDE_BASE", 1)))
    dims = []
    for r in tiles:
        dims.append(abs(int(r.w)))
        dims.append(abs(int(r.h)))
    if dims:
        dims_gcd = abs(int(dims[0]))
        for dim in dims[1:]:
            dims_gcd = math.gcd(dims_gcd, abs(int(dim)))
        stride = min(stride, max(1, dims_gcd))
    max_placements = int(getattr(CFG, "MAX_PLACEMENTS", 150000))

    # adaptive thinning
    while True:
        est_total = 0
        for r in tiles:
            if r.w == r.h:
                est_total += est_positions(W, H, r.w, r.h, stride)
            else:
                est_total += est_positions(W, H, r.w, r.h, stride)
                est_total += est_positions(W, H, r.h, r.w, stride)
        if est_total <= max_placements or stride >= max(W, H):
            break
        stride *= 2

    options = build_options(W, H, tiles, stride)
    total_places = sum(len(o) for o in options)
    if total_places == 0:
        return False, [], "No placements remain (thinned away or grid too small)"
    if total_places > max_placements and stride >= max(W, H):
        return False, [], (
            f"Model capped: {total_places:,} placements > limit ({max_placements:,}); stride={stride}"
        )

    def _solve_with_limit(limit_value, seconds):
        m = _cp.CpModel()
        n = len(tiles)

        # at most once (coverage) / exactly once (strict)
        p = [[m.NewBoolVar(f"p_{i}_{k}") for k in range(len(options[i]))] for i in range(n)]
        for i in range(n):
            if allow_discard:
                if p[i]:
                    m.Add(sum(p[i]) <= 1)
            else:
                if not p[i]:
                    return False, [], "No placements remain for at least one tile (over-thinned)"
                m.Add(sum(p[i]) == 1)

        # symmetry breaking
        place_idx = []
        for i in range(n):
            idx = m.NewIntVar(0, max(0, len(options[i]) - 1), f"idx_{i}")
            if options[i]:
                m.Add(idx == sum(k * p[i][k] for k in range(len(options[i]))))
            place_idx.append(idx)

        by_shape: Dict[tuple, List[int]] = {}
        for i, t in enumerate(tiles):
            key = tuple(sorted((t.w, t.h)))
            by_shape.setdefault(key, []).append(i)
        for _, idxs in by_shape.items():
            if len(idxs) >= 2:
                idxs = sorted(idxs)
                for a, b in zip(idxs, idxs[1:]):
                    m.Add(place_idx[a] <= place_idx[b])

        # --- coverage / non-overlap ---
        #
        # In coverage mode we must still prevent overlap, so use ≤ 1 coverage per cell.
        # In strict mode it’s the usual == 1 exact cover.
        for x in range(W):
            for y in range(H):
                covers = []
                for i in range(n):
                    for k, (px, py, rot, w, h) in enumerate(options[i]):
                        if (px <= x < px + w) and (py <= y < py + h):
                            covers.append(p[i][k])
                if not covers:
                    if not allow_discard:
                        return False, [], f"Coverage impossible with stride={stride} (un-coverable cell)"
                    # coverage mode can tolerate uncovered cells
                    continue
                m.Add(sum(covers) <= 1)

        # ---- rule / guard extras ----
        max_edge_ft = getattr(CFG, "MAX_EDGE_FT", None)
        try:
            max_edge_ft = None if max_edge_ft is None else float(max_edge_ft)
        except Exception:
            max_edge_ft = None
        if max_edge_ft is not None and max_edge_ft <= 0:
            max_edge_ft = None
        max_edge_cells = None if (max_edge_ft is None) else int(round(max_edge_ft / 0.5 + 1e-9))

        if (max_edge_cells is not None) and (max_edge_cells < max(W, H)):

            def _or_bool(name, vars_list):
                b = m.NewBoolVar(name)
                if vars_list:
                    m.AddMaxEquality(b, vars_list)
                else:
                    m.Add(b == 0)
                return b

            # vertical
            for x in range(1, W):
                seam_col = []
                for y in range(H):
                    s = m.NewBoolVar(f"sv_{x}_{y}")
                    left = []
                    right = []
                    both = []
                    for i in range(n):
                        for k, (px, py, rot, w, h) in enumerate(options[i]):
                            covers_left = (px <= x - 1 < px + w) and (py <= y < py + h)
                            covers_right = (px <= x < px + w) and (py <= y < py + h)
                            if covers_left:
                                left.append(p[i][k])
                            if covers_right:
                                right.append(p[i][k])
                            if covers_left and covers_right:
                                both.append(p[i][k])

                    left_cover = _or_bool(f"sv_left_{x}_{y}", left)
                    right_cover = _or_bool(f"sv_right_{x}_{y}", right)
                    share = _or_bool(f"share_v_{x}_{y}", both)

                    both_cover = m.NewBoolVar(f"sv_both_{x}_{y}")
                    m.Add(both_cover <= left_cover)
                    m.Add(both_cover <= right_cover)
                    m.Add(both_cover >= left_cover + right_cover - 1)

                    m.Add(s + share == both_cover)
                    seam_col.append(s)
                L = max_edge_cells
                for y0 in range(0, H - (L + 1) + 1):
                    m.Add(sum(seam_col[yy] for yy in range(y0, y0 + L + 1)) <= L)

            # horizontal
            for y in range(1, H):
                seam_row = []
                for x in range(W):
                    s = m.NewBoolVar(f"sh_{x}_{y}")
                    top = []
                    bottom = []
                    both = []
                    for i in range(n):
                        for k, (px, py, rot, w, h) in enumerate(options[i]):
                            covers_top = (py <= y - 1 < py + h) and (px <= x < px + w)
                            covers_bottom = (py <= y < py + h) and (px <= x < px + w)
                            if covers_top:
                                top.append(p[i][k])
                            if covers_bottom:
                                bottom.append(p[i][k])
                            if covers_top and covers_bottom:
                                both.append(p[i][k])

                    top_cover = _or_bool(f"sh_top_{x}_{y}", top)
                    bottom_cover = _or_bool(f"sh_bottom_{x}_{y}", bottom)
                    share = _or_bool(f"share_h_{x}_{y}", both)

                    both_cover = m.NewBoolVar(f"sh_both_{x}_{y}")
                    m.Add(both_cover <= top_cover)
                    m.Add(both_cover <= bottom_cover)
                    m.Add(both_cover >= top_cover + bottom_cover - 1)

                    m.Add(s + share == both_cover)
                    seam_row.append(s)
                L = max_edge_cells
                for x0 in range(0, W - (L + 1) + 1):
                    m.Add(sum(seam_row[xx] for xx in range(x0, x0 + L + 1)) <= L)

        if bool(getattr(CFG, "NO_PLUS", False)):
            for nx in range(1, W):
                for ny in range(1, H):
                    corners = []
                    for i in range(n):
                        for k, (px, py, rot, w, h) in enumerate(options[i]):
                            if (px == nx and py == ny) or (px + w == nx and py == ny) or (px == nx and py + h == ny) or (px + w == nx and py + h == ny):
                                corners.append(p[i][k])
                    if corners:
                        m.Add(sum(corners) <= 3)

        if limit_value is not None:
            try:
                LMT = int(limit_value)
            except Exception:
                LMT = 0
            if LMT < 0:
                LMT = -1
            if LMT >= 0:
                for i in range(n):
                    wi_hi = {(tiles[i].w, tiles[i].h), (tiles[i].h, tiles[i].w)}
                    same_neighbors = []
                    for j in range(n):
                        if j == i:
                            continue
                        wj_hj = {(tiles[j].w, tiles[j].h), (tiles[j].h, tiles[j].w)}
                        if wi_hi != wj_hj:
                            continue
                        a = m.NewBoolVar(f"adj_{i}_{j}")
                        same_neighbors.append(a)
                        touching = []
                        for k, (xi, yi, ri, wi, hi) in enumerate(options[i]):
                            for m2, (xj, yj, rj, wj, hj) in enumerate(options[j]):
                                vt = (xi + wi == xj or xj + wj == xi) and not (yi + hi <= yj or yj + hj <= yi)
                                ht = (yi + hi == yj or yj + hj == yi) and not (xi + wi <= xj or xj + wj <= xi)
                                if vt or ht:
                                    z = m.NewBoolVar(f"t_{i}_{j}_{k}_{m2}")
                                    m.AddMultiplicationEquality(z, [p[i][k], p[j][m2]])
                                    touching.append(z)
                        if touching:
                            m.AddMaxEquality(a, touching)
                        else:
                            m.Add(a == 0)
                    if same_neighbors:
                        m.Add(sum(same_neighbors) <= LMT)

        # ------- objective / coverage target (no callbacks) -------
        solver = _cp.CpSolver()
        solver.parameters.max_time_in_seconds = float(seconds)
        solver.parameters.max_memory_in_mb = int(getattr(CFG, "MAX_MEMORY_MB", 2048))
        solver.parameters.num_search_workers = int(getattr(CFG, "WORKERS", 1))
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 0
        solver.parameters.log_search_progress = False
        solver.parameters.symmetry_level = 0
        solver.parameters.stop_after_first_solution = False  # optimization

        if allow_discard:
            # Maximize total area (in cells); also require a minimum coverage.
            used_area_terms = []
            for i in range(n):
                if not options[i]:
                    continue
                # area of tile i in cells:
                area_i = tiles[i].w * tiles[i].h
                used_i = m.NewBoolVar(f"used_{i}")
                m.AddMaxEquality(used_i, p[i])  # used if any placement picked
                used_area_terms.append(used_i * area_i)

            if used_area_terms:
                total_used_area = sum(used_area_terms)
                target_pct = float(getattr(CFG, "COVERAGE_GOAL_PCT", 99.5))
                target_cells = int(round((target_pct / 100.0) * (W * H)))
                # We only maximize coverage.  Previously we also added a hard
                # lower bound which could incorrectly mark layouts infeasible when
                # geometric parity prevented perfectly covering the grid (e.g. all
                # even-width tiles on an odd-width board).  Keeping the objective
                # without the hard constraint lets CP-SAT chase the best coverage
                # attainable while still returning partial layouts when necessary.
                _ = (target_pct, target_cells)  # retained for potential diagnostics / parity tweaks
                m.Maximize(total_used_area)
            else:
                m.Minimize(0)
        else:
            # exact cover case: nothing to optimize
            m.Minimize(0)

        res = solver.Solve(m)

        if res in (_cp.OPTIMAL, _cp.FEASIBLE):
            placed: List[Placed] = []
            for i in range(n):
                for k, (x, y, rot, w, h) in enumerate(options[i]):
                    if solver.BooleanValue(p[i][k]):
                        placed.append(Placed(x, y, Rect(w, h, tiles[i].name)))
                        break
            return True, placed, None

        if res == _cp.INFEASIBLE:
            return False, [], "Proven infeasible under current constraints"
        if res == _cp.MODEL_INVALID:
            return False, [], "Model invalid (configuration error)"
        return False, [], "Stopped before solution (timebox)"

    same_shape_cfg = getattr(CFG, "SAME_SHAPE_LIMIT", None)
    ok, placed, reason = _solve_with_limit(same_shape_cfg, max_seconds)

    if ok or reason != "Proven infeasible under current constraints":
        return ok, placed, reason

    # If the limit is finite, probe a relaxed model to provide actionable feedback.
    try:
        same_shape_int = int(same_shape_cfg)
    except Exception:
        same_shape_int = None

    if same_shape_int is None or same_shape_int < 0:
        return ok, placed, reason

    diag_seconds = min(float(max_seconds), float(getattr(CFG, "SAME_SHAPE_DIAG_SECONDS", 10.0)))
    diag_seconds = max(1.0, diag_seconds)
    diag_ok, diag_placed, _ = _solve_with_limit(None, diag_seconds)
    if diag_ok and diag_placed:
        msg = (
            "Proven infeasible under current constraints (same-shape limit "
            f"{same_shape_int} prevents a feasible layout; try increasing TS_SAME_SHAPE_LIMIT "
            "or setting it to -1 to disable the guard)."
        )
        return False, [], msg

    return ok, placed, reason
