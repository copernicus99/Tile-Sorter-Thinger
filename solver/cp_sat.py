import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Union, Optional, Sequence
from ortools.sat.python import cp_model as _cp

from models import Placed, Rect
from config import CFG, CELL

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

_RNG: Optional[random.Random] = None
_PLACEMENT_CACHE: Dict[Tuple[int, int, int, int, int], Tuple[Tuple[int, int, int, int, int], ...]] = {}


def _no_plus_guard_enabled(cfg) -> bool:
    """Return ``True`` when the plus-intersection guard should be active."""

    try:
        if not bool(getattr(cfg, "NO_PLUS", False)):
            return False
    except Exception:
        return False

    try:
        if bool(getattr(cfg, "TEST_MODE", False)):
            return False
    except Exception:
        pass

    return True


def _resolve_guard_backoffs(
    attempt,
    *,
    edge_guard_cells: Optional[int],
    plus_guard_enabled: bool,
):
    """Retry helper that disables guards when they block feasibility.

    ``attempt`` is expected to be a callable accepting ``edge_guard_cells`` and
    ``plus_guard_enabled`` keyword arguments, returning ``(ok, placed, reason)``.

    The helper retries in two stages:

    #. Drop the seam guard (``edge_guard_cells``) when present.
    #. Drop the plus-intersection guard when it was active.

    Retries stop early if a run returns a reason other than
    ``"Proven infeasible under current constraints"`` so callers preserve the
    most informative message from CP-SAT.
    """

    ok, placed, reason = attempt(
        edge_guard_cells=edge_guard_cells,
        plus_guard_enabled=plus_guard_enabled,
    )

    infeasible = reason == "Proven infeasible under current constraints"
    if ok or not infeasible:
        return ok, placed, reason

    retry_plan = []
    if edge_guard_cells is not None:
        retry_plan.append({
            "edge_guard_cells": None,
            "plus_guard_enabled": plus_guard_enabled,
        })
    if plus_guard_enabled:
        retry_plan.append({
            "edge_guard_cells": edge_guard_cells,
            "plus_guard_enabled": False,
        })

    for overrides in retry_plan:
        retry_ok, retry_placed, retry_reason = attempt(**overrides)
        if retry_ok:
            return True, retry_placed, None
        if retry_reason != "Proven infeasible under current constraints":
            return retry_ok, retry_placed, retry_reason

    return ok, placed, reason


def _compute_edge_guard_cells(
    max_edge_ft: Optional[float],
    *,
    cell_size: float,
    board_max: int,
    test_mode: bool,
) -> Optional[int]:
    """Return the seam guard length expressed in cells or ``None`` when disabled.

    The guard is skipped when:

    * the configuration does not specify a maximum edge length
    * the configured value is non-positive
    * ``TEST_MODE`` is enabled (local iteration should be lenient)
    * the grid dimension is smaller than the guard length, meaning the
      constraint would never trigger anyway

    ``board_max`` is the larger of ``W``/``H`` and is used to avoid applying the
    guard to trivially small boards.  ``cell_size`` comes from ``config.CELL`` so
    environments that tweak the base unit (e.g. 0.25ft grid cells) still yield a
    correct conversion.
    """

    if test_mode:
        return None

    try:
        value_ft = None if max_edge_ft is None else float(max_edge_ft)
    except Exception:
        value_ft = None

    if value_ft is None or value_ft <= 0:
        return None

    try:
        cell = float(cell_size)
    except Exception:
        cell = 0.0

    if cell <= 0:
        return None

    guard_cells = int(round(value_ft / cell + 1e-9))
    if guard_cells <= 0:
        return None

    try:
        board_dim = int(board_max)
    except Exception:
        board_dim = 0

    if board_dim <= guard_cells:
        return None

    return guard_cells


def _system_rng() -> random.Random:
    global _RNG
    if _RNG is None:
        try:
            _RNG = random.SystemRandom()
        except NotImplementedError:
            _RNG = random.Random()
    return _RNG


def _placement_key(W: int, H: int, w: int, h: int, stride: int) -> Tuple[int, int, int, int, int]:
    return (int(W), int(H), int(w), int(h), max(1, int(stride)))


def _compute_locs(W: int, H: int, w: int, h: int, stride: int) -> Tuple[Tuple[int, int, int, int, int], ...]:
    key = _placement_key(W, H, w, h, stride)
    cached = _PLACEMENT_CACHE.get(key)
    if cached is not None:
        return cached

    sx = max(1, stride)
    sy = max(1, stride)

    xs: Sequence[int]
    ys: Sequence[int]

    if w >= W:
        xs = (0,)
    else:
        xs = tuple(range(0, W - w + 1, sx))

    if h >= H:
        ys = (0,)
    else:
        ys = tuple(range(0, H - h + 1, sy))

    locs = tuple((x, y, 0, w, h) for x in xs for y in ys)
    _PLACEMENT_CACHE[key] = locs
    return locs


def build_options(W: int, H: int, tiles: List[Rect], stride: int, *, rng: Optional[random.Random] = None, randomize: bool = False):
    opts = []
    phase_seed = (W * 1315423911 ^ H * 2654435761) & 0xFFFFFFFF
    max_opts_per_tile = int(getattr(CFG, "MAX_OPTIONS_PER_TILE", 2000))
    max_opts_per_rect = int(getattr(CFG, "MAX_OPTIONS_PER_RECT", 2000))
    rng_local = _system_rng() if (randomize and rng is None) else rng

    for idx, r in enumerate(tiles):
        t = []
        cfgs = [(r.w, r.h, 0)] if r.w == r.h else [(r.w, r.h, 0), (r.h, r.w, 1)]
        total_for_rect = 0

        if randomize and rng_local is not None and len(cfgs) > 1:
            rng_local.shuffle(cfgs)

        for (w, h, rot) in cfgs:
            locs = [(*loc[:2], rot, loc[3], loc[4]) for loc in _compute_locs(W, H, w, h, stride)]

            if randomize and rng_local is not None and len(locs) > 1:
                rng_local.shuffle(locs)

            if len(locs) > max_opts_per_tile:
                if randomize and rng_local is not None:
                    locs = locs[:max_opts_per_tile]
                else:
                    step = max(1, len(locs) // max_opts_per_tile)
                    offset = (phase_seed + idx * 97 + w * 17 + h * 23 + rot * 31) % step
                    locs = locs[offset::step][:max_opts_per_tile]

            t.extend(locs)
            total_for_rect += len(locs)

        if total_for_rect > max_opts_per_rect and len(t) > max_opts_per_rect:
            if randomize and rng_local is not None:
                rng_local.shuffle(t)
                t = t[:max_opts_per_rect]
            else:
                step = max(1, len(t) // max_opts_per_rect)
                offset = (phase_seed + idx * 31337) % step
                t = t[offset::step][:max_opts_per_rect]

        if randomize and rng_local is not None and len(t) > 1:
            rng_local.shuffle(t)

        opts.append(t)

    return opts

# ---------------- main solve ----------------
def try_pack_exact_cover(
    W: int,
    H: int,
    multiset: Union[Iterable, Dict],
    allow_discard: bool = False,
    max_seconds: float = 30.0,
    *,
    initial_hint: Optional[List[Placed]] = None,
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

    randomize = bool(getattr(CFG, "RANDOMIZE_PLACEMENTS", False))
    rng = _system_rng() if randomize else None

    tiles = list(tiles)
    if randomize and rng is not None and len(tiles) > 1:
        rng.shuffle(tiles)

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

    # forced placements: whole-board or strip fills that fully determine layout
    forced: List[Placed] = []
    remaining_tiles: List[Rect] = []

    for rect in tiles:
        if (rect.w == W and rect.h == H) or (rect.w == H and rect.h == W):
            orientation = Rect(W, H, rect.name)
            forced.append(Placed(0, 0, orientation))
        else:
            remaining_tiles.append(rect)

    if len(forced) > 1:
        # Multiple whole-board tiles cannot coexist.
        return False, [], "Bad demand: multiple tiles match entire board"

    if forced and remaining_tiles:
        return False, [], "Bad demand: full-board tile conflicts with other tiles"

    if forced and not remaining_tiles:
        return True, forced, None

    if not forced:
        remaining_tiles = list(tiles)

    # Detect strip fills that deterministically cover the board
    def _orient_strip(rect: Rect, target: str) -> Optional[Rect]:
        if target == "width":
            if rect.w == W:
                return rect
            if rect.h == W:
                return Rect(rect.h, rect.w, rect.name)
        if target == "height":
            if rect.h == H:
                return rect
            if rect.w == H:
                return Rect(rect.h, rect.w, rect.name)
        return None

    width_strips: List[Rect] = []
    other_tiles: List[Rect] = []
    for rect in remaining_tiles:
        oriented = _orient_strip(rect, "width")
        if oriented is not None and oriented.w == W:
            width_strips.append(oriented)
        else:
            other_tiles.append(rect)

    if width_strips and not other_tiles:
        total_height = sum(r.h for r in width_strips)
        if total_height == H:
            y = 0
            strip_forced: List[Placed] = []
            for rect in sorted(width_strips, key=lambda r: (-r.h, r.name or "")):
                strip_forced.append(Placed(0, y, Rect(rect.w, rect.h, rect.name)))
                y += rect.h
            if y == H:
                strip_forced.extend(forced)
                return True, strip_forced, None

    height_strips: List[Rect] = []
    other_tiles_h: List[Rect] = []
    for rect in remaining_tiles:
        oriented = _orient_strip(rect, "height")
        if oriented is not None and oriented.h == H:
            height_strips.append(oriented)
        else:
            other_tiles_h.append(rect)

    if height_strips and not other_tiles_h:
        total_width = sum(r.w for r in height_strips)
        if total_width == W:
            x = 0
            strip_forced = []
            for rect in sorted(height_strips, key=lambda r: (-r.w, r.name or "")):
                strip_forced.append(Placed(x, 0, Rect(rect.w, rect.h, rect.name)))
                x += rect.w
            if x == W:
                strip_forced.extend(forced)
                return True, strip_forced, None

    tiles = remaining_tiles

    options = build_options(W, H, tiles, stride, rng=rng, randomize=randomize)
    total_places = sum(len(o) for o in options)
    if total_places == 0:
        return False, [], "No placements remain (thinned away or grid too small)"
    if total_places > max_placements and stride >= max(W, H):
        return False, [], (
            f"Model capped: {total_places:,} placements > limit ({max_placements:,}); stride={stride}"
        )

    # Reorder tiles so the most constrained shapes appear first.  CP-SAT spends
    # far less time exploring symmetric assignments when the tiles with the
    # fewest legal placements lead the search.  Retain a shuffled tie-breaker
    # when randomization is enabled so we keep the stochastic flavour while
    # still prioritizing constrained pieces.
    ordering = list(range(len(tiles)))
    if ordering:
        if randomize and rng is not None:
            rng.shuffle(ordering)
        ordering.sort(key=lambda i: (len(options[i]) or 0, -(tiles[i].w * tiles[i].h)))
        if any(idx != i for i, idx in enumerate(ordering)):
            tiles = [tiles[i] for i in ordering]
            options = [options[i] for i in ordering]

    def _solve_with_limit(limit_value, seconds, *, edge_guard_cells, plus_guard_enabled):
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
        cell_to_vars: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        corner_to_vars: Dict[Tuple[int, int], List[_cp.IntVar]] = defaultdict(list)
        for i in range(n):
            for k, (px, py, _rot, w, h) in enumerate(options[i]):
                for dx in range(w):
                    for dy in range(h):
                        cx = px + dx
                        cy = py + dy
                        cell_to_vars[(cx, cy)].append((i, k))
                corner_to_vars[(px, py)].append(p[i][k])
                corner_to_vars[(px + w, py)].append(p[i][k])
                corner_to_vars[(px, py + h)].append(p[i][k])
                corner_to_vars[(px + w, py + h)].append(p[i][k])

        for x in range(W):
            for y in range(H):
                placements_here = cell_to_vars.get((x, y))
                if not placements_here:
                    if not allow_discard:
                        return False, [], f"Coverage impossible with stride={stride} (un-coverable cell)"
                    continue
                vars_here = [p[i][k] for i, k in placements_here]
                if allow_discard:
                    m.AddAtMostOne(vars_here)
                else:
                    m.Add(sum(vars_here) == 1)

        # ---- rule / guard extras ----
        max_edge_cells = edge_guard_cells

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
                    left_ids = cell_to_vars.get((x - 1, y), [])
                    right_ids = cell_to_vars.get((x, y), [])

                    left = [p[i][k] for i, k in left_ids]
                    right = [p[i][k] for i, k in right_ids]
                    left_set = set(left_ids)
                    right_set = set(right_ids)
                    both_pairs = left_set & right_set
                    both = [p[i][k] for i, k in both_pairs]

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
                    top_ids = cell_to_vars.get((x, y - 1), [])
                    bottom_ids = cell_to_vars.get((x, y), [])

                    top = [p[i][k] for i, k in top_ids]
                    bottom = [p[i][k] for i, k in bottom_ids]
                    top_set = set(top_ids)
                    bottom_set = set(bottom_ids)
                    both_pairs = top_set & bottom_set
                    both = [p[i][k] for i, k in both_pairs]

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

        if plus_guard_enabled:
            for nx in range(1, W):
                for ny in range(1, H):
                    corners = corner_to_vars.get((nx, ny))
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
                vertical_left = defaultdict(list)
                vertical_right = defaultdict(list)
                horizontal_bottom = defaultdict(list)
                horizontal_top = defaultdict(list)

                for i in range(n):
                    for k, (px, py, _rot, w, h) in enumerate(options[i]):
                        vertical_left[px].append((py, py + h, i, k))
                        vertical_right[px + w].append((py, py + h, i, k))
                        horizontal_bottom[py].append((px, px + w, i, k))
                        horizontal_top[py + h].append((px, px + w, i, k))

                def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
                    return not (a1 <= b0 or b1 <= a0)

                adjacency_pairs: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)

                for edge_x, right_list in vertical_right.items():
                    left_list = vertical_left.get(edge_x, [])
                    if not left_list:
                        continue
                    for (ay0, ay1, ai, ak) in right_list:
                        for (by0, by1, bi, bk) in left_list:
                            if _overlaps(ay0, ay1, by0, by1) and ai != bi:
                                key = (min(ai, bi), max(ai, bi))
                                adjacency_pairs[key].append((ai, ak, bi, bk))

                for edge_y, top_list in horizontal_top.items():
                    bottom_list = horizontal_bottom.get(edge_y, [])
                    if not bottom_list:
                        continue
                    for (ax0, ax1, ai, ak) in top_list:
                        for (bx0, bx1, bi, bk) in bottom_list:
                            if _overlaps(ax0, ax1, bx0, bx1) and ai != bi:
                                key = (min(ai, bi), max(ai, bi))
                                adjacency_pairs[key].append((ai, ak, bi, bk))

                tile_adj_vars: Dict[int, List[_cp.IntVar]] = defaultdict(list)

                for (ai, bi), combos in adjacency_pairs.items():
                    wi_hi = {(tiles[ai].w, tiles[ai].h), (tiles[ai].h, tiles[ai].w)}
                    wj_hj = {(tiles[bi].w, tiles[bi].h), (tiles[bi].h, tiles[bi].w)}
                    if wi_hi != wj_hj:
                        continue

                    bools = []
                    for (i_idx, k_idx, j_idx, m_idx) in combos:
                        z = m.NewBoolVar(f"t_{i_idx}_{j_idx}_{k_idx}_{m_idx}")
                        m.AddMultiplicationEquality(z, [p[i_idx][k_idx], p[j_idx][m_idx]])
                        bools.append(z)

                    if not bools:
                        continue

                    pair_var = m.NewBoolVar(f"adj_{ai}_{bi}")
                    m.AddMaxEquality(pair_var, bools)
                    tile_adj_vars[ai].append(pair_var)
                    tile_adj_vars[bi].append(pair_var)

                for idx, vars_list in tile_adj_vars.items():
                    if vars_list:
                        m.Add(sum(vars_list) <= LMT)

        # ------- objective / coverage target (no callbacks) -------
        solver = _cp.CpSolver()
        solver.parameters.max_time_in_seconds = float(seconds)
        solver.parameters.max_memory_in_mb = int(getattr(CFG, "MAX_MEMORY_MB", 2048))
        solver.parameters.num_search_workers = int(getattr(CFG, "WORKERS", 1))
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 0
        solver.parameters.log_search_progress = False
        solver.parameters.symmetry_level = 0
        branching_mode = str(getattr(CFG, "SEARCH_BRANCHING", "AUTOMATIC")).upper()
        branching_map = {
            "AUTOMATIC": getattr(_cp, "AUTOMATIC_SEARCH", _cp.PORTFOLIO_SEARCH),
            "PORTFOLIO": getattr(_cp, "PORTFOLIO_SEARCH", _cp.PORTFOLIO_SEARCH),
            "FIXED": getattr(_cp, "FIXED_SEARCH", _cp.PORTFOLIO_SEARCH),
            "LP": getattr(_cp, "LP_SEARCH", _cp.PORTFOLIO_SEARCH),
        }
        solver.parameters.search_branching = branching_map.get(branching_mode, _cp.PORTFOLIO_SEARCH)
        solver.parameters.random_seed = int(getattr(CFG, "RANDOM_SEED", 0))
        solver.parameters.stop_after_first_solution = bool(
            getattr(CFG, "STOP_AFTER_FIRST_SOLUTION", False)
        )

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

        # Apply solution hints when provided
        if initial_hint:
            hint_remaining = list(initial_hint)
            matched = [False] * len(hint_remaining)
            for i, rect in enumerate(tiles):
                for idx_hint, placed_hint in enumerate(hint_remaining):
                    if matched[idx_hint]:
                        continue
                    rect_hint = placed_hint.rect
                    if rect_hint is None:
                        continue
                    if rect_hint.name and rect.name and rect_hint.name != rect.name:
                        continue
                    dims_match = (
                        (rect_hint.w == rect.w and rect_hint.h == rect.h)
                        or (rect_hint.w == rect.h and rect_hint.h == rect.w)
                    )
                    if not dims_match:
                        continue
                    for k, (px, py, rot, w, h) in enumerate(options[i]):
                        if placed_hint.x == px and placed_hint.y == py and (
                            (rect_hint.w == w and rect_hint.h == h)
                            or (rect_hint.w == h and rect_hint.h == w)
                        ):
                            m.AddHint(p[i][k], 1)
                            matched[idx_hint] = True
                            break
                    if matched[idx_hint]:
                        break

        res = solver.Solve(m)

        if res in (_cp.OPTIMAL, _cp.FEASIBLE):
            placed: List[Placed] = []
            for i in range(n):
                for k, (x, y, rot, w, h) in enumerate(options[i]):
                    if solver.BooleanValue(p[i][k]):
                        placed.append(Placed(x, y, Rect(w, h, tiles[i].name)))
                        break
            placed.extend(forced)
            return True, placed, None

        if res == _cp.INFEASIBLE:
            return False, [], "Proven infeasible under current constraints"
        if res == _cp.MODEL_INVALID:
            return False, [], "Model invalid (configuration error)"
        return False, [], "Stopped before solution (timebox)"

    same_shape_cfg = getattr(CFG, "SAME_SHAPE_LIMIT", None)

    try:
        max_edge_ft_cfg = getattr(CFG, "MAX_EDGE_FT", None)
    except Exception:
        max_edge_ft_cfg = None

    try:
        test_mode_flag = bool(getattr(CFG, "TEST_MODE", False))
    except Exception:
        test_mode_flag = False

    edge_guard_cells = _compute_edge_guard_cells(
        max_edge_ft_cfg,
        cell_size=CELL,
        board_max=max(W, H),
        test_mode=test_mode_flag,
    )
    plus_guard_enabled = _no_plus_guard_enabled(CFG)

    def _attempt(edge_guard_cells_param, plus_guard_enabled_param):
        return _solve_with_limit(
            same_shape_cfg,
            max_seconds,
            edge_guard_cells=edge_guard_cells_param,
            plus_guard_enabled=plus_guard_enabled_param,
        )

    ok, placed, reason = _resolve_guard_backoffs(
        _attempt,
        edge_guard_cells=edge_guard_cells,
        plus_guard_enabled=plus_guard_enabled,
    )

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
    diag_ok, diag_placed, _ = _solve_with_limit(
        None,
        diag_seconds,
        edge_guard_cells=edge_guard_cells,
        plus_guard_enabled=plus_guard_enabled,
    )
    if diag_ok and diag_placed:
        msg = (
            "Proven infeasible under current constraints (same-shape limit "
            f"{same_shape_int} prevents a feasible layout; try increasing TS_SAME_SHAPE_LIMIT "
            "or setting it to -1 to disable the guard)."
        )
        return False, [], msg

    return ok, placed, reason
