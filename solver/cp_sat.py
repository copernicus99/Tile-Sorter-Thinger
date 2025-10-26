import math
import random
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Union, Optional, Sequence, Set
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


def _placements_satisfy_guards(
    W: int,
    H: int,
    placements: Sequence[Placed],
    guards: Optional[Dict[str, object]] = None,
) -> bool:
    """Return ``True`` when the placements respect all active guards."""

    if not placements:
        return True

    guard_cfg = guards or {}
    same_shape_limit = guard_cfg.get("same_shape_limit")
    try:
        same_shape_limit = None if same_shape_limit is None else int(same_shape_limit)
    except Exception:
        same_shape_limit = None

    edge_limit_cells = guard_cfg.get("max_edge_cells")
    try:
        edge_limit_cells = None if edge_limit_cells is None else int(edge_limit_cells)
    except Exception:
        edge_limit_cells = None

    no_plus = bool(guard_cfg.get("no_plus"))

    total_cells = max(0, int(W)) * max(0, int(H))
    if total_cells <= 0:
        return False

    grid: List[int] = [-1] * total_cells
    shapes: List[Tuple[int, int]] = []

    for idx, placed in enumerate(placements):
        rect = getattr(placed, "rect", None)
        if rect is None:
            return False
        try:
            px = int(placed.x)
            py = int(placed.y)
            w = int(rect.w)
            h = int(rect.h)
        except Exception:
            return False
        if px < 0 or py < 0 or w <= 0 or h <= 0:
            return False
        if px + w > W or py + h > H:
            return False
        shape_key = (min(w, h), max(w, h))
        shapes.append(shape_key)
        for dy in range(h):
            row_offset = (py + dy) * W
            for dx in range(w):
                cell = row_offset + px + dx
                if cell < 0 or cell >= total_cells:
                    return False
                if grid[cell] != -1:
                    # Overlap indicates an inconsistent placement.
                    return False
                grid[cell] = idx

    def _seam_guard_passes() -> bool:
        if edge_limit_cells is None or edge_limit_cells < 0:
            return True
        max_dim = max(W, H)
        if edge_limit_cells >= max_dim:
            return True
        limit = edge_limit_cells
        # vertical seams
        for x in range(1, W):
            run = 0
            for y in range(H):
                left = grid[y * W + (x - 1)]
                right = grid[y * W + x]
                if left != -1 and right != -1 and left != right:
                    run += 1
                    if run > limit:
                        return False
                else:
                    run = 0
        # horizontal seams
        for y in range(1, H):
            run = 0
            row_above = (y - 1) * W
            row_here = y * W
            for x in range(W):
                top = grid[row_above + x]
                bottom = grid[row_here + x]
                if top != -1 and bottom != -1 and top != bottom:
                    run += 1
                    if run > limit:
                        return False
                else:
                    run = 0
        return True

    def _no_plus_guard_passes() -> bool:
        if not no_plus:
            return True
        for y in range(1, H):
            row_above = (y - 1) * W
            row_here = y * W
            for x in range(1, W):
                nw = grid[row_above + (x - 1)]
                ne = grid[row_above + x]
                sw = grid[row_here + (x - 1)]
                se = grid[row_here + x]
                if -1 in (nw, ne, sw, se):
                    continue
                if len({nw, ne, sw, se}) >= 4:
                    return False
        return True

    def _same_shape_guard_passes() -> bool:
        if same_shape_limit is None or same_shape_limit < 0:
            return True
        adjacency: Dict[int, set[int]] = defaultdict(set)
        for y in range(H):
            row = y * W
            for x in range(W):
                here = grid[row + x]
                if here == -1:
                    continue
                if x + 1 < W:
                    right = grid[row + x + 1]
                    if right != -1 and right != here and shapes[here] == shapes[right]:
                        adjacency[here].add(right)
                        adjacency[right].add(here)
                if y + 1 < H:
                    down = grid[(y + 1) * W + x]
                    if down != -1 and down != here and shapes[here] == shapes[down]:
                        adjacency[here].add(down)
                        adjacency[down].add(here)
        for neighbors in adjacency.values():
            if len(neighbors) > same_shape_limit:
                return False
        return True

    return (
        _seam_guard_passes()
        and _no_plus_guard_passes()
        and _same_shape_guard_passes()
    )


def _grid_fill_exact_cover(
    W: int,
    H: int,
    tiles: Sequence[Rect],
    *,
    deadline: Optional[float] = None,
    guards: Optional[Dict[str, object]] = None,
    forced_slack: Optional[Set[int]] = None,
) -> Optional[List[Placed]]:
    """Fast DFS tiler that anchors every placement at the next uncovered cell.

    The routine is intentionally conservative: it only attempts a fill when the
    board dimensions are positive and every rectangle has positive area.  The
    branching factor stays tiny by grouping identical tiles together and by
    always selecting the lowest/top-left empty cell as the next anchor point.
    This mirrors Algorithm X's "choose column with minimum candidates" heuristic
    but is dramatically cheaper to evaluate for racks containing only a handful
    of distinct shapes.  All tiles must be placed, yet the board may contain
    slack area; any uncovered cells simply remain empty once every tile has been
    positioned.  Callers may supply ``forced_slack`` to pre-mark board cells that
    can never be covered by any placement; this keeps the DFS from repeatedly
    attempting to anchor tiles on those unreachable locations.
    """

    try:
        W = int(W)
        H = int(H)
    except Exception:
        setattr(_grid_fill_exact_cover, "timed_out", False)
        return None

    if W <= 0 or H <= 0:
        setattr(_grid_fill_exact_cover, "timed_out", False)
        return None

    total_area = W * H
    tile_area = 0
    groups: Dict[Tuple[int, int], Dict[str, object]] = {}
    for rect in tiles:
        try:
            rw = abs(int(rect.w))
            rh = abs(int(rect.h))
        except Exception:
            setattr(_grid_fill_exact_cover, "timed_out", False)
            return None
        if rw <= 0 or rh <= 0:
            setattr(_grid_fill_exact_cover, "timed_out", False)
            return None
        tile_area += rw * rh
        key = (min(rw, rh), max(rw, rh))
        entry = groups.setdefault(key, {"count": 0, "names": []})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["names"].append(rect.name)

    if tile_area > total_area:
        setattr(_grid_fill_exact_cover, "timed_out", False)
        return None

    if not groups:
        setattr(_grid_fill_exact_cover, "timed_out", False)
        return []

    tile_types: List[Dict[str, object]] = []
    timed_out = False

    same_shape_limit: Optional[int] = None
    adjacency_enabled = False
    if isinstance(guards, dict):
        maybe_limit = guards.get("same_shape_limit")
        try:
            if maybe_limit is not None:
                same_shape_limit = int(maybe_limit)
        except Exception:
            same_shape_limit = None
        if same_shape_limit is not None and same_shape_limit >= 0:
            adjacency_enabled = True
        else:
            same_shape_limit = None

    def _deadline_exceeded() -> bool:
        nonlocal timed_out
        if deadline is None:
            return False
        if time.time() >= deadline:
            timed_out = True
            return True
        return False

    for (min_side, max_side), entry in groups.items():
        count = int(entry.get("count", 0))
        if count <= 0:
            continue
        names_list: List[Optional[str]] = list(entry.get("names", []))  # type: ignore[arg-type]
        if len(names_list) < count:
            names_list.extend([None] * (count - len(names_list)))
        tile_types.append(
            {
                "min": min_side,
                "max": max_side,
                "count": count,
                "used": 0,
                "names": names_list,
                "area": min_side * max_side,
            }
        )

    if not tile_types:
        return []

    tile_types.sort(
        key=lambda t: (
            int(t["area"]),
            int(t["max"]),
            int(t["min"]),
        ),
        reverse=True,
    )

    for idx, tile in enumerate(tile_types):
        tile["shape_id"] = idx

    offset_cache: Dict[Tuple[int, int, int], Tuple[int, ...]] = {}

    def _offsets_for(width: int, height: int) -> Tuple[int, ...]:
        key = (width, height, W)
        cached = offset_cache.get(key)
        if cached is not None:
            return cached
        offsets = tuple(dy * W + dx for dy in range(height) for dx in range(width))
        offset_cache[key] = offsets
        return offsets

    for tile in tile_types:
        if _deadline_exceeded():
            break
        min_side = int(tile["min"])
        max_side = int(tile["max"])
        orientations: List[Tuple[int, int, Tuple[int, ...]]] = []
        dims = [(min_side, max_side)]
        if min_side != max_side:
            dims.append((max_side, min_side))
        dims.sort(key=lambda wh: (-wh[1], -wh[0]))
        for w, h in dims:
            offsets = _offsets_for(w, h)
            orientations.append((w, h, offsets))
        tile["orientations"] = orientations

    tiles_total = sum(int(tile["count"]) for tile in tile_types)
    grid = bytearray(W * H)
    owners: List[int] = [-1] * (W * H)
    placements: List[Placed] = []
    placed_tiles = 0
    placement_shapes: List[int] = []
    placement_neighbors: List[Set[int]] = []

    forced_slack = {int(cell) for cell in (forced_slack or set())}
    if forced_slack:
        for cell in forced_slack:
            if 0 <= cell < W * H:
                grid[cell] = 1

    def _search(filled_cells: int) -> bool:
        nonlocal placed_tiles
        if _deadline_exceeded():
            return False
        if placed_tiles == tiles_total:
            if _placements_satisfy_guards(W, H, placements, guards):
                return True
            return False
        next_idx = grid.find(0)
        if next_idx == -1:
            return False
        y, x = divmod(next_idx, W)
        for tile in tile_types:
            if tile["used"] >= tile["count"]:
                continue
            if _deadline_exceeded():
                return False
            for w, h, offsets in tile["orientations"]:
                if x + w > W or y + h > H:
                    continue
                base = next_idx
                fits = True
                for off in offsets:
                    if grid[base + off]:
                        fits = False
                        break
                if not fits:
                    continue
                tile["used"] += 1
                placed_tiles += 1
                name_idx = tile["used"] - 1
                names_list = tile["names"]
                name = None
                if isinstance(names_list, list) and 0 <= name_idx < len(names_list):
                    name = names_list[name_idx]
                placements.append(Placed(x, y, Rect(w, h, name)))
                placement_shapes.append(int(tile["shape_id"]))
                placement_neighbors.append(set())
                new_idx = len(placements) - 1
                for off in offsets:
                    grid[base + off] = 1
                    owners[base + off] = new_idx

                neighbor_ids: Set[int] = set()
                if adjacency_enabled and same_shape_limit is not None:
                    shape_id = placement_shapes[new_idx]
                    for off in offsets:
                        cell = base + off
                        cy, cx = divmod(cell, W)
                        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nx = cx + dx
                            ny = cy + dy
                            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                                continue
                            neighbor_owner = owners[ny * W + nx]
                            if neighbor_owner == -1 or neighbor_owner == new_idx:
                                continue
                            if placement_shapes[neighbor_owner] != shape_id:
                                continue
                            neighbor_ids.add(neighbor_owner)

                    if len(neighbor_ids) > same_shape_limit:
                        # guard violation — undo and continue with next placement
                        for off in offsets:
                            grid[base + off] = 0
                            owners[base + off] = -1
                        placements.pop()
                        placement_shapes.pop()
                        placement_neighbors.pop()
                        placed_tiles -= 1
                        tile["used"] -= 1
                        continue

                    violation = False
                    for neighbor in neighbor_ids:
                        neighbor_set = placement_neighbors[neighbor]
                        if new_idx not in neighbor_set and len(neighbor_set) >= same_shape_limit:
                            violation = True
                            break
                    if violation:
                        for off in offsets:
                            grid[base + off] = 0
                            owners[base + off] = -1
                        placements.pop()
                        placement_shapes.pop()
                        placement_neighbors.pop()
                        placed_tiles -= 1
                        tile["used"] -= 1
                        continue

                    for neighbor in neighbor_ids:
                        placement_neighbors[new_idx].add(neighbor)
                        placement_neighbors[neighbor].add(new_idx)

                if _search(filled_cells + w * h):
                    return True
                if adjacency_enabled and same_shape_limit is not None:
                    for neighbor in list(placement_neighbors[new_idx]):
                        placement_neighbors[neighbor].discard(new_idx)
                    placement_neighbors[new_idx].clear()
                for off in offsets:
                    grid[base + off] = 0
                    owners[base + off] = -1
                placements.pop()
                placement_shapes.pop()
                placement_neighbors.pop()
                placed_tiles -= 1
                tile["used"] -= 1
        return False

    solved = _search(0)
    setattr(_grid_fill_exact_cover, "timed_out", timed_out)
    if solved:
        return placements
    if timed_out:
        return None
    return None


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

    The tuple returned includes the guard configuration that produced the final
    ``(ok, placed, reason)`` triple so callers can detect when guards were fully
    disabled and trigger deterministic fallbacks accordingly.
    """

    ok, placed, reason = attempt(
        edge_guard_cells=edge_guard_cells,
        plus_guard_enabled=plus_guard_enabled,
    )

    result_edge = edge_guard_cells
    result_plus = plus_guard_enabled

    infeasible = reason == "Proven infeasible under current constraints"
    guard_active = (edge_guard_cells is not None) or bool(plus_guard_enabled)
    inconclusive_reason = reason in (
        None,
        "Stopped before solution (timebox)",
        "Stopped before solution (timebox / crash?)",
    )

    if ok:
        return ok, placed, reason, result_edge, result_plus

    if not guard_active:
        return ok, placed, reason, result_edge, result_plus

    if not (infeasible or (not placed and inconclusive_reason)):
        return ok, placed, reason, result_edge, result_plus

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
        override_edge = overrides.get("edge_guard_cells", edge_guard_cells)
        override_plus = overrides.get("plus_guard_enabled", plus_guard_enabled)
        retry_ok, retry_placed, retry_reason = attempt(**overrides)
        if retry_ok:
            return True, retry_placed, None, override_edge, override_plus
        if retry_reason not in (
            "Proven infeasible under current constraints",
            "Stopped before solution (timebox)",
            "Stopped before solution (timebox / crash?)",
        ):
            return retry_ok, retry_placed, retry_reason, override_edge, override_plus
        result_edge = override_edge
        result_plus = override_plus
        ok, placed, reason = retry_ok, retry_placed, retry_reason

    return ok, placed, reason, result_edge, result_plus


def _compute_edge_guard_cells(
    max_edge_ft: Optional[float],
    *,
    cell_size: float,
    board_max: int,
    test_mode: bool,
    max_tile_side_ft: Optional[float] = None,
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
        tile_side_ft = None if max_tile_side_ft is None else float(max_tile_side_ft)
    except Exception:
        tile_side_ft = None

    if tile_side_ft is not None and tile_side_ft > value_ft + 1e-9:
        # The configured guard would block the largest tile entirely, so relax it.
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


def _backtracking_exact_cover(
    W: int,
    H: int,
    tiles: Sequence[Rect],
    options: Sequence[Sequence[Tuple[int, int, int, int, int]]],
    *,
    max_seconds: Optional[float] = None,
    guards: Optional[Dict[str, object]] = None,
    forced_slack: Optional[Set[int]] = None,
) -> Optional[List[Placed]]:
    """Deterministic exact-cover search used as a last-resort fallback.

    The helper targets small boards where CP-SAT either times out or is
    unavailable.  It only supports the pure exact-cover case (no discards) and
    now evaluates guard constraints once a full placement is assembled.  Two
    search strategies are attempted:

    #.  A greedy tile-filling DFS that walks the grid cell-by-cell.  This is
        extremely effective for well-behaved racks (e.g. the crash regression
        puzzle) because the branching factor stays tiny when always anchoring
        placements at the next uncovered cell.
    #.  The existing Algorithm X style exact-cover search which acts as a
        general-purpose safety net when the greedy search fails or encounters
        one of the configurable guard limits.
    """

    try:
        W = int(W)
        H = int(H)
    except Exception:
        return None

    if W <= 0 or H <= 0:
        return None

    n = len(tiles)
    if n == 0:
        return []

    try:
        max_cells_cfg = int(getattr(CFG, "BACKTRACK_MAX_CELLS", 900))
    except Exception:
        max_cells_cfg = 900
    try:
        max_tiles_cfg = int(getattr(CFG, "BACKTRACK_MAX_TILES", 16))
    except Exception:
        max_tiles_cfg = 16
    try:
        node_limit_cfg = int(getattr(CFG, "BACKTRACK_NODE_LIMIT", 300000))
    except Exception:
        node_limit_cfg = 300000

    try:
        time_limit = float(max_seconds) if max_seconds is not None else None
    except Exception:
        time_limit = None
    if time_limit is not None and time_limit <= 0:
        time_limit = None

    stats = {
        "board": (W, H),
        "tiles": n,
        "max_cells": max_cells_cfg,
        "max_tiles": max_tiles_cfg,
        "node_limit": node_limit_cfg,
        "nodes": 0,
        "limit_hit": False,
        "time_limit": time_limit,
        "timed_out": False,
        "symmetry_pruned": 0,
    }
    setattr(_backtracking_exact_cover, "last_stats", dict(stats))

    max_cells_cfg = max(1, max_cells_cfg)
    max_tiles_cfg = max(1, max_tiles_cfg)
    node_limit_cfg = max(1, node_limit_cfg)

    area_cells = W * H
    tile_area = 0
    try:
        for rect in tiles:
            tile_area += abs(int(rect.w)) * abs(int(rect.h))
    except Exception:
        stats.update({"reason": "tile_dims_invalid"})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    slack_cells = max(0, area_cells - tile_area)
    forced_slack = {int(cell) for cell in (forced_slack or set()) if 0 <= int(cell) < area_cells}
    stats["slack_cells"] = slack_cells
    stats["forced_slack_cells"] = len(forced_slack)
    stats["free_slack_cells"] = max(0, slack_cells - len(forced_slack))

    if tile_area > area_cells:
        stats.update({"reason": "tile_area_exceeds_board"})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    if area_cells > max_cells_cfg:
        stats.update({"reason": "max_cells_exceeded"})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    # Permit modestly larger tile counts on small boards before giving up.
    if n > max_tiles_cfg:
        stats.update({"reason": "max_tiles_exceeded"})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    if len(options) != n:
        stats.update({"reason": "options_mismatch"})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    group_ids: List[int] = [0] * n
    group_lookup: Dict[Tuple[int, int], int] = {}
    for idx, rect in enumerate(tiles):
        try:
            rw = abs(int(rect.w))
            rh = abs(int(rect.h))
        except Exception:
            stats.update({"reason": "tile_dims_invalid"})
            setattr(_backtracking_exact_cover, "last_stats", dict(stats))
            return None
        key = (min(rw, rh), max(rw, rh))
        group_ids[idx] = group_lookup.setdefault(key, len(group_lookup))

    guard_dict: Dict[str, object]
    if isinstance(guards, dict):
        guard_dict = {k: v for k, v in guards.items() if v is not None}
    else:
        guard_dict = {}

    same_shape_cfg = guard_dict.get("same_shape_limit")
    try:
        same_shape_limit = None if same_shape_cfg is None else int(same_shape_cfg)
    except Exception:
        same_shape_limit = None

    coverage_ratio = (tile_area / area_cells) if area_cells else 0.0
    stats["coverage_ratio"] = coverage_ratio
    stats["same_shape_limit"] = same_shape_limit

    try:
        relax_allowed = bool(getattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE", True))
    except Exception:
        relax_allowed = True
    try:
        relax_threshold = float(
            getattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE_THRESHOLD", 0.92)
        )
    except Exception:
        relax_threshold = 0.92

    guard_variants: List[Tuple[Optional[Dict[str, object]], bool]] = []
    base_guard = guard_dict or None
    guard_variants.append((base_guard, False))
    if (
        relax_allowed
        and same_shape_limit is not None
        and same_shape_limit >= 0
        and coverage_ratio >= relax_threshold
    ):
        relaxed = dict(guard_dict)
        if "same_shape_limit" in relaxed:
            relaxed.pop("same_shape_limit", None)
            guard_variants.append((relaxed or None, True))

    if forced_slack:
        cell_lookup: Dict[int, int] = {}
        next_idx = 0
        for cell in range(area_cells):
            if cell in forced_slack:
                continue
            cell_lookup[cell] = next_idx
            next_idx += 1
        total_cells = next_idx
    else:
        cell_lookup = {cell: cell for cell in range(area_cells)}
        total_cells = area_cells

    tile_offset = total_cells
    row_columns: List[Tuple[int, ...]] = []
    row_info: List[Tuple[int, Tuple[int, int, int, int, int]]] = []
    row_groups: List[int] = []
    row_keys: List[Tuple[int, int, int, int, int]] = []
    column_to_rows: Dict[int, List[int]] = defaultdict(list)

    for tile_idx, placements in enumerate(options):
        if not placements:
            stats.update({"reason": "empty_options"})
            setattr(_backtracking_exact_cover, "last_stats", dict(stats))
            return None
        for opt in placements:
            px, py, _rot, w, h = opt
            columns: List[int] = [tile_offset + tile_idx]
            skip_row = False
            for dy in range(h):
                row_offset = (py + dy) * W
                for dx in range(w):
                    cell = row_offset + px + dx
                    mapped = cell_lookup.get(cell)
                    if mapped is None:
                        skip_row = True
                        break
                    columns.append(mapped)
                if skip_row:
                    break
            if skip_row:
                continue
            row_idx = len(row_columns)
            cols_tuple = tuple(columns)
            row_columns.append(cols_tuple)
            row_info.append((tile_idx, opt))
            row_groups.append(group_ids[tile_idx])
            row_keys.append((py, px, w, h, _rot))
            for col in cols_tuple:
                column_to_rows[col].append(row_idx)

    total_columns = tile_offset + n
    is_primary: List[bool] = [False] * total_columns
    for col in range(tile_offset, total_columns):
        is_primary[col] = True

    missing_primary = [
        col
        for col in range(tile_offset, total_columns)
        if (col not in column_to_rows) or (not column_to_rows[col])
    ]
    if missing_primary:
        stats.update({"reason": "uncovered_tile", "missing_primary": len(missing_primary)})
        setattr(_backtracking_exact_cover, "last_stats", dict(stats))
        return None

    last_stats: Optional[Dict[str, object]] = None

    def _run_search(active_guards, attempt_deadline):
        active_cols: set[int] = set(range(total_columns))
        active_primary: set[int] = {col for col in active_cols if is_primary[col]}
        active_rows: set[int] = set(range(len(row_columns)))
        solution_rows: List[int] = []
        nodes_searched = 0
        limit_hit = False
        timed_out = False
        best_solution: Optional[List[Placed]] = None
        group_assignments: Dict[int, Dict[int, Tuple[int, int, int, int, int]]] = defaultdict(dict)
        symmetry_pruned = 0

        def _deadline_exceeded() -> bool:
            nonlocal timed_out
            if attempt_deadline is None:
                return False
            if time.time() >= attempt_deadline:
                timed_out = True
                return True
            return False

        def _choose_column() -> Optional[int]:
            best_col = None
            best_count = None
            if _deadline_exceeded():
                return None
            if not active_primary:
                return None
            for col in list(active_primary):
                candidates = 0
                for row_idx in column_to_rows[col]:
                    if row_idx in active_rows:
                        candidates += 1
                        if best_count is not None and candidates >= best_count:
                            break
                if candidates == 0:
                    return col
                if best_col is None or candidates < best_count:
                    best_col = col
                    best_count = candidates
                    if best_count == 1:
                        break
            return best_col

        def _rows_to_placements(rows: Sequence[int]) -> Optional[List[Placed]]:
            placed_by_tile: List[Optional[Placed]] = [None] * n
            for row_idx in rows:
                tile_idx, opt = row_info[row_idx]
                px, py, _rot, w, h = opt
                placed_by_tile[tile_idx] = Placed(
                    px, py, Rect(w, h, tiles[tile_idx].name)
                )
            if any(p is None for p in placed_by_tile):
                return None
            return [p for p in placed_by_tile if p is not None]

        def _search() -> bool:
            nonlocal nodes_searched, limit_hit, best_solution, symmetry_pruned
            if _deadline_exceeded():
                return False
            if not active_primary:
                placements = _rows_to_placements(solution_rows)
                if placements is None:
                    return False
                if not _placements_satisfy_guards(W, H, placements, active_guards):
                    return False
                best_solution = placements
                return True
            col = _choose_column()
            if col is None:
                return False

            candidates = [r for r in column_to_rows[col] if r in active_rows]
            if not candidates:
                return False

            for row_idx in candidates:
                if _deadline_exceeded():
                    return False
                if nodes_searched >= node_limit_cfg:
                    limit_hit = True
                    return False
                nodes_searched += 1
                solution_rows.append(row_idx)

                tile_idx, _ = row_info[row_idx]
                group_id = row_groups[row_idx]
                key = row_keys[row_idx]
                group_map = group_assignments[group_id]
                violation = False
                for other_idx, other_key in group_map.items():
                    if other_idx == tile_idx:
                        continue
                    if other_idx < tile_idx and other_key > key:
                        violation = True
                        break
                    if other_idx > tile_idx and other_key < key:
                        violation = True
                        break
                if violation:
                    symmetry_pruned += 1
                    solution_rows.pop()
                    continue
                previous_key = group_map.get(tile_idx)
                group_map[tile_idx] = key

                removed_cols: List[int] = []
                removed_rows: List[int] = []
                for c in row_columns[row_idx]:
                    if c in active_cols:
                        active_cols.remove(c)
                        removed_cols.append(c)
                        if is_primary[c]:
                            active_primary.discard(c)
                        for r in column_to_rows[c]:
                            if r in active_rows:
                                active_rows.remove(r)
                                removed_rows.append(r)

                if _search():
                    return True

                for r in reversed(removed_rows):
                    active_rows.add(r)
                for c in reversed(removed_cols):
                    active_cols.add(c)
                    if is_primary[c]:
                        active_primary.add(c)
                solution_rows.pop()
                if previous_key is None:
                    group_map.pop(tile_idx, None)
                else:
                    group_map[tile_idx] = previous_key

            return False

        solved = _search()
        result_stats: Dict[str, object] = {
            "nodes": nodes_searched,
            "limit_hit": limit_hit,
            "timed_out": timed_out,
            "symmetry_pruned": symmetry_pruned,
        }
        if solved and best_solution:
            result_stats.update({
                "placed": len(best_solution),
                "result": "solved",
                "coverage_cells": sum(p.rect.w * p.rect.h for p in best_solution),
            })
            return best_solution, result_stats
        if timed_out:
            result_stats["reason"] = "time_limit"
        elif limit_hit:
            result_stats["reason"] = "node_limit"
        else:
            result_stats.setdefault("reason", "guard_blocked")
        return None, result_stats

    for active_guard, relaxed_flag in guard_variants:
        attempt_stats = dict(stats)
        attempt_stats["same_shape_guard_relaxed"] = bool(relaxed_flag)
        attempt_stats["guards_active"] = bool(active_guard)
        attempt_stats["guard_keys"] = sorted(active_guard.keys()) if active_guard else []

        if time_limit is not None:
            attempt_deadline = time.time() + max(0.0, time_limit)
        else:
            attempt_deadline = None

        greedy = _grid_fill_exact_cover(
            W,
            H,
            tiles,
            deadline=attempt_deadline,
            guards=active_guard,
            forced_slack=forced_slack,
        )
        greedy_timed_out = bool(getattr(_grid_fill_exact_cover, "timed_out", False))
        if greedy is not None:
            attempt_stats.update({
                "method": "grid_fill",
                "result": "solved",
                "nodes": n,
                "limit_hit": False,
                "timed_out": False,
                "symmetry_pruned": 0,
            })
            setattr(_backtracking_exact_cover, "last_stats", dict(attempt_stats))
            return greedy
        if greedy_timed_out:
            attempt_stats.update({"reason": "time_limit", "timed_out": True, "symmetry_pruned": 0})
            last_stats = attempt_stats
            setattr(_backtracking_exact_cover, "last_stats", dict(attempt_stats))
            continue

        placements, search_stats = _run_search(active_guard, attempt_deadline)
        attempt_stats.update(search_stats)
        setattr(_backtracking_exact_cover, "last_stats", dict(attempt_stats))
        if placements is not None:
            return placements
        last_stats = attempt_stats

    if last_stats is not None:
        setattr(_backtracking_exact_cover, "last_stats", dict(last_stats))
    return None


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


def build_options(
    W: int,
    H: int,
    tiles: List[Rect],
    stride: int,
    *,
    rng: Optional[random.Random] = None,
    randomize: bool = False,
):
    opts: List[List[Tuple[int, int, int, int, int]]] = []
    phase_seed = (W * 1315423911 ^ H * 2654435761) & 0xFFFFFFFF
    max_opts_per_tile = int(getattr(CFG, "MAX_OPTIONS_PER_TILE", 2000))
    max_opts_per_rect = int(getattr(CFG, "MAX_OPTIONS_PER_RECT", 2000))
    rng_local = _system_rng() if (randomize and rng is None) else rng

    board_cells = max(0, int(W)) * max(0, int(H))
    coverage: List[int] = [0] * board_cells if board_cells else []
    tile_option_counts: List[int] = []
    duplicates_pruned = 0
    thinned = False
    total_options = 0

    for idx, r in enumerate(tiles):
        t: List[Tuple[int, int, int, int, int]] = []
        cfgs = [(r.w, r.h, 0)] if r.w == r.h else [(r.w, r.h, 0), (r.h, r.w, 1)]
        total_for_rect = 0

        if randomize and rng_local is not None and len(cfgs) > 1:
            rng_local.shuffle(cfgs)

        for (w, h, rot) in cfgs:
            raw_locs = _compute_locs(W, H, w, h, stride)
            locs = [(*loc[:2], rot, loc[3], loc[4]) for loc in raw_locs]

            if randomize and rng_local is not None and len(locs) > 1:
                rng_local.shuffle(locs)

            if len(locs) > max_opts_per_tile:
                thinned = True
                if randomize and rng_local is not None:
                    locs = locs[:max_opts_per_tile]
                else:
                    step = max(1, len(locs) // max_opts_per_tile)
                    offset = (phase_seed + idx * 97 + w * 17 + h * 23 + rot * 31) % step
                    locs = locs[offset::step][:max_opts_per_tile]

            seen: Set[Tuple[int, int, int, int]] = set()
            for loc in locs:
                px, py, _rot, lw, lh = loc
                key = (px, py, lw, lh)
                if key in seen:
                    duplicates_pruned += 1
                    continue
                seen.add(key)
                t.append(loc)
                if coverage:
                    for dy in range(lh):
                        row_offset = (py + dy) * W
                        for dx in range(lw):
                            idx_cell = row_offset + px + dx
                            if 0 <= idx_cell < len(coverage):
                                coverage[idx_cell] += 1
                total_for_rect += 1

        if total_for_rect > max_opts_per_rect and len(t) > max_opts_per_rect:
            thinned = True
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
        tile_option_counts.append(len(t))
        total_options += len(t)

    forced_slack: Set[int] = set()
    if coverage and not thinned:
        forced_slack = {cell for cell, count in enumerate(coverage) if count == 0}

    meta: Dict[str, object] = {
        "option_count": total_options,
        "tile_option_counts": tile_option_counts,
        "coverage": coverage if coverage else None,
        "forced_slack": forced_slack,
        "duplicates_pruned": duplicates_pruned,
        "thinned": thinned,
    }

    return opts, meta

# ---------------- main solve ----------------
def try_pack_exact_cover(
    W: int,
    H: int,
    multiset: Union[Iterable, Dict],
    allow_discard: bool = False,
    max_seconds: float = 30.0,
    *,
    initial_hint: Optional[List[Placed]] = None,
    force_backtracking: bool = False,
) -> Tuple[bool, List[Placed], str]:
    """Exact-cover model; if allow_discard=True, coverage model (no callbacks)."""

    result_ok: bool = False
    result_reason: Optional[str] = None
    result_placed: List[Placed] = []
    board_W: Optional[int] = None
    board_H: Optional[int] = None
    tiles_requested: Optional[int] = None
    backtracking_attempts: List[Dict[str, object]] = []

    meta: Dict[str, object] = {
        "allow_discard": bool(allow_discard),
        "initial_hint_count": len(initial_hint or []),
        "backtracking_attempts": backtracking_attempts,
        "solved_via": None,
    }

    def _finish(ok_value: bool, placements_value: List[Placed], reason_value: Optional[str]):
        nonlocal result_ok, result_reason, result_placed
        result_ok = bool(ok_value)
        result_reason = reason_value
        result_placed = list(placements_value) if placements_value else []
        return ok_value, placements_value, reason_value

    try:
        try:
            board_W = int(W)
            board_H = int(H)
        except Exception:
            meta["error"] = "grid_not_integers"
            return _finish(False, [], "Bad grid: W/H must be integers")

        W, H = board_W, board_H
        meta["board"] = {"W": W, "H": H}

        if W <= 0 or H <= 0:
            meta["error"] = "grid_non_positive"
            return _finish(False, [], "Bad grid: W/H must be positive")

        ok, tiles, reason = _expand_multiset(multiset)
        if not ok:
            meta["error"] = "demand_parse"
            meta["parse_reason"] = reason
            return _finish(False, [], reason)
        if not tiles:
            meta["error"] = "demand_empty"
            return _finish(False, [], "Bad demand: nothing parsed from request")

        meta["force_backtracking"] = bool(force_backtracking)

        randomize = bool(getattr(CFG, "RANDOMIZE_PLACEMENTS", False))
        rng = _system_rng() if randomize else None

        tiles = list(tiles)
        tiles_requested = len(tiles)
        meta["tiles_requested"] = tiles_requested

        if force_backtracking and allow_discard:
            meta["error"] = "backtracking_force_discard"
            return _finish(False, [], "Backtracking rescue unavailable when discards are allowed")

        if randomize and rng is not None and len(tiles) > 1:
            rng.shuffle(tiles)

        stride = max(1, int(getattr(CFG, "GRID_STRIDE_BASE", 1)))
        dims: List[int] = []
        for r in tiles:
            dims.append(abs(int(r.w)))
            dims.append(abs(int(r.h)))
        if dims:
            dims_gcd = abs(int(dims[0]))
            for dim in dims[1:]:
                dims_gcd = math.gcd(dims_gcd, abs(int(dim)))
            stride = min(stride, max(1, dims_gcd))
        max_placements = int(getattr(CFG, "MAX_PLACEMENTS", 150000))

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

        meta["stride"] = stride

        forced: List[Placed] = []
        remaining_tiles: List[Rect] = []

        for rect in tiles:
            if (rect.w == W and rect.h == H) or (rect.w == H and rect.h == W):
                orientation = Rect(W, H, rect.name)
                forced.append(Placed(0, 0, orientation))
            else:
                remaining_tiles.append(rect)

        meta["forced_tiles"] = len(forced)

        if len(forced) > 1:
            meta["error"] = "conflicting_full_board_tiles"
            return _finish(False, [], "Bad demand: multiple tiles match entire board")

        if forced and remaining_tiles:
            meta["error"] = "full_board_conflict"
            return _finish(False, [], "Bad demand: full-board tile conflicts with other tiles")

        if forced and not remaining_tiles:
            meta["solved_via"] = "forced_full_board"
            return _finish(True, list(forced), None)

        if not forced:
            remaining_tiles = list(tiles)

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
                    meta["solved_via"] = "forced_strip_width"
                    return _finish(True, strip_forced, None)

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
                    meta["solved_via"] = "forced_strip_height"
                    return _finish(True, strip_forced, None)

        tiles = remaining_tiles

        max_tile_side_cells = 0
        if tiles:
            max_tile_side_cells = max(max(abs(int(r.w)), abs(int(r.h))) for r in tiles)
        max_tile_side_ft = max_tile_side_cells * CELL if max_tile_side_cells else 0.0

        same_shape_cfg = getattr(CFG, "SAME_SHAPE_LIMIT", None)
        try:
            same_shape_int = int(same_shape_cfg)
        except Exception:
            same_shape_int = None

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
            max_tile_side_ft=max_tile_side_ft,
        )
        plus_guard_enabled = _no_plus_guard_enabled(CFG)

        guards_require_cp_sat = (
            plus_guard_enabled
            or (edge_guard_cells is not None)
            or (same_shape_int is not None and same_shape_int >= 0)
        )

        guard_settings: Dict[str, object] = {}
        if edge_guard_cells is not None:
            guard_settings["max_edge_cells"] = edge_guard_cells
        if plus_guard_enabled:
            guard_settings["no_plus"] = True
        if same_shape_int is not None and same_shape_int >= 0:
            guard_settings["same_shape_limit"] = same_shape_int
        if not guard_settings:
            guard_settings = {}

        options, options_meta = build_options(
            W, H, tiles, stride, rng=rng, randomize=randomize
        )
        total_places = int(options_meta.get("option_count", sum(len(o) for o in options)))
        meta["option_count"] = total_places
        forced_slack_cells: Set[int] = set(options_meta.get("forced_slack") or ())
        meta["forced_slack_cells"] = len(forced_slack_cells)
        meta["options_thinned"] = bool(options_meta.get("thinned"))
        meta["option_duplicates_pruned"] = int(options_meta.get("duplicates_pruned", 0))
        tile_option_counts = options_meta.get("tile_option_counts")
        if isinstance(tile_option_counts, list) and len(tile_option_counts) == len(options):
            meta["tile_option_counts"] = list(tile_option_counts)
        if total_places == 0:
            meta["error"] = "no_options"
            return _finish(False, [], "No placements remain (thinned away or grid too small)")
        if total_places > max_placements and stride >= max(W, H):
            meta["error"] = "model_capped"
            meta["capped_placements"] = total_places
            return _finish(False, [], (
                f"Model capped: {total_places:,} placements > limit ({max_placements:,}); stride={stride}"
            ))

        ordering = list(range(len(tiles)))
        if ordering:
            if randomize and rng is not None:
                rng.shuffle(ordering)
            ordering.sort(key=lambda i: (len(options[i]) or 0, -(tiles[i].w * tiles[i].h)))
            if any(idx != i for i, idx in enumerate(ordering)):
                tiles = [tiles[i] for i in ordering]
                options = [options[i] for i in ordering]
                if "tile_option_counts" in meta:
                    toc_list = meta.get("tile_option_counts")
                    if isinstance(toc_list, list) and len(toc_list) == len(ordering):
                        meta["tile_option_counts"] = [toc_list[i] for i in ordering]

        backtracking_pref = bool(getattr(CFG, "BACKTRACK_PROBE_FIRST", True))
        if guards_require_cp_sat and backtracking_pref:
            backtracking_pref = False
            meta["backtracking_prefilter_blocked"] = "guards"
        meta["backtracking_prefilter"] = backtracking_pref

        def _run_backtracking(stage: str) -> Optional[List[Placed]]:
            attempt: Dict[str, object] = {"stage": stage}
            start_bt = time.time()
            placements = _backtracking_exact_cover(
                W,
                H,
                tiles,
                options,
                max_seconds=max_seconds,
                guards=guard_settings,
                forced_slack=forced_slack_cells,
            )
            elapsed_bt = time.time() - start_bt
            attempt["elapsed"] = elapsed_bt
            stats = getattr(_backtracking_exact_cover, "last_stats", None)
            if isinstance(stats, dict):
                attempt.update({
                    "nodes": stats.get("nodes"),
                    "limit_hit": stats.get("limit_hit"),
                    "reason": stats.get("reason"),
                    "same_shape_guard_relaxed": stats.get("same_shape_guard_relaxed"),
                    "coverage_ratio": stats.get("coverage_ratio"),
                    "forced_slack_cells": stats.get("forced_slack_cells"),
                    "free_slack_cells": stats.get("free_slack_cells"),
                })
            if placements is not None:
                placements = list(placements)
                placements.extend(forced)
                attempt["placed"] = len(placements)
                attempt["result"] = "solved"
                backtracking_attempts.append(attempt)
                return placements
            attempt["placed"] = 0
            attempt["result"] = "failed"
            backtracking_attempts.append(attempt)
            return None

        backtracking_pref_failed = False
        if not allow_discard and backtracking_pref:
            pref_solution = _run_backtracking("prefilter")
            if pref_solution is not None:
                if force_backtracking:
                    meta["cp_sat"] = {"skipped": True}
                meta["solved_via"] = "backtracking_prefilter"
                return _finish(True, pref_solution, None)
            backtracking_pref_failed = True

        if force_backtracking:
            meta["forced_backtracking"] = True
            rescue = _run_backtracking("forced_rescue")
            if rescue is not None:
                meta["solved_via"] = "backtracking_forced"
                meta["cp_sat"] = {"skipped": True}
                return _finish(True, rescue, None)

            meta["backtracking_rescue_failed"] = True
            msg = "Backtracking rescue failed"
            if backtracking_attempts:
                last_attempt = backtracking_attempts[-1]
                details: List[str] = []
                if isinstance(last_attempt, dict):
                    reason_tag = last_attempt.get("reason")
                    if reason_tag:
                        details.append(str(reason_tag))
                    if last_attempt.get("limit_hit"):
                        details.append("limit hit")
                if details:
                    msg = f"{msg} ({', '.join(details)})"
            meta["error"] = "backtracking_rescue_failed"
            meta["cp_sat"] = {"skipped": True}
            return _finish(False, [], msg)

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

        if (
            edge_guard_cells is None
            and max_edge_ft_cfg not in (None, 0)
            and max_tile_side_ft > 0
            and max_tile_side_ft > float(max_edge_ft_cfg)
        ):
            meta["edge_guard_relaxed"] = {
                "configured_max_edge_ft": float(max_edge_ft_cfg),
                "max_tile_side_ft": max_tile_side_ft,
            }
        plus_guard_enabled = _no_plus_guard_enabled(CFG)
    
        def _attempt(*, edge_guard_cells, plus_guard_enabled):
            return _solve_with_limit(
                same_shape_cfg,
                max_seconds,
                edge_guard_cells=edge_guard_cells,
                plus_guard_enabled=plus_guard_enabled,
            )
    
        cp_sat_start = time.time()
        ok, placed, reason, active_edge_guard, active_plus_guard = _resolve_guard_backoffs(
            _attempt,
            edge_guard_cells=edge_guard_cells,
            plus_guard_enabled=plus_guard_enabled,
        )
        cp_sat_elapsed = time.time() - cp_sat_start
        meta["cp_sat"] = {
            "ok": bool(ok),
            "reason": reason,
            "edge_guard": active_edge_guard,
            "plus_guard": bool(active_plus_guard),
            "elapsed": cp_sat_elapsed,
            "placements": len(placed) if placed else 0,
        }
    
        def _coverage_cells(placements: List[Placed]) -> int:
            return sum(p.rect.w * p.rect.h for p in placements) if placements else 0
    
        target_cells = W * H
        best_ok, best_placed, best_reason = ok, placed, reason
        best_coverage = _coverage_cells(placed)
        meta["cp_sat_coverage_cells"] = best_coverage
    
        guard_active = (active_edge_guard is not None) or bool(active_plus_guard)
    
        if (
            allow_discard
            and guard_active
            and target_cells > 0
            and best_coverage < target_cells
            and not best_ok
        ):
            fallback_plan: List[Tuple[Optional[int], bool]] = []
            if active_edge_guard is not None:
                fallback_plan.append((None, active_plus_guard))
            if active_plus_guard:
                fallback_plan.append((active_edge_guard, False))
            if active_edge_guard is not None and active_plus_guard:
                fallback_plan.append((None, False))
    
            for guard_edge, guard_plus in fallback_plan:
                relax_ok, relax_placed, relax_reason = _attempt(
                    edge_guard_cells=guard_edge,
                    plus_guard_enabled=guard_plus,
                )
                relax_cov = _coverage_cells(relax_placed)
                improved = relax_ok and relax_cov > best_coverage
                if not improved and not relax_ok and not best_ok:
                    improved = relax_cov > best_coverage
                if improved:
                    best_ok = relax_ok
                    best_placed = relax_placed
                    best_reason = relax_reason
                    best_coverage = relax_cov
                    if best_coverage >= target_cells:
                        break
    
            ok, placed, reason = best_ok, best_placed, best_reason
    
        guard_active = (active_edge_guard is not None) or bool(active_plus_guard)
    
        if ok and placed:
            meta.setdefault("solved_via", "cp_sat")
    
        if not ok and not allow_discard and not guard_active:
            if backtracking_pref_failed:
                meta["backtracking_skipped"] = "prefilter_failed"
            else:
                fallback_solution = _run_backtracking("post_cp_sat")
                if fallback_solution is not None:
                    meta["solved_via"] = "backtracking_post_cp_sat"
                    return _finish(True, fallback_solution, None)
    
        if ok or reason != "Proven infeasible under current constraints":
            return _finish(ok, placed, reason)
    
        # If the limit is finite, probe a relaxed model to provide actionable feedback.
        try:
            same_shape_int = int(same_shape_cfg)
        except Exception:
            same_shape_int = None
    
        if same_shape_int is None or same_shape_int < 0:
            return _finish(ok, placed, reason)
    
        diag_seconds = min(float(max_seconds), float(getattr(CFG, "SAME_SHAPE_DIAG_SECONDS", 10.0)))
        diag_seconds = max(1.0, diag_seconds)
        diag_ok, diag_placed, _ = _solve_with_limit(
            None,
            diag_seconds,
            edge_guard_cells=active_edge_guard,
            plus_guard_enabled=active_plus_guard,
        )
        if diag_ok and diag_placed:
            msg = (
                "Proven infeasible under current constraints (same-shape limit "
                f"{same_shape_int} prevents a feasible layout; try increasing TS_SAME_SHAPE_LIMIT "
                "or setting it to -1 to disable the guard)."
            )
            return _finish(False, [], msg)
    
        return _finish(ok, placed, reason)
    
    finally:
        if board_W is not None and board_H is not None:
            meta.setdefault("board", {"W": board_W, "H": board_H})
        if tiles_requested is not None:
            meta.setdefault("tiles_requested", tiles_requested)
        meta["ok"] = bool(result_ok)
        meta["reason"] = result_reason
        meta["placed"] = len(result_placed)
        if meta.get("solved_via") is None and result_ok:
            meta["solved_via"] = "cp_sat"
        setattr(try_pack_exact_cover, "last_meta", meta)
