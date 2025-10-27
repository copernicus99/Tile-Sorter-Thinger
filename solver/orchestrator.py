# Orchestrator: option-driven CP-SAT workflow (A–F sequence)
from __future__ import annotations

import os
import sys
import time
import traceback
import math
import threading
import heapq
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set

import multiprocessing as mp

from models import Rect, Placed, ft_to_cells
from tiles import parse_demand
from config import CFG, CELL
from progress import (
    set_phase, set_phase_total, set_attempt, set_grid, set_progress_pct,
    set_best_used, set_coverage_pct, set_elapsed, set_status, set_message,
    log_attempt_detail,
)
from solver.cp_sat import try_pack_exact_cover

# Ensure Windows-safe start method
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass


# ---------- helpers ----------

@dataclass
class CandidateBoard:
    W: int  # cells
    H: int  # cells
    label: str  # e.g. '12.0 × 12.0 ft'


def _fmt_ft(cells: int) -> float:
    return round(cells * CELL + 1e-9, 1)


def _coerce_bag_ft(maybe: Any) -> Dict[Tuple[float, float], int]:
    """
    Convert a variety of inbound shapes into {(wf, hf): count}.
    Accepts:
      - {(wf, hf): count}
      - [((wf, hf), count), ...]
      - (ok, bag_map)
      - raw form-like dict -> tiles.parse_demand
    """
    if isinstance(maybe, tuple) and len(maybe) == 2 and isinstance(maybe[0], (bool, int)):
        ok_flag, bag = maybe
        if not ok_flag:
            return {}
        maybe = bag

    if isinstance(maybe, dict):
        if maybe and all(isinstance(k, tuple) and len(k) == 2 for k in maybe.keys()):
            return {(float(k[0]), float(k[1])): int(v) for k, v in maybe.items()}
        parsed = parse_demand(maybe)
        if isinstance(parsed, tuple):
            if len(parsed) == 2:
                ok, bag_ft = parsed
                return bag_ft if ok else {}
            if len(parsed) == 3:
                bag_list, _decoded, _err = parsed
                if bag_list:
                    out: Dict[Tuple[float, float], int] = {}
                    for wf, hf, cnt in bag_list:
                        try:
                            key = (float(wf), float(hf))
                            out[key] = out.get(key, 0) + int(cnt)
                        except Exception:
                            continue
                    if out:
                        return out
                return {}
        return parsed

    if hasattr(maybe, "__iter__"):
        try:
            out: Dict[Tuple[float, float], int] = {}
            for pair in maybe:
                (wf, hf), cnt = pair
                out[(float(wf), float(hf))] = out.get((float(wf), float(hf)), 0) + int(cnt)
            if out:
                return out
        except Exception:
            pass

    return {}


def _bag_ft_to_cells(bag_ft: Dict[Tuple[float, float], int]) -> Dict[Tuple[int, int], int]:
    bag_cells: Dict[Tuple[int, int], int] = {}
    for (wf, hf), cnt in bag_ft.items():
        Wc = ft_to_cells(wf)
        Hc = ft_to_cells(hf)
        bag_cells[(Wc, Hc)] = bag_cells.get((Wc, Hc), 0) + int(cnt)
    return bag_cells


def _align_up_to_multiple(value: int, step: int) -> int:
    if step <= 1:
        return int(value)
    value_i = int(value)
    return int(((value_i + step - 1) // step) * step)


def _align_down_to_multiple(value: int, step: int) -> int:
    if step <= 1:
        return int(value)
    value_i = int(value)
    return int((value_i // step) * step)


def _grid_step_from_bag(bag_cells: Dict[Tuple[int, int], int]) -> int:
    step = 0
    for (w, h) in bag_cells.keys():
        step = math.gcd(step, int(abs(w)))
        step = math.gcd(step, int(abs(h)))
    return max(1, step)


def _square_candidates(min_side: int, max_side: int, *, descending: bool, multiple_of: int = 1) -> List[CandidateBoard]:
    min_side = max(6, int(min_side))
    max_side = max(min_side, int(max_side))
    if descending:
        rng = range(max_side, min_side - 1, -1)
    else:
        rng = range(min_side, max_side + 1)
    out: List[CandidateBoard] = []
    for s in rng:
        if multiple_of > 1 and s % multiple_of != 0:
            continue
        out.append(CandidateBoard(s, s, f"{_fmt_ft(s)} × {_fmt_ft(s)} ft"))
    return out


def _mirrored_probe_order(values: List[int]) -> List[int]:
    """Return a high/low interleaving (pop-in/out) ordering.

    Phase D is meant to “fit tiles to a 10×10 grid,” so we start from the base
    grid (the largest candidate) and then mirror probes toward the smaller
    boards.  This alternates popping from the high and low ends of the sorted
    sequence to avoid biasing the search toward only shrinking or only growing
    boards.
    """

    if not values:
        return []

    ordered: List[int] = []
    lo = 0
    hi = len(values) - 1
    while hi >= lo:
        ordered.append(values[hi])
        if hi == lo:
            break
        ordered.append(values[lo])
        hi -= 1
        lo += 1
    return ordered


def _should_retry_phase(reason: Optional[str]) -> bool:
    """Return True when a phase attempt should be retried on the same board."""

    if not reason:
        return False

    text = str(reason).strip().lower()
    if not text:
        return False

    if "proven infeasible" in text:
        return False

    retry_tokens = (
        "timebox",
        "timeout",
        "stopped before solution",
        "subprocess",
        "crash",
        "capped",
        "limit",
        "killed",
    )

    return any(token in text for token in retry_tokens)


def _is_crash_reason(reason: Optional[str]) -> bool:
    if not reason:
        return False
    text = str(reason).strip().lower()
    if not text:
        return False
    crash_tokens = (
        "subprocess ended early",
        "parent recv exception",
        "native solver crash",
        "timebox / crash",
        "child exit",
    )
    return any(token in text for token in crash_tokens)


def _phase_c_candidates(base_side: int, *, grid_step: int) -> List[CandidateBoard]:
    """Return the Phase C base board sized to the provided baseline."""

    side = max(6, int(base_side))
    if grid_step > 1:
        side = _align_up_to_multiple(side, grid_step)
    label = f"{_fmt_ft(side)} × {_fmt_ft(side)} ft"
    return [CandidateBoard(side, side, label)]


_PHASE_D_MAX_CANDIDATES = 15


def _phase_d_candidates(
    shrink_floor: int,
    base_side: int,
    *,
    grid_step: int,
    area_cells: int,
) -> List[CandidateBoard]:
    """Return Phase D boards by walking neighbors around the base grid.

    Phase D is the "pop-in/out" search over grids near the base 10×10 board.
    Earlier iterations of the orchestrator hard-coded this phase to probe only
    the 10×10 grid which prevented it from exploring slightly wider/taller
    boards required by some puzzles.  This helper now performs a small best-
    first search around the base grid, bounded by the maximum tile dimension
    (``shrink_floor``) on the lower side and by both the total demand area and
    a modest expansion budget on the upper side.  The resulting sequence keeps
    the base board first, then alternates between shrinking and expanding to
    mirror the desired "pop-in/out" behaviour while respecting the
    ``_PHASE_D_MAX_CANDIDATES`` limit.
    """

    step = max(1, int(grid_step))
    base = max(6, int(base_side))

    floor_val = max(6, int(shrink_floor))
    if floor_val > base:
        floor_val = base

    if area_cells <= 0:
        area_side = base
    else:
        area_side = _align_up_to_multiple(_ceil_sqrt_cells(area_cells), step)

    expand_cap = max(base, area_side)
    expand_cap = max(expand_cap, base + step * 4)
    expand_cap = _align_up_to_multiple(expand_cap, step)

    heap: List[Tuple[int, int, int]] = []
    heapq.heappush(heap, (0, base, base))
    seen: set[Tuple[int, int]] = set()
    out: List[CandidateBoard] = []

    min_area = max(0, int(area_cells))

    while heap and len(out) < _PHASE_D_MAX_CANDIDATES:
        priority, Wc, Hc = heapq.heappop(heap)
        key = (Wc, Hc)
        if key in seen:
            continue
        seen.add(key)

        Wc = max(6, int(Wc))
        Hc = max(6, int(Hc))
        if Wc * Hc >= min_area:
            out.append(CandidateBoard(Wc, Hc, f"{_fmt_ft(Wc)} × {_fmt_ft(Hc)} ft"))

        for dW, dH in ((-step, 0), (step, 0), (0, -step), (0, step)):
            next_W = Wc + dW
            next_H = Hc + dH
            if next_W < floor_val or next_H < floor_val:
                continue
            if next_W > expand_cap or next_H > expand_cap:
                continue
            if (next_W, next_H) in seen:
                continue
            next_priority = max(abs(next_W - base), abs(next_H - base))
            heapq.heappush(heap, (next_priority, next_W, next_H))

    if not out:
        fallback_side = max(
            base,
            _align_up_to_multiple(_ceil_sqrt_cells(min_area), step),
        )
        out.append(
            CandidateBoard(
                fallback_side,
                fallback_side,
                f"{_fmt_ft(fallback_side)} × {_fmt_ft(fallback_side)} ft",
            )
        )

    return out


def _rectangular_candidates(widths: Iterable[int], heights: Iterable[int], *, descending: bool, multiple_of: int = 1) -> List[CandidateBoard]:
    out: List[CandidateBoard] = []
    seen = set()
    for w in widths:
        for h in heights:
            w_i = max(6, int(w))
            h_i = max(6, int(h))
            key = (w_i, h_i)
            if key in seen:
                continue
            seen.add(key)
            if multiple_of > 1 and (w_i % multiple_of != 0 or h_i % multiple_of != 0):
                continue
            out.append(CandidateBoard(w_i, h_i, f"{_fmt_ft(w_i)} × {_fmt_ft(h_i)} ft"))
    key_fn = lambda cb: (cb.W * cb.H, max(cb.W, cb.H), cb.W)
    return sorted(out, key=key_fn, reverse=descending)


def _ceil_sqrt_cells(area_cells: int) -> int:
    if area_cells <= 0:
        return 6
    return int(math.ceil(area_cells ** 0.5))


def _area_sqft(area_cells: int) -> float:
    return float(area_cells) * (CELL ** 2)


def _descending_values(start: int, end: int, *, step: int = 1) -> List[int]:
    start_i = int(start)
    end_i = int(end)
    if start_i < end_i:
        start_i, end_i = end_i, start_i
    values: List[int] = []
    for val in range(start_i, end_i - 1, -1):
        if step <= 1 or val % step == 0:
            values.append(val)
    if not values:
        values.append(start_i)
    return values


# ---------- CP-SAT isolation ----------

def _cp_worker(conn, W, H, bag, seconds, allow_discard, hint, stderr_path=None):
    stderr_handle = None
    try:
        if stderr_path:
            try:
                fd = os.open(
                    stderr_path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o644,
                )
                try:
                    os.dup2(fd, 2)
                finally:
                    if fd not in (0, 1, 2):
                        os.close(fd)
                stderr_handle = open(
                    stderr_path,
                    "a",
                    buffering=1,
                    encoding="utf-8",
                    errors="replace",
                )
                sys.stderr = stderr_handle
            except Exception:
                stderr_handle = None
        os.environ["ORTOOLS_CP_SAT_NUM_THREADS"] = "1"
        os.environ["NUM_CPUS"] = "1"
        ok, placed, reason = try_pack_exact_cover(
            W=W,
            H=H,
            multiset=bag,
            allow_discard=allow_discard,
            max_seconds=seconds,
            initial_hint=hint,
        )
        out = []
        for p in placed:
            out.append((p.x, p.y, (p.rect.w, p.rect.h, getattr(p.rect, "name", None))))
        meta = getattr(try_pack_exact_cover, "last_meta", None)
        conn.send((ok, out, reason, meta))
    except Exception as e:
        try:
            conn.send((False, [], f"cp-sat exception: {type(e).__name__}: {e}", None))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if stderr_handle:
            try:
                stderr_handle.flush()
            except Exception:
                pass
            try:
                stderr_handle.close()
            except Exception:
                pass


def _safe_parent_poll(parent, timeout):
    try:
        return parent.poll(timeout)
    except BrokenPipeError:
        return True


def _terminate_process(proc: mp.Process, grace: float = 0.2) -> None:
    """Best-effort helper that tears down ``proc`` within ``grace`` seconds."""

    try:
        proc.join(timeout=grace)
    except Exception:
        pass

    if not proc.is_alive():
        return

    terminate = getattr(proc, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except Exception:
            pass

    try:
        proc.join(timeout=grace)
    except Exception:
        pass

    if not proc.is_alive():
        return

    kill = getattr(proc, "kill", None)
    if callable(kill):
        try:
            kill()
        except Exception:
            pass

    try:
        proc.join(timeout=grace)
    except Exception:
        pass


def _run_cp_sat_isolated(
    W: int,
    H: int,
    bag: Dict[Tuple[int, int], int],
    seconds: float,
    allow_discard: bool,
    *,
    hint: Optional[List[Placed]] = None,
):
    try:
        stderr_tmp = tempfile.NamedTemporaryFile(
            prefix="cp_sat_",
            suffix=".stderr",
            delete=False,
        )
        stderr_path = stderr_tmp.name
    except Exception:
        stderr_path = None
    else:
        stderr_tmp.close()

    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_cp_worker,
        args=(child, W, H, bag, seconds, allow_discard, hint, stderr_path),
    )
    proc.daemon = True
    proc.start()
    try:
        child.close()
    except Exception:
        pass

    ok = False
    placed: List[Placed] = []
    reason = None
    meta: Optional[Dict[str, object]] = None
    payload_received = False
    payload_kind: Optional[str] = None
    payload_raw_type: Optional[str] = None
    placement_count = 0

    try:
        budget = max(0.0, float(seconds))
    except Exception:
        budget = 0.0
    # Allow a small grace window for process spin-up / teardown.
    deadline = time.monotonic() + max(1.0, budget + 1.0)

    while proc.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait_slice = min(0.1, remaining)
        if _safe_parent_poll(parent, wait_slice):
            break

    try:
        if parent.poll(0):
            payload = parent.recv()
            if isinstance(payload, tuple) and len(payload) == 4:
                ok_msg, placed_msg, reason_msg, meta_msg = payload
            else:
                ok_msg, placed_msg, reason_msg = payload
                meta_msg = None
            ok = bool(ok_msg)
            out: List[Placed] = []
            for x, y, (w, h, nm) in placed_msg:
                out.append(Placed(x, y, Rect(w, h, nm)))
            placed = out
            placement_count = len(placed_msg) if hasattr(placed_msg, "__len__") else len(out)
            reason = reason_msg
            if isinstance(meta_msg, dict):
                meta = meta_msg
            payload_received = True
            payload_kind = "result"
            payload_raw_type = type(payload).__name__
        else:
            reason = "Stopped before solution (timebox / crash?)"
    except (EOFError, BrokenPipeError):
        reason = "Subprocess ended early (pipe closed)"
    except Exception as e:
        reason = f"Parent recv exception: {type(e).__name__}: {e}"

    try:
        if proc.is_alive():
            _terminate_process(proc)
    finally:
        try:
            parent.close()
        except Exception:
            pass

    if proc.exitcode == -1073741819:  # 0xC0000005
        ok = False
        if not payload_received or not reason:
            reason = "Native solver crash (0xC0000005) – isolated; server is fine"

    stderr_tail: Optional[str] = None
    if stderr_path and os.path.exists(stderr_path):
        try:
            with open(stderr_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                limit = 2048
                fh.seek(max(0, size - limit))
                data = fh.read()
            stderr_tail = data.decode("utf-8", errors="replace")
        except Exception:
            stderr_tail = None
        finally:
            with suppress(OSError):
                os.remove(stderr_path)
    elif stderr_path:
        with suppress(OSError):
            os.remove(stderr_path)

    isolation_meta: Dict[str, object] = {
        "exitcode": proc.exitcode,
        "payload_received": payload_received,
        "payload_kind": payload_kind,
        "payload_raw_type": payload_raw_type,
        "placement_count": placement_count,
        "stderr_tail": (stderr_tail or "").strip(),
    }

    meta_out: Dict[str, object] = {}
    if isinstance(meta, dict):
        meta_out.update(meta)
    meta_out["isolation"] = isolation_meta

    return ok, placed, reason, meta_out


def _run_backtracking_rescue(
    W: int,
    H: int,
    bag: Dict[Tuple[int, int], int],
    *,
    hint: Optional[List[Placed]] = None,
    seconds: Optional[float] = None,
):
    guard_descriptions: List[str] = []

    test_mode = False
    try:
        test_mode = bool(getattr(CFG, "TEST_MODE", False))
    except Exception:
        test_mode = False

    if not test_mode:
        try:
            max_edge_ft = getattr(CFG, "MAX_EDGE_FT", None)
        except Exception:
            max_edge_ft = None
        try:
            edge_value = None if max_edge_ft is None else float(max_edge_ft)
        except Exception:
            edge_value = None
        if edge_value is not None and edge_value > 0:
            guard_descriptions.append(f"edge length ≤ {edge_value:g} ft")

        try:
            no_plus_flag = bool(getattr(CFG, "NO_PLUS", False))
        except Exception:
            no_plus_flag = False
        if no_plus_flag:
            guard_descriptions.append("no-plus intersections")

        try:
            same_shape_limit = getattr(CFG, "SAME_SHAPE_LIMIT", None)
        except Exception:
            same_shape_limit = None
        try:
            same_shape_value = (
                None if same_shape_limit is None else int(same_shape_limit)
            )
        except Exception:
            same_shape_value = None
        if same_shape_value is not None and same_shape_value >= 0:
            guard_descriptions.append(
                f"same-shape adjacency limit {same_shape_value}"
            )

    try:
        ok, placed, reason = try_pack_exact_cover(
            W=W,
            H=H,
            multiset=bag,
            allow_discard=False,
            max_seconds=float(seconds) if seconds is not None else float(getattr(CFG, "TIME_F", 900.0)),
            initial_hint=hint,
            force_backtracking=True,
        )
    except Exception as exc:
        return (
            False,
            [],
            f"Backtracking rescue exception: {type(exc).__name__}: {exc}",
            None,
        )

    meta = getattr(try_pack_exact_cover, "last_meta", None)
    if isinstance(meta, dict) and guard_descriptions:
        meta.setdefault("guards", guard_descriptions)
    return ok, placed, reason, meta


# ---------- public entrypoint ----------

def solve_orchestrator(*args, **kwargs):
    """
    Returns: (ok, placed, W_ft, H_ft, strategy, reason, meta)
    Strategy values are the option letters A–F from the specification.
    """
    t0 = time.time()

    log_attempt_detail(
        "Guard configuration",
        TS_TEST_MODE=1 if getattr(CFG, "TEST_MODE", False) else 0,
        TS_MAX_EDGE_FT=float(getattr(CFG, "MAX_EDGE_FT", 0.0)),
        TS_NO_PLUS=1 if getattr(CFG, "NO_PLUS", False) else 0,
        TS_SAME_SHAPE_LIMIT=int(getattr(CFG, "SAME_SHAPE_LIMIT", -1)),
    )
    try:
        inbound = None
        if args:
            inbound = args[0]
        elif "bag" in kwargs:
            inbound = kwargs["bag"]
        elif "bag_ft" in kwargs:
            inbound = kwargs["bag_ft"]
        elif "form_data" in kwargs:
            inbound = kwargs["form_data"]

        bag_ft = _coerce_bag_ft(inbound)
        if not bag_ft:
            set_status("Error")
            return (False, [], 0.0, 0.0, "error",
                    "Bad demand: nothing parsed from request", {})

        bag_cells = _bag_ft_to_cells(bag_ft)
        grid_step = _grid_step_from_bag(bag_cells)
        demand_count = sum(bag_cells.values())
        area_cells = sum((w * h) * c for (w, h), c in bag_cells.items())
        area_sqft = _area_sqft(area_cells)

        max_tile_w = max((abs(int(w)) for (w, _h) in bag_cells.keys()), default=0)
        max_tile_h = max((abs(int(h)) for (_w, h) in bag_cells.keys()), default=0)
        max_tile_side_raw = max(max_tile_w, max_tile_h)
        max_tile_side = max(6, max_tile_side_raw)
        max_tile_side = _align_up_to_multiple(max_tile_side, grid_step)

        base_area_sqft = max(1.0, float(getattr(CFG, "BASE_GRID_AREA_SQFT", 1000.0)))
        base_side_cells = max(6, ft_to_cells(math.sqrt(base_area_sqft)))
        base_side_cells = max(base_side_cells, max_tile_side)
        base_side_cells = _align_up_to_multiple(base_side_cells, grid_step)
        sqrt_cells = _ceil_sqrt_cells(area_cells)
        sqrt_cells_raw = sqrt_cells
        sqrt_cells = max(sqrt_cells, max_tile_side)
        sqrt_cells = _align_up_to_multiple(sqrt_cells, grid_step)
        base_side_cells = max(base_side_cells, sqrt_cells)

        log_attempt_detail(
            "Run setup",
            demand_count=demand_count,
            area_sqft=round(area_sqft, 2),
            grid_step=grid_step,
            max_tile_side=max_tile_side,
            base_side_cells=base_side_cells,
            sqrt_cells=sqrt_cells,
        )

        set_status("Solving")
        set_phase("S0")
        set_phase_total(1)
        set_attempt("Computing demand")
        set_grid("—")
        set_progress_pct(0.0)
        set_best_used(0)
        set_coverage_pct(0.0)

        guard_block_reason = "Proven infeasible under current constraints"
        guard_failure_reason: Optional[str] = None

        try:
            probe_board = _align_up_to_multiple(
                max(sqrt_cells_raw, max(max_tile_side_raw, 1)), grid_step
            )
            relax_original = getattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE", True)
            setattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE", False)
            probe_ok, _, probe_reason = try_pack_exact_cover(
                probe_board,
                probe_board,
                bag_cells,
                allow_discard=False,
                max_seconds=1.0,
                force_backtracking=True,
            )
            setattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE", relax_original)
            guard_blocked = False
            if isinstance(probe_reason, str):
                guard_blocked = probe_reason.startswith(guard_block_reason) or (
                    "guard_blocked" in probe_reason.lower()
                )
            if (not probe_ok) and (probe_reason == guard_block_reason or guard_blocked):
                guard_failure_reason = guard_block_reason
        except Exception:
            try:
                setattr(CFG, "BACKTRACK_RELAX_SAME_SHAPE", relax_original)
            except Exception:
                pass
            pass

        best_used_tiles = 0
        best_cover_pct = 0.0
        hint_cache: Dict[Tuple[int, int], List[Placed]] = {}

        def _record_best(used_tiles: int, coverage_pct: float) -> None:
            nonlocal best_used_tiles, best_cover_pct
            coverage_pct = max(0.0, float(coverage_pct))
            used_tiles = max(0, int(used_tiles))
            better = False
            changed = False
            if coverage_pct > best_cover_pct + 1e-9:
                better = True
            elif abs(coverage_pct - best_cover_pct) <= 1e-9 and used_tiles > best_used_tiles:
                better = True
            if better:
                best_cover_pct = coverage_pct
                best_used_tiles = used_tiles
                changed = True
            else:
                if used_tiles > best_used_tiles:
                    best_used_tiles = used_tiles
                    changed = True
                if coverage_pct > best_cover_pct:
                    best_cover_pct = coverage_pct
                    changed = True
            set_best_used(best_used_tiles)
            set_coverage_pct(best_cover_pct)
            if changed:
                log_attempt_detail(
                    "Best progress",
                    used_tiles=best_used_tiles,
                    coverage_pct=round(best_cover_pct, 2),
                )

        def _run_phase(
            label: str,
            candidates: List[CandidateBoard],
            seconds: float,
            allow_discard: bool,
            *,
            prefer_large: bool,
            continue_on_partial: bool = False,
            require_full_duration: bool = False,
        ) -> Tuple[bool, Optional[CandidateBoard], List[Placed], float, int, Optional[str]]:
            nonlocal hint_cache
            if seconds <= 0 or not candidates:
                set_phase(label)
                set_phase_total(int(max(0.0, seconds)))
                set_progress_pct(100.0 if not candidates else 0.0)
                reason = "No board candidates available" if not candidates else "Phase time budget is zero"
                return False, None, [], 0.0, 0, reason

            set_phase(label)
            set_phase_total(int(seconds))
            phase_start = time.time()
            set_progress_pct(0.0)

            def _sort_key(cb: CandidateBoard) -> Tuple[int, float]:
                has_hint = 0 if (cb.W, cb.H) in hint_cache else 1
                area = cb.W * cb.H
                size_rank = -area if prefer_large else area
                return (has_hint, size_rank)

            queue: List[CandidateBoard] = sorted(list(candidates), key=_sort_key)
            base_candidates: List[CandidateBoard] = list(queue)
            total = len(queue)
            best_tuple: Optional[Tuple[CandidateBoard, List[Placed], float, int]] = None
            last_reason: Optional[str] = None
            attempt_idx = 0
            retry_counts: Dict[Tuple[int, int], int] = {}
            max_retries = max(0, int(getattr(CFG, "PHASE_RETRY_LIMIT", 3)))
            min_retry_remaining = min(60.0, max(10.0, 0.2 * seconds)) if seconds > 0 else 0.0
            recent_attempts: List[float] = []
            partial_best: Dict[Tuple[int, int], float] = {}
            partial_stalls: Dict[Tuple[int, int], int] = {}
            plateau_limit = max(1, int(getattr(CFG, "PARTIAL_PLATEAU_LIMIT", 5)))
            progress_boards: Set[Tuple[int, int]] = set()
            rescue_attempted: Set[Tuple[int, int]] = set()

            def _update_phase_progress() -> None:
                elapsed_phase = time.time() - phase_start
                if seconds > 0:
                    pct = max(0.0, min(100.0, 100.0 * (elapsed_phase / seconds)))
                    set_progress_pct(pct)

            progress_stop = threading.Event()

            def _progress_ticker() -> None:
                while not progress_stop.wait(0.5):
                    _update_phase_progress()

            ticker: Optional[threading.Thread] = None
            if seconds > 0:
                ticker = threading.Thread(
                    target=_progress_ticker,
                    name="phase-progress",
                    daemon=True,
                )
                ticker.start()

            try:
                while True:
                    elapsed_phase = time.time() - phase_start
                    if elapsed_phase >= seconds:
                        _update_phase_progress()
                        break

                    if not queue:
                        if not require_full_duration:
                            break
                        if not base_candidates:
                            break
                        if not progress_boards:
                            break
                        queue = [
                            cand
                            for cand in base_candidates
                            if (cand.W, cand.H) in progress_boards
                        ]
                        if not queue:
                            break
                        queue = sorted(queue, key=_sort_key)
                        total = max(total, attempt_idx + len(queue))
                        continue

                    remaining = seconds - elapsed_phase
                    cb = queue.pop(0)
                    attempt_idx += 1
                    key = (cb.W, cb.H)
                    remaining_candidates = max(1, len(queue) + 1)
                    ideal_slice = remaining / float(remaining_candidates)
                    adaptive_slice = ideal_slice
                    recent_window = recent_attempts[-3:]
                    if recent_window:
                        observed = max(recent_window)
                        adaptive_slice = min(adaptive_slice, observed * 1.25)
                    per_attempt = min(max(5.0, adaptive_slice), remaining)

                    set_attempt(
                        cb.label,
                        budget=per_attempt,
                        idx=attempt_idx,
                        total=max(total, attempt_idx + len(queue)),
                    )

                    board_area = cb.W * cb.H if cb else 0
                    if not allow_discard and board_area < area_cells:
                        log_attempt_detail(
                            "Attempt skipped",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            reason="Board area smaller than total tile demand",
                            board_area=board_area,
                            demand_area=area_cells,
                        )
                        set_message(
                            f"Phase {label} skipped {cb.label}: demand exceeds board area"
                        )
                        continue

                    set_grid(cb.label)
                    log_attempt_detail(
                        "Attempt dispatched",
                        phase=label,
                        board=f"{cb.W}×{cb.H}",
                        queue_size=len(queue),
                        remaining_round=round(remaining, 2),
                        adaptive_slice=round(adaptive_slice, 2),
                        budget=round(per_attempt, 2),
                    )
                    attempt_started = time.time()
                    hint = hint_cache.get((cb.W, cb.H))
                    ok, placed, reason, solve_meta = _run_cp_sat_isolated(
                        W=cb.W,
                        H=cb.H,
                        bag=bag_cells,
                        seconds=per_attempt,
                        allow_discard=allow_discard,
                        hint=hint,
                    )
                    retries = retry_counts.get(key, 0)
                    board_area = cb.W * cb.H if cb else 0
                    used_area = (
                        sum(p.rect.w * p.rect.h for p in placed)
                        if placed
                        else 0
                    )
                    crash_like = _is_crash_reason(reason)

                    if crash_like and placed:
                        log_attempt_detail(
                            "Crash partial discarded",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            placements=len(placed),
                            coverage_pct=round(
                                (100.0 * used_area / float(board_area))
                                if board_area > 0 and used_area > 0
                                else 0.0,
                                2,
                            ),
                        )
                        placed = []
                        used_area = 0

                    trigger_crash_rescue = (
                        not ok
                        and crash_like
                        and key not in rescue_attempted
                        and (board_area == 0 or used_area < board_area)
                    )
                    trigger_retry_rescue = (
                        not ok
                        and not allow_discard
                        and not placed
                        and retries >= max_retries
                        and _should_retry_phase(last_reason)
                    )

                    rescue_supported = (not allow_discard) or (board_area >= area_cells)

                    if (trigger_crash_rescue or trigger_retry_rescue) and rescue_supported:
                        rescue_attempted.add(key)
                        trigger = "crash" if trigger_crash_rescue else "retry_limit"
                        log_attempt_detail(
                            "Fallback started",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            reason=reason or "",
                            retries=retries,
                            max_retries=max_retries,
                            trigger=trigger,
                        )
                        set_message(
                            f"Phase {label} CP-SAT failed for {cb.label}; running deterministic rescue"
                        )
                        fb_start = time.time()
                        fb_ok, fb_placed, fb_reason, fb_meta = _run_backtracking_rescue(
                            cb.W,
                            cb.H,
                            bag_cells,
                            hint=hint,
                            seconds=max(per_attempt, 5.0),
                        )
                        fb_elapsed = time.time() - fb_start
                        if fb_ok and fb_placed:
                            ok = True
                            placed = fb_placed
                            used_area = sum(p.rect.w * p.rect.h for p in placed)
                            reason = fb_reason
                            solve_meta = fb_meta or solve_meta
                            log_attempt_detail(
                                "Fallback solved",
                                phase=label,
                                board=f"{cb.W}×{cb.H}",
                                elapsed=round(fb_elapsed, 2),
                                placements=len(fb_placed),
                                solved_via=(fb_meta or {}).get("solved_via"),
                            )
                        else:
                            reason = fb_reason or reason
                            solve_meta = fb_meta or solve_meta
                            retry_counts[key] = max_retries
                            log_attempt_detail(
                                "Fallback failed",
                                phase=label,
                                board=f"{cb.W}×{cb.H}",
                                elapsed=round(fb_elapsed, 2),
                                reason=reason or "",
                            )
                    elif trigger_crash_rescue or trigger_retry_rescue:
                        rescue_attempted.add(key)
                        log_attempt_detail(
                            "Fallback skipped",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            reason=reason or "",
                            allow_discard=allow_discard,
                            board_area=board_area,
                            demand_area=area_cells,
                        )

                    attempt_elapsed = time.time() - attempt_started
                    recent_attempts.append(attempt_elapsed)
                    if len(recent_attempts) > 10:
                        recent_attempts.pop(0)
                    set_message(
                        f"Phase {label} attempt {attempt_idx}/{max(total, attempt_idx + len(queue))} ran {attempt_elapsed:.1f}s (budget ≤{per_attempt:.1f}s)"
                    )

                    last_reason = str(reason) if reason else None

                    meta_payload = solve_meta if isinstance(solve_meta, dict) else {}
                    isolation_meta = None
                    if isinstance(meta_payload, dict):
                        isolation_meta = meta_payload.get("isolation")
                    detail_kwargs: Dict[str, Any] = {
                        "phase": label,
                        "board": f"{cb.W}×{cb.H}",
                        "ok": ok,
                        "reason": reason or "",
                        "allow_discard": allow_discard,
                    }
                    if isinstance(meta_payload, dict) and meta_payload:
                        detail_kwargs.update(
                            {
                                "solved_via": meta_payload.get("solved_via"),
                                "placements": meta_payload.get("placed"),
                                "coverage_cells": meta_payload.get("cp_sat_coverage_cells"),
                                "guard_edge": meta_payload.get("cp_sat", {}).get("edge_guard")
                                if isinstance(meta_payload.get("cp_sat"), dict)
                                else None,
                                "guard_plus": meta_payload.get("cp_sat", {}).get("plus_guard")
                                if isinstance(meta_payload.get("cp_sat"), dict)
                                else None,
                            }
                        )
                    if isolation_meta and isinstance(isolation_meta, dict):
                        tail = isolation_meta.get("stderr_tail")
                        detail_kwargs.update(
                            {
                                "isolation_exitcode": isolation_meta.get("exitcode"),
                                "isolation_payload_received": isolation_meta.get("payload_received"),
                                "isolation_payload_kind": isolation_meta.get("payload_kind"),
                                "isolation_payload_type": isolation_meta.get("payload_raw_type"),
                                "isolation_placement_count": isolation_meta.get("placement_count"),
                                "isolation_stderr_tail": tail[-512:] if isinstance(tail, str) else None,
                            }
                        )
                    if detail_kwargs:
                        log_attempt_detail("Solver detail", **detail_kwargs)

                    if placed:
                        hint_cache[(cb.W, cb.H)] = list(placed)

                    used_tiles = len(placed) if placed else 0
                    used_area = (
                        sum(p.rect.w * p.rect.h for p in placed)
                        if placed
                        else used_area
                    )
                    coverage_pct = (
                        100.0 * used_area / float(board_area)
                        if board_area > 0 and used_area > 0
                        else 0.0
                    )

                    if used_tiles > 0:
                        _record_best(used_tiles, coverage_pct)
                        progress_boards.add(key)

                    if allow_discard and placed:
                        if best_tuple is None:
                            best_tuple = (cb, list(placed), coverage_pct, used_tiles)
                        else:
                            best_cb, _, best_cov, best_used = best_tuple
                            better = False
                            if coverage_pct > best_cov + 1e-9:
                                better = True
                            elif abs(coverage_pct - best_cov) <= 1e-9:
                                current_area = cb.W * cb.H
                                best_area = best_cb.W * best_cb.H
                                if prefer_large and current_area > best_area:
                                    better = True
                                elif not prefer_large and current_area < best_area:
                                    better = True
                                elif used_tiles > best_used:
                                    better = True
                            if better:
                                best_tuple = (cb, list(placed), coverage_pct, used_tiles)

                        if board_area > 0 and used_area >= board_area:
                            set_progress_pct(100.0)
                            return True, cb, placed, coverage_pct, used_tiles, None

                    elif ok and placed:
                        if area_cells == 0 or used_area >= area_cells:
                            set_progress_pct(100.0)
                            return True, cb, placed, coverage_pct, used_tiles, None

                    elapsed_now = time.time() - phase_start
                    remaining_after = seconds - elapsed_now
                    total = max(total, attempt_idx + len(queue))
                    retries = retry_counts.get(key, 0)
                    improved = False
                    if board_area > 0 and used_area > 0:
                        prev_best_cov = partial_best.get(key, 0.0)
                        if coverage_pct > prev_best_cov + 1e-9:
                            partial_best[key] = coverage_pct
                            partial_stalls[key] = 0
                            improved = True
                        else:
                            partial_stalls[key] = partial_stalls.get(key, 0) + 1
                    if (
                        continue_on_partial
                        and board_area > 0
                        and used_area > 0
                        and used_area < board_area
                        and remaining_after > 1.0
                    ):
                        stall_count = partial_stalls.get(key, 0)
                        if improved or stall_count < plateau_limit:
                            queue.insert(0, cb)
                            queue = sorted(queue, key=_sort_key)
                            total = max(total, attempt_idx + len(queue))
                            if improved:
                                detail = f"improved to {coverage_pct:.2f}%"
                            else:
                                detail = f"stalled at {coverage_pct:.2f}% (retry {stall_count}/{plateau_limit})"
                            set_message(
                                f"Phase {label} re-queue {cb.label} to chase full coverage – {detail}"
                            )
                            log_attempt_detail(
                                "Partial retry",
                                phase=label,
                                board=f"{cb.W}×{cb.H}",
                                coverage_pct=round(coverage_pct, 2),
                                stall_count=stall_count,
                                plateau_limit=plateau_limit,
                                improved=improved,
                            )
                            _update_phase_progress()
                            continue
                        base_candidates = [
                            cand for cand in base_candidates if (cand.W, cand.H) != key
                        ]
                        progress_boards.discard(key)
                        queue = [cand for cand in queue if (cand.W, cand.H) != key]
                        total = max(total, attempt_idx + len(queue))
                        set_message(
                            f"Phase {label} plateau on {cb.label} at {coverage_pct:.2f}% – skipping further retries"
                        )
                        log_attempt_detail(
                            "Partial plateau",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            coverage_pct=round(coverage_pct, 2),
                            stall_count=partial_stalls.get(key, 0),
                            plateau_limit=plateau_limit,
                        )
                    if (
                        not ok
                        and _should_retry_phase(last_reason)
                        and remaining_after >= max(5.0, min_retry_remaining)
                        and retries < max_retries
                        and (
                            not continue_on_partial
                            or used_area > 0
                            or key in progress_boards
                        )
                    ):
                        retry_counts[key] = retries + 1
                        queue.insert(0, cb)
                        queue = sorted(queue, key=_sort_key)
                        total = max(total, attempt_idx + len(queue))
                        set_message(
                            f"Phase {label} re-queue {cb.label} after '{last_reason}' ({retry_counts[key]}/{max_retries})"
                        )
                        log_attempt_detail(
                            "Attempt retry",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            reason=last_reason or "",
                            retries=retry_counts[key],
                            max_retries=max_retries,
                            remaining_after=round(remaining_after, 2),
                        )
                        blockers: List[str] = []
                        if allow_discard:
                            blockers.append("allow_discard=True")
                        if placed:
                            blockers.append(f"placements={len(placed)}")
                        if retries < max_retries:
                            blockers.append(f"retry_count={retries}/{max_retries}")
                        if not _is_crash_reason(last_reason):
                            blockers.append("reason_not_crash_like")
                        if blockers:
                            log_attempt_detail(
                                "Fallback skipped",
                                phase=label,
                                board=f"{cb.W}×{cb.H}",
                                blocked_by="; ".join(blockers),
                                reason=last_reason or "",
                                retries=retries,
                                max_retries=max_retries,
                            )
                        _update_phase_progress()
                        continue
                    elif (
                        not ok
                        and _should_retry_phase(last_reason)
                        and retries >= max_retries
                    ):
                        log_attempt_detail(
                            "Retry limit reached",
                            phase=label,
                            board=f"{cb.W}×{cb.H}",
                            reason=last_reason or "",
                            retries=retries,
                            max_retries=max_retries,
                        )

                    if require_full_duration:
                        wait_remaining = max(0.0, per_attempt - attempt_elapsed)
                        if wait_remaining > 0:
                            future_remaining = seconds - (time.time() - phase_start)
                            if future_remaining > 0:
                                time.sleep(min(wait_remaining, future_remaining))

                    _update_phase_progress()
            finally:
                progress_stop.set()
                _update_phase_progress()
                if ticker is not None:
                    ticker.join(timeout=0.2)

            if best_tuple and best_tuple[3] > 0:
                best_cb, best_placed, best_cov, best_used = best_tuple
                _record_best(best_used, best_cov)
                set_progress_pct(100.0)
                hint_cache[(best_cb.W, best_cb.H)] = list(best_placed)
                return True, best_cb, list(best_placed), best_cov, best_used, None

            set_progress_pct(100.0)
            fallback_reason = last_reason or "Exhausted all board candidates without a feasible layout"
            return False, None, [], 0.0, 0, fallback_reason

        success_meta: Optional[Dict[str, Any]] = None
        final_reason: Optional[str] = None
        final_strategy: Optional[str] = None
        final_placed: List[Placed] = []
        final_board: Optional[CandidateBoard] = None

        message_success = "This is the best that can be done."

        if area_sqft < base_area_sqft:
            decrement = grid_step if grid_step > 1 else 1
            max_side_small = max(6, base_side_cells - decrement)
            if max_side_small < sqrt_cells:
                max_side_small = sqrt_cells
            max_side_small = _align_down_to_multiple(max(max_side_small, 6), grid_step)
            if max_side_small < sqrt_cells:
                max_side_small = sqrt_cells
            candidates_A = [
                cb
                for cb in _square_candidates(
                    sqrt_cells,
                    max_side_small,
                    descending=True,
                    multiple_of=grid_step,
                )
                if cb.W * cb.H >= area_cells
            ]
            res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                "A", candidates_A, float(getattr(CFG, "TIME_A", 600.0)), False, prefer_large=True
            )
            if res_reason:
                final_reason = res_reason
                if res_reason == guard_block_reason:
                    guard_failure_reason = res_reason
                    set_status("Error")
                    set_message(res_reason)
                    return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
            if res_ok and res_board:
                final_strategy = "A"
                final_board = res_board
                final_placed = res_placed
                success_meta = {
                    "note": message_success,
                    "coverage_pct": round(res_cov, 2),
                    "best_used": res_used,
                }
            else:
                min_side_B = max(6, sqrt_cells - 6)
                min_side_B = max(6, _align_down_to_multiple(min_side_B, grid_step))
                widths = _descending_values(max_side_small, min_side_B, step=grid_step)
                heights = _descending_values(max_side_small, min_side_B, step=grid_step)
                candidates_B = _rectangular_candidates(
                    widths,
                    heights,
                    descending=True,
                    multiple_of=grid_step,
                )[:30]
                res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                    "B", candidates_B, float(getattr(CFG, "TIME_B", 600.0)), True, prefer_large=True
                )
                if res_reason:
                    final_reason = res_reason
                    if res_reason == guard_block_reason:
                        guard_failure_reason = res_reason
                        set_status("Error")
                        set_message(res_reason)
                        return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
                if res_ok and res_board:
                    final_strategy = "B"
                    final_board = res_board
                    final_placed = res_placed
                    success_meta = {
                        "note": message_success,
                        "coverage_pct": round(res_cov, 2),
                        "best_used": res_used,
                    }
        else:
            base_candidate = _phase_c_candidates(base_side_cells, grid_step=grid_step)[0]
            base_area_cells = base_candidate.W * base_candidate.H
            if area_cells > base_area_cells:
                log_attempt_detail(
                    "Phase C skipped",
                    reason="Demand exceeds base board",
                    demand_cells=area_cells,
                    board_cells=base_area_cells,
                    demand_sqft=round(area_sqft, 2),
                    board_sqft=round(_area_sqft(base_area_cells), 2),
                )
                set_message("Demand exceeds 10.0 × 10.0 ft grid; expanding search")
                res_ok = False
                res_board = None
                res_placed = []
                res_cov = 0.0
                res_used = 0
                res_reason = "Demand exceeds base Phase C board"
            else:
                res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                    "C",
                    [base_candidate],
                    float(getattr(CFG, "TIME_C", 300.0)),
                    False,
                    prefer_large=False,
                    continue_on_partial=True,
                    require_full_duration=True,
                )
            if res_reason:
                final_reason = res_reason
                if res_reason == guard_block_reason:
                    guard_failure_reason = res_reason
                    set_status("Error")
                    set_message(res_reason)
                    return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
            if res_ok and res_board:
                final_strategy = "C"
                final_board = res_board
                final_placed = res_placed
                success_meta = {
                    "note": message_success,
                    "coverage_pct": round(res_cov, 2),
                    "best_used": res_used,
                }
            else:
                shrink_floor = max(6, max_tile_side)
                shrink_floor = _align_up_to_multiple(shrink_floor, grid_step)
                if shrink_floor > base_side_cells:
                    shrink_floor = base_side_cells
                candidates_D = _phase_d_candidates(
                    shrink_floor=shrink_floor,
                    base_side=base_side_cells,
                    grid_step=grid_step,
                    area_cells=area_cells,
                )
                if not candidates_D:
                    candidates_D = [base_candidate]
                res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                    "D",
                    candidates_D,
                    float(getattr(CFG, "TIME_D", 300.0)),
                    True,
                    prefer_large=True,
                    continue_on_partial=True,
                    require_full_duration=True,
                )
                if res_reason:
                    final_reason = res_reason
                    if res_reason == guard_block_reason:
                        guard_failure_reason = res_reason
                        set_status("Error")
                        set_message(res_reason)
                        return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
                if res_ok and res_board:
                    final_strategy = "D"
                    final_board = res_board
                    final_placed = res_placed
                    success_meta = {
                        "note": message_success,
                        "coverage_pct": round(res_cov, 2),
                        "best_used": res_used,
                    }
                else:
                    expand_upper_raw = max(base_side_cells + 12, sqrt_cells)
                    expand_upper = _align_up_to_multiple(expand_upper_raw, grid_step)
                    min_expand = base_side_cells + (grid_step if grid_step > 1 else 1)
                    candidates_E = [
                        cb
                        for cb in _square_candidates(
                            min_expand,
                            expand_upper,
                            descending=False,
                            multiple_of=grid_step,
                        )
                        if cb.W * cb.H >= area_cells
                    ][:30]
                    res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                        "E", candidates_E, float(getattr(CFG, "TIME_E", 900.0)), False, prefer_large=False
                    )
                    if res_reason:
                        final_reason = res_reason
                        if res_reason == guard_block_reason:
                            guard_failure_reason = res_reason
                            set_status("Error")
                            set_message(res_reason)
                            return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
                    if res_ok and res_board:
                        final_strategy = "E"
                        final_board = res_board
                        final_placed = res_placed
                        success_meta = {
                            "note": message_success,
                            "coverage_pct": round(res_cov, 2),
                            "best_used": res_used,
                        }
                    else:
                        candidates_F = _square_candidates(
                            min_expand,
                            expand_upper,
                            descending=False,
                            multiple_of=grid_step,
                        )[:30]
                        res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                            "F", candidates_F, float(getattr(CFG, "TIME_F", 900.0)), True, prefer_large=False
                        )
                        if res_reason:
                            final_reason = res_reason
                            if res_reason == guard_block_reason:
                                guard_failure_reason = res_reason
                                set_status("Error")
                                set_message(res_reason)
                                return (False, [], 0.0, 0.0, "error", res_reason, {"reason": res_reason})
                        if res_ok and res_board:
                            final_strategy = "F"
                            final_board = res_board
                            final_placed = res_placed
                            success_meta = {
                                "note": message_success,
                                "coverage_pct": round(res_cov, 2),
                                "best_used": res_used,
                            }

        if final_strategy and final_board and not guard_failure_reason:
            elapsed = time.time() - t0
            set_status("Solved")
            set_elapsed(elapsed)
            set_message(message_success)
            if success_meta is None:
                success_meta = {}
            success_meta.update({
                "demand_count": demand_count,
                "placed_count": len(final_placed),
                "elapsed": elapsed,
                "W": final_board.W,
                "H": final_board.H,
                "W_ft": _fmt_ft(final_board.W),
                "H_ft": _fmt_ft(final_board.H),
            })
            return (
                True,
                final_placed,
                _fmt_ft(final_board.W),
                _fmt_ft(final_board.H),
                final_strategy,
                None,
                success_meta,
            )

        set_status("Error")
        failure_reason = guard_failure_reason or final_reason or "There is no solution"
        if failure_reason != guard_block_reason:
            try:
                guard_board = max(sqrt_cells, max(max_tile_side_raw, 1))
                guard_board = _align_up_to_multiple(guard_board, grid_step)
                guard_ok, _, guard_reason = try_pack_exact_cover(
                    guard_board,
                    guard_board,
                    bag_cells,
                    allow_discard=False,
                    max_seconds=2.0,
                    force_backtracking=False,
                )
                if (not guard_ok) and guard_reason == guard_block_reason:
                    failure_reason = guard_block_reason
            except Exception:
                pass
        set_message(failure_reason)
        return (False, [], 0.0, 0.0, "error", failure_reason, {"reason": failure_reason})

    except Exception as e:
        set_status("Error")
        reason = f"orchestrator exception: {type(e).__name__}: {e}"
        traceback.print_exc()
        return (False, [], 0.0, 0.0, "error", reason, {"trace": reason})
