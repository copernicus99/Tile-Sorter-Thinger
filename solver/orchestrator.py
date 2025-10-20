# Orchestrator: option-driven CP-SAT workflow (A–F sequence)
from __future__ import annotations

import os
import time
import traceback
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable

import multiprocessing as mp

from models import Rect, Placed, ft_to_cells
from tiles import parse_demand
from config import CFG, CELL
from progress import (
    set_phase, set_phase_total, set_attempt, set_grid, set_progress_pct,
    set_best_used, set_coverage_pct, set_elapsed, set_status, set_message
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

def _cp_worker(conn, W, H, bag, seconds, allow_discard):
    try:
        os.environ["ORTOOLS_CP_SAT_NUM_THREADS"] = "1"
        os.environ["NUM_CPUS"] = "1"
        ok, placed, reason = try_pack_exact_cover(
            W=W, H=H, multiset=bag, allow_discard=allow_discard, max_seconds=seconds
        )
        out = []
        for p in placed:
            out.append((p.x, p.y, (p.rect.w, p.rect.h, getattr(p.rect, "name", None))))
        conn.send((ok, out, reason))
    except Exception as e:
        try:
            conn.send((False, [], f"cp-sat exception: {type(e).__name__}: {e}"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _safe_parent_poll(parent, timeout):
    try:
        return parent.poll(timeout)
    except BrokenPipeError:
        return True


def _run_cp_sat_isolated(W: int, H: int, bag: Dict[Tuple[int, int], int],
                         seconds: float, allow_discard: bool):
    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_cp_worker, args=(child, W, H, bag, seconds, allow_discard))
    proc.daemon = True
    proc.start()
    try:
        child.close()
    except Exception:
        pass

    ok = False
    placed: List[Placed] = []
    reason = None
    got_message = False

    deadline = time.time() + max(1.0, float(seconds) + 1.0)
    while time.time() < deadline and proc.is_alive():
        if _safe_parent_poll(parent, 0.05):
            break
        time.sleep(0.05)

    try:
        if parent.poll(0):
            got_message = True
            ok_msg, placed_msg, reason_msg = parent.recv()
            ok = bool(ok_msg)
            out: List[Placed] = []
            for x, y, (w, h, nm) in placed_msg:
                out.append(Placed(x, y, Rect(w, h, nm)))
            placed = out
            reason = reason_msg
        else:
            reason = "Stopped before solution (timebox / crash?)"
    except (EOFError, BrokenPipeError):
        reason = "Subprocess ended early (pipe closed)"
    except Exception as e:
        reason = f"Parent recv exception: {type(e).__name__}: {e}"

    try:
        proc.join(timeout=0.2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.2)
    finally:
        try:
            parent.close()
        except Exception:
            pass

    if proc.exitcode == -1073741819:  # 0xC0000005
        ok = False
        if not got_message or not reason:
            reason = "Native solver crash (0xC0000005) – isolated; server is fine"

    return ok, placed, reason


# ---------- public entrypoint ----------

def solve_orchestrator(*args, **kwargs):
    """
    Returns: (ok, placed, W_ft, H_ft, strategy, reason, meta)
    Strategy values are the option letters A–F from the specification.
    """
    t0 = time.time()
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

        base_area_sqft = max(1.0, float(getattr(CFG, "BASE_GRID_AREA_SQFT", 1000.0)))
        base_side_cells = max(6, ft_to_cells(math.sqrt(base_area_sqft)))
        base_side_cells = _align_up_to_multiple(base_side_cells, grid_step)
        sqrt_cells = _ceil_sqrt_cells(area_cells)
        sqrt_cells = _align_up_to_multiple(sqrt_cells, grid_step)

        set_status("Solving")
        set_phase("S0")
        set_phase_total(1)
        set_attempt("Computing demand")
        set_grid("—")
        set_progress_pct(0.0)
        set_best_used(0)
        set_coverage_pct(0.0)

        best_used_tiles = 0
        best_cover_pct = 0.0

        def _record_best(used_tiles: int, coverage_pct: float) -> None:
            nonlocal best_used_tiles, best_cover_pct
            coverage_pct = max(0.0, float(coverage_pct))
            used_tiles = max(0, int(used_tiles))
            better = False
            if coverage_pct > best_cover_pct + 1e-9:
                better = True
            elif abs(coverage_pct - best_cover_pct) <= 1e-9 and used_tiles > best_used_tiles:
                better = True
            if better:
                best_cover_pct = coverage_pct
                best_used_tiles = used_tiles
            else:
                if used_tiles > best_used_tiles:
                    best_used_tiles = used_tiles
                if coverage_pct > best_cover_pct:
                    best_cover_pct = coverage_pct
            set_best_used(best_used_tiles)
            set_coverage_pct(best_cover_pct)

        def _run_phase(
            label: str,
            candidates: List[CandidateBoard],
            seconds: float,
            allow_discard: bool,
            *,
            prefer_large: bool,
        ) -> Tuple[bool, Optional[CandidateBoard], List[Placed], float, int, Optional[str]]:
            if seconds <= 0 or not candidates:
                set_phase(label)
                set_phase_total(int(max(0.0, seconds)))
                set_progress_pct(100.0 if not candidates else 0.0)
                reason = "No board candidates available" if not candidates else "Phase time budget is zero"
                return False, None, [], 0.0, 0, reason

            set_phase(label)
            set_phase_total(int(seconds))
            phase_start = time.time()
            total = len(candidates)
            best_tuple: Optional[Tuple[CandidateBoard, List[Placed], float, int]] = None
            last_reason: Optional[str] = None

            for idx, cb in enumerate(candidates, 1):
                elapsed_phase = time.time() - phase_start
                remaining = seconds - elapsed_phase
                if remaining <= 0:
                    break

                per_attempt = min(max(5.0, seconds / max(1, total)), remaining)
                set_attempt(cb.label)
                set_grid(cb.label)
                ok, placed, reason = _run_cp_sat_isolated(
                    W=cb.W, H=cb.H, bag=bag_cells,
                    seconds=per_attempt,
                    allow_discard=allow_discard
                )

                if reason:
                    last_reason = str(reason)

                coverage_pct = 0.0
                used_tiles = len(placed) if placed else 0
                if demand_count > 0:
                    coverage_pct = 100.0 * used_tiles / float(demand_count)
                elif used_tiles > 0:
                    coverage_pct = 100.0

                if ok and placed:
                    _record_best(used_tiles, coverage_pct)
                    if not allow_discard and used_tiles >= demand_count:
                        set_progress_pct(100.0)
                        return True, cb, placed, coverage_pct, used_tiles, None

                    if allow_discard:
                        if best_tuple is None:
                            best_tuple = (cb, placed, coverage_pct, used_tiles)
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
                                best_tuple = (cb, placed, coverage_pct, used_tiles)

                        if used_tiles >= demand_count and demand_count > 0:
                            set_progress_pct(100.0)
                            return True, cb, placed, coverage_pct, used_tiles, None

                set_progress_pct(min(100.0, 100.0 * idx / max(1, total)))

                if (time.time() - phase_start) >= seconds:
                    break

            if best_tuple and best_tuple[3] > 0:
                best_cb, best_placed, best_cov, best_used = best_tuple
                _record_best(best_used, best_cov)
                set_progress_pct(100.0)
                return True, best_cb, best_placed, best_cov, best_used, None

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
            base_candidate = CandidateBoard(
                base_side_cells,
                base_side_cells,
                f"{_fmt_ft(base_side_cells)} × {_fmt_ft(base_side_cells)} ft",
            )
            res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                "C",
                [base_candidate],
                float(getattr(CFG, "TIME_C", 300.0)),
                False,
                prefer_large=False,
            )
            if res_reason:
                final_reason = res_reason
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
                res_ok, res_board, res_placed, res_cov, res_used, res_reason = _run_phase(
                    "D",
                    [base_candidate],
                    float(getattr(CFG, "TIME_D", 300.0)),
                    True,
                    prefer_large=True,
                )
                if res_reason:
                    final_reason = res_reason
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
                        if res_ok and res_board:
                            final_strategy = "F"
                            final_board = res_board
                            final_placed = res_placed
                            success_meta = {
                                "note": message_success,
                                "coverage_pct": round(res_cov, 2),
                                "best_used": res_used,
                            }

        if final_strategy and final_board:
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
        failure_reason = final_reason or "There is no solution"
        set_message(failure_reason)
        return (False, [], 0.0, 0.0, "error", failure_reason, {"reason": failure_reason})

    except Exception as e:
        set_status("Error")
        reason = f"orchestrator exception: {type(e).__name__}: {e}"
        traceback.print_exc()
        return (False, [], 0.0, 0.0, "error", reason, {"trace": reason})
