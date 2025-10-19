# Orchestrator: CP-SAT first, then guaranteed greedy fallback
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import multiprocessing as mp

from models import Rect, Placed, ft_to_cells
from tiles import parse_demand
from config import CFG
from progress import (
    set_phase, set_phase_total, set_attempt, set_grid, set_progress_pct,
    set_best_used, set_coverage_pct, set_elapsed, set_status
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
    return round(cells / 2.0, 1)  # 2 cells/ft -> 0.5 ft per cell


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
        if isinstance(parsed, tuple) and len(parsed) == 2:
            ok, bag_ft = parsed
            return bag_ft if ok else {}
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


def _make_candidate_list(area_cells: int) -> List[CandidateBoard]:
    out: List[CandidateBoard] = []
    side = max(6, int(round(area_cells ** 0.5)))
    candidates = set()

    for d in range(-2, 3):
        s = max(6, side + d)
        candidates.add((s, s))

    for w in range(side - 4, side + 5):
        for h in (side - 2, side + 2):
            if w >= 6 and h >= 6:
                candidates.add((w, h))

    for (Wc, Hc) in sorted(candidates):
        out.append(CandidateBoard(Wc, Hc, f"{_fmt_ft(Wc)} × {_fmt_ft(Hc)} ft"))
    return out


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


# ---------- Guaranteed greedy fallback ----------

def _greedy_expand(bag_cells: Dict[Tuple[int, int], int]) -> Tuple[int, int, List[Placed]]:
    """
    Always returns a layout by packing left->right and expanding the board.
    No overlaps; no rotation; keeps given orientations.
    """
    # expand the multiset into a list of tiles, largest first for nicer rows
    tiles: List[Tuple[int, int]] = []
    for (w, h), c in bag_cells.items():
        tiles.extend([(w, h)] * int(c))
    tiles.sort(key=lambda wh: (wh[1], wh[0]), reverse=True)  # by height, then width

    if not tiles:
        return 0, 0, []

    W = max(w for w, _ in tiles)
    H = 0
    x = 0
    y = 0
    row_h = 0
    placed: List[Placed] = []

    for (w, h) in tiles:
        # wrap row if needed
        if x + w > W and x > 0:
            y += row_h
            x = 0
            row_h = 0
        # expand height if this tile doesn't fit vertically
        if y + h > H:
            H = y + h
        # we also expand width if the very first tile in a row is wider than current W
        if x == 0 and w > W:
            W = w
        # place tile
        placed.append(Placed(x, y, Rect(w, h, f"{w}x{h}")))
        x += w
        row_h = max(row_h, h)
        # if next tile would overflow width, the next loop will wrap automatically

    return W, H, placed


# ---------- public entrypoint ----------

def solve_orchestrator(*args, **kwargs):
    """
    Returns: (ok, placed, W_ft, H_ft, strategy, reason, meta)
    Strategy values:
      - "F"   : CP-SAT phase
      - "G"   : Greedy fallback (guaranteed)
      - "error": failure before placement
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
        demand_count = sum(bag_cells.values())
        area_cells = sum((w * h) * c for (w, h), c in bag_cells.items())

        set_phase("S0")
        set_phase_total(int(getattr(CFG, "TIME_S0", 30)))
        set_attempt(f"{_fmt_ft(0)} × {_fmt_ft(0)} ft (area)")
        set_grid("—")
        set_progress_pct(0.0)
        set_status("Solving")

        boards = _make_candidate_list(area_cells)
        if not boards:
            side = max(6, int(round((area_cells) ** 0.5)))
            boards = [
                CandidateBoard(side, side, f"{_fmt_ft(side)} × {_fmt_ft(side)} ft"),
                CandidateBoard(side + 2, side + 2, f"{_fmt_ft(side+2)} × {_fmt_ft(side+2)} ft"),
            ]

        timebox_F = float(getattr(CFG, "TIME_F", 900))
        per_board = max(10.0, min(180.0, timebox_F / max(1, len(boards))))
        set_phase("F")
        set_phase_total(int(timebox_F))

        best_used_pct = 0.0
        best_used_tiles = 0
        best_solution: Optional[Tuple[int, int, List[Placed]]] = None

        for idx, cb in enumerate(boards, 1):
            set_attempt(cb.label)
            set_grid(cb.label)

            ok1, placed1, _ = _run_cp_sat_isolated(
                W=cb.W, H=cb.H, bag=bag_cells,
                seconds=min(15.0, per_board * 0.25),
                allow_discard=False
            )
            if ok1:
                used = len(placed1)
                best_used_tiles = used
                best_used_pct = 100.0 * used / max(1, demand_count)
                best_solution = (cb.W, cb.H, placed1)
                set_best_used(best_used_tiles)
                set_coverage_pct(best_used_pct)
                break

            ok2, placed2, _ = _run_cp_sat_isolated(
                W=cb.W, H=cb.H, bag=bag_cells,
                seconds=per_board,
                allow_discard=True
            )
            if ok2 and placed2:
                used = len(placed2)
                used_pct = 100.0 * used / max(1, demand_count)
                if used_pct > best_used_pct + 1e-9:
                    best_used_pct = used_pct
                    best_used_tiles = used
                    best_solution = (cb.W, cb.H, placed2)

            set_best_used(best_used_tiles)
            set_coverage_pct(best_used_pct)
            set_progress_pct(min(100.0, 100.0 * idx / max(1, len(boards))))

            if (time.time() - t0) >= timebox_F:
                break

        # If CP-SAT found something, return it
        if best_solution:
            Wc, Hc, placed = best_solution
            meta = {
                "demand_count": demand_count,
                "best_used": best_used_tiles,
                "coverage_pct": round(best_used_pct, 2),
                "elapsed": time.time() - t0,
            }
            set_status("Solved")
            set_elapsed(meta["elapsed"])
            return (True, placed, _fmt_ft(Wc), _fmt_ft(Hc), "F", None, meta)

        # ---------- Greedy fallback (guaranteed) ----------
        set_phase("G")
        set_phase_total(1)
        set_attempt("Greedy")
        Wc, Hc, placed = _greedy_expand(bag_cells)
        meta = {
            "demand_count": demand_count,
            "best_used": len(placed),
            "coverage_pct": round(100.0 * len(placed) / max(1, demand_count), 2),
            "elapsed": time.time() - t0,
        }
        set_best_used(len(placed))
        set_coverage_pct(meta["coverage_pct"])
        set_grid(f"{_fmt_ft(Wc)} × {_fmt_ft(Hc)} ft")
        set_progress_pct(100.0)
        set_status("Solved")
        set_elapsed(meta["elapsed"])
        return (True, placed, _fmt_ft(Wc), _fmt_ft(Hc), "G", None, meta)

    except Exception as e:
        set_status("Error")
        reason = f"orchestrator exception: {type(e).__name__}: {e}"
        traceback.print_exc()
        return (False, [], 0.0, 0.0, "error", reason, {"trace": reason})
