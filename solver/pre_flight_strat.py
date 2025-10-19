# solver/pre_flight_strat.py  (Python 3.9 compatible)
"""
Light-weight, fast "S*" strategies that run before the heavier C/D/F
orchestration. Each probe is time-boxed and heavily guarded to avoid
building massive models. CP-SAT is executed via a child process to
isolate native crashes on Windows.
"""
import time
import math
from functools import reduce
from math import gcd
from typing import Dict, List, Tuple, Optional

from models import Meta, Rect, ft_to_cells, cells_to_ft
from tiles import NAME_TO_RECT_DIMS
from progress import _set_progress
from solver.constructive import quick_banded_fill_width10
from solver.cp_isolate import run_cp_sat_isolated
from config import CFG as DEFAULT_CFG


# ---------------- helpers ----------------

def _edge_label(cfg) -> str:
    return "off (test mode)" if getattr(cfg, "MAX_EDGE_FT", None) is None else f"{cfg.MAX_EDGE_FT:.1f} ft"


def _meta(start: float, cfg) -> Meta:
    # side-effect free meta; no file I/O here
    return Meta(template="", elapsed_sec=time.time() - start, edge_label=_edge_label(cfg))


def _fmt_grid(Wc: int, Hc: int) -> str:
    return f"{cells_to_ft(Wc):.1f} × {cells_to_ft(Hc):.1f} ft"


def _set_grid_fields(Wc: int, Hc: int, attempt_label: Optional[str] = None) -> None:
    g = _fmt_grid(Wc, Hc)
    _set_progress(
        attempt=attempt_label if attempt_label is not None else g,
        grid=g,
        grid_ft=g,
        grid_w_ft=round(cells_to_ft(Wc), 6),
        grid_h_ft=round(cells_to_ft(Hc), 6),
        grid_w_cells=int(Wc),
        grid_h_cells=int(Hc),
    )


def _demand_to_bag(demand: Dict[str, int]) -> List[Rect]:
    bag: List[Rect] = []
    for name, count in demand.items():
        w, h = NAME_TO_RECT_DIMS[name]
        bag.extend([Rect(w, h, name)] * int(count))
    return bag


def _est_placements(Wc: int, Hc: int, bag: List[Rect], stride: int = 1) -> int:
    """
    Rough estimate: count of candidate placements with optional stride thinning.
    """
    total = 0
    for r in bag:
        # try both orientations if rectangle is not square
        for (w, h) in ((r.w, r.h), (r.h, r.w)) if r.w != r.h else ((r.w, r.h),):
            if Wc >= w and Hc >= h:
                nx = (Wc - w) // max(1, stride) + 1
                ny = (Hc - h) // max(1, stride) + 1
                total += max(0, nx) * max(0, ny)
    return total


def _quick_try(W_ft: float,
               H_ft: float,
               *,
               allow_discard: bool,
               sec: int,
               label: str,
               strat: str,
               bag: List[Rect],
               start: float,
               cfg) -> Optional[Tuple[bool, List, int, int, str, Optional[str], Meta]]:
    """
    Uniform wrapper to run a short CP-SAT probe in an *isolated* child process.
    Returns the standard solver tuple on success; otherwise None (meaning:
    move to the next strategy).
    """
    Wc, Hc = ft_to_cells(W_ft), ft_to_cells(H_ft)
    if Wc <= 0 or Hc <= 0:
        return None

    try:
        _set_progress(strategy=strat, step_started=time.time(), time_limit=sec)
        _set_grid_fields(Wc, Hc, attempt_label=label)

        ok, placed, reason, crash = run_cp_sat_isolated(
            Wc, Hc, bag, allow_discard, sec
        )

        if crash:
            # surface the crash-like note to the overlay, then continue
            _set_progress(reason=f"{strat} skipped ({crash})", status="solving")

        if ok:
            return True, placed, Wc, Hc, strat, None, _meta(start, cfg)
        return None
    except MemoryError:
        _set_progress(reason=f"{strat} skipped (memory)", status="solving")
        return None
    except Exception as e:  # pragma: no cover (diagnostic)
        _set_progress(reason=f"{strat} skipped: {e}", status="solving")
        return None


# ---------------- S-strategies ----------------

def run_pre_flight(demand: Dict[str, int], start: float, cfg=DEFAULT_CFG):
    """
    Try a short sequence of safe, targeted strategies before the heavier
    C/D/F flow.  Returns a standard solver tuple on success, or None to
    continue with main orchestration.
    """
    # S0 — constructive 10×H (banded), near-instant when layout exists
    _set_progress(
        status="solving",
        strategy="S0",
        step_started=start,
        time_limit=getattr(cfg, "TIME_S0", 30),
        attempt="10 × H (banded)",
        grid="10.0 × ? ft",
        grid_ft="10.0 × ? ft",
        grid_w_ft=10.0,
        grid_h_ft=None,
        grid_w_cells=ft_to_cells(10.0),
        grid_h_cells=None,
    )
    quick, Wc0, Hc0 = quick_banded_fill_width10(demand)
    if quick:
        _set_grid_fields(Wc0, Hc0, attempt_label="10 × H (banded)")
        return True, quick, Wc0, Hc0, "S0", None, _meta(start, cfg)

    # Shared context
    bag = _demand_to_bag(demand)
    total_cells = sum(r.w * r.h for r in bag)  # area in *cells*
    TEN = ft_to_cells(10.0)

    # S1 — 10×H from area (NO discards)  [guards + correct units]
    if TEN:
        for Hc in (math.ceil(total_cells / TEN), math.floor(total_cells / TEN)):
            if not Hc or Hc <= 0:
                continue
            H_ft = cells_to_ft(Hc)

            # Guard 1: avoid skyscraper models
            if H_ft > getattr(cfg, "S1_MAX_HEIGHT_FT", 16.0):
                _set_progress(reason=f"S1 skipped: H={H_ft:.1f}ft exceeds cap", status="solving")
                continue

            # Guard 2: placement estimate
            est = _est_placements(TEN, Hc, bag, stride=1)
            if est > getattr(cfg, "S_MAX_EST_PLACEMENTS", 150000):
                _set_progress(reason=f"S1 skipped: est placements {est:,} > cap", status="solving")
                continue

            r = _quick_try(
                10.0, H_ft,
                allow_discard=False,
                sec=getattr(cfg, "TIME_S1", 45),
                label="10 × H (area)",
                strat="S1",
                bag=bag,
                start=start,
                cfg=cfg,
            )
            if r:
                return r

    # S3 — gcd/parity snap height (NO discards)  [guards + correct units]
    if bag and TEN:
        g_h = reduce(gcd, [t.h for t in bag]) or 1
        Hc_exact = total_cells // TEN
        if Hc_exact * TEN == total_cells:
            Hc_snap = max(1, (Hc_exact // g_h) * g_h)
            H_ft = cells_to_ft(Hc_snap)

            if H_ft > getattr(cfg, "S1_MAX_HEIGHT_FT", 16.0):
                _set_progress(reason=f"S3 skipped: H={H_ft:.1f}ft exceeds cap", status="solving")
            else:
                est = _est_placements(TEN, Hc_snap, bag, stride=1)
                if est > getattr(cfg, "S_MAX_EST_PLACEMENTS", 150000):
                    _set_progress(reason=f"S3 skipped: est placements {est:,} > cap", status="solving")
                else:
                    r = _quick_try(
                        10.0, H_ft,
                        allow_discard=False,
                        sec=getattr(cfg, "TIME_S3", 30),
                        label="10 × H (gcd-snap)",
                        strat="S3",
                        bag=bag,
                        start=start,
                        cfg=cfg,
                    )
                    if r:
                        return r

    # S2 — targeted rectangles (discards allowed)
    for wf, hf in [(10.0, 12.0), (12.0, 10.0), (11.0, 12.0), (12.0, 11.0)]:
        r = _quick_try(
            wf, hf,
            allow_discard=True,
            sec=getattr(cfg, "TIME_S2", 60),
            label=f"{wf:.1f} × {hf:.1f} ft (targeted)",
            strat="S2",
            bag=bag,
            start=start,
            cfg=cfg,
        )
        if r:
            return r

    # S4 — just-over-area rectangles (discards allowed)
    for wf, hf in [(10.0, 13.0), (13.0, 10.0)]:
        r = _quick_try(
            wf, hf,
            allow_discard=True,
            sec=getattr(cfg, "TIME_S4", 45),
            label=f"{wf:.1f} × {hf:.1f} ft (slack)",
            strat="S4",
            bag=bag,
            start=start,
            cfg=cfg,
        )
        if r:
            return r

    # S5 — near-square with mild slack (discards allowed)
    root = math.sqrt(float(total_cells))
    for pad in (1.0, 1.5):
        Wf, Hf = round(root + pad, 1), round(root - pad, 1)
        r = _quick_try(
            Wf, Hf,
            allow_discard=True,
            sec=getattr(cfg, "TIME_S5", 45),
            label=f"{Wf:.1f} × {Hf:.1f} ft (near-square slack)",
            strat="S5",
            bag=bag,
            start=start,
            cfg=cfg,
        )
        if r:
            return r

    # S6 — skinny/tall probes (NO discards)
    for w in (8.0, 9.0, 12.0):
        if w <= 0:
            continue
        Hc = math.ceil(total_cells / ft_to_cells(w))
        H_ft = cells_to_ft(Hc)
        r = _quick_try(
            w, H_ft,
            allow_discard=False,
            sec=getattr(cfg, "TIME_S6", 30),
            label=f"{w:.1f} × H (skinny)",
            strat="S6",
            bag=bag,
            start=start,
            cfg=cfg,
        )
        if r:
            return r

    # No pre-flight hit → let the main orchestrator proceed.
    return None
