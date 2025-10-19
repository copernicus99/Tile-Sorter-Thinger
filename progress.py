from __future__ import annotations

import time
import threading
from typing import Any, Dict

# ------------------------------
# Thread-safe global progress state
# ------------------------------

PROGRESS_LOCK = threading.Lock()

# Single source of truth for the modal
PROGRESS: Dict[str, Any] = {
    "status": "Idle",          # Idle | Solving | Solved | Error
    "phase": "",               # e.g. S0 | F | G
    "phase_total": "",         # total for the phase, string or number
    "attempt": "",             # e.g. "21.5 × 22.5 ft"
    "grid": "",                # e.g. "21.5 × 22.5 ft"
    "percent": 0.0,            # 0..100 float
    "best_used": 0,            # tiles placed best-so-far
    "coverage_pct": 0.0,       # % of demand covered
    "elapsed_start": None,     # t0 (float) when solving started
    "elapsed": 0.0,            # seconds snapshot
    "message": "",             # optional note
    # compatibility / counters
    "demand_count": 0,
}

# ------------------------------
# Helpers
# ------------------------------

def _now() -> float:
    return time.time()

def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"

def reset() -> None:
    with PROGRESS_LOCK:
        PROGRESS.update({
            "status": "Idle",
            "phase": "",
            "phase_total": "",
            "attempt": "",
            "grid": "",
            "percent": 0.0,
            "best_used": 0,
            "coverage_pct": 0.0,
            "elapsed_start": None,
            "elapsed": 0.0,
            "message": "",
            "demand_count": 0,
        })

def start_timer() -> None:
    with PROGRESS_LOCK:
        PROGRESS["elapsed_start"] = _now()
        PROGRESS["elapsed"] = 0.0

def _touch_elapsed_locked() -> None:
    t0 = PROGRESS.get("elapsed_start")
    if t0 is not None:
        PROGRESS["elapsed"] = _now() - float(t0)

# ------------------------------
# Setters (tolerant)
# ------------------------------

def set_status(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["status"] = str(v)

def set_phase(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["phase"] = "" if v is None else str(v)

def set_phase_total(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["phase_total"] = "" if v is None else str(v)

def set_attempt(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["attempt"] = "" if v is None else str(v)

def set_grid(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["grid"] = "" if v is None else str(v)

def set_progress_pct(pct: Any) -> None:
    try:
        f = float(pct)
    except Exception:
        f = 0.0
    f = max(0.0, min(100.0, f))
    with PROGRESS_LOCK:
        PROGRESS["percent"] = f
        _touch_elapsed_locked()

def set_best_used(n: Any) -> None:
    try:
        i = int(n)
    except Exception:
        i = 0
    with PROGRESS_LOCK:
        PROGRESS["best_used"] = max(0, i)

def set_coverage_pct(pct: Any) -> None:
    try:
        f = float(pct)
    except Exception:
        f = 0.0
    f = max(0.0, min(100.0, f))
    with PROGRESS_LOCK:
        PROGRESS["coverage_pct"] = f

def set_elapsed(seconds: Any) -> None:
    try:
        f = float(seconds)
    except Exception:
        f = 0.0
    with PROGRESS_LOCK:
        PROGRESS["elapsed"] = max(0.0, f)

def set_message(msg: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["message"] = "" if msg is None else str(msg)

def set_done() -> None:
    with PROGRESS_LOCK:
        _touch_elapsed_locked()
        PROGRESS["status"] = "Solved"
        PROGRESS["percent"] = 100.0

# ------------------------------
# Backward-compat shims
# ------------------------------

def set_demand_count(n: Any) -> None:
    try:
        i = int(n)
    except Exception:
        i = 0
    with PROGRESS_LOCK:
        PROGRESS["demand_count"] = max(0, i)

def set_attempt_wh(w: Any, h: Any) -> None:
    try:
        wf = float(w)
        hf = float(h)
        s = f"{wf:g} × {hf:g} ft"
    except Exception:
        s = ""
    set_attempt(s)

def set_grid_wh(w: Any, h: Any) -> None:
    try:
        wf = float(w)
        hf = float(h)
        s = f"{wf:g} × {hf:g} ft"
    except Exception:
        s = ""
    set_grid(s)

# ALIAS: some older code imports `_start_ticker` — map it to start_timer
def _start_ticker() -> None:
    start_timer()

# ALIAS: tolerate older `_set_progress(...)` calls.
# Accept either a single percentage or keyword updates.
def _set_progress(value: Any = None, **kw: Any) -> None:
    if value is not None:
        set_progress_pct(value)
    # optional bulk updates via keywords
    m = {
        "phase": set_phase,
        "phase_total": set_phase_total,
        "attempt": set_attempt,
        "grid": set_grid,
        "percent": set_progress_pct,
        "best_used": set_best_used,
        "coverage_pct": set_coverage_pct,
        "elapsed": set_elapsed,
        "status": set_status,
        "message": set_message,
    }
    for k, v in kw.items():
        fn = m.get(k)
        if fn:
            fn(v)

# ------------------------------
# Snapshots for the UI
# ------------------------------

def snapshot() -> Dict[str, Any]:
    with PROGRESS_LOCK:
        _touch_elapsed_locked()
        return {
            "status": PROGRESS["status"],
            "phase": PROGRESS["phase"],
            "phase_total": PROGRESS["phase_total"],
            "attempt": PROGRESS["attempt"],
            "grid": PROGRESS["grid"],
            "percent": PROGRESS["percent"],
            "best_used": PROGRESS["best_used"],
            "coverage_pct": PROGRESS["coverage_pct"],
            "elapsed": PROGRESS["elapsed"],
            "elapsed_str": _fmt_elapsed(PROGRESS["elapsed"]),
            "message": PROGRESS["message"],
            "demand_count": PROGRESS["demand_count"],
        }

def as_json() -> Dict[str, Any]:
    # Alias used by /progress3
    return snapshot()
