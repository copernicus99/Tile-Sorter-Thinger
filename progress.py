from __future__ import annotations

import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------
# Thread-safe global progress state
# ------------------------------

PROGRESS_LOCK = threading.Lock()


# Cache the last non-empty values surfaced to the UI so the overlay keeps
# showing useful context even if the underlying progress fields are cleared
# temporarily (for example, during reset).
DISPLAY_CACHE = {
    "phase": "",
    "phase_total": "",
    "attempt": "",
    "grid": "",
}


def _init_logger() -> logging.Logger:
    logger = logging.getLogger("solver.attempt_log")
    if logger.handlers:
        return logger

    log_path = Path(__file__).resolve().parent / "logs" / "solver_attempts.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception:
        # If the logger cannot be initialised we silently continue; progress
        # tracking should not break the solver.
        logger.handlers.clear()
    return logger


ATTEMPT_LOGGER = _init_logger()


def _log_enabled() -> bool:
    return bool(ATTEMPT_LOGGER.handlers)


def _fmt_seconds(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        return f"{float(seconds):.2f}s"
    except Exception:
        return None


def _emit_log(event: str, **fields: Any) -> None:
    if not _log_enabled():
        return
    extras = [
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None and value != ""
    ]
    try:
        if extras:
            ATTEMPT_LOGGER.info("%s | %s", event, " ".join(extras))
        else:
            ATTEMPT_LOGGER.info("%s", event)
    except Exception:
        # Logging failures must never bubble back to callers.
        pass


LOG_STATE: Dict[str, Any] = {
    "run_start": None,
    "phase": "",
    "phase_start": None,
    "attempt": "",
    "attempt_start": None,
    "grid": "",
}


def _finalize_attempt_locked(now: Optional[float] = None, *, reason: Optional[str] = None) -> None:
    attempt = LOG_STATE.get("attempt")
    if not attempt:
        return
    if now is None:
        now = _now()
    start = LOG_STATE.get("attempt_start")
    duration = None
    if isinstance(start, (int, float)):
        duration = max(0.0, float(now) - float(start))
    _emit_log(
        "Attempt finished",
        phase=LOG_STATE.get("phase") or "",
        attempt=attempt,
        grid=LOG_STATE.get("grid") or "",
        duration=_fmt_seconds(duration),
        reason=reason,
    )
    LOG_STATE["attempt"] = ""
    LOG_STATE["attempt_start"] = None


def _log_attempt_transition_locked(new_attempt: str) -> None:
    prev_attempt = LOG_STATE.get("attempt") or ""
    if new_attempt == prev_attempt:
        return
    now = _now()
    if prev_attempt:
        _finalize_attempt_locked(now, reason="switch")
    LOG_STATE["attempt"] = new_attempt
    if new_attempt:
        LOG_STATE["attempt_start"] = now
        _emit_log(
            "Attempt started",
            phase=LOG_STATE.get("phase") or "",
            attempt=new_attempt,
            grid=(LOG_STATE.get("grid") or new_attempt),
        )
    else:
        LOG_STATE["attempt_start"] = None


def _log_phase_transition_locked(new_phase: str) -> None:
    prev_phase = LOG_STATE.get("phase") or ""
    if new_phase == prev_phase:
        return
    now = _now()
    if LOG_STATE.get("attempt"):
        _finalize_attempt_locked(now, reason="phase_change")
    if prev_phase and LOG_STATE.get("phase_start"):
        duration = max(0.0, now - float(LOG_STATE["phase_start"]))
        _emit_log(
            "Phase finished",
            phase=prev_phase,
            duration=_fmt_seconds(duration),
        )
    LOG_STATE["phase"] = new_phase
    LOG_STATE["phase_start"] = now
    if new_phase:
        _emit_log("Phase started", phase=new_phase)


def _update_grid_locked(new_grid: str) -> None:
    prev_grid = LOG_STATE.get("grid") or ""
    if new_grid == prev_grid:
        return
    LOG_STATE["grid"] = new_grid
    if new_grid:
        _emit_log(
            "Grid updated",
            phase=LOG_STATE.get("phase") or "",
            attempt=LOG_STATE.get("attempt") or new_grid,
            grid=new_grid,
        )

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
    "done": False,             # run completed
    "ok": None,               # success flag if known
    "result_url": "",         # optional navigation target
    # compatibility / counters
    "demand_count": 0,
}

# ------------------------------
# Helpers
# ------------------------------


def _remember_display(name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text:
        DISPLAY_CACHE[name] = text


def _clear_display_cache() -> None:
    for key in DISPLAY_CACHE.keys():
        DISPLAY_CACHE[key] = ""


def _display_value(name: str) -> str:
    raw = PROGRESS.get(name, "")
    if raw not in (None, ""):
        return str(raw)
    cached = DISPLAY_CACHE.get(name, "")
    return str(cached) if cached not in (None, "") else ""

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
        now = _now()
        _finalize_attempt_locked(now, reason="reset")
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
            "done": False,
            "ok": None,
            "result_url": "",
            "demand_count": 0,
        })
        LOG_STATE.update({
            "phase": "",
            "phase_start": None,
            "attempt": "",
            "attempt_start": None,
            "grid": "",
            "run_start": None,
        })
        _clear_display_cache()
        _emit_log("Progress reset")

def start_timer() -> None:
    with PROGRESS_LOCK:
        now = _now()
        PROGRESS["elapsed_start"] = now
        PROGRESS["elapsed"] = 0.0
        LOG_STATE["run_start"] = now
        _emit_log("Run timer started")

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
        phase_str = "" if v is None else str(v)
        PROGRESS["phase"] = phase_str
        if phase_str:
            _remember_display("phase", phase_str)
        _log_phase_transition_locked(phase_str)

def set_phase_total(v: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["phase_total"] = "" if v is None else str(v)
        if PROGRESS["phase_total"]:
            _remember_display("phase_total", PROGRESS["phase_total"])

def set_attempt(v: Any) -> None:
    with PROGRESS_LOCK:
        attempt_str = "" if v is None else str(v)
        PROGRESS["attempt"] = attempt_str
        if attempt_str:
            _remember_display("attempt", attempt_str)
        _log_attempt_transition_locked(attempt_str)

def set_grid(v: Any) -> None:
    with PROGRESS_LOCK:
        grid_str = "" if v is None else str(v)
        PROGRESS["grid"] = grid_str
        if grid_str:
            _remember_display("grid", grid_str)
        _update_grid_locked(grid_str)

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

def set_result_url(url: Any) -> None:
    with PROGRESS_LOCK:
        PROGRESS["result_url"] = "" if url is None else str(url)

def set_done(ok: Any = None, *, reason: Any = None, message: Any = None) -> None:
    """Mark the run complete, tolerating legacy arguments.

    Historically ``set_done`` accepted a truthy/falsey positional flag and an
    optional ``reason`` keyword.  Recent callers in :mod:`app` still pass those
    arguments, so we accept them here instead of raising ``TypeError``.  The
    boolean flag controls the final status when provided; otherwise we leave the
    status unchanged (defaulting to ``"Solved"`` for backwards-compatibility).
    Any supplied ``reason``/``message`` is surfaced via the ``message`` field.
    """

    final_status: Optional[str] = None
    ok_flag: Optional[bool] = None
    if ok is not None:
        try:
            ok_flag = bool(ok)
            final_status = "Solved" if ok_flag else "Error"
        except Exception:
            final_status = None

    final_message = message if message is not None else reason

    with PROGRESS_LOCK:
        _touch_elapsed_locked()
        now = _now()
        if final_status is not None:
            PROGRESS["status"] = final_status
            if ok_flag is None:
                ok_flag = final_status == "Solved"
        elif PROGRESS.get("status") in ("", "Idle", None):
            PROGRESS["status"] = "Solved"
            if ok_flag is None:
                ok_flag = True
        PROGRESS["percent"] = 100.0
        if final_message is not None:
            PROGRESS["message"] = str(final_message)
        PROGRESS["done"] = True
        if ok_flag is not None:
            PROGRESS["ok"] = ok_flag
        _finalize_attempt_locked(now, reason="run_complete")
        run_start = LOG_STATE.get("run_start")
        if isinstance(run_start, (int, float)):
            total = max(0.0, now - float(run_start))
        else:
            total = None
        LOG_STATE.update({
            "run_start": None,
            "phase_start": None,
            "phase": PROGRESS.get("phase", ""),
            "grid": PROGRESS.get("grid", ""),
        })
        coverage_pct = PROGRESS.get("coverage_pct")
        coverage_str = None
        if coverage_pct is not None:
            try:
                coverage_str = f"{float(coverage_pct):.2f}%"
            except Exception:
                coverage_str = str(coverage_pct)
        _emit_log(
            "Run finished",
            status=PROGRESS.get("status"),
            ok=PROGRESS.get("ok"),
            duration=_fmt_seconds(total),
            best_used=PROGRESS.get("best_used"),
            coverage=coverage_str,
            message=PROGRESS.get("message"),
        )

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
            "phase_display": _display_value("phase"),
            "phase_total_display": _display_value("phase_total"),
            "attempt": PROGRESS["attempt"],
            "attempt_display": _display_value("attempt"),
            "grid": PROGRESS["grid"],
            "grid_display": _display_value("grid"),
            "percent": PROGRESS["percent"],
            "best_used": PROGRESS["best_used"],
            "coverage_pct": PROGRESS["coverage_pct"],
            "elapsed": PROGRESS["elapsed"],
            "elapsed_str": _fmt_elapsed(PROGRESS["elapsed"]),
            "message": PROGRESS["message"],
            "done": PROGRESS["done"],
            "ok": PROGRESS["ok"],
            "result_url": PROGRESS["result_url"],
            "demand_count": PROGRESS["demand_count"],
        }

def as_json() -> Dict[str, Any]:
    # Alias used by /progress3
    return snapshot()
