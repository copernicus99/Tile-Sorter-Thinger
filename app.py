# app.py — phase numbers visible immediately; progress no-cache
from __future__ import annotations
import os
import time
import threading
from typing import Any, Dict, Iterable, List, Tuple, Optional

from flask import Flask, request, render_template, send_from_directory, jsonify, url_for

from solver.orchestrator import solve_orchestrator
from tiles import parse_demand, fmt_decoded_items
from config import CFG
from io_files import write_coords, write_layout_view_html
from render import render_result
from models import Placed

from progress import (
    reset as progress_reset,
    as_json as progress_json,
    _start_ticker as progress_start,
    set_status, set_phase, set_phase_total, set_attempt, set_grid,
    set_elapsed, set_progress_pct, set_coverage_pct,
    set_best_used, set_demand_count, set_done, set_result_url,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_output_paths(configured: str, fallback: str) -> Tuple[str, str, str]:
    name = (configured or "").strip() or fallback
    if os.path.isabs(name):
        full_path = name
    else:
        full_path = os.path.abspath(os.path.join(BASE_DIR, name))
    directory = os.path.dirname(full_path) or BASE_DIR
    filename = os.path.basename(full_path) or fallback
    return full_path, directory, filename


_COORDS_FULL_PATH, COORDS_DIR, COORDS_FILENAME = _resolve_output_paths(
    CFG.COORDS_OUT, "coords.txt"
)
_LAYOUT_FULL_PATH, LAYOUT_DIR, LAYOUT_FILENAME = _resolve_output_paths(
    CFG.LAYOUT_HTML, "layout_view.html"
)

LAST_RESULT: Dict[str, Any] = {
    "ok": False,
    "strategy": "error",
    "W_ft": 0.0,
    "H_ft": 0.0,
    "W": 0,
    "H": 0,
    "edge_label": "None",
    "placed_count": 0,
    "demand_count": 0,
    "elapsed_str": "0s",
    "svg": "",
    "demand_items": [],
    "coords_filename": COORDS_FILENAME,
    "layout_filename": LAYOUT_FILENAME,
}

app = Flask(__name__, static_folder=".", template_folder="templates")


@app.after_request
def _no_cache_progress(resp):
    try:
        if request.path == "/progress3":
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "tile_selection_form.html")


@app.route("/result/latest")
def result_latest():
    return render_template("result.html", **LAST_RESULT)


@app.route("/styles.css")
def styles_css():
    return send_from_directory(BASE_DIR, "styles.css")


@app.route("/styles_add_rcgrid.css")
def styles_add_rcgrid_css():
    return send_from_directory(BASE_DIR, "styles_add_rcgrid.css")


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 1:
        return "0s"
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"{s}s"
    h, m = divmod(m, 60)
    if h == 0:
        return f"{m}m {s}s"
    return f"{h}h {m}m {s}s"


def _merge_like_mapping() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        merged.update(payload)

    try:
        form_dict = request.form.to_dict(flat=False)
    except Exception:
        form_dict = dict(request.form or {})
    for k, v in form_dict.items():
        merged.setdefault(k, v if isinstance(v, list) else [v])

    try:
        args_dict = request.args.to_dict(flat=False)
    except Exception:
        args_dict = dict(request.args or {})
    for k, v in args_dict.items():
        merged.setdefault(k, v if isinstance(v, list) else [v])

    return merged


def _bag_list_to_map(bag_list: Iterable[Tuple[float, float, int]]) -> Dict[Tuple[float, float], int]:
    bag_map: Dict[Tuple[float, float], int] = {}
    for w, h, cnt in bag_list:
        key = (float(w), float(h))
        bag_map[key] = bag_map.get(key, 0) + int(cnt)
    return bag_map


def _extract_grid_from_payload(p: Dict[str, Any]) -> Tuple[float, float, int, int]:
    W_ft = float(p.get("W_ft") or 0.0)
    H_ft = float(p.get("H_ft") or 0.0)
    W = int(p.get("W") or 0)
    H = int(p.get("H") or 0)
    if (not W_ft or not H_ft) and isinstance(p.get("grid_ft"), (list, tuple)) and len(p["grid_ft"]) == 2:
        try:
            W_ft = float(p["grid_ft"][0])
            H_ft = float(p["grid_ft"][1])
        except Exception:
            pass
    return W_ft, H_ft, W, H


def _normalize_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        out = dict(result)
        if "ok" not in out:
            out["ok"] = bool(out.get("svg") or out.get("placed_count") or out.get("placements"))
        if not out.get("ok") and "strategy" not in out:
            out["strategy"] = out.get("reason") or out.get("message") or "No solution (unspecified)."
        return out

    if isinstance(result, (tuple, list)):
        ok_flag: Optional[bool] = None
        payload: Optional[Dict[str, Any]] = None
        placements: Optional[List[Placed]] = None
        string_values: List[str] = []

        seq = list(result)
        for elem in seq:
            if isinstance(elem, bool) and ok_flag is None:
                ok_flag = elem
            elif isinstance(elem, str):
                string_values.append(elem)
            elif isinstance(elem, dict) and payload is None:
                payload = elem
            elif isinstance(elem, (list, tuple)) and placements is None:
                maybe_list = list(elem)
                if not maybe_list or all(isinstance(p, Placed) for p in maybe_list):
                    placements = maybe_list  # type: ignore[assignment]

        if payload is None:
            for elem in seq:
                if isinstance(elem, (tuple, list)):
                    for sub in elem:
                        if isinstance(sub, dict):
                            payload = sub
                            break
                    if payload is not None:
                        break

        out = dict(payload or {})

        if placements is None and isinstance(out.get("placements"), (list, tuple)):
            maybe_list = list(out["placements"])  # type: ignore[index]
            if not maybe_list or all(isinstance(p, Placed) for p in maybe_list):
                placements = maybe_list  # type: ignore[assignment]

        if placements is not None:
            out["placements"] = placements

        if ok_flag is None:
            ok_flag = bool(out.get("svg") or out.get("placed_count") or out.get("placements"))

        out["ok"] = bool(ok_flag)

        strategy_token: Optional[str] = None
        reason_text: Optional[str] = None
        if string_values:
            if len(string_values) == 1:
                if out["ok"]:
                    strategy_token = string_values[0]
                else:
                    reason_text = string_values[0]
            else:
                strategy_token = string_values[0]
                reason_text = string_values[-1]
                if not out["ok"] and strategy_token and strategy_token.lower() == "error":
                    strategy_token = None

        if out["ok"]:
            if strategy_token and "strategy" not in out:
                out["strategy"] = strategy_token
        else:
            fallback_reason = out.get("reason") or reason_text
            if isinstance(fallback_reason, str) and fallback_reason:
                out["strategy"] = fallback_reason
                out.setdefault("reason", fallback_reason)
            elif string_values:
                tail = string_values[-1]
                if tail and tail.lower() != "error":
                    out["strategy"] = tail
                else:
                    out["strategy"] = "No solution (unspecified)."
            else:
                out.setdefault("strategy", "No solution (unspecified).")

        return out

    return {"ok": False, "strategy": f"unexpected result from orchestrator: {type(result).__name__}"}


def _call_orchestrator_forgiving(
    bag_map: Dict[Tuple[float, float], int],
    W_ft: Optional[float],
    H_ft: Optional[float],
) -> Any:
    attempts: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
    if W_ft and H_ft:
        attempts.extend([
            ("kwargs bag_ft+grid_ft", (), {"bag_ft": bag_map, "grid_ft": (W_ft, H_ft)}),
            ("kwargs bag+grid_ft",   (), {"bag": bag_map, "grid_ft": (W_ft, H_ft)}),
            ("pos (bag,(W,H))",      (bag_map, (W_ft, H_ft)), {}),
            ("pos (bag,W,H)",        (bag_map, W_ft, H_ft), {}),
        ])
    attempts.extend([
        ("kwargs bag_ft", (), {"bag_ft": bag_map}),
        ("kwargs bag",    (), {"bag": bag_map}),
        ("pos (bag,)",    (bag_map,), {}),
    ])
    last_err: Optional[Exception] = None
    for _label, args, kwargs in attempts:
        try:
            return solve_orchestrator(*args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No valid orchestrator call pattern found.")


def _finalize_solver_progress(ok_flag: bool, strategy_text: str) -> None:
    """Write the terminal solver status without clobbering failure states."""

    set_status("Solved" if ok_flag else "error")
    set_done(ok_flag, reason=strategy_text)


def _start_overlay_spinner(t0: float) -> threading.Event:
    stop_evt = threading.Event()

    def _snapshot_fields() -> Dict[str, str]:
        try:
            snap = progress_json()
        except Exception:
            return {}
        normalize = lambda v: "" if v is None else str(v)
        return {
            "phase": normalize(snap.get("phase")),
            "phase_total": normalize(snap.get("phase_total")),
            "attempt": normalize(snap.get("attempt")),
            "grid": normalize(snap.get("grid")),
        }

    baseline: Optional[Dict[str, str]] = None

    def _solver_started(current: Dict[str, str]) -> bool:
        if not baseline:
            return False
        # Any change away from the placeholder bootstrap values means the
        # orchestrator has started publishing real progress.
        if current.get("phase") and current.get("phase") != baseline.get("phase"):
            return True
        if current.get("phase_total") and current.get("phase_total") != baseline.get("phase_total"):
            return True
        if current.get("attempt") and current.get("attempt") != baseline.get("attempt"):
            return True
        grid = current.get("grid")
        if grid and grid not in {baseline.get("grid"), "—"}:
            return True
        return False

    def _run():
        nonlocal baseline
        max_pct = 0.0
        while not stop_evt.is_set():
            if stop_evt.wait(0.5):
                break
            fields = _snapshot_fields()
            if not baseline and fields:
                baseline = fields
            elif fields and _solver_started(fields):
                break
            try:
                dt = time.time() - t0
                pct = max(5.0, min(35.0, (dt / 45.0) * 60.0))
                if pct > max_pct:
                    max_pct = pct
                set_progress_pct(max_pct)
            except Exception:
                continue
        stop_evt.set()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop_evt


@app.route("/solve", methods=["POST"])
def solve():
    # Reset & start progress service
    progress_reset()
    progress_start()

    # Make sure numeric phase fields are never blank
    set_status("Solving")
    set_phase(1)          # current phase (int)
    set_phase_total(6)    # total phases (int) — safe default
    set_attempt("")
    set_grid("")
    set_progress_pct(0)
    set_coverage_pct(None)
    set_best_used(0)

    t0 = time.time()
    spinner_stop = _start_overlay_spinner(t0)

    like = _merge_like_mapping()
    bag_list, decoded, err = parse_demand(like)
    try:
        set_demand_count(sum(n for _, _, n in bag_list) if bag_list else 0)
    except Exception:
        set_demand_count(0)

    if err or not bag_list:
        seen_keys = ", ".join(list(like.keys())[:8]) or "—"
        reason = f"Bad demand: {err or 'nothing parsed from request'} (saw keys: {seen_keys})"
        set_status("error"); set_done(True, reason=reason)
        spinner_stop.set()
        LAST_RESULT.update({
            "ok": False,
            "strategy": reason,
            "W_ft": 0.0, "H_ft": 0.0, "W": 0, "H": 0,
            "edge_label": "None",
            "placed_count": 0,
            "demand_count": 0,
            "elapsed_str": _fmt_elapsed(time.time() - t0),
            "svg": "",
            "demand_items": fmt_decoded_items(decoded),
            "coords_filename": COORDS_FILENAME,
            "layout_filename": LAYOUT_FILENAME,
        })
        set_result_url(url_for("result_latest"))
        return render_template("result.html", **LAST_RESULT)

    bag_map = _bag_list_to_map(bag_list)

    W_ft = H_ft = None
    grid_ft_val = like.get("grid_ft")
    if isinstance(grid_ft_val, (list, tuple)) and len(grid_ft_val) == 2:
        try:
            W_ft, H_ft = float(grid_ft_val[0]), float(grid_ft_val[1])
        except Exception:
            W_ft = H_ft = None
    else:
        for wk, hk in (("grid_ft_w", "grid_ft_h"), ("attempt_w", "attempt_h")):
            wv = like.get(wk, [])
            hv = like.get(hk, [])
            if isinstance(wv, list) and wv and isinstance(hv, list) and hv:
                try:
                    W_ft, H_ft = float(wv[0]), float(hv[0])
                    break
                except Exception:
                    W_ft = H_ft = None

    if W_ft and H_ft:
        set_attempt(f"{W_ft:g} × {H_ft:g} ft")
        set_grid(f"{W_ft:g} × {H_ft:g} ft")

    try:
        result_raw: Any = _call_orchestrator_forgiving(bag_map, W_ft, H_ft)
    except Exception as e:
        reason = f"orchestrator exception: {type(e).__name__}: {e}"
        set_status("error"); set_done(True, reason=reason)
        spinner_stop.set()
        LAST_RESULT.update({
            "ok": False,
            "strategy": reason,
            "W_ft": 0.0, "H_ft": 0.0, "W": 0, "H": 0,
            "edge_label": "None",
            "placed_count": 0,
            "demand_count": sum(n for _, _, n in bag_list),
            "elapsed_str": _fmt_elapsed(time.time() - t0),
            "svg": "",
            "demand_items": fmt_decoded_items(decoded),
            "coords_filename": COORDS_FILENAME,
            "layout_filename": LAYOUT_FILENAME,
        })
        set_result_url(url_for("result_latest"))
        return render_template("result.html", **LAST_RESULT)

    result = _normalize_result(result_raw)

    rW_ft, rH_ft, rW, rH = _extract_grid_from_payload(result)
    if rW_ft and rH_ft:
        set_grid(f"{rW_ft:g} × {rH_ft:g} ft")

    strategy_text = result.get("strategy") or result.get("reason") or "No solution (unspecified)."
    ok_flag = bool(result.get("ok"))
    if ok_flag and result.get("note"):
        strategy_text = str(result["note"])
    _finalize_solver_progress(ok_flag, strategy_text)
    set_elapsed(time.time() - t0)
    spinner_stop.set()

    placements_list = list(result.get("placements") or [])
    svg_markup = result.get("svg", "")
    coords_name = COORDS_FILENAME
    layout_name = LAYOUT_FILENAME
    grid_label_text = None

    if rW_ft and rH_ft:
        if rW and rH:
            grid_label_text = f"{rW_ft:g} × {rH_ft:g} ft ({rW} × {rH} cells)"
        else:
            grid_label_text = f"{rW_ft:g} × {rH_ft:g} ft"
    elif rW and rH:
        grid_label_text = f"{rW} × {rH} cells"

    if placements_list:
        try:
            svg_markup, legend_html = render_result(placements_list, int(result.get("W") or 0), int(result.get("H") or 0))
        except Exception:
            legend_html = ""
        else:
            try:
                coords_path = write_coords(placements_list, int(result.get("W") or 0), int(result.get("H") or 0), BASE_DIR)
                coords_name = os.path.basename(coords_path) or COORDS_FILENAME
            except Exception:
                coords_name = COORDS_FILENAME
            try:
                layout_path = write_layout_view_html(
                    svg_markup, legend_html, BASE_DIR, grid_label=grid_label_text
                )
                layout_name = os.path.basename(layout_path) or LAYOUT_FILENAME
            except Exception:
                layout_name = LAYOUT_FILENAME

    LAST_RESULT.update({
        "ok": ok_flag,
        "strategy": strategy_text,
        "W_ft": rW_ft,
        "H_ft": rH_ft,
        "W": rW,
        "H": rH,
        "edge_label": result.get("edge_label", "None"),
        "placed_count": result.get("placed_count", 0),
        "demand_count": sum(n for _, _, n in bag_list),
        "elapsed_str": _fmt_elapsed(time.time() - t0),
        "svg": svg_markup,
        "demand_items": fmt_decoded_items(decoded),
        "coords_filename": coords_name,
        "layout_filename": layout_name,
    })
    set_result_url(url_for("result_latest"))
    return render_template("result.html", **LAST_RESULT)


@app.route("/download/coords")
def download_coords():
    return send_from_directory(COORDS_DIR, COORDS_FILENAME, as_attachment=True)


@app.route("/download/html")
def download_html():
    return send_from_directory(LAYOUT_DIR, LAYOUT_FILENAME, as_attachment=True)


@app.route("/progress3")
def progress3():
    return jsonify(progress_json())


if __name__ == "__main__":
    progress_start()
    app.run(debug=False)
