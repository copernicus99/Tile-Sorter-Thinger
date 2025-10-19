# tiles.py — robust demand parser
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Tuple, Optional

Float = float
BagItem = Tuple[Float, Float, int]  # (w_ft, h_ft, count)

_NUM = r"(?:\d+(?:\.\d+)?)"
# Accept keys like qty_2x3, tile_2x2.5, 2x3, size-2x3, count[2x3], row_2x3_qty, etc.
_ANY_KEY_SIZE_RE = re.compile(rf"(?P<w>{_NUM})\s*[xX]\s*(?P<h>{_NUM})")


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _as_listish(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _getlist(container: Any, key: str) -> List[Any]:
    if container is None:
        return []
    # merged mapping uses lists already
    if isinstance(container, dict) and key in container:
        return _as_listish(container[key])
    # fallback for MultiDict-like
    if hasattr(container, "getlist"):
        try:
            return list(container.getlist(key))
        except Exception:
            return []
    return []


def _append(bag: List[BagItem], decoded: List[Tuple[str, int]], w: float, h: float, n: int):
    bag.append((w, h, n))
    decoded.append((f"{w:g}x{h:g}", n))


def parse_demand(form_like: Any) -> Tuple[List[BagItem], List[Tuple[str, int]], Optional[str]]:
    """
    Return (bag_ft, decoded_items, error_message_or_None).
    Accepts many shapes (JSON or form). See app.py for how we build form_like.
    """
    bag: List[BagItem] = []
    decoded: List[Tuple[str, int]] = []

    if not form_like:
        return [], [], "nothing parsed from request"

    # --- Shape 1: explicit JSON tiles list --------------------------------
    if isinstance(form_like, dict) and isinstance(form_like.get("tiles"), list):
        for t in form_like["tiles"]:
            w = _to_float(t.get("w"))
            h = _to_float(t.get("h"))
            n = _to_int(t.get("count"))
            if w and h and n and n > 0:
                _append(bag, decoded, float(w), float(h), int(n))
        if bag:
            return bag, decoded, None

    # --- Shape 2: parallel arrays in JSON ---------------------------------
    if isinstance(form_like, dict) and {"w", "h", "count"} <= set(form_like.keys()):
        wL = _as_listish(form_like.get("w"))
        hL = _as_listish(form_like.get("h"))
        nL = _as_listish(form_like.get("count"))
        for w, h, n in zip(wL, hL, nL):
            wf, hf, ni = _to_float(w), _to_float(h), _to_int(n)
            if wf and hf and ni and ni > 0:
                _append(bag, decoded, float(wf), float(hf), int(ni))
        if bag:
            return bag, decoded, None

    # --- Shape 3: form arrays ---------------------------------------------
    for wK, hK, nK in (
        ("w_ft[]", "h_ft[]", "count[]"),
        ("w[]", "h[]", "n[]"),
        ("w_ft", "h_ft", "count"),
    ):
        wL, hL, nL = _getlist(form_like, wK), _getlist(form_like, hK), _getlist(form_like, nK)
        if not (wL and hL and nL):
            continue
        for w, h, n in zip(wL, hL, nL):
            wf, hf, ni = _to_float(w), _to_float(h), _to_int(n)
            if wf and hf and ni and ni > 0:
                _append(bag, decoded, float(wf), float(hf), int(ni))
        if bag:
            return bag, decoded, None

    # --- Shape 4: per-size keys anywhere in the key name ------------------
    def _iter_items():
        if isinstance(form_like, dict):
            for k, v in form_like.items():
                yield k, v
        elif hasattr(form_like, "items"):
            for k, v in form_like.items():
                yield k, v

    for k, v in _iter_items():
        m = _ANY_KEY_SIZE_RE.search(str(k))
        if not m:
            continue
        wf = _to_float(m.group("w"))
        hf = _to_float(m.group("h"))
        if not (wf and hf):
            continue
        # value may be list / tuple / scalar
        vv = v[0] if isinstance(v, (list, tuple)) and v else v
        ni = _to_int(vv)
        if ni and ni > 0:
            _append(bag, decoded, float(wf), float(hf), int(ni))

    if bag:
        return bag, decoded, None

    return [], [], "nothing parsed from request"


def fmt_decoded_items(decoded: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return sorted(decoded, key=lambda t: (float(t[0].split("x")[0]), float(t[0].split("x")[1])))
