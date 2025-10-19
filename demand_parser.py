# demand_parser.py
import re
from typing import Any, Dict, Optional, Tuple

PARSER_SOURCE = __file__

_NUM = r"\d+(?:[._,]\d+)?"
_SIZE_RE = re.compile(
    rf"(?P<w>{_NUM})\s*[x×]\s*(?P<h>{_NUM})",
    re.IGNORECASE,
)

def _to_float(tok: str) -> float:
    return float(tok.replace("_", ".").replace(",", "."))

def _extract_size(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    m = _SIZE_RE.search(s)
    if not m:
        return None
    w = _to_float(m.group("w"))
    h = _to_float(m.group("h"))
    return (w, h)

def _normalize_key_for_size(key: str) -> str:
    k = (key or "").strip()

    for pre in ("q_", "qty_", "quantity_", "count_", "cnt_"):
        if k.lower().startswith(pre):
            k = k[len(pre):]
            break

    k = k.replace("×", "x")
    k = k.replace(" ", "")
    k = k.replace("_x_", "x")
    k = k.replace("__", "_")
    return k

def parse_demand(form_data: Dict[str, Any]):
    """
    Parse the posted form into { (width_ft, height_ft): count }.
    Accepts sizes in either the key or the value; count must be numeric.
    """
    bag: Dict[Tuple[float, float], int] = {}
    if not isinstance(form_data, dict):
        return (False, {})

    for raw_k, raw_v in form_data.items():
        if raw_v is None:
            continue

        v_str = str(raw_v).strip()
        if v_str == "":
            continue

        try:
            count = int(float(v_str))
        except Exception:
            count = None

        nk = _normalize_key_for_size(str(raw_k))
        size = _extract_size(nk)

        if size is None:
            size = _extract_size(v_str.replace("×", "x"))

        if size is None:
            continue
        if count is None or count <= 0:
            continue

        w, h = size
        key = (float(w), float(h))
        bag[key] = bag.get(key, 0) + int(count)

    if not bag:
        return (False, {})

    return (True, bag)
