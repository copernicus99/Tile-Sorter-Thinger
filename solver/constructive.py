# solver/constructive.py
from typing import Dict, List, Tuple, Optional
from tiles import NAME_TO_RECT_DIMS
from models import Placed, Rect, ft_to_cells, cells_to_ft

def quick_banded_fill_width10(demand: Dict[str, int]) -> Tuple[Optional[List[Placed]], int, int]:
    """
    Full-width 10 ft banded constructive solver.

    If the given tile set can be arranged in horizontal bands that exactly tile a 10 ft width
    (i.e., each tile width divides 10 ft, possibly after rotation), and the total area is an
    exact multiple of 10, this constructs a placement instantly.

    Returns:
        (placed, Wcells, Hcells) on success, or (None, 0, 0) if not applicable.
    """
    W_ft = 10.0

    # Build a (width_ft, height_ft, count, name) list, preferring shorter height for banding.
    items: List[List[object]] = []
    for name, cnt in demand.items():
        w_c, h_c = NAME_TO_RECT_DIMS[name]
        w_ft, h_ft = cells_to_ft(w_c), cells_to_ft(h_c)
        # Prefer orientation with smaller side as band height.
        if h_ft > w_ft:
            w_ft, h_ft = h_ft, w_ft
        items.append([w_ft, h_ft, int(cnt), name])

    # Each band's tile width must divide 10 ft
    for w_ft, _, cnt, _ in items:
        if cnt > 0 and abs(W_ft % float(w_ft)) > 1e-9:
            return None, 0, 0

    # Total height must be total_area / 10
    total_area = 0.0
    for w_ft, h_ft, cnt, _ in items:
        total_area += float(w_ft) * float(h_ft) * int(cnt)
    if abs(total_area % W_ft) > 1e-9:
        return None, 0, 0
    target_H_ft = total_area / W_ft

    # Greedy tallest-bands-first
    items.sort(key=lambda t: float(t[1]), reverse=True)

    placed: List[Placed] = []
    y = 0.0
    remaining = target_H_ft

    for i, (w_ft, h_ft, cnt, name) in enumerate(items):
        w_ft = float(w_ft); h_ft = float(h_ft); cnt = int(cnt)
        across = int(round(W_ft / w_ft)) if w_ft > 0 else 0  # tiles per full row
        max_rows = (cnt // across) if across else 0
        if max_rows <= 0:
            continue

        usable_rows = min(max_rows, int((remaining + 1e-9) // h_ft))
        for _ in range(usable_rows):
            x = 0.0
            for __ in range(across):
                placed.append(
                    Placed(
                        ft_to_cells(x),
                        ft_to_cells(y),
                        Rect(ft_to_cells(w_ft), ft_to_cells(h_ft), name)  # keep label in chosen orientation
                    )
                )
                x += w_ft
            items[i][2] = int(items[i][2]) - across
            y += h_ft
            remaining = round(target_H_ft - y, 6)
            if remaining <= 1e-9:
                break
        if remaining <= 1e-9:
            break

    if remaining > 1e-6:
        # Couldn’t reach exact target height with full rows
        return None, 0, 0

    return placed, ft_to_cells(W_ft), ft_to_cells(target_H_ft)
